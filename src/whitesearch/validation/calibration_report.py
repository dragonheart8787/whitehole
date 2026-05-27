"""Generate fixed-layout calibration reports (coverage, SBC, PPC, prior audit)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..dataio.loader import load_observation_data
from ..inference import BilbyRunner
from ..inference.evidence import prior_sensitivity_audit
from ..likelihoods import GWLikelihood
from ..models import get_model
from ..simulators import GravitationalWaveSimulator, get_simulator
from ..validation.gw_diagnostics import run_gw_diagnostics, write_gw_diagnostics
from ..validation.injection import InjectionRecovery
from ..validation.ppc import PosteriorPredictiveCheck
from ..validation.sbc import SBCRunner

logger = logging.getLogger(__name__)


def generate_calibration_report(
    outdir: str | Path,
    *,
    model: str = "bounce",
    channel: str = "gw",
    data_source: str = "mock",
    event: str | None = None,
    n_injections: int = 20,
    n_sbc: int = 12,
    n_ppc: int = 50,
    nlive: int = 50,
    seed: int = 42,
    reference_amplitude: bool = False,
    likelihood_mode: str = "mf",
    force_toy: bool = True,
) -> Path:
    """Write full calibration artifact tree under ``outdir``."""
    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    use_mf = likelihood_mode == "mf"

    context = {
        "sample_rate": 4096,
        "duration": 4.0,
        "t_merger": 1.0,
        "low_freq_cutoff": 20.0,
        "rng_seed": seed,
    }

    runner = BilbyRunner(
        nlive=nlive,
        outdir=str(root / "inference"),
        seed=seed,
        force_toy=force_toy,
    )
    model_obj = get_model(model)
    sim = get_simulator(channel)
    ll = GWLikelihood(model, use_full_likelihood=not use_mf)

    # --- diagnostics: mock vs gwosc ---
    mock_dir = root / "mock_vs_real"
    mock_dir.mkdir(exist_ok=True)
    rows = []
    for src, ev in [("mock", None), ("gwosc", event or "GW150914")]:
        try:
            obs, prov = load_observation_data(
                src, channel, event=ev, inject_model=model, seed=seed,
                context=context, allow_mock_fallback=(src == "gwosc"),
                reference_amplitude=reference_amplitude,
            )
            obs_d = obs if isinstance(obs, dict) else {
                "strain": obs.data,
                **getattr(obs, "metadata", {}),
            }
            diag = run_gw_diagnostics(
                obs_d,
                context,
                ["null", "bounce", "bh_ringdown"],
                seed=seed,
                use_mf=use_mf,
            )
            write_gw_diagnostics(mock_dir / f"diagnostics_{src}.json", diag)
            rows.append({
                "source": src,
                "actual": prov.actual_source,
                "strain_rms_used": diag.get("strain_rms_used"),
                "null_lnL": diag.get("null_lnL"),
            })
        except Exception as exc:
            rows.append({"source": src, "error": str(exc)})
    pd.DataFrame(rows).to_csv(mock_dir / "strain_rms_comparison.csv", index=False)

    # --- coverage (injection recovery) ---
    ir = InjectionRecovery(simulator=sim, runner=runner, n_injections=n_injections, rng_seed=seed)
    ir_result = ir.run_injections(model_obj, ll, context, save_dir=root / "injections")
    cov_df = ir_result.summary()
    cov_df.to_csv(root / "coverage.csv", index=False)

    # --- SBC ---
    sbc_dir = root / "sbc"
    sbc_dir.mkdir(exist_ok=True)
    sbc = SBCRunner(n_simulations=n_sbc, n_posterior_samples=30, rng_seed=seed)
    sbc_result = sbc.run(model_obj, sim, ll, runner, context)
    sbc_result.summary().to_csv(sbc_dir / "sbc_summary.csv", index=False)
    try:
        sbc_result.plot(save_path=str(sbc_dir / "rank_histograms.png"))
    except Exception as exc:
        logger.warning("SBC plot failed: %s", exc)

    # --- PPC on mock ---
    obs_mock, _ = load_observation_data(
        "mock", channel, inject_model=model, seed=seed, context=context,
        reference_amplitude=reference_amplitude,
    )
    fit_result = runner.run(ll, obs_mock, context, model_obj, label="ppc_fit")
    ppc_dir = root / "ppc"
    ppc_dir.mkdir(exist_ok=True)
    import numpy as np
    obs_for_ppc = GravitationalWaveSimulator().simulate(
        model_obj.sample_prior(np.random.default_rng(seed)),
        context,
        rng=np.random.default_rng(seed),
    )
    ppc = PosteriorPredictiveCheck(sim, n_replicates=n_ppc, rng_seed=seed)
    ppc_result = ppc.run(obs_for_ppc, fit_result, context)
    ppc_result.summary().to_csv(ppc_dir / "ppc_summary.csv", index=False)
    try:
        ppc_result.plot(save_path=str(ppc_dir / "ppc_summary.png"))
    except Exception as exc:
        logger.warning("PPC plot failed: %s", exc)

    # --- prior sensitivity ---
    audit_df = prior_sensitivity_audit(
        model_obj, ll, obs_mock, context, runner, n_audits=3,
    )
    audit_df.to_csv(root / "prior_audit.csv", index=False)

    obs_dict = obs_mock if isinstance(obs_mock, dict) else {
        "strain": obs_mock.data,
        **getattr(obs_mock, "metadata", {}),
    }
    diag_main = run_gw_diagnostics(
        obs_dict,
        context,
        ["null", model, "bh_ringdown"],
        seed=seed,
        use_mf=use_mf,
    )
    write_gw_diagnostics(root / "diagnostics.json", diag_main)

    _write_index_md(root, cov_df, sbc_result, audit_df, reference_amplitude, likelihood_mode)
    return root


def _write_index_md(
    root: Path,
    cov_df: pd.DataFrame,
    sbc_result: Any,
    audit_df: pd.DataFrame,
    reference_amplitude: bool,
    likelihood_mode: str,
) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_cov_ok = int(cov_df["coverage_ok"].sum()) if "coverage_ok" in cov_df.columns else 0
    n_sbc_ok = sum(sbc_result.calibrated.values()) if hasattr(sbc_result, "calibrated") else 0
    n_audit_bad = int(audit_df["sensitive"].sum()) if "sensitive" in audit_df.columns else 0

    text = f"""# WhiteSearch Calibration Report

Generated: {ts}

| Check | Result |
|-------|--------|
| Coverage (90% CI) | {n_cov_ok} / {len(cov_df)} parameters in [0.8, 1.0] |
| SBC (KS p>0.05) | {n_sbc_ok} / {len(getattr(sbc_result, 'calibrated', {}))} parameters |
| Prior sensitivity (|ΔlnZ|<1) | {len(audit_df) - n_audit_bad} / {len(audit_df)} combos OK |
| Likelihood mode | {likelihood_mode} |
| Reference amplitude scaling | {reference_amplitude} |

## Artifacts

- `coverage.csv` — injection / recovery
- `sbc/sbc_summary.csv`, `sbc/rank_histograms.png`
- `ppc/ppc_summary.csv`
- `prior_audit.csv`
- `mock_vs_real/` — mock vs GWOSC diagnostics
- `diagnostics.json` — likelihood finiteness on mock data

> Development report — not publication-grade unless run with dynesty and validated data.
"""
    (root / "index.md").write_text(text, encoding="utf-8")
