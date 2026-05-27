"""Generate fixed-layout calibration reports (coverage, SBC, PPC, prior audit)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..dataio.loader import load_observation_data
from ..inference import BilbyRunner
from ..inference.evidence import prior_sensitivity_audit
from ..likelihoods import GWLikelihood
from ..models import get_model
from ..simulators import get_simulator
from ..simulators.base import SimData
from ..validation.gw_diagnostics import run_gw_diagnostics, write_gw_diagnostics
from ..validation.injection import InjectionRecovery
from ..validation.ppc import PosteriorPredictiveCheck
from ..validation.sbc import SBCRunner

logger = logging.getLogger(__name__)

# Fixed artifact paths (relative to report root)
CALIBRATION_ARTIFACTS = {
    "index": "index.md",
    "report": "report.json",
    "config": "config.json",
    "diagnostics": "diagnostics.json",
    "coverage_csv": "coverage.csv",
    "coverage_png": "coverage.png",
    "prior_audit": "prior_audit.csv",
    "sbc_summary": "sbc/sbc_summary.csv",
    "sbc_combined": "sbc/rank_histograms.png",
    "ppc_summary": "ppc/ppc_summary.csv",
    "mock_strain_csv": "mock_vs_real/strain_rms_comparison.csv",
    "mock_lnz_csv": "mock_vs_real/lnZ_comparison.csv",
    "mock_diag_mock": "mock_vs_real/diagnostics_mock.json",
    "mock_diag_gwosc": "mock_vs_real/diagnostics_gwosc.json",
}

COMPARE_MODELS = ("null", "bounce", "bh_ringdown")


def observation_to_simdata(obs: Any, channel: str = "gw") -> SimData:
    """Convert GW observation dict (or SimData) to SimData for PPC."""
    if isinstance(obs, SimData):
        return obs
    strain = obs["strain"] if isinstance(obs, dict) else obs.data
    meta = (
        {k: v for k, v in obs.items() if k != "strain"}
        if isinstance(obs, dict)
        else dict(getattr(obs, "metadata", {}))
    )
    return SimData(channel=channel, data=strain, metadata=meta)


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
    profile: str = "quick",
) -> tuple[Path, dict[str, Any]]:
    """Write full calibration artifact tree under ``outdir``.

    Returns
    -------
    root : Path
    report : dict — parsed ``report.json`` contents
    """
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

    config = {
        "model": model,
        "channel": channel,
        "data_source": data_source,
        "event": event,
        "n_injections": n_injections,
        "n_sbc": n_sbc,
        "n_ppc": n_ppc,
        "nlive": nlive,
        "seed": seed,
        "reference_amplitude": reference_amplitude,
        "likelihood_mode": likelihood_mode,
        "force_toy": force_toy,
        "profile": profile,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (root / CALIBRATION_ARTIFACTS["config"]).write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    runner = BilbyRunner(
        nlive=nlive,
        outdir=str(root / "inference"),
        seed=seed,
        force_toy=force_toy,
    )
    model_obj = get_model(model)
    sim = get_simulator(channel)
    ll = GWLikelihood(model, use_full_likelihood=not use_mf)

    # --- mock vs real ---
    mvr_section = _run_mock_vs_real(
        root,
        channel=channel,
        model=model,
        event=event,
        seed=seed,
        context=context,
        reference_amplitude=reference_amplitude,
        use_mf=use_mf,
        runner=runner,
    )

    # --- coverage (injection recovery) on chosen data source ---
    ir = InjectionRecovery(
        simulator=sim, runner=runner, n_injections=n_injections, rng_seed=seed
    )
    ir_result = ir.run_injections(model_obj, ll, context, save_dir=root / "injections")
    cov_df = ir_result.summary()
    cov_df.to_csv(root / CALIBRATION_ARTIFACTS["coverage_csv"], index=False)
    _plot_coverage(cov_df, root / CALIBRATION_ARTIFACTS["coverage_png"])
    coverage_section = _evaluate_coverage(cov_df)

    # --- SBC ---
    sbc_dir = root / "sbc"
    sbc_dir.mkdir(exist_ok=True)
    sbc = SBCRunner(n_simulations=n_sbc, n_posterior_samples=30, rng_seed=seed)
    sbc_result = sbc.run(model_obj, sim, ll, runner, context)
    sbc_result.summary().to_csv(sbc_dir / "sbc_summary.csv", index=False)
    sbc_paths: list[str] = []
    try:
        sbc_result.plot(save_path=str(sbc_dir / "rank_histograms.png"))
        for p in sbc_result.plot_all(sbc_dir):
            sbc_paths.append(str(p.relative_to(root)).replace("\\", "/"))
    except Exception as exc:
        logger.warning("SBC plot failed: %s", exc)
    sbc_section = _evaluate_sbc(sbc_result)

    # --- PPC (same observation as fit) ---
    obs_main, _ = load_observation_data(
        data_source,
        channel,
        event=event,
        inject_model=model,
        seed=seed,
        context=context,
        allow_mock_fallback=(data_source == "gwosc"),
        reference_amplitude=reference_amplitude,
    )
    obs_sim = observation_to_simdata(obs_main, channel)
    fit_result = runner.run(ll, obs_main, context, model_obj, label="ppc_fit")
    ppc_dir = root / "ppc"
    ppc_dir.mkdir(exist_ok=True)
    ppc = PosteriorPredictiveCheck(sim, n_replicates=n_ppc, rng_seed=seed)
    ppc_result = ppc.run(obs_sim, fit_result, context)
    ppc_result.summary().to_csv(ppc_dir / "ppc_summary.csv", index=False)
    ppc_paths: list[str] = []
    try:
        ppc_result.plot(save_path=str(ppc_dir / "ppc_summary.png"))
        for p in ppc_result.plot_all(ppc_dir):
            ppc_paths.append(str(p.relative_to(root)).replace("\\", "/"))
    except Exception as exc:
        logger.warning("PPC plot failed: %s", exc)
    ppc_section = _evaluate_ppc(ppc_result)

    # --- prior sensitivity ---
    audit_df = prior_sensitivity_audit(
        model_obj, ll, obs_main, context, runner, n_audits=3,
    )
    audit_df.to_csv(root / CALIBRATION_ARTIFACTS["prior_audit"], index=False)
    audit_section = _evaluate_prior_audit(audit_df)

    obs_dict = obs_main if isinstance(obs_main, dict) else {
        "strain": obs_main.data,
        **getattr(obs_main, "metadata", {}),
    }
    diag_main = run_gw_diagnostics(
        obs_dict,
        context,
        ["null", model, "bh_ringdown"],
        seed=seed,
        use_mf=use_mf,
    )
    write_gw_diagnostics(root / CALIBRATION_ARTIFACTS["diagnostics"], diag_main)

    sections = {
        "coverage": coverage_section,
        "sbc": sbc_section,
        "ppc": ppc_section,
        "prior_audit": audit_section,
        "mock_vs_real": mvr_section,
    }
    overall = (
        "PASS"
        if all(s.get("pass") for s in sections.values() if s.get("pass") is not None)
        else "FAIL"
    )
    report = {
        "overall": overall,
        "sections": sections,
        "artifacts": CALIBRATION_ARTIFACTS,
        "plots": {"sbc": sbc_paths, "ppc": ppc_paths},
    }
    (root / CALIBRATION_ARTIFACTS["report"]).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    _write_index_md(
        root,
        report=report,
        cov_df=cov_df,
        sbc_paths=sbc_paths,
        ppc_paths=ppc_paths,
        config=config,
    )
    return root, report


def _run_mock_vs_real(
    root: Path,
    *,
    channel: str,
    model: str,
    event: str | None,
    seed: int,
    context: dict[str, Any],
    reference_amplitude: bool,
    use_mf: bool,
    runner: BilbyRunner,
) -> dict[str, Any]:
    mock_dir = root / "mock_vs_real"
    mock_dir.mkdir(exist_ok=True)
    strain_rows: list[dict[str, Any]] = []
    lnz_rows: list[dict[str, Any]] = []
    gwosc_ok = False

    for src, ev in [("mock", None), ("gwosc", event or "GW150914")]:
        try:
            obs, prov = load_observation_data(
                src,
                channel,
                event=ev,
                inject_model=model,
                seed=seed,
                context=context,
                allow_mock_fallback=(src == "gwosc"),
                reference_amplitude=reference_amplitude,
            )
            obs_d = obs if isinstance(obs, dict) else {
                "strain": obs.data,
                **getattr(obs, "metadata", {}),
            }
            diag = run_gw_diagnostics(
                obs_d,
                context,
                list(COMPARE_MODELS),
                seed=seed,
                use_mf=use_mf,
            )
            write_gw_diagnostics(mock_dir / f"diagnostics_{src}.json", diag)
            strain_rows.append({
                "source": src,
                "status": "ok",
                "actual": prov.actual_source,
                "strain_rms_used": diag.get("strain_rms_used"),
                "null_lnL": diag.get("null_lnL"),
            })
            if src == "gwosc":
                gwosc_ok = True

            for mname in COMPARE_MODELS:
                m_obj = get_model(mname)
                m_ll = GWLikelihood(mname, use_full_likelihood=not use_mf)
                try:
                    res = runner.run(
                        m_ll, obs, context, m_obj, label=f"lnz_{src}_{mname}"
                    )
                    lnz_rows.append({
                        "source": src,
                        "model": mname,
                        "status": "ok",
                        "ln_Z": res.log_evidence,
                        "approximate": res.metadata.get("is_approximate_evidence", False),
                    })
                except Exception as exc:
                    lnz_rows.append({
                        "source": src,
                        "model": mname,
                        "status": "error",
                        "error": str(exc),
                    })
        except Exception as exc:
            strain_rows.append({
                "source": src,
                "status": "skipped" if src == "gwosc" else "error",
                "error": str(exc),
            })
            for mname in COMPARE_MODELS:
                lnz_rows.append({
                    "source": src,
                    "model": mname,
                    "status": "skipped",
                    "error": str(exc),
                })

    pd.DataFrame(strain_rows).to_csv(
        mock_dir / "strain_rms_comparison.csv", index=False
    )
    pd.DataFrame(lnz_rows).to_csv(mock_dir / "lnZ_comparison.csv", index=False)

    return {
        "pass": None,
        "gwosc_available": gwosc_ok,
        "note": "Informational; GWOSC may be skipped without failing overall report",
    }


def _plot_coverage(cov_df: pd.DataFrame, path: Path) -> None:
    if cov_df.empty or "coverage_90pct" not in cov_df.columns:
        return
    fig, ax = plt.subplots(figsize=(max(6, len(cov_df) * 0.5), 4))
    params = cov_df["parameter"].astype(str)
    vals = cov_df["coverage_90pct"].values
    colors = [
        "seagreen" if cov_df.iloc[i].get("coverage_ok", False) else "coral"
        for i in range(len(cov_df))
    ]
    ax.bar(params, vals, color=colors)
    ax.axhline(0.9, color="black", ls="--", label="nominal 90%")
    ax.axhspan(0.8, 1.0, alpha=0.1, color="green", label="PASS band [0.8, 1.0]")
    ax.set_ylabel("Coverage (90% CI)")
    ax.set_xlabel("Parameter")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _evaluate_coverage(cov_df: pd.DataFrame) -> dict[str, Any]:
    if cov_df.empty or "coverage_ok" not in cov_df.columns:
        return {"pass": False, "n_ok": 0, "n_total": 0}
    n_ok = int(cov_df["coverage_ok"].sum())
    n_total = len(cov_df)
    return {
        "pass": n_ok == n_total and n_total > 0,
        "n_ok": n_ok,
        "n_total": n_total,
        "threshold": "coverage in [0.8, 1.0]",
    }


def _evaluate_sbc(sbc_result: Any) -> dict[str, Any]:
    calibrated = getattr(sbc_result, "calibrated", {})
    n_ok = sum(1 for v in calibrated.values() if v)
    n_total = len(calibrated)
    return {
        "pass": n_ok == n_total and n_total > 0,
        "n_ok": n_ok,
        "n_total": n_total,
        "threshold": "KS p > 0.05",
        "pvalues": getattr(sbc_result, "uniformity_pvalues", {}),
    }


def _evaluate_ppc(ppc_result: Any) -> dict[str, Any]:
    df = ppc_result.summary()
    if df.empty:
        return {"pass": False, "n_ok": 0, "n_total": 0}
    n_ok = int((~df["suspicious"]).sum()) if "suspicious" in df.columns else 0
    n_total = len(df)
    return {
        "pass": n_ok == n_total and n_total > 0,
        "n_ok": n_ok,
        "n_total": n_total,
        "threshold": "p-value in (0.05, 0.95)",
    }


def _evaluate_prior_audit(audit_df: pd.DataFrame) -> dict[str, Any]:
    if audit_df.empty or "sensitive" not in audit_df.columns:
        return {"pass": False, "n_ok": 0, "n_total": 0}
    n_bad = int(audit_df["sensitive"].sum())
    n_total = len(audit_df)
    return {
        "pass": n_bad == 0,
        "n_ok": n_total - n_bad,
        "n_total": n_total,
        "threshold": "|delta_ln_Z| < 1",
    }


def _write_index_md(
    root: Path,
    *,
    report: dict[str, Any],
    cov_df: pd.DataFrame,
    sbc_paths: list[str],
    ppc_paths: list[str],
    config: dict[str, Any],
) -> None:
    ts = config.get("generated_utc", datetime.now(timezone.utc).isoformat())
    overall = report.get("overall", "FAIL")
    sections = report.get("sections", {})

    def row(name: str, key: str) -> str:
        s = sections.get(key, {})
        status = "PASS" if s.get("pass") else "FAIL"
        return f"| {name} | {status} | {s.get('n_ok', '—')} / {s.get('n_total', '—')} | {s.get('threshold', '')} |"

    sbc_imgs = "\n".join(f"![SBC]({p})" for p in sbc_paths[:6])
    ppc_imgs = "\n".join(f"![PPC]({p})" for p in ppc_paths[:6])
    if len(sbc_paths) > 6:
        sbc_imgs += f"\n\n_({len(sbc_paths) - 6} more in `sbc/`.)_"
    if len(ppc_paths) > 6:
        ppc_imgs += f"\n\n_({len(ppc_paths) - 6} more in `ppc/`.)_"

    mvr = sections.get("mock_vs_real", {})
    gwosc_note = (
        "GWOSC loaded OK."
        if mvr.get("gwosc_available")
        else "GWOSC skipped or failed — see `mock_vs_real/strain_rms_comparison.csv`."
    )

    text = f"""# WhiteSearch Calibration Report

**Overall: {overall}**

Generated: {ts}

| Section | Status | OK / Total | Criterion |
|---------|--------|------------|-----------|
{row("Coverage", "coverage")}
{row("SBC", "sbc")}
{row("PPC", "ppc")}
{row("Prior audit", "prior_audit")}
| Mock vs real | INFO | — | {gwosc_note} |

## Configuration

- Model: `{config.get("model")}` | Channel: `{config.get("channel")}`
- Data (coverage): `{config.get("data_source")}` | Profile: `{config.get("profile")}`
- Likelihood: `{config.get("likelihood_mode")}` | force_toy: `{config.get("force_toy")}`
- n_injections={config.get("n_injections")}, n_sbc={config.get("n_sbc")}, nlive={config.get("nlive")}

## Figures

### Coverage

![Coverage](coverage.png)

### SBC rank histograms

{sbc_imgs or "_See `sbc/rank_hist_*.png`._"}

### PPC

{ppc_imgs or "_See `ppc/ppc_*.png`._"}

## Artifacts

| File | Description |
|------|-------------|
| `report.json` | Machine-readable PASS/FAIL |
| `config.json` | Run parameters |
| `coverage.csv`, `coverage.png` | Injection / recovery |
| `sbc/sbc_summary.csv`, `sbc/rank_hist_*.png` | Simulation-based calibration |
| `ppc/ppc_summary.csv`, `ppc/ppc_*.png` | Posterior predictive checks |
| `prior_audit.csv` | Prior sensitivity |
| `mock_vs_real/strain_rms_comparison.csv` | Strain RMS mock vs GWOSC |
| `mock_vs_real/lnZ_comparison.csv` | ln Z by source and model |
| `diagnostics.json` | Likelihood finiteness |

> Development report — use `--profile standard --no-force-toy` for dynesty-backed checks.
"""
    (root / CALIBRATION_ARTIFACTS["index"]).write_text(text, encoding="utf-8")
