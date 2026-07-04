"""GW channel diagnostics for likelihood scale and dynesty readiness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..likelihoods.gw_likelihood import GWLikelihood
from ..likelihoods.gw_units import inner_product_norm, time_to_freq
from ..models import get_model
from ..utils.math_utils import matched_filter_snr


def run_gw_diagnostics(
    data: dict[str, Any],
    context: dict[str, Any],
    models: list[str],
    *,
    n_prior_draws: int = 200,
    seed: int = 0,
    use_mf: bool = False,
    finite_threshold: float | None = None,
) -> dict[str, Any]:
    """Compute diagnostics shared across null / signal models.

    Parameters
    ----------
    finite_threshold : float | None
        Optional absolute lnL cutoff for the frac_finite statistic.  By
        default a draw counts as finite when its lnL is finite and not at
        the likelihood's rejection floor (``ll_min``) — a scale-free rule.
        The previous hard-coded ``-1e5`` cutoff was tuned to mock-scale
        data and classified every real-data draw (lnL ~ -1e6 and below)
        as non-finite regardless of whether the template was accepted.
    """
    strain = np.asarray(data["strain"], dtype=np.float64)
    sr = float(data.get("sample_rate", 4096.0))
    psd = np.asarray(data["psd"], dtype=np.float64)
    dt = 1.0 / sr
    freqs, strain_f, df = time_to_freq(strain, dt)

    out: dict[str, Any] = {
        "strain_rms_raw": data.get("strain_rms_raw"),
        "strain_rms_bp": data.get("strain_rms_bp"),
        "strain_rms_used": float(np.std(strain)),
        "amplitude_scale_applied": data.get("amplitude_scale_applied", 1.0),
        "reference_amplitude": data.get("reference_amplitude", False),
        "t_merger_s": data.get("t_merger"),
        "psd_median": float(np.median(psd[psd > 0])) if np.any(psd > 0) else None,
        "whitening_ok": data.get("preprocess_quality", {}).get("whitening_ok"),
        "models": {},
    }

    ll = GWLikelihood("null", use_full_likelihood=not use_mf)
    out["null_lnL"] = ll.loglike({}, data, context)

    rng = np.random.default_rng(seed)
    for mname in models:
        model = get_model(mname)
        ll_m = GWLikelihood(mname, use_full_likelihood=not use_mf)
        logls: list[float] = []
        snrs: list[float] = []

        for _ in range(n_prior_draws):
            theta = model.sample_prior(rng)
            lp = model.log_prior(theta)
            if not np.isfinite(lp):
                continue
            val = ll_m.loglike(theta, data, context)
            logls.append(val)
            if mname != "null" and len(model.parameter_names) > 0:
                try:
                    times = np.arange(len(strain)) * dt
                    h = ll_m._build_template(
                        theta, times, float(data.get("t_merger", 0.5)),
                        freqs, sr / 2, float(context.get("low_freq_cutoff", 20.0)),
                    )
                    if h is not None:
                        _, hf, _ = time_to_freq(h, dt)
                        snrs.append(matched_filter_snr(hf, strain_f, psd, df))
                except Exception:
                    pass

        if finite_threshold is not None:
            finite = [x for x in logls if np.isfinite(x) and x > finite_threshold]
        else:
            # Scale-free: usable = finite and not the rejection sentinel.
            finite = [x for x in logls if np.isfinite(x) and x != ll_m.ll_min]
        out["models"][mname] = {
            "median_lnL": float(np.median(finite)) if finite else None,
            "frac_finite": len(finite) / max(n_prior_draws, 1),
            "max_mf_snr": float(np.max(snrs)) if snrs else None,
            "n_params": len(model.parameter_names),
        }

    return out


def write_gw_diagnostics(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
