"""Bayes factor computation, interpretation, and prior sensitivity audit.

Core function: compute_bayes_factor(results_dict)
Audit function: prior_sensitivity_audit(model, likelihood, data, context, n_audits)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .bilby_runner import InferenceResult, BilbyRunner
from ..likelihoods.base import BaseLikelihood
from ..models.base import BaseModel, ParameterSpec

logger = logging.getLogger(__name__)


# ── Kass-Raftery evidence scale ───────────────────────────────────────────────

KR_SCALE = [
    (0.0, "not worth mentioning"),
    (1.0, "positive"),
    (3.0, "strong"),
    (5.0, "very strong (BF > 150)"),
]


def interpret_ln_bf(ln_bf: float) -> str:
    label = KR_SCALE[0][1]
    for threshold, desc in KR_SCALE:
        if ln_bf >= threshold:
            label = desc
    return label


# ── Main evidence computation ─────────────────────────────────────────────────

def compute_bayes_factor(
    results: dict[str, InferenceResult],
    wh_model: str,
    null_model: str = "null",
    alt_model: str | None = None,
) -> dict[str, Any]:
    """Compute Bayes factors and apply pass/fail gates.

    Parameters
    ----------
    results : dict[model_name → InferenceResult]
    wh_model : str — name of the white hole model result
    null_model : str — name of the null-hypothesis result
    alt_model : str | None — name of the best astrophysical alternative

    Returns
    -------
    dict with ln_BF values, uncertainties, interpretations, and gate outcomes
    """
    if wh_model not in results:
        raise KeyError(f"White hole model {wh_model!r} not in results.")
    if null_model not in results:
        raise KeyError(f"Null model {null_model!r} not in results.")

    wh = results[wh_model]
    null = results[null_model]

    ln_bf_null = wh.log_evidence - null.log_evidence
    err_null = np.sqrt(wh.log_evidence_err**2 + null.log_evidence_err**2)

    output: dict[str, Any] = {
        "wh_model": wh_model,
        "ln_Z_wh": wh.log_evidence,
        "ln_Z_null": null.log_evidence,
        "ln_BF_vs_null": ln_bf_null,
        "ln_BF_vs_null_err": err_null,
        "interpretation_vs_null": interpret_ln_bf(ln_bf_null),
        "gate_internal_passed": ln_bf_null > 3.0,
        "gate_publication_passed": ln_bf_null > 5.0,
    }

    if alt_model and alt_model in results:
        alt = results[alt_model]
        ln_bf_alt = wh.log_evidence - alt.log_evidence
        err_alt = np.sqrt(wh.log_evidence_err**2 + alt.log_evidence_err**2)
        output.update(
            {
                "alt_model": alt_model,
                "ln_Z_alt": alt.log_evidence,
                "ln_BF_vs_alt": ln_bf_alt,
                "ln_BF_vs_alt_err": err_alt,
                "interpretation_vs_alt": interpret_ln_bf(ln_bf_alt),
                "gate_alt_internal_passed": ln_bf_alt > 1.0,
                "gate_alt_publication_passed": ln_bf_alt > 3.0,
            }
        )

    return output


# ── Prior sensitivity audit ───────────────────────────────────────────────────

def prior_sensitivity_audit(
    model: BaseModel,
    likelihood: BaseLikelihood,
    data: Any,
    context: dict[str, Any],
    runner: BilbyRunner,
    n_audits: int = 5,
    prior_scale_factors: list[float] | None = None,
) -> pd.DataFrame:
    """Run inference under perturbed priors and report evidence sensitivity.

    For each parameter, inflate / deflate the prior width by a set of scale
    factors and recompute ln Z.  A stable result should show |Δ ln Z| < 1 nat
    across all perturbations.

    Parameters
    ----------
    prior_scale_factors : list[float]
        Multiplicative factors for prior width.  Default: [0.5, 1.0, 2.0].

    Returns
    -------
    DataFrame with columns: parameter, scale_factor, ln_Z, delta_ln_Z
    """
    if prior_scale_factors is None:
        prior_scale_factors = [0.5, 1.0, 2.0]

    base_result = runner.run(likelihood, data, context, model, label="audit_base")
    rows = []

    for spec in model.parameters():
        for scale in prior_scale_factors:
            perturbed_model = _perturb_prior(model, spec.name, scale)
            try:
                result = runner.run(
                    likelihood, data, context, perturbed_model,
                    label=f"audit_{spec.name}_scale{scale:.2f}",
                )
                delta = result.log_evidence - base_result.log_evidence
            except Exception as exc:
                logger.warning("Prior audit failed for %s scale %.1f: %s", spec.name, scale, exc)
                delta = float("nan")
                result = base_result

            rows.append(
                {
                    "parameter": spec.name,
                    "scale_factor": scale,
                    "ln_Z": result.log_evidence,
                    "delta_ln_Z": delta,
                    "sensitive": abs(delta) > 1.0,
                }
            )

    df = pd.DataFrame(rows)
    n_sensitive = df["sensitive"].sum()
    logger.info(
        "Prior sensitivity audit: %d / %d parameter-scale combinations show |Δ ln Z| > 1 nat",
        n_sensitive,
        len(df),
    )
    return df


def _perturb_prior(
    model: BaseModel,
    param_name: str,
    scale: float,
) -> BaseModel:
    """Return a copy of the model with the named parameter's prior scaled."""
    import copy

    new_model = copy.deepcopy(model)
    for spec in new_model.parameters():
        if spec.name != param_name:
            continue
        kwargs = dict(spec.prior_kwargs)
        if spec.prior_type in ("uniform", "log_uniform"):
            mid = 0.5 * (kwargs["low"] + kwargs["high"])
            half = 0.5 * (kwargs["high"] - kwargs["low"]) * scale
            kwargs["low"] = mid - half
            kwargs["high"] = mid + half
        elif spec.prior_type in ("normal",):
            kwargs["std"] = kwargs["std"] * scale
        spec.prior_kwargs = kwargs
    return new_model


# ── Candidate ranking ─────────────────────────────────────────────────────────

def rank_candidates(
    candidate_results: list[dict[str, Any]],
    sort_by: str = "ln_BF_vs_null",
) -> pd.DataFrame:
    """Sort and rank a list of per-candidate Bayes factor dicts.

    Parameters
    ----------
    candidate_results : list of dicts from compute_bayes_factor()
    sort_by : column to sort by

    Returns
    -------
    DataFrame with rank column added
    """
    df = pd.DataFrame(candidate_results)
    df = df.sort_values(sort_by, ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df
