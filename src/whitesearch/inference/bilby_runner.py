"""Bilby + dynesty inference runner.

Wraps BaseLikelihood into a bilby.Likelihood and runs nested sampling to
produce both posterior samples and log Bayesian evidence log Z.

When bilby is not installed, a minimal toy nested sampler is provided
(importance-sampling approximation) so the pipeline can be tested end-to-end.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..likelihoods.base import BaseLikelihood
from ..models.base import BaseModel

logger = logging.getLogger(__name__)

try:
    import bilby
    import dynesty  # noqa: F401
    BILBY_AVAILABLE = True
except ImportError:
    BILBY_AVAILABLE = False
    logger.warning(
        "bilby/dynesty not installed; using toy nested sampler. "
        "Install with: pip install bilby dynesty"
    )


@dataclass
class InferenceResult:
    """Container for a single run_sampler result.

    Attributes
    ----------
    log_evidence : float
        Log marginal likelihood ln Z.
    log_evidence_err : float
        Uncertainty on ln Z (from dynesty).
    posterior : pd.DataFrame
        Posterior samples (one column per parameter).
    log_likelihood_samples : ndarray
        Log-likelihood evaluated at each posterior sample.
    metadata : dict
        Run configuration and timing.
    """

    log_evidence: float
    log_evidence_err: float
    posterior: pd.DataFrame
    log_likelihood_samples: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ln_bf_vs(self) -> Any:
        """Helper for computing log Bayes factors against this result."""
        return _BayesFactorHelper(self)

    def credible_intervals(self, level: float = 0.9) -> dict[str, tuple[float, float]]:
        """Return credible intervals for all parameters."""
        from ..utils.math_utils import compute_credible_interval
        result = {}
        for col in self.posterior.columns:
            samples = self.posterior[col].values
            result[col] = compute_credible_interval(samples, level)
        return result

    def median_params(self) -> dict[str, float]:
        return self.posterior.median().to_dict()

    def __repr__(self) -> str:
        return (
            f"InferenceResult(ln_Z={self.log_evidence:.2f}±{self.log_evidence_err:.2f}, "
            f"n_posterior={len(self.posterior)})"
        )


class _BayesFactorHelper:
    def __init__(self, result: InferenceResult) -> None:
        self._result = result

    def __sub__(self, other: InferenceResult) -> tuple[float, float]:
        """Compute (ln BF, σ_ln BF) = (ln Z_self − ln Z_other, σ_combined)."""
        ln_bf = self._result.log_evidence - other.log_evidence
        err = np.sqrt(
            self._result.log_evidence_err**2 + other.log_evidence_err**2
        )
        return float(ln_bf), float(err)


class BilbyRunner:
    """Runs Bayesian inference using Bilby + dynesty nested sampling.

    Parameters
    ----------
    sampler : str
        Bilby sampler backend.  Recommended: 'dynesty' (default).
    nlive : int
        Number of live points for nested sampling.
    outdir : str | Path
        Output directory for bilby result files.
    resume : bool
        Resume from a previous run if checkpoint exists.
    seed : int
        Random seed for reproducibility.
    sampler_kwargs : dict
        Additional keyword arguments passed to bilby.run_sampler().
    """

    def __init__(
        self,
        sampler: str = "dynesty",
        nlive: int = 500,
        outdir: str | Path = "artifacts/bilby",
        resume: bool = True,
        seed: int = 42,
        force_toy: bool = False,
        **sampler_kwargs: Any,
    ) -> None:
        self.sampler = sampler
        self.nlive = nlive
        self.outdir = Path(outdir)
        self.resume = resume
        self.seed = seed
        env_force = os.environ.get("WHITESEARCH_FORCE_TOY", "").strip().lower() in (
            "1", "true", "yes",
        )
        self.force_toy = force_toy or env_force
        self.sampler_kwargs = sampler_kwargs

    def run(
        self,
        likelihood: BaseLikelihood,
        data: Any,
        context: dict[str, Any],
        model: BaseModel,
        label: str = "whitesearch",
    ) -> InferenceResult:
        """Run nested sampling and return an InferenceResult.

        Parameters
        ----------
        likelihood : BaseLikelihood
        data : observed data object
        context : instrument / observation configuration
        model : BaseModel (supplies bilby priors)
        label : str — unique label for this run
        """
        if not model.parameter_names:
            return self._analytic_zero_parameter_evidence(
                likelihood, data, context, model, label=label
            )

        if not BILBY_AVAILABLE or self.force_toy or self.sampler == "toy":
            if BILBY_AVAILABLE and (self.force_toy or self.sampler == "toy"):
                logger.warning("Using toy sampler (force_toy=True).")
            else:
                logger.warning("Using toy sampler (bilby not installed).")
            return self._toy_sampler(likelihood, data, context, model, label=label)

        t0 = time.time()
        self.outdir.mkdir(parents=True, exist_ok=True)

        bilby_likelihood = self._wrap_likelihood(likelihood, data, context)
        priors = model.to_bilby_priors()

        logger.info(
            "Running bilby/%s with %d live points for model=%s label=%s",
            self.sampler,
            self.nlive,
            model.name,
            label,
        )

        result = bilby.run_sampler(
            likelihood=bilby_likelihood,
            priors=priors,
            sampler=self.sampler,
            nlive=self.nlive,
            label=label,
            outdir=str(self.outdir),
            resume=self.resume,
            seed=self.seed,
            save=True,
            **self.sampler_kwargs,
        )

        elapsed = time.time() - t0
        logger.info(
            "Inference complete in %.1f s: ln Z = %.2f ± %.2f",
            elapsed,
            result.log_evidence,
            result.log_evidence_err,
        )

        posterior = result.posterior.copy()
        if "log_likelihood" in posterior.columns:
            ll_samples = posterior.pop("log_likelihood").values
        else:
            ll_samples = np.zeros(len(posterior))

        return InferenceResult(
            log_evidence=float(result.log_evidence),
            log_evidence_err=float(result.log_evidence_err),
            posterior=posterior,
            log_likelihood_samples=ll_samples,
            metadata={
                "sampler": self.sampler,
                "is_approximate_evidence": False,
                "nlive": self.nlive,
                "label": label,
                "elapsed_s": elapsed,
                "seed": self.seed,
            },
        )

    def compare_models(
        self,
        results: dict[str, InferenceResult],
        reference: str = "null",
    ) -> pd.DataFrame:
        """Compute pairwise Bayes factors relative to a reference model.

        Returns
        -------
        DataFrame with columns: model, ln_Z, ln_Z_err, ln_BF, ln_BF_err,
            BF_interpretation
        """
        rows = []
        ref = results.get(reference)
        if ref is None:
            raise KeyError(f"Reference model {reference!r} not in results.")

        for name, res in results.items():
            ln_bf = res.log_evidence - ref.log_evidence
            err = np.sqrt(res.log_evidence_err**2 + ref.log_evidence_err**2)
            rows.append(
                {
                    "model": name,
                    "ln_Z": res.log_evidence,
                    "ln_Z_err": res.log_evidence_err,
                    "ln_BF": ln_bf,
                    "ln_BF_err": err,
                    "BF_interpretation": self._interpret_bf(ln_bf),
                }
            )
        return pd.DataFrame(rows).sort_values("ln_BF", ascending=False)

    @staticmethod
    def _interpret_bf(ln_bf: float) -> str:
        """Kass & Raftery (1995) evidence scale for ln BF."""
        if ln_bf < 1.0:
            return "not worth mentioning"
        elif ln_bf < 3.0:
            return "positive"
        elif ln_bf < 5.0:
            return "strong"
        else:
            return "very strong (BF > 150)"

    def _wrap_likelihood(
        self,
        likelihood: BaseLikelihood,
        data: Any,
        context: dict[str, Any],
    ) -> "bilby.core.likelihood.Likelihood":
        """Wrap a BaseLikelihood in a bilby.Likelihood subclass."""

        params = {name: None for name in likelihood.parameter_names}
        ws_ll = likelihood
        ws_data = data
        ws_ctx = context

        class _Wrapper(bilby.core.likelihood.Likelihood):
            def __init__(self):
                super().__init__(parameters=params)

            def log_likelihood(self) -> float:
                val = ws_ll.loglike(self.parameters, ws_data, ws_ctx)
                return float(val) if np.isfinite(val) else -1e30

        return _Wrapper()

    def _analytic_zero_parameter_evidence(
        self,
        likelihood: BaseLikelihood,
        data: Any,
        context: dict[str, Any],
        model: BaseModel,
        label: str = "whitesearch",
    ) -> InferenceResult:
        """Exact log-evidence for models with no free parameters (e.g. null)."""
        theta: dict[str, float] = {}
        ll = float(likelihood.loglike(theta, data, context))
        logger.info(
            "Analytic evidence for %s (0 parameters): ln Z = %.3f",
            model.name,
            ll,
        )
        return InferenceResult(
            log_evidence=ll,
            log_evidence_err=0.0,
            posterior=pd.DataFrame(),
            log_likelihood_samples=np.array([ll]),
            metadata={
                "sampler": "analytic_zero_parameter",
                "is_approximate_evidence": False,
                "label": label,
                "seed": self.seed,
            },
        )

    def _toy_sampler(
        self,
        likelihood: BaseLikelihood,
        data: Any,
        context: dict[str, Any],
        model: BaseModel,
        label: str = "whitesearch",
        n_samples: int = 2000,
    ) -> InferenceResult:
        """Importance-sampling approximation when bilby is unavailable."""
        rng = np.random.default_rng(self.seed)
        param_names = model.parameter_names
        samples = []
        log_weights = []

        for _ in range(n_samples):
            theta = model.sample_prior(rng)
            ll = likelihood.loglike(theta, data, context)
            lp = model.log_prior(theta)
            samples.append(theta)
            log_weights.append(ll + lp if np.isfinite(ll) else -1e30)

        log_weights_arr = np.array(log_weights, dtype=float)
        # Importance weight normalisation
        max_lw = np.max(log_weights_arr)
        weights = np.exp(log_weights_arr - max_lw)
        weights /= weights.sum()

        # Effective sample size estimate
        ln_Z = float(max_lw + np.log(np.sum(np.exp(log_weights_arr - max_lw))) - np.log(n_samples))
        ln_Z_err = 1.0 / np.sqrt(np.sum(weights ** 2) * n_samples)

        # Weighted posterior samples
        idx = rng.choice(n_samples, size=min(1000, n_samples), replace=True, p=weights)
        posterior_rows = [samples[i] for i in idx]
        if param_names:
            posterior = pd.DataFrame(posterior_rows, columns=param_names)
        else:
            posterior = pd.DataFrame(index=range(len(posterior_rows)))

        return InferenceResult(
            log_evidence=ln_Z,
            log_evidence_err=ln_Z_err,
            posterior=posterior,
            log_likelihood_samples=log_weights_arr[idx],
            metadata={
                "sampler": "toy_importance_sampling",
                "is_approximate_evidence": True,
                "n_prior_samples": n_samples,
                "label": label,
                "seed": self.seed,
            },
        )
