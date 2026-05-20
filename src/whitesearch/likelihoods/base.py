"""Abstract base class for WhiteSearch likelihoods and Bilby wrapper."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseLikelihood(ABC):
    """Abstract likelihood base class.

    Every channel-specific likelihood must implement ``loglike()``.
    An optional ``to_bilby_likelihood()`` factory method provides Bilby
    integration when bilby is installed.
    """

    @property
    @abstractmethod
    def parameter_names(self) -> list[str]:
        """Return the parameter names expected by ``loglike()``."""

    @abstractmethod
    def loglike(
        self,
        theta: dict[str, float],
        data: Any,
        context: dict[str, Any],
    ) -> float:
        """Compute log p(data | theta, model).

        Parameters
        ----------
        theta : dict[str, float]
            Current parameter point.
        data : Any
            Observed data object (strain array, SimData, DataFrame, etc.).
        context : dict
            Instrument / observation configuration.

        Returns
        -------
        float
            Log-likelihood value.  Must return -inf for out-of-support theta.
        """

    def to_bilby_likelihood(
        self,
        data: Any,
        context: dict[str, Any],
    ) -> Any:
        """Return a bilby.core.likelihood.Likelihood wrapping this object."""
        try:
            import bilby  # noqa: F401
        except ImportError:
            raise ImportError("bilby must be installed for to_bilby_likelihood()")
        return _BilbyLikelihoodWrapper(self, data, context)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(params={self.parameter_names})"


class _BilbyLikelihoodWrapper:
    """Internal adapter that wraps a BaseLikelihood for bilby's run_sampler."""

    def __init__(
        self,
        ws_likelihood: BaseLikelihood,
        data: Any,
        context: dict[str, Any],
    ) -> None:
        import bilby

        self._ws_likelihood = ws_likelihood
        self._data = data
        self._context = context
        self.parameters = {name: None for name in ws_likelihood.parameter_names}

        # Make this look like a bilby.Likelihood to run_sampler
        self.__class__ = type(
            "BilbyWrappedLikelihood",
            (bilby.core.likelihood.Likelihood,),
            {
                "log_likelihood": self._log_likelihood,
                "parameters": self.parameters,
            },
        )
        super().__init__(parameters=self.parameters)

    def _log_likelihood(self) -> float:
        lp = self._ws_likelihood.loglike(self.parameters, self._data, self._context)
        return float(lp) if np.isfinite(lp) else -np.inf


def poisson_loglike(
    counts: np.ndarray,
    mu: np.ndarray,
) -> float:
    """Log-likelihood for independent Poisson counts.

    ∑_i [ k_i log(μ_i) − μ_i − log(k_i!) ]

    Uses the Stirling approximation for large k.
    """
    counts = np.asarray(counts, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)

    if np.any(mu <= 0):
        return -np.inf

    from scipy.special import gammaln
    ll = np.sum(counts * np.log(mu + 1e-300) - mu - gammaln(counts + 1))
    return float(ll)


def gaussian_loglike(
    data: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """Log-likelihood for independent Gaussian observations.

    ∑_i [ −½ ((d_i − μ_i) / σ_i)² − log(σ_i √(2π)) ]
    """
    data = np.asarray(data, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    if np.any(sigma <= 0):
        return -np.inf

    ll = -0.5 * np.sum(((data - mu) / sigma) ** 2 + np.log(2.0 * np.pi * sigma**2))
    return float(ll)


def von_mises_loglike(
    phases: np.ndarray,
    mu_phases: np.ndarray,
    kappa: float,
) -> float:
    """Log-likelihood for wrapped Gaussian (von Mises) phase observations.

    Used for closure phases where Gaussian approx fails at high noise.

    log p(φ) = κ cos(φ − μ) − log(2π I_0(κ))
    """
    from scipy.special import i0

    ll = np.sum(kappa * np.cos(phases - mu_phases)) - len(phases) * np.log(
        2.0 * np.pi * i0(kappa)
    )
    return float(ll)
