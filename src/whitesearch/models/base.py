"""Abstract base classes for white hole and alternative models.

Every concrete model must implement BaseModel, which provides:
  - parameters()      → list of ParameterSpec with prior definitions
  - sample_prior()    → sample a parameter dict from the prior
  - summary_stats()   → compute observable summary statistics
  - to_bilby_priors() → convert to bilby.core.prior.PriorDict (if bilby is available)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


PriorType = Literal[
    "uniform", "log_uniform", "normal", "half_normal",
    "cos_uniform", "volume_uniform", "beta", "discrete_uniform",
]


@dataclass
class ParameterSpec:
    """Specification for a single model parameter including its prior.

    Attributes
    ----------
    name : str
        Parameter identifier (must be a valid Python identifier).
    prior_type : PriorType
        Prior distribution family.
    unit : str
        Physical unit string for display/logging.
    description : str
        Human-readable description of the parameter.
    prior_kwargs : dict
        Keyword arguments specific to the prior_type.
        - uniform:         low, high
        - log_uniform:     low, high  (both > 0)
        - normal:          mean, std
        - half_normal:     sigma
        - cos_uniform:     (no extra args; returns angle in radians)
        - volume_uniform:  low, high  (D_L; prior ∝ D^2)
        - beta:            a, b
        - discrete_uniform: values  (list of allowed int/float values)
    latex : str
        LaTeX representation for plot labels.
    """

    name: str
    prior_type: PriorType
    unit: str = ""
    description: str = ""
    prior_kwargs: dict[str, Any] = field(default_factory=dict)
    latex: str = ""

    def sample(self, rng: np.random.Generator) -> float:
        """Draw one sample from the prior."""
        match self.prior_type:
            case "uniform":
                return float(rng.uniform(self.prior_kwargs["low"], self.prior_kwargs["high"]))
            case "log_uniform":
                log_low = np.log(self.prior_kwargs["low"])
                log_high = np.log(self.prior_kwargs["high"])
                return float(np.exp(rng.uniform(log_low, log_high)))
            case "normal":
                return float(rng.normal(self.prior_kwargs["mean"], self.prior_kwargs["std"]))
            case "half_normal":
                return float(np.abs(rng.normal(0.0, self.prior_kwargs["sigma"])))
            case "cos_uniform":
                cos_val = rng.uniform(-1.0, 1.0)
                return float(np.arccos(cos_val))
            case "volume_uniform":
                low, high = self.prior_kwargs["low"], self.prior_kwargs["high"]
                u = rng.uniform(0.0, 1.0)
                return float((low**3 + u * (high**3 - low**3)) ** (1.0 / 3.0))
            case "beta":
                return float(rng.beta(self.prior_kwargs["a"], self.prior_kwargs["b"]))
            case "discrete_uniform":
                vals = self.prior_kwargs["values"]
                return float(rng.choice(vals))
            case _:
                raise ValueError(f"Unknown prior_type: {self.prior_type!r}")

    def log_prior(self, value: float) -> float:
        """Evaluate log p(value) under the prior."""
        match self.prior_type:
            case "uniform":
                low, high = self.prior_kwargs["low"], self.prior_kwargs["high"]
                if low <= value <= high:
                    return -np.log(high - low)
                return -np.inf
            case "log_uniform":
                low, high = self.prior_kwargs["low"], self.prior_kwargs["high"]
                if low <= value <= high:
                    return -np.log(value) - np.log(np.log(high / low))
                return -np.inf
            case "normal":
                mean, std = self.prior_kwargs["mean"], self.prior_kwargs["std"]
                return float(-0.5 * ((value - mean) / std) ** 2 - np.log(std * np.sqrt(2 * np.pi)))
            case "half_normal":
                sigma = self.prior_kwargs["sigma"]
                if value < 0:
                    return -np.inf
                return float(-0.5 * (value / sigma) ** 2 - np.log(sigma * np.sqrt(np.pi / 2)))
            case "cos_uniform":
                if 0.0 <= value <= np.pi:
                    return float(np.log(np.sin(value) / 2.0))
                return -np.inf
            case "volume_uniform":
                low, high = self.prior_kwargs["low"], self.prior_kwargs["high"]
                if low <= value <= high:
                    norm = (high**3 - low**3) / 3.0
                    return float(np.log(value**2) - np.log(norm))
                return -np.inf
            case "beta":
                from scipy.special import betaln
                a, b = self.prior_kwargs["a"], self.prior_kwargs["b"]
                if 0.0 < value < 1.0:
                    return float((a - 1) * np.log(value) + (b - 1) * np.log(1 - value) - betaln(a, b))
                return -np.inf
            case "discrete_uniform":
                vals = self.prior_kwargs["values"]
                if value in vals:
                    return -np.log(len(vals))
                return -np.inf
            case _:
                return -np.inf

    def to_bilby_prior(self) -> Any:
        """Convert to a bilby.core.prior.Prior object if bilby is available."""
        try:
            import bilby.core.prior as bp
        except ImportError:
            raise ImportError("bilby must be installed for to_bilby_prior()")

        match self.prior_type:
            case "uniform":
                return bp.Uniform(
                    minimum=self.prior_kwargs["low"],
                    maximum=self.prior_kwargs["high"],
                    name=self.name,
                    latex_label=self.latex or self.name,
                    unit=self.unit,
                )
            case "log_uniform":
                return bp.LogUniform(
                    minimum=self.prior_kwargs["low"],
                    maximum=self.prior_kwargs["high"],
                    name=self.name,
                    latex_label=self.latex or self.name,
                    unit=self.unit,
                )
            case "normal":
                return bp.Gaussian(
                    mu=self.prior_kwargs["mean"],
                    sigma=self.prior_kwargs["std"],
                    name=self.name,
                    latex_label=self.latex or self.name,
                    unit=self.unit,
                )
            case "cos_uniform":
                return bp.Cosine(
                    name=self.name,
                    latex_label=self.latex or self.name,
                    unit=self.unit,
                )
            case "volume_uniform":
                return bp.PowerLaw(
                    alpha=2,
                    minimum=self.prior_kwargs["low"],
                    maximum=self.prior_kwargs["high"],
                    name=self.name,
                    latex_label=self.latex or self.name,
                    unit=self.unit,
                )
            case "discrete_uniform":
                return bp.DeltaFunction(
                    peak=self.prior_kwargs["values"][0],
                    name=self.name,
                )
            case _:
                return bp.Uniform(
                    minimum=0.0,
                    maximum=1.0,
                    name=self.name,
                )


class BaseModel(ABC):
    """Abstract base class for all white hole and alternative models.

    Subclasses must implement ``parameters()``, ``summary_stats()``, and
    optionally override ``sample_prior()`` if the default (independent
    sampling from each ParameterSpec) is not appropriate.
    """

    name: str = "BaseModel"
    channel: str = "generic"  # 'gw', 'radio', 'xray', 'image', 'joint'

    @abstractmethod
    def parameters(self) -> list[ParameterSpec]:
        """Return ordered list of ParameterSpec objects defining the model."""

    @property
    def parameter_names(self) -> list[str]:
        return [p.name for p in self.parameters()]

    def sample_prior(self, rng: np.random.Generator | None = None) -> dict[str, float]:
        """Sample one parameter vector from the prior.

        Override for correlated priors or hierarchical structures.
        """
        if rng is None:
            rng = np.random.default_rng()
        return {p.name: p.sample(rng) for p in self.parameters()}

    def log_prior(self, params: dict[str, float]) -> float:
        """Evaluate log p(params) = ∑ log p_i(θ_i) (assumes independence)."""
        lp = 0.0
        for spec in self.parameters():
            lp += spec.log_prior(params.get(spec.name, -np.inf))
            if not np.isfinite(lp):
                return -np.inf
        return lp

    @abstractmethod
    def summary_stats(self, params: dict[str, float]) -> dict[str, float]:
        """Compute observable summary statistics at the given parameter point."""

    def to_bilby_priors(self) -> Any:
        """Return a bilby.core.prior.PriorDict for this model."""
        try:
            import bilby.core.prior as bp
        except ImportError:
            raise ImportError("bilby must be installed for to_bilby_priors()")
        return bp.PriorDict({p.name: p.to_bilby_prior() for p in self.parameters()})

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, channel={self.channel!r})"
