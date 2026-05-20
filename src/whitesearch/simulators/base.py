"""Abstract base class and data container for forward simulators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class SimData:
    """Container for simulated observational data from one channel.

    Attributes
    ----------
    channel : str
        Channel identifier: 'gw', 'radio', 'xray', 'image'.
    data : Any
        The primary data array (strain, dynamic spectrum, image, etc.).
    metadata : dict
        Auxiliary information: sample rate, frequency axis, PSD, etc.
    params_true : dict
        The parameter values used to generate the data (for injection/recovery).
    noise_realisation : ndarray | None
        The noise array added to the signal (for diagnostics).
    """

    channel: str
    data: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    params_true: dict[str, float] = field(default_factory=dict)
    noise_realisation: NDArray | None = None

    def __repr__(self) -> str:
        shape = np.shape(self.data) if self.data is not None else "None"
        return (
            f"SimData(channel={self.channel!r}, shape={shape}, "
            f"n_params={len(self.params_true)})"
        )


class BaseSimulator(ABC):
    """Abstract forward simulator.

    Subclasses implement ``simulate(params, context)`` which maps a parameter
    dict to a SimData object.  The ``context`` dict holds instrumental settings
    (sample rate, frequency range, uv-coverage, etc.) loaded from config files.

    Design contract
    ---------------
    - Must be deterministic given the same ``rng_seed`` in context.
    - Signal and noise should be separable (store noise in ``noise_realisation``).
    - Must tolerate parameters at the boundary of the prior support.
    """

    channel: str = "generic"

    @abstractmethod
    def simulate(
        self,
        params: dict[str, float],
        context: dict[str, Any],
        rng: np.random.Generator | None = None,
    ) -> SimData:
        """Run the forward simulation.

        Parameters
        ----------
        params : dict
            Model parameters (floats).
        context : dict
            Instrument / observation configuration.
        rng : np.random.Generator | None
            Random number generator for noise.  If None, a new one is created.

        Returns
        -------
        SimData
            Simulated data with signal + noise.
        """

    def signal_only(
        self,
        params: dict[str, float],
        context: dict[str, Any],
    ) -> SimData:
        """Return noiseless signal (useful for debugging and PPC).

        Default implementation injects zeros for noise; override for efficiency.
        """
        rng = np.random.default_rng(0)
        orig = self.simulate(params, context, rng=rng)
        # Replace data with signal by subtracting stored noise
        if orig.noise_realisation is not None:
            pure_signal = orig.data - orig.noise_realisation
            return SimData(
                channel=orig.channel,
                data=pure_signal,
                metadata=orig.metadata,
                params_true=orig.params_true,
                noise_realisation=None,
            )
        return orig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(channel={self.channel!r})"
