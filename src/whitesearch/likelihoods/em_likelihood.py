"""Electromagnetic burst likelihood for radio and X-ray channels.

Radio (FRB-like):
  Each frequency-time pixel d[i,j] ~ N(F_signal(theta)[i,j], sigma_noise^2)
  Marginalised over off-burst background via per-channel baseline subtraction.

X-ray / Gamma-ray:
  Poisson likelihood: C_i ~ Poisson(mu_i(theta) * A * dt + B * dt)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import BaseLikelihood, poisson_loglike, gaussian_loglike
from ..utils.constants import K_DM


class RadioBurstLikelihood(BaseLikelihood):
    """Gaussian dynamic-spectrum likelihood for FRB-like radio bursts.

    The likelihood treats each frequency-time pixel as an independent
    Gaussian observation with known RMS noise (sigma_noise).
    Off-burst channels provide the noise estimate; the signal model
    is a forward-simulated dynamic spectrum from EMBurstSimulator.
    """

    def __init__(self, model_name: str = "pbh_tunneling") -> None:
        self.model_name = model_name

    @property
    def parameter_names(self) -> list[str]:
        if self.model_name == "null":
            return []
        if self.model_name == "pbh_tunneling":
            return [
                "log10_M_g", "log10_f_pbh", "log10_k_tunnel",
                "log10_eta_r", "z", "DM_host",
                "log10_W_int_ms", "log10_tau_sc_ms", "spectral_index",
            ]
        # magnetar_flare
        return [
            "log10_fluence_jy_ms", "log10_W_ms", "DM",
            "log10_tau_sc_ms", "spectral_index",
        ]

    def loglike(
        self,
        theta: dict[str, float],
        data: Any,
        context: dict[str, Any],
    ) -> float:
        """Compute log p(d | θ) for the radio dynamic spectrum.

        Parameters
        ----------
        theta : dict
        data : SimData with .data shape (n_freq, n_time) or raw dict
        context : dict with 'freqs_mhz', 'times_s', 'sigma_noise_jy'
        """
        if self.model_name == "null":
            return self._null_radio_loglike(data, context)

        from ..simulators.em_burst import EMBurstSimulator

        # Observed data
        if hasattr(data, "data"):
            obs = np.asarray(data.data, dtype=np.float64)
            meta = data.metadata
        else:
            obs = np.asarray(data["data"], dtype=np.float64)
            meta = data

        sigma = float(meta.get("sigma_noise_jy", context.get("sigma_noise_jy", 1.0)))

        # Build model dynamic spectrum (noiseless signal)
        sim = EMBurstSimulator()
        sim_data = sim.simulate(theta, context, rng=np.random.default_rng(0))
        signal = np.asarray(sim_data.data - sim_data.noise_realisation, dtype=np.float64)

        # Clip to match shapes
        min_shape = tuple(min(a, b) for a, b in zip(obs.shape, signal.shape))
        obs = obs[: min_shape[0], : min_shape[1]]
        signal = signal[: min_shape[0], : min_shape[1]]

        # Gaussian likelihood (vectorised)
        ll = -0.5 * np.sum(((obs - signal) / sigma) ** 2)
        ll -= 0.5 * obs.size * np.log(2.0 * np.pi * sigma**2)
        return float(ll)

    def _null_radio_loglike(self, data: Any, context: dict[str, Any]) -> float:
        """Pure noise dynamic spectrum (zero signal)."""
        if hasattr(data, "data"):
            obs = np.asarray(data.data, dtype=np.float64)
            meta = data.metadata
        else:
            obs = np.asarray(data["data"], dtype=np.float64)
            meta = data
        sigma = float(meta.get("sigma_noise_jy", context.get("sigma_noise_jy", 1.0)))
        ll = -0.5 * np.sum((obs / sigma) ** 2)
        ll -= 0.5 * obs.size * np.log(2.0 * np.pi * sigma**2)
        return float(ll)

    def predictive_summary_stats(
        self,
        theta: dict[str, float],
        context: dict[str, Any],
    ) -> dict[str, float]:
        """Summary statistics for PPC: DM, width, fluence, spectral index."""
        dm = 100.0 + theta.get("z", 0.0) * 855.0 + theta.get("DM_host", 50.0)
        W_int = 10.0 ** theta.get("log10_W_int_ms", 1.0)
        tau_sc = 10.0 ** theta.get("log10_tau_sc_ms", 0.0)
        W_obs = np.sqrt(W_int**2 + tau_sc**2)
        return {
            "DM_total": dm,
            "W_int_ms": W_int,
            "W_obs_ms": W_obs,
            "spectral_index": theta.get("spectral_index", -1.5),
        }


class XRayBurstLikelihood(BaseLikelihood):
    """Poisson likelihood for X-ray / gamma-ray photon counts.

    C_i ~ Poisson(μ_i(θ) + B_i)
    where μ_i(θ) is the model signal rate and B_i is the background.
    """

    def __init__(self, model_name: str = "pbh_tunneling") -> None:
        self.model_name = model_name

    @property
    def parameter_names(self) -> list[str]:
        return ["log10_fluence_erg_cm2", "log10_T90_s", "log10_eta_gamma"]

    def loglike(
        self,
        theta: dict[str, float],
        data: Any,
        context: dict[str, Any],
    ) -> float:
        """Compute Poisson log-likelihood for X-ray count light curve.

        Parameters
        ----------
        data : SimData or dict with 'data' (counts array) and metadata
        """
        from ..simulators.em_burst import XRayLightCurveSimulator

        if hasattr(data, "data"):
            obs_counts = np.asarray(data.data, dtype=np.float64)
            meta = data.metadata
        else:
            obs_counts = np.asarray(data["counts"], dtype=np.float64)
            meta = data

        bg_rate = float(meta.get("bg_rate_cps", context.get("bg_rate_cps", 0.5)))
        area = float(meta.get("area_cm2", context.get("area_cm2", 1000.0)))
        dt = float(meta.get("dt_s", context.get("dt_s", 1.0)))
        mu_bg = np.full(len(obs_counts), bg_rate * dt)

        # Build signal model
        sim = XRayLightCurveSimulator()
        sim_data = sim.simulate(theta, context, rng=np.random.default_rng(0))
        mu_signal = np.asarray(sim_data.metadata["mu_signal"], dtype=np.float64)

        n = min(len(obs_counts), len(mu_signal))
        mu_total = mu_signal[:n] + mu_bg[:n]
        counts = obs_counts[:n]

        ll = poisson_loglike(counts, mu_total)
        return ll

    def predictive_summary_stats(
        self,
        theta: dict[str, float],
        context: dict[str, Any],
    ) -> dict[str, float]:
        return {
            "fluence_erg_cm2": float(10.0 ** theta.get("log10_fluence_erg_cm2", -8.0)),
            "T90_s": float(10.0 ** theta.get("log10_T90_s", 0.0)),
        }
