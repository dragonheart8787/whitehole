"""VLBI visibility likelihood for the EHT image/shadow channel.

Visibility amplitude likelihood: Gaussian
Closure phase likelihood: von Mises (wrapped Gaussian)

Reference: Thompson, Moran & Swenson, "Interferometry and Synthesis in Radio Astronomy"
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import BaseLikelihood, gaussian_loglike, von_mises_loglike
from ..simulators.image_shadow import ImageShadowSimulator


class VisibilityLikelihood(BaseLikelihood):
    """Combined visibility-amplitude + closure-phase VLBI likelihood.

    log L = log L_amp + log L_phase

    log L_amp  = ∑_b −½ (|V_obs_b| − |V_model_b|)² / σ_b²
    log L_phase = ∑_t κ cos(φ_obs_t − φ_model_t) − log(2π I₀(κ))

    where κ is estimated from S/N.
    """

    def __init__(
        self,
        use_closure_phases: bool = True,
        closure_kappa: float = 10.0,
        robust_data: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        use_closure_phases : bool
            Include closure phase likelihood (insensitive to station gains).
        closure_kappa : float
            von Mises concentration parameter κ (higher = tighter around μ).
        robust_data : bool
            Use Student-t likelihood for visibility amplitudes (outlier robust).
        """
        self.use_closure = use_closure_phases
        self.kappa = closure_kappa
        self.robust = robust_data

    @property
    def parameter_names(self) -> list[str]:
        return [
            "M", "a_star", "D_L", "i", "position_angle",
            "ring_width_frac", "log10_brightness",
        ]

    def loglike(
        self,
        theta: dict[str, float],
        data: Any,
        context: dict[str, Any],
    ) -> float:
        """Compute log p(visibilities | θ).

        Parameters
        ----------
        data : SimData or dict
            Must contain 'data' (complex visibility array) and
            metadata with 'sigma' (per-baseline noise), 'closure_phases' (observed).
        """
        if hasattr(data, "data"):
            obs_vis = np.asarray(data.data, dtype=complex)
            meta = data.metadata
        else:
            obs_vis = np.asarray(data["visibilities"], dtype=complex)
            meta = data

        # Per-baseline thermal noise
        sigma_vis = float(meta.get("thermal_noise_jy", context.get("thermal_noise_jy", 0.05)))
        sigma_arr = np.full(len(obs_vis), sigma_vis)
        if "sigma" in meta:
            sigma_arr = np.asarray(meta["sigma"], dtype=float)

        # Build model visibilities
        sim = ImageShadowSimulator()
        sim_data = sim.simulate(theta, context, rng=np.random.default_rng(0))
        model_vis = np.asarray(sim_data.data, dtype=complex)

        n = min(len(obs_vis), len(model_vis))
        obs_vis = obs_vis[:n]
        model_vis = model_vis[:n]
        sigma_arr = sigma_arr[:n]

        # ── Amplitude likelihood ───────────────────────────────────────────────
        obs_amp = np.abs(obs_vis)
        model_amp = np.abs(model_vis)

        if self.robust:
            ll_amp = self._student_t_loglike(obs_amp, model_amp, sigma_arr, nu=3.0)
        else:
            ll_amp = gaussian_loglike(obs_amp, model_amp, sigma_arr)

        # ── Closure phase likelihood ───────────────────────────────────────────
        ll_phase = 0.0
        if self.use_closure:
            obs_closure = meta.get("closure_phases", None)
            model_closure = sim_data.metadata.get("closure_phases", None)
            if obs_closure is not None and model_closure is not None:
                obs_cp = np.asarray(obs_closure, dtype=float)
                mod_cp = np.asarray(model_closure, dtype=float)
                n_cp = min(len(obs_cp), len(mod_cp))
                ll_phase = von_mises_loglike(obs_cp[:n_cp], mod_cp[:n_cp], self.kappa)

        return ll_amp + ll_phase

    @staticmethod
    def _student_t_loglike(
        data: NDArray,
        mu: NDArray,
        sigma: NDArray,
        nu: float = 3.0,
    ) -> float:
        """Robust Student-t log-likelihood for outlier-tolerant visibility fitting."""
        from scipy.special import gammaln

        z = (data - mu) / sigma
        ll = (
            gammaln(0.5 * (nu + 1.0))
            - gammaln(0.5 * nu)
            - 0.5 * np.log(nu * np.pi * sigma**2)
            - 0.5 * (nu + 1.0) * np.log(1.0 + z**2 / nu)
        )
        return float(np.sum(ll))

    def predictive_summary_stats(
        self,
        theta: dict[str, float],
        context: dict[str, Any],
    ) -> dict[str, float]:
        """Ring diameter, axial ratio, brightness for PPC."""
        from ..utils.constants import G, C, M_SUN, MPC_M, MUAS_RAD

        M = theta["M"]
        a = theta.get("a_star", 0.0)
        D_L = theta["D_L"]
        i = theta.get("i", 0.0)

        rg = G * M * M_SUN / C**2
        b_c = 3.0 * np.sqrt(3.0) * rg
        theta_d = 2.0 * b_c / (D_L * MPC_M) / MUAS_RAD

        return {
            "theta_d_muas": theta_d,
            "axial_ratio": float(np.abs(np.cos(i))),
            "ring_width_muas": theta_d * theta.get("ring_width_frac", 0.1) / 2.0,
            "brightness": float(10.0 ** theta.get("log10_brightness", 0.0)),
        }
