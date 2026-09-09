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

    #: Parameters the ring model in ImageShadowSimulator actually reads.
    RING_PARAMETERS = [
        "M", "a_star", "D_L", "i", "position_angle",
        "ring_width_frac", "log10_brightness",
    ]

    def __init__(
        self,
        model_name: str = "gr_eternal",
        use_closure_phases: bool = True,
        closure_kappa: float = 10.0,
        robust_data: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        model_name : str
            Which model's parameters to report from ``parameter_names``.  Added
            so the null hypothesis can be fit on this channel at all: every
            other channel's likelihood has a "null" branch, and without one the
            intersection check refused to build priors for it.  See
            docs/XRAY_IMAGE_PREFLIGHT_AUDIT.md X.4.
        use_closure_phases : bool
            Include the closure-phase term.  True means the caller asserts the
            data carries closure phases, and their absence is an error rather
            than a silent fallback to amplitude-only; pass False to declare an
            amplitude-only analysis deliberately.
        closure_kappa : float
            von Mises concentration parameter κ (higher = tighter around μ).
        robust_data : bool
            Use Student-t likelihood for visibility amplitudes (outlier robust).
        """
        self.model_name = model_name
        self.use_closure = use_closure_phases
        self.kappa = closure_kappa
        self.robust = robust_data
        #: Provenance of the most recent evaluation's closure-phase decision.
        self.last_closure_config: dict[str, Any] = {}

    @property
    def parameter_names(self) -> list[str]:
        if self.model_name == "null":
            return []
        return list(self.RING_PARAMETERS)

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

        # Build model visibilities ON THE DATA'S OWN BASELINES.  Previously the
        # observation's uv_coverage was never forwarded, so ImageShadowSimulator
        # fell back to _default_eht_uv() and the model was sampled at different
        # (u, v) points from the data it was being differenced against.  That did
        # not bite only because the mock record happens to reuse the default
        # coverage; any real uvfits would.  See
        # docs/XRAY_IMAGE_PREFLIGHT_AUDIT.md X.8.1.
        model_context = dict(context)
        model_context["uv_coverage"] = self._require_uv_coverage(meta, context)
        sim = ImageShadowSimulator()
        sim_data = sim.simulate(theta, model_context, rng=np.random.default_rng(0))
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
        # Fail closed rather than degrade silently: use_closure=True asserts the
        # data carries closure phases, so their absence is an error.  An
        # amplitude-only analysis is declared by constructing the likelihood
        # with use_closure_phases=False, and either way the decision is recorded
        # in last_closure_config.  See docs/XRAY_IMAGE_PREFLIGHT_AUDIT.md X.8.2.
        ll_phase = 0.0
        if not self.use_closure:
            self.last_closure_config = {
                "used_closure_phase": False,
                "reason": "caller declared an amplitude-only analysis",
                "n_closure_phases": 0,
            }
        else:
            obs_closure = meta.get("closure_phases", None)
            model_closure = sim_data.metadata.get("closure_phases", None)
            if obs_closure is None:
                raise KeyError(
                    "VisibilityLikelihood was constructed with "
                    "use_closure_phases=True but the observation carries no "
                    "'closure_phases'. Supply them, or construct the likelihood "
                    "with use_closure_phases=False to declare an "
                    "amplitude-only analysis. Observation keys: "
                    f"{sorted(meta) if isinstance(meta, dict) else type(meta).__name__}"
                )
            if model_closure is None:  # pragma: no cover - simulator always sets it
                raise KeyError(
                    "ImageShadowSimulator produced no 'closure_phases' for the "
                    "model visibilities; cannot form the closure-phase term."
                )
            obs_cp = np.asarray(obs_closure, dtype=float)
            mod_cp = np.asarray(model_closure, dtype=float)
            n_cp = min(len(obs_cp), len(mod_cp))
            ll_phase = von_mises_loglike(obs_cp[:n_cp], mod_cp[:n_cp], self.kappa)
            self.last_closure_config = {
                "used_closure_phase": True,
                "reason": "observation supplied closure phases",
                "n_closure_phases": int(n_cp),
            }

        return ll_amp + ll_phase

    @staticmethod
    def _require_uv_coverage(meta: Any, context: dict[str, Any]) -> NDArray:
        """Baselines the observation was actually sampled on.

        Fail-closed: falling back to a default array would compare the model and
        the data at different points in the uv plane, which is the image-channel
        version of a forward-model mismatch.
        """
        for source in (meta, context):
            if isinstance(source, dict) and source.get("uv_coverage") is not None:
                return np.asarray(source["uv_coverage"], dtype=float)
        raise KeyError(
            "VisibilityLikelihood needs the observation's 'uv_coverage' to "
            "build model visibilities on the same baselines; neither the data "
            "metadata nor the context provides it. Observation keys: "
            f"{sorted(meta) if isinstance(meta, dict) else type(meta).__name__}"
        )

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
