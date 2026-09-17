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
from ..simulators.image_shadow import (
    ImageShadowSimulator,
    UnrepresentableRingError,
    build_ring_image,
)
from ..utils.targets import require_target


class VisibilityLikelihood(BaseLikelihood):
    """Combined visibility-amplitude + closure-phase VLBI likelihood.

    log L = log L_amp + log L_phase

    log L_amp  = ∑_b −½ (|V_obs_b| − |V_model_b|)² / σ_b²
    log L_phase = ∑_t κ cos(φ_obs_t − φ_model_t) − log(2π I₀(κ))

    where κ is estimated from S/N.
    """

    #: Parameters the geometric ring hypothesis (gr_eternal) actually reads
    #: in ImageShadowSimulator.  ``D_L`` is absent by design: the source
    #: distance is a per-target known constant taken from the observation or
    #: the context, not a sampled parameter -- see utils.targets.
    RING_PARAMETERS = [
        "M", "a_star", "i", "position_angle",
        "ring_width_frac", "log10_total_flux_jy",
    ]

    #: Parameters the accretion-flow hypothesis (bh_accretion) reads.  It
    #: shares the geometry and replaces the two free emission parameters with
    #: an accretion rate plus a jet-footpoint contrast.
    ACCRETION_PARAMETERS = [
        "M", "a_star", "i", "position_angle",
        "log10_mdot_edd", "jet_power_frac",
    ]

    #: Which parameter vector each model on this channel is fit with.
    #: Fail-closed: an unlisted model name raises rather than silently
    #: inheriting the ring vector, which is how bh_accretion previously
    #: advertised parameters its forward model never read.
    MODEL_PARAMETERS: dict[str, list[str]] = {
        "null": [],
        "gr_eternal": RING_PARAMETERS,
        "bh_accretion": ACCRETION_PARAMETERS,
    }

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
        #: How many evaluations hit a ring the image grid cannot represent.
        self.n_unrepresentable: int = 0
        self.last_unrepresentable: str | None = None

    @property
    def parameter_names(self) -> list[str]:
        try:
            return list(self.MODEL_PARAMETERS[self.model_name])
        except KeyError:
            raise KeyError(
                f"VisibilityLikelihood has no forward model for "
                f"{self.model_name!r}. Known: {sorted(self.MODEL_PARAMETERS)}."
            ) from None

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
        # Same metadata-first, context-second resolution the GW likelihood uses
        # for per-event known quantities (t_merger, band edges), except that a
        # missing target raises instead of falling back: the target fixes the
        # source distance, and the two supported targets differ by 3.31 dex.
        model_context["target"] = require_target(meta, context).name

        # A geometry the image grid cannot represent has no model prediction,
        # so the sampler must reject the point rather than crash the run.  The
        # previous parameterisation reached the same place numerically: those
        # rings imaged to ~zero flux and scored an arbitrarily bad likelihood.
        # Counted, not swallowed -- `n_unrepresentable` says how often it fired.
        # Checked against `model_context`, so the target (hence the distance,
        # hence the ring radius) is the metadata-first one.
        try:
            build_ring_image(theta, model_context)
        except UnrepresentableRingError as exc:
            self.n_unrepresentable += 1
            self.last_unrepresentable = str(exc)
            return float("-inf")

        sim = ImageShadowSimulator()
        sim_data = sim.simulate(theta, model_context, rng=np.random.default_rng(0))
        # The MODEL TEMPLATE is the noise-free prediction, `vis_signal` -- not
        # `sim_data.data`, which is `vis_signal + noise` and therefore an
        # *observation*, not a prediction.  Using `.data` gave every template a
        # fixed thermal-noise realisation (the rng seed below is constant), so
        # the likelihood compared measured-with-noise against
        # predicted-with-different-noise.  Because |V| is a positively biased
        # function of added noise, it also inflated the template amplitude and
        # pulled the recovered flux down.  Measured at typical SNR: +0.27%
        # inflation before, 0% after.  See audit X.19.
        #
        # The rng is still passed, and still fixed, so the discarded noise draw
        # cannot make the likelihood stochastic between evaluations.
        model_vis = np.asarray(sim_data.metadata["vis_signal"], dtype=complex)

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
            # Same correction: the model's closure phases must come from the
            # noise-free prediction.  `closure_phases` in the metadata is the
            # observation-side quantity, computed from the noisy visibilities.
            model_closure = sim_data.metadata.get("closure_phases_signal", None)
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
                    "ImageShadowSimulator produced no 'closure_phases_signal' "
                    "for the model visibilities; cannot form the closure-phase "
                    "term."
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
    ) -> dict[str, Any]:
        """Ring diameter, axial ratio, thickness, flux and brightness for PPC.

        Built by ``build_ring_image``, the same function the simulator uses, so
        the posterior-predictive numbers cannot be computed from a slightly
        different forward model than the one that was fit.  That includes the
        ``I0 = F / G`` inversion: the sampled amplitude is the integrated flux
        and the peak surface brightness is derived from it.

        The return is ``dict[str, Any]`` rather than ``dict[str, float]``
        because it also carries ``emission_model``, the provenance string
        naming which hypothesis produced these numbers.

        Derived with the same two functions the simulator uses
        (``_shadow_radius_muas`` and ``ring_emission_from_params``) rather than
        a local copy of either expression, and with the same per-target
        distance, so a posterior-predictive check cannot be comparing against a
        slightly different forward model than the one that was fit.
        """
        built = build_ring_image(theta, context)
        return {
            "theta_d_muas": 2.0 * built.r_ring_muas,
            "axial_ratio": built.axial_ratio,
            "ring_width_muas": built.w_ring_muas,
            "total_flux_jy": built.total_flux_jy,
            "brightness": built.brightness,
            "geometry_factor_muas2": built.geometry_factor_muas2,
            "emission_model": built.emission.emission_model,
        }
