"""Alternative (null-hypothesis and astrophysical foreground) models.

Every white-hole candidate must be tested against these models.
A Bayes factor BF(WH / alt) > exp(3) is the internal upgrade gate;
> exp(5) is the publication gate.

Models implemented here
-----------------------
- NullHypothesis        : pure Gaussian noise (no signal)
- MagnetarFlare         : coherent radio burst from magnetar giant flare
- GRBAfterglowFRB       : GRB prompt emission or magnetar-powered FRB
- StandardBHRingdown    : GR black hole merger ringdown (GW channel)
- BHAccretion           : black hole accretion-flow emission (image channel)
"""

from __future__ import annotations

import numpy as np

from .base import BaseModel, ParameterSpec
from .gr_eternal import SPIN_MAX
from ..utils.targets import ImageTarget, get_target


class NullHypothesis(BaseModel):
    """Pure Gaussian noise — zero-signal model.

    No free parameters; the log-likelihood under this model is just
    −½ Σ (d_i / σ_i)², which the likelihood module computes directly.
    """

    name = "NullHypothesis"
    channel = "generic"

    def parameters(self) -> list[ParameterSpec]:
        return []  # no signal parameters

    def summary_stats(self, params: dict[str, float]) -> dict[str, float]:
        return {}


class MagnetarFlare(BaseModel):
    """Magnetar giant flare producing an FRB-like coherent radio burst.

    Used as the primary alternative to PBH tunneling in the radio channel.
    """

    name = "MagnetarFlare"
    channel = "radio"

    def parameters(self) -> list[ParameterSpec]:
        return [
            ParameterSpec(
                name="log10_fluence_jy_ms",
                prior_type="uniform",
                prior_kwargs={"low": -2.0, "high": 5.0},
                unit="log10(Jy ms)",
                description="Log10 burst fluence",
                latex=r"$\log_{10}\mathcal{F}$",
            ),
            ParameterSpec(
                # Named for the key EMBurstSimulator reads and pbh_tunneling
                # already declares.  Was 'log10_W_ms', which the simulator
                # never saw, so the width was silently pinned at 10 ms --
                # docs/RADIO_PREFLIGHT_AUDIT.md R.4.
                name="log10_W_int_ms",
                prior_type="uniform",
                prior_kwargs={"low": -1.0, "high": 3.0},
                unit="log10(ms)",
                description="Log10 intrinsic burst width",
                latex=r"$\log_{10} W$",
            ),
            ParameterSpec(
                name="DM",
                prior_type="log_uniform",
                prior_kwargs={"low": 10.0, "high": 3000.0},
                unit="pc/cm^3",
                description="Dispersion measure",
                latex=r"$\mathrm{DM}$",
            ),
            ParameterSpec(
                name="log10_tau_sc_ms",
                prior_type="uniform",
                prior_kwargs={"low": -3.0, "high": 2.0},
                unit="log10(ms)",
                description="Log10 scattering timescale at 1 GHz",
                latex=r"$\log_{10}\tau_\mathrm{sc}$",
            ),
            ParameterSpec(
                name="spectral_index",
                prior_type="normal",
                prior_kwargs={"mean": -1.6, "std": 1.2},
                unit="dimensionless",
                description="Radio spectral index",
                latex=r"$\alpha$",
            ),
            ParameterSpec(
                name="log10_rm",
                prior_type="uniform",
                prior_kwargs={"low": 0.0, "high": 6.0},
                unit="log10(rad/m^2)",
                description="Log10 of rotation measure",
                latex=r"$\log_{10}\mathrm{RM}$",
            ),
            ParameterSpec(
                name="linear_pol_frac",
                prior_type="uniform",
                prior_kwargs={"low": 0.0, "high": 1.0},
                unit="dimensionless",
                description="Linear polarisation fraction",
                latex=r"$\Pi_L$",
            ),
        ]

    def summary_stats(self, params: dict[str, float]) -> dict[str, float]:
        return {
            "fluence_jy_ms": 10.0 ** params["log10_fluence_jy_ms"],
            "W_ms": 10.0 ** params["log10_W_int_ms"],
            "DM": params["DM"],
            "tau_sc_ms": 10.0 ** params["log10_tau_sc_ms"],
            "spectral_index": params["spectral_index"],
            "linear_pol_frac": params["linear_pol_frac"],
        }


class GRBAfterglowFRB(BaseModel):
    """GRB prompt emission or afterglow FRB association.

    Alternative for FRB events with possible gamma-ray counterpart.
    """

    name = "GRBAfterglowFRB"
    channel = "radio"

    def parameters(self) -> list[ParameterSpec]:
        return [
            ParameterSpec(
                name="log10_fluence_jy_ms",
                prior_type="uniform",
                prior_kwargs={"low": -3.0, "high": 5.0},
                unit="log10(Jy ms)",
                description="Radio burst fluence",
                latex=r"$\log_{10}\mathcal{F}$",
            ),
            ParameterSpec(
                name="log10_T90_s",
                prior_type="uniform",
                prior_kwargs={"low": -3.0, "high": 3.0},
                unit="log10(s)",
                description="Log10 of prompt T90 duration",
                latex=r"$\log_{10}T_{90}$",
            ),
            ParameterSpec(
                name="z",
                prior_type="log_uniform",
                prior_kwargs={"low": 0.001, "high": 10.0},
                unit="dimensionless",
                description="Redshift",
                latex=r"$z$",
            ),
            ParameterSpec(
                # Same rename as magnetar's width, for the same reason: the
                # simulator reads 'spectral_index', so 'spectral_index_radio'
                # was never seen -- docs/RADIO_PREFLIGHT_AUDIT.md R.4.
                name="spectral_index",
                prior_type="normal",
                prior_kwargs={"mean": -0.6, "std": 0.5},
                unit="dimensionless",
                description="Afterglow radio spectral index",
                latex=r"$\alpha_r$",
            ),
            ParameterSpec(
                name="DM",
                prior_type="log_uniform",
                prior_kwargs={"low": 10.0, "high": 3000.0},
                unit="pc/cm^3",
                description="Dispersion measure",
                latex=r"$\mathrm{DM}$",
            ),
        ]

    def summary_stats(self, params: dict[str, float]) -> dict[str, float]:
        return {
            "fluence_jy_ms": 10.0 ** params["log10_fluence_jy_ms"],
            "T90_s": 10.0 ** params["log10_T90_s"],
            "z": params["z"],
            "spectral_index_radio": params["spectral_index"],
            "DM": params["DM"],
        }


# Analysis band the GW likelihood can actually see, taken from
# configs/instruments/ligo.yaml -> preprocessing.{low,high}_freq_cutoff, which
# is what GWLikelihood._parse_data() falls back to and what
# GWLikelihood._band_mask() then clips at min(high_freq_cutoff, 0.95*nyquist)
# (1945.6 Hz at the configured 4096 Hz sample rate, so high_freq_cutoff binds).
BAND_LOW_HZ = 20.0
BAND_HIGH_HZ = 1700.0


class StandardBHRingdown(BaseModel):
    """Phenomenological GR black hole ringdown (GW-channel comparison model).

    Damped-sinusoid ringdown at the Kerr QNM frequency and quality factor for
    ``(M, a_star)`` with **no** frequency or quality-factor deviations, and a
    free peak strain amplitude ``log10_A``.

    Why the amplitude is a free nuisance parameter rather than a function of
    ``(D_L, i)``: a single-detector, ringdown-only analysis cannot separate
    luminosity distance from inclination -- they enter only through one overall
    amplitude -- so this model does not attempt the physical decomposition that
    ``BlackToWhiteBounce`` makes.  ``bh_ringdown`` is the comparison model in
    this project, not the target of a distance/inclination measurement.
    ``D_L`` and ``i`` were previously declared here but never entered
    ``GWLikelihood._build_template()``'s bh_ringdown branch (which uses
    ``A_rd = 10**log10_A``), so the declared parameter vector did not match the
    likelihood's forward model.

    The ``M`` prior is bounded so that the Kerr QNM frequency stays inside the
    analysis band for *every* spin in the ``a_star`` prior; see
    ``_m_prior_bounds_for_band()``.
    """

    name = "StandardBHRingdown"
    channel = "gw"

    SPIN_MAX = 0.998

    @staticmethod
    def _m_prior_bounds_for_band(
        f_low: float = BAND_LOW_HZ,
        f_high: float = BAND_HIGH_HZ,
        spin_max: float = 0.998,
    ) -> tuple[float, float]:
        """Widest ``M`` range whose QNM frequency stays inside ``[f_low, f_high]``.

        ``kerr_qnm_frequency`` gives ``f = k(a) / M`` with ``k`` monotonically
        increasing in spin, so across ``a_star in [0, spin_max]`` the frequency
        for a given mass spans ``[k(0)/M, k(spin_max)/M]``.  Requiring both ends
        to stay in band for every spin gives
        ``M >= k(spin_max)/f_high`` and ``M <= k(0)/f_low``.

        This is deliberately conservative: the bound is set by the worst-case
        spin, so a lower-spin remnant lighter than the returned minimum would
        still be in band.  Making the constraint spin-dependent would need a
        joint ``(M, a_star)`` prior, which the independent-``ParameterSpec``
        prior machinery does not express.
        """
        from ..utils.math_utils import kerr_qnm_frequency

        k_low_spin = kerr_qnm_frequency(1.0, 0.0)[0]        # f at M = 1 Msun, a = 0
        k_high_spin = kerr_qnm_frequency(1.0, spin_max)[0]  # f at M = 1 Msun, a = spin_max
        return k_high_spin / f_high, k_low_spin / f_low

    def parameters(self) -> list[ParameterSpec]:
        return [
            ParameterSpec(
                name="M",
                prior_type="log_uniform",
                # [20, 590] sits inside the exact band-derived bounds
                # [19.13, 594.88] returned by _m_prior_bounds_for_band() at the
                # configured 20-1700 Hz band, with a small margin.  The old
                # [5, 1000] range mapped to f_rd in [11.9, 6505] Hz, so 17% of
                # prior draws were either rejected outright by _build_template()
                # or landed above the band mask, where the likelihood is blind.
                prior_kwargs={"low": 20.0, "high": 590.0},
                unit="M_sun",
                description="Final BH mass (bounded so f_QNM stays in the analysis band)",
                latex=r"$M$",
            ),
            ParameterSpec(
                name="a_star",
                prior_type="uniform",
                prior_kwargs={"low": 0.0, "high": self.SPIN_MAX},
                unit="dimensionless",
                description="Dimensionless spin",
                latex=r"$a_*$",
            ),
            ParameterSpec(
                name="log10_A",
                prior_type="uniform",
                prior_kwargs={"low": -24.0, "high": -18.0},
                unit="log10(strain)",
                description="Log10 peak ringdown strain (free amplitude nuisance parameter)",
                latex=r"$\log_{10} A$",
            ),
        ]

    def summary_stats(self, params: dict[str, float]) -> dict[str, float]:
        from ..utils.math_utils import kerr_qnm_frequency

        M = params["M"]
        a = params["a_star"]
        f_gr, q_gr = kerr_qnm_frequency(M, a)
        return {
            "f_qnm_hz": f_gr,
            "q_qnm": q_gr,
            "delta_f_hz": 0.0,
            "delta_Q": 0.0,
            "A": 10.0 ** params["log10_A"],
        }


class BHAccretion(BaseModel):
    """Black hole accretion-flow emission (image-channel alternative).

    The primary alternative to the GR eternal white hole on the image channel,
    and a genuinely different hypothesis rather than a relabelled one: here the
    ring's brightness and thickness are both consequences of a single accretion
    rate, and the ring is not azimuthally symmetric.

    Phenomenological in the same sense as ``StandardBHRingdown`` on the GW
    channel -- a declared scaling law with a free normalisation, not a
    radiative-transfer calculation:

    ``log10_mdot_edd``
        Accretion rate in Eddington units.  Sets the peak surface brightness
        (rising with rate) *and* the emission-ring thickness (falling with
        rate, as a radiatively inefficient thick flow gives way to a thin
        disc).  ``gr_eternal`` carries those as two independent free
        parameters; tying them is the substance of this hypothesis.
    ``jet_power_frac``
        Fraction of the accretion power emerging in the jet, which brightens
        the ring segment at the jet footpoint -- i.e. along the projected spin
        axis.  ``gr_eternal``'s ring has no azimuthal structure at all.

    Both laws live in ``simulators.image_shadow.ring_emission_from_params``,
    which the simulator and ``VisibilityLikelihood`` both call, so there is one
    copy of each expression rather than two that can drift.

    ``D_L`` is NOT a parameter.  Like ``gr_eternal``, this model reads the
    source distance from the analysis target; see ``utils.targets``.

    Parameters
    ----------
    target : str | ImageTarget | None
        Which source is being analysed ('M87*' or 'SgrA*').
    """

    name = "BHAccretion"
    channel = "image"

    #: See GREternalWhiteHole.requires_target.
    requires_target = True

    def __init__(self, target: "str | ImageTarget | None" = None) -> None:
        self.target = get_target(target) if target is not None else None

    @property
    def resolved_target(self) -> ImageTarget:
        """The analysis target, or raise (fail-closed; see GREternalWhiteHole)."""
        if self.target is None:
            raise ValueError(
                f"{self.name} needs to know which target it describes before it "
                "can state a mass prior or a shadow size: the source distance "
                "is a known constant on this channel, not a sampled parameter. "
                "Construct it as get_model('bh_accretion', target='M87*'), or "
                "use models.model_for_context(name, context)."
            )
        return self.target

    def parameters(self) -> list[ParameterSpec]:
        tgt = self.resolved_target
        return [
            ParameterSpec(
                name="M",
                prior_type="log_uniform",
                prior_kwargs={
                    "low": tgt.mass_prior_low_msun,
                    "high": tgt.mass_prior_high_msun,
                },
                unit="M_sun",
                description=(
                    f"BH mass of {tgt.name}; same per-target prior as "
                    f"gr_eternal so the two hypotheses are compared on the same "
                    f"geometry ({tgt.mass_source})"
                ),
                latex=r"$M$",
            ),
            ParameterSpec(
                name="a_star",
                prior_type="uniform",
                prior_kwargs={"low": 0.0, "high": SPIN_MAX},
                unit="dimensionless",
                description="Dimensionless spin",
                latex=r"$a_*$",
            ),
            ParameterSpec(
                name="i",
                prior_type="cos_uniform",
                prior_kwargs={},
                unit="rad",
                description="Inclination angle (0 = face-on); sets ring axial ratio",
                latex=r"$i$",
            ),
            ParameterSpec(
                name="position_angle",
                prior_type="uniform",
                prior_kwargs={"low": 0.0, "high": np.pi},
                unit="rad",
                description=(
                    "Position angle of the projected spin axis, which is also "
                    "where the jet footpoint brightens the ring"
                ),
                latex=r"$\xi$",
            ),
            ParameterSpec(
                name="log10_mdot_edd",
                prior_type="uniform",
                prior_kwargs={"low": -5.0, "high": 0.0},
                unit="log10(M_Edd)",
                description=(
                    "Log10 accretion rate in Eddington units; sets both ring "
                    "brightness and ring thickness"
                ),
                latex=r"$\log_{10}\dot{m}$",
            ),
            ParameterSpec(
                name="jet_power_frac",
                prior_type="log_uniform",
                prior_kwargs={"low": 0.01, "high": 1.0},
                unit="dimensionless",
                description=(
                    "Fraction of accretion power in the jet; sets the azimuthal "
                    "brightness contrast at the jet footpoint"
                ),
                latex=r"$f_\mathrm{jet}$",
            ),
        ]

    def summary_stats(self, params: dict[str, float]) -> dict[str, float]:
        from ..utils.constants import G, C, M_SUN, MPC_M, MUAS_RAD
        from ..simulators.image_shadow import ring_emission_from_params

        M = params["M"]
        a = params["a_star"]
        D_L = self.resolved_target.distance_mpc

        rg = G * M * M_SUN / C**2
        b_c = 3.0 * np.sqrt(3.0) * rg  # Schwarzschild approx
        theta_d_muas = 2.0 * b_c / (D_L * MPC_M) / MUAS_RAD

        # Same function the simulator and the likelihood call, so the reported
        # brightness/thickness are the ones actually imaged.
        emission = ring_emission_from_params(params)

        return {
            "theta_d_muas": theta_d_muas,
            "ring_width_muas": theta_d_muas * emission.ring_width_frac / 2.0,
            "ring_brightness": emission.brightness,
            "axial_ratio": float(np.abs(np.cos(params["i"]))),
            "jet_contrast": 1.0 + emission.asym_amp,
            "mdot_edd": 10.0 ** params["log10_mdot_edd"],
            "spin": a,
            "D_L_mpc": D_L,
            "jet_power_frac": params["jet_power_frac"],
        }
