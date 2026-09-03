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
- BHAccretion           : standard thin-disk black hole accretion (image channel)
"""

from __future__ import annotations

import numpy as np

from .base import BaseModel, ParameterSpec


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
                name="log10_W_ms",
                prior_type="uniform",
                prior_kwargs={"low": -1.0, "high": 3.0},
                unit="log10(ms)",
                description="Log10 observed burst width",
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
            "W_ms": 10.0 ** params["log10_W_ms"],
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
                name="spectral_index_radio",
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
            "spectral_index_radio": params["spectral_index_radio"],
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
    """Standard thin-disk black hole accretion (image channel alternative).

    Used as the primary alternative to GR eternal white hole in image comparisons.
    """

    name = "BHAccretion"
    channel = "image"

    def parameters(self) -> list[ParameterSpec]:
        return [
            ParameterSpec(
                name="M",
                prior_type="log_uniform",
                prior_kwargs={"low": 1e6, "high": 1e10},
                unit="M_sun",
                description="BH mass",
                latex=r"$M$",
            ),
            ParameterSpec(
                name="a_star",
                prior_type="uniform",
                prior_kwargs={"low": 0.0, "high": 0.998},
                unit="dimensionless",
                description="Dimensionless spin",
                latex=r"$a_*$",
            ),
            ParameterSpec(
                name="D_L",
                prior_type="log_uniform",
                prior_kwargs={"low": 1.0, "high": 2000.0},
                unit="Mpc",
                description="Luminosity distance",
                latex=r"$D_L$",
            ),
            ParameterSpec(
                name="i",
                prior_type="cos_uniform",
                prior_kwargs={},
                unit="rad",
                description="Inclination angle",
                latex=r"$i$",
            ),
            ParameterSpec(
                name="log10_mdot_edd",
                prior_type="uniform",
                prior_kwargs={"low": -5.0, "high": 0.0},
                unit="log10(M_Edd)",
                description="Log10 accretion rate in Eddington units",
                latex=r"$\log_{10}\dot{m}$",
            ),
            ParameterSpec(
                name="jet_power_frac",
                prior_type="log_uniform",
                prior_kwargs={"low": 0.01, "high": 1.0},
                unit="dimensionless",
                description="Fraction of accretion power in jet",
                latex=r"$f_\mathrm{jet}$",
            ),
        ]

    def summary_stats(self, params: dict[str, float]) -> dict[str, float]:
        from ..utils.constants import G, C, M_SUN, MPC_M, MUAS_RAD
        from ..utils.math_utils import kerr_qnm_frequency

        M = params["M"]
        a = params["a_star"]
        D_L = params["D_L"]

        rg = G * M * M_SUN / C**2
        b_c = 3.0 * np.sqrt(3.0) * rg  # Schwarzschild approx
        theta_d_muas = 2.0 * b_c / (D_L * MPC_M) / MUAS_RAD

        return {
            "theta_d_muas": theta_d_muas,
            "mdot_edd": 10.0 ** params["log10_mdot_edd"],
            "spin": a,
            "D_L_mpc": D_L,
            "jet_power_frac": params["jet_power_frac"],
        }
