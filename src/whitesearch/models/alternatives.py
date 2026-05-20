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


class StandardBHRingdown(BaseModel):
    """Standard GR black hole ringdown (alternative to bounce model in GW channel).

    No frequency or quality-factor deviations from Kerr QNM predictions.
    """

    name = "StandardBHRingdown"
    channel = "gw"

    def parameters(self) -> list[ParameterSpec]:
        return [
            ParameterSpec(
                name="M",
                prior_type="log_uniform",
                prior_kwargs={"low": 5.0, "high": 1000.0},
                unit="M_sun",
                description="Final BH mass",
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
                name="log10_A",
                prior_type="uniform",
                prior_kwargs={"low": -24.0, "high": -18.0},
                unit="log10(strain)",
                description="Log10 peak ringdown strain",
                latex=r"$\log_{10} A$",
            ),
            ParameterSpec(
                name="D_L",
                prior_type="volume_uniform",
                prior_kwargs={"low": 10.0, "high": 10000.0},
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
            "D_L_mpc": params["D_L"],
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
