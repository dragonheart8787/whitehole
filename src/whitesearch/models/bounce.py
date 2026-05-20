"""Black-to-white bounce model.

A black hole core undergoes a quantum-gravity-mediated transition to a white hole
after a conversion timescale τ_bounce.  Key references:
  - Haggard & Rovelli (2015): non-singular black-hole / white-hole metric
  - Bianchi, Christodoulou et al. (2018): LQG bounce amplitude
  - Lifetime scaling: τ ~ k M^p, with p ∈ {4, 5} under debate

Primary observable channel: gravitational waves (ringdown deviation, bounce burst).
Secondary: coherent radio or MeV gamma outflow from outgoing shell.
"""

from __future__ import annotations

import numpy as np

from .base import BaseModel, ParameterSpec
from ..utils.constants import (
    G, C, M_SUN, MPC_M, GYR_S,
    F_QNM_SCHW, T_QNM_SCHW, Q_QNM_SCHW,
)
from ..utils.math_utils import kerr_qnm_frequency


class BlackToWhiteBounce(BaseModel):
    """Black-to-white bounce parametric model.

    The GW signal has two components:
      1. Standard ringdown (h_rd) modified by bounce parameter ε_f, ε_Q
      2. Bounce burst (h_bounce) at t = τ_bounce with amplitude ∝ M / D_L
    """

    name = "BlackToWhiteBounce"
    channel = "gw"

    def parameters(self) -> list[ParameterSpec]:
        return [
            ParameterSpec(
                name="M",
                prior_type="log_uniform",
                prior_kwargs={"low": 5.0, "high": 1000.0},
                unit="M_sun",
                description="Initial black hole mass at merger",
                latex=r"$M$",
            ),
            ParameterSpec(
                name="a_star",
                prior_type="uniform",
                prior_kwargs={"low": 0.0, "high": 0.998},
                unit="dimensionless",
                description="Final BH dimensionless spin",
                latex=r"$a_*$",
            ),
            ParameterSpec(
                name="log10_tau_bounce_yr",
                prior_type="uniform",
                prior_kwargs={"low": -3.0, "high": 10.0},
                unit="log10(yr)",
                description="Log10 of BH-to-WH conversion timescale in years",
                latex=r"$\log_{10}(\tau_{b}/\mathrm{yr})$",
            ),
            ParameterSpec(
                name="log10_ell_q",
                prior_type="uniform",
                prior_kwargs={"low": 0.0, "high": 6.0},
                unit="log10(l_Planck)",
                description="Log10 of quantum length scale in Planck units",
                latex=r"$\log_{10}(\ell_q/l_P)$",
            ),
            ParameterSpec(
                name="p_lifetime",
                prior_type="discrete_uniform",
                prior_kwargs={"values": [4, 5]},
                unit="dimensionless",
                description="Lifetime scaling exponent: τ ~ k M^p",
                latex=r"$p$",
            ),
            ParameterSpec(
                name="eps_f",
                prior_type="uniform",
                prior_kwargs={"low": -0.3, "high": 0.3},
                unit="dimensionless",
                description="Fractional shift in ringdown frequency from GR: f = f_GR * (1 + ε_f)",
                latex=r"$\varepsilon_f$",
            ),
            ParameterSpec(
                name="eps_Q",
                prior_type="uniform",
                prior_kwargs={"low": -0.5, "high": 0.5},
                unit="dimensionless",
                description="Fractional shift in quality factor from GR: Q = Q_GR * (1 + ε_Q)",
                latex=r"$\varepsilon_Q$",
            ),
            ParameterSpec(
                name="log10_A_bounce",
                prior_type="uniform",
                prior_kwargs={"low": -25.0, "high": -18.0},
                unit="log10(strain)",
                description="Log10 peak strain of the bounce burst",
                latex=r"$\log_{10} A_b$",
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
            ParameterSpec(
                name="eta_r",
                prior_type="log_uniform",
                prior_kwargs={"low": 1e-10, "high": 1.0},
                unit="dimensionless",
                description="Radio emission efficiency of white hole outflow",
                latex=r"$\eta_r$",
            ),
            ParameterSpec(
                name="eta_gamma",
                prior_type="log_uniform",
                prior_kwargs={"low": 1e-10, "high": 1.0},
                unit="dimensionless",
                description="Gamma-ray emission efficiency of white hole outflow",
                latex=r"$\eta_\gamma$",
            ),
        ]

    # ── Derived quantities ─────────────────────────────────────────────────────

    def gr_ringdown_params(
        self,
        M_msun: float,
        a_star: float,
    ) -> tuple[float, float]:
        """Return (f_GR [Hz], Q_GR) from GR Kerr QNM fitting formulae."""
        return kerr_qnm_frequency(M_msun, a_star)

    def modified_ringdown_params(
        self,
        M_msun: float,
        a_star: float,
        eps_f: float,
        eps_Q: float,
    ) -> tuple[float, float]:
        """Return bounce-modified (f_mod, Q_mod)."""
        f_gr, q_gr = self.gr_ringdown_params(M_msun, a_star)
        return f_gr * (1.0 + eps_f), q_gr * (1.0 + eps_Q)

    def tau_bounce_s(
        self,
        log10_tau_yr: float,
    ) -> float:
        """Convert log10(τ [yr]) to τ [s]."""
        return float(10.0 ** log10_tau_yr * GYR_S / 1e9)

    # ── Summary statistics ─────────────────────────────────────────────────────

    def summary_stats(self, params: dict[str, float]) -> dict[str, float]:
        """Observable summary statistics for the GW channel.

        Includes ringdown frequency deviation, quality factor deviation,
        and bounce burst properties.
        """
        M = params["M"]
        a = params["a_star"]
        eps_f = params["eps_f"]
        eps_Q = params["eps_Q"]

        f_gr, q_gr = self.gr_ringdown_params(M, a)
        f_mod, q_mod = self.modified_ringdown_params(M, a, eps_f, eps_Q)
        tau = self.tau_bounce_s(params["log10_tau_bounce_yr"])

        # GW horizon amplitude: h ~ G M c^{-2} / D_L (characteristic strain)
        D_L_m = params["D_L"] * MPC_M
        h_char = G * M * M_SUN / (C**2 * D_L_m)

        # Bounce burst timing relative to merger
        tau_decay_s = q_gr / (np.pi * f_gr)  # ringdown e-folding time

        return {
            "f_gr_hz": f_gr,
            "q_gr": q_gr,
            "f_mod_hz": f_mod,
            "q_mod": q_mod,
            "delta_f_hz": f_mod - f_gr,
            "delta_Q": q_mod - q_gr,
            "tau_bounce_s": tau,
            "tau_over_tau_decay": tau / tau_decay_s if tau_decay_s > 0 else np.inf,
            "h_char": h_char,
            "A_bounce": float(10.0 ** params["log10_A_bounce"]),
            "mass_msun": M,
            "spin": a,
            "dist_mpc": params["D_L"],
        }
