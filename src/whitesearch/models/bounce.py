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
# Single source of truth for the GW analysis band (from
# configs/instruments/ligo.yaml preprocessing.{low,high}_freq_cutoff).
from .alternatives import BAND_HIGH_HZ, BAND_LOW_HZ
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

    SPIN_MAX = 0.998
    EPS_F_RANGE = (-0.3, 0.3)

    @classmethod
    def _m_prior_bounds_for_band(
        cls,
        f_low: float = BAND_LOW_HZ,
        f_high: float = BAND_HIGH_HZ,
    ) -> tuple[float, float]:
        """Widest ``M`` range whose ringdown frequency stays inside the band.

        ``kerr_qnm_frequency`` gives ``f = k(a) / M`` with ``k`` monotonically
        increasing in spin, and the bounce model multiplies it by
        ``(1 + eps_f)``.  Across ``a_star in [0, SPIN_MAX]`` and
        ``eps_f in EPS_F_RANGE`` the frequency for a given mass therefore spans
        ``[k(0)(1+eps_f_min)/M, k(SPIN_MAX)(1+eps_f_max)/M]``.  Requiring both
        ends to stay in band for every ``(a_star, eps_f)`` gives

            M >= k(SPIN_MAX) * (1 + eps_f_max) / f_high
            M <= k(0)        * (1 + eps_f_min) / f_low

        Same construction as ``StandardBHRingdown._m_prior_bounds_for_band()``,
        with the extra ``eps_f`` factor bounce applies to the GR frequency.

        Deliberately conservative: the bounds are set by the worst-case
        ``(spin, eps_f)`` corner, so a lower-spin or negative-``eps_f`` remnant
        lighter than the returned minimum would still be in band.  Making the
        constraint joint would need a joint ``(M, a_star, eps_f)`` prior, which
        the independent-``ParameterSpec`` machinery cannot express.
        """
        f_at_unit_mass_low_spin = kerr_qnm_frequency(1.0, 0.0)[0]
        f_at_unit_mass_high_spin = kerr_qnm_frequency(1.0, cls.SPIN_MAX)[0]
        eps_f_min, eps_f_max = cls.EPS_F_RANGE
        return (
            f_at_unit_mass_high_spin * (1.0 + eps_f_max) / f_high,
            f_at_unit_mass_low_spin * (1.0 + eps_f_min) / f_low,
        )

    def parameters(self) -> list[ParameterSpec]:
        return [
            ParameterSpec(
                name="M",
                prior_type="log_uniform",
                # [25, 415] sits inside the exact band-derived bounds
                # [24.872086, 416.417493] from _m_prior_bounds_for_band() at
                # the configured 20-1700 Hz band, with a small margin.  These
                # are the conservative bounds set by the highest spin combined
                # with the eps_f upper bound (and the lowest spin with the
                # eps_f lower bound at the top end).  The old [5, 1000] range
                # mapped to f_rd in [8.97, 6860.64] Hz, so 14.06% of prior
                # draws were rejected outright by
                # GWLikelihood._build_template() and 16.46% fell outside the
                # band mask, where the likelihood is blind.  See
                # docs/BOUNCE_PREFLIGHT_AUDIT.md section B.5.
                prior_kwargs={"low": 25.0, "high": 415.0},
                unit="M_sun",
                description=(
                    "Initial black hole mass at merger "
                    "(bounded so f_QNM*(1+eps_f) stays in the analysis band)"
                ),
                latex=r"$M$",
            ),
            ParameterSpec(
                name="a_star",
                prior_type="uniform",
                prior_kwargs={"low": 0.0, "high": self.SPIN_MAX},
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
                prior_kwargs={"low": self.EPS_F_RANGE[0], "high": self.EPS_F_RANGE[1]},
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
