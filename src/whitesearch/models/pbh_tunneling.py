"""Primordial Black Hole quantum tunneling to white hole model.

A PBH formed in the early universe undergoes quantum tunneling to a white hole
after a Hawking-evaporation-plus-tunneling lifetime τ ~ k M².  When the mass
reaches a critical value today, it explosively releases energy across radio,
optical, and MeV gamma-ray channels.

Key references:
  - Barrau, Cailleteau et al. (2014): PBH → WH tunneling
  - Barrau, Rovelli & Vidotto (2014): flash of gamma-rays
  - Bianchi et al. (2023): FRB contribution from PBH WH

Primary observable channel: radio (FRB-like coherent burst) + MeV gamma.
"""

from __future__ import annotations

import numpy as np

from .base import BaseModel, ParameterSpec
from ..utils.constants import (
    G, C, HBAR, M_SUN, M_PLANCK, MPC_M,
    K_DM, DM_IGM_PER_Z, H0, OMEGA_M, OMEGA_LAMBDA,
)


# PBH mass range for observable events today (in grams and solar masses)
# τ_Hawking ~ 5120π G² M³ / (ℏ c⁴) → observable for M ~ 10^{14–15} g
PBH_MASS_MIN_G = 1e13   # grams
PBH_MASS_MAX_G = 1e16   # grams
PBH_MASS_MIN_MSUN = PBH_MASS_MIN_G / 2e33
PBH_MASS_MAX_MSUN = PBH_MASS_MAX_G / 2e33


class PBHTunnelingWhiteHole(BaseModel):
    """PBH quantum tunneling parametric model for radio/EM channel.

    A single-event FRB-like burst is produced when a PBH of mass M tunnels
    to a white hole.  The event rate and fluence depend on f_PBH, M, and
    the tunneling coefficient k.
    """

    name = "PBHTunnelingWhiteHole"
    channel = "radio"

    def parameters(self) -> list[ParameterSpec]:
        return [
            ParameterSpec(
                name="log10_M_g",
                prior_type="uniform",
                prior_kwargs={"low": np.log10(PBH_MASS_MIN_G), "high": np.log10(PBH_MASS_MAX_G)},
                unit="log10(g)",
                description="Log10 of PBH mass in grams",
                latex=r"$\log_{10}(M/\mathrm{g})$",
            ),
            ParameterSpec(
                name="log10_f_pbh",
                prior_type="uniform",
                prior_kwargs={"low": -10.0, "high": 0.0},
                unit="log10(dimensionless)",
                description="Log10 of PBH fraction of dark matter",
                latex=r"$\log_{10} f_\mathrm{PBH}$",
            ),
            ParameterSpec(
                name="log10_k_tunnel",
                prior_type="uniform",
                prior_kwargs={"low": -3.0, "high": 3.0},
                unit="log10(dimensionless)",
                description="Log10 of quantum-gravity tunneling coefficient k (τ ~ k M^2)",
                latex=r"$\log_{10} k$",
            ),
            ParameterSpec(
                name="log10_eta_r",
                prior_type="uniform",
                prior_kwargs={"low": -10.0, "high": 0.0},
                unit="log10(dimensionless)",
                description="Log10 of coherent radio emission efficiency",
                latex=r"$\log_{10}\eta_r$",
            ),
            ParameterSpec(
                name="log10_eta_gamma",
                prior_type="uniform",
                prior_kwargs={"low": -10.0, "high": 0.0},
                unit="log10(dimensionless)",
                description="Log10 of gamma-ray emission efficiency",
                latex=r"$\log_{10}\eta_\gamma$",
            ),
            ParameterSpec(
                name="z",
                prior_type="log_uniform",
                prior_kwargs={"low": 1e-4, "high": 5.0},
                unit="dimensionless",
                description="Redshift of the event",
                latex=r"$z$",
            ),
            ParameterSpec(
                name="DM_host",
                prior_type="log_uniform",
                prior_kwargs={"low": 1.0, "high": 1000.0},
                unit="pc/cm^3",
                description="Host galaxy DM contribution",
                latex=r"$\mathrm{DM_{host}}$",
            ),
            ParameterSpec(
                name="log10_W_int_ms",
                prior_type="uniform",
                prior_kwargs={"low": -1.0, "high": 3.0},
                unit="log10(ms)",
                description="Log10 of intrinsic burst width in ms",
                latex=r"$\log_{10}(W_\mathrm{int}/\mathrm{ms})$",
            ),
            ParameterSpec(
                name="log10_tau_sc_ms",
                prior_type="uniform",
                prior_kwargs={"low": -2.0, "high": 2.0},
                unit="log10(ms)",
                description="Log10 of scattering timescale in ms (at 1 GHz)",
                latex=r"$\log_{10}(\tau_\mathrm{sc}/\mathrm{ms})$",
            ),
            ParameterSpec(
                name="spectral_index",
                prior_type="normal",
                prior_kwargs={"mean": -1.5, "std": 1.0},
                unit="dimensionless",
                description="Radio spectral index α (F_ν ∝ ν^α)",
                latex=r"$\alpha$",
            ),
        ]

    # ── Derived physical quantities ────────────────────────────────────────────

    @staticmethod
    def hawking_lifetime_s(M_g: float) -> float:
        """Hawking evaporation lifetime τ_H ≈ 5120π G² M³ / (ℏ c⁴) [s].

        Parameters
        ----------
        M_g : float — PBH mass in grams
        """
        M_kg = M_g * 1e-3
        return float(5120.0 * np.pi * G**2 * M_kg**3 / (HBAR * C**4))

    @staticmethod
    def tunneling_lifetime_s(M_g: float, k: float) -> float:
        """Quantum tunneling lifetime τ_t = k (M/M_Planck)² × t_Planck [s].

        This is the Planck-star model lifetime.
        """
        from ..utils.constants import T_PLANCK
        M_kg = M_g * 1e-3
        ratio = M_kg / M_PLANCK
        return float(k * ratio**2 * T_PLANCK)

    @staticmethod
    def igm_dm(z: float) -> float:
        """Average IGM DM contribution: DM_IGM ≈ 855 z [pc/cm^3] (Macquart+ 2020)."""
        return DM_IGM_PER_Z * z

    def total_dm(self, params: dict[str, float]) -> float:
        """DM_total = DM_MW (100 pc/cm^3 placeholder) + DM_IGM + DM_host."""
        dm_mw = 100.0  # pc/cm^3 — placeholder; real runs use NE2001/YMW16
        dm_igm = self.igm_dm(params["z"])
        dm_host = params["DM_host"]
        return dm_mw + dm_igm + dm_host

    def burst_fluence_jy_ms(self, params: dict[str, float], nu_ref_ghz: float = 1.0) -> float:
        """Estimate radio burst fluence [Jy ms] at reference frequency ν_ref.

        F ≈ η_r * E_tot / (4π D_L² * Δν * W)
        where Δν ≈ 1 GHz (typical bandwidth) and W is the observed width.
        """
        from ..utils.constants import JY, GPC_M

        M_g = 10.0 ** params["log10_M_g"]
        M_kg = M_g * 1e-3
        eta_r = 10.0 ** params["log10_eta_r"]
        z = params["z"]

        # Total released energy ~ M c²
        E_tot = M_kg * C**2  # J

        # Comoving / luminosity distance (simple flat ΛCDM approximation)
        # D_L ≈ z * c / H0  for z ≪ 1  (use proper integrator for z > 0.5)
        D_L_mpc = self._dl_mpc(z)
        D_L_m = D_L_mpc * MPC_M

        # Observed width
        W_int_ms = 10.0 ** params["log10_W_int_ms"]
        tau_sc_ms = 10.0 ** params["log10_tau_sc_ms"]
        W_obs_ms = np.sqrt(W_int_ms**2 + tau_sc_ms**2)

        # Bandwidth (1 GHz default)
        delta_nu_hz = 1.0e9

        fluence_j_per_hz = eta_r * E_tot / (4.0 * np.pi * D_L_m**2 * delta_nu_hz)
        # Convert to Jy·ms: 1 Jy·ms = 1e-26 W/m²/Hz * 1e-3 s = 1e-29 J/m²/Hz
        fluence_jy_ms = fluence_j_per_hz / (JY * 1e-3)

        return float(fluence_jy_ms)

    @staticmethod
    def _dl_mpc(z: float) -> float:
        """Flat ΛCDM luminosity distance using a simple trapezoidal integral."""
        if z <= 0:
            return 0.0
        z_arr = np.linspace(0, z, 500)
        integrand = 1.0 / np.sqrt(
            OMEGA_M * (1.0 + z_arr) ** 3 + OMEGA_LAMBDA
        )
        chi_integral = np.trapezoid(integrand, z_arr)
        c_km_s = C / 1e3  # km/s
        d_c_mpc = (c_km_s / H0) * chi_integral
        return float((1.0 + z) * d_c_mpc)

    # ── Summary statistics ─────────────────────────────────────────────────────

    def summary_stats(self, params: dict[str, float]) -> dict[str, float]:
        M_g = 10.0 ** params["log10_M_g"]
        tau_h = self.hawking_lifetime_s(M_g)
        k = 10.0 ** params["log10_k_tunnel"]
        tau_t = self.tunneling_lifetime_s(M_g, k)
        dm_total = self.total_dm(params)
        fluence = self.burst_fluence_jy_ms(params)
        D_L = self._dl_mpc(params["z"])

        return {
            "M_g": M_g,
            "tau_hawking_s": tau_h,
            "tau_tunneling_s": tau_t,
            "tau_eff_s": min(tau_h, tau_t),
            "DM_total": dm_total,
            "W_int_ms": float(10.0 ** params["log10_W_int_ms"]),
            "tau_sc_ms": float(10.0 ** params["log10_tau_sc_ms"]),
            "W_obs_ms": float(
                np.sqrt(
                    (10.0 ** params["log10_W_int_ms"]) ** 2
                    + (10.0 ** params["log10_tau_sc_ms"]) ** 2
                )
            ),
            "fluence_jy_ms": fluence,
            "spectral_index": params["spectral_index"],
            "z": params["z"],
            "D_L_mpc": D_L,
        }
