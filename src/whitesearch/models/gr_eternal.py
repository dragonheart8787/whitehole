"""GR Eternal White Hole model.

Represents the time-reversed region of the Schwarzschild/Kerr maximal extension.
Serves as the theoretical baseline rather than the primary observational target:
a GR eternal white hole has no natural astrophysical formation channel, but defines
the simplest geometry for image/shadow comparisons.

Observable: primarily the image channel (shadow angular diameter, photon ring
ellipticity, brightness distribution) and possible quasi-thermal radio continuum.
"""

from __future__ import annotations

import numpy as np

from .base import BaseModel, ParameterSpec
from ..utils.constants import G, C, M_SUN, MPC_M, MUAS_RAD


class GREternalWhiteHole(BaseModel):
    """Parametric GR eternal white hole with Schwarzschild/Kerr geometry.

    The shadow angular radius is derived analytically; emission is modelled
    as a thin ring of Gaussian brightness around the photon orbit.

    Parameters
    ----------
    include_charge : bool
        Include the electric charge Q as a free parameter (Kerr-Newman).
        Default False (pure Kerr).
    """

    name = "GREternalWhiteHole"
    channel = "image"

    def __init__(self, include_charge: bool = False) -> None:
        self.include_charge = include_charge

    def parameters(self) -> list[ParameterSpec]:
        params = [
            ParameterSpec(
                name="M",
                prior_type="log_uniform",
                prior_kwargs={"low": 1e6, "high": 1e10},
                unit="M_sun",
                description="BH/WH mass (supermassive; target: VLBI-resolvable sources)",
                latex=r"$M$",
            ),
            ParameterSpec(
                name="a_star",
                prior_type="uniform",
                prior_kwargs={"low": 0.0, "high": 0.998},
                unit="dimensionless",
                description="Dimensionless spin |a*| = |J|c / (GM^2)",
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
                description="Inclination angle (0 = face-on)",
                latex=r"$i$",
            ),
            ParameterSpec(
                name="position_angle",
                prior_type="uniform",
                prior_kwargs={"low": 0.0, "high": np.pi},
                unit="rad",
                description="Position angle of the spin axis on sky",
                latex=r"$\xi$",
            ),
            ParameterSpec(
                name="log10_ne",
                prior_type="uniform",
                prior_kwargs={"low": -7.0, "high": 2.0},
                unit="log10(cm^{-3})",
                description="Log10 of external electron number density",
                latex=r"$\log_{10} n_e$",
            ),
            ParameterSpec(
                name="log10_B",
                prior_type="uniform",
                prior_kwargs={"low": -9.0, "high": 0.0},
                unit="log10(Gauss)",
                description="Log10 of ambient magnetic field strength",
                latex=r"$\log_{10} B$",
            ),
            ParameterSpec(
                name="ring_width_frac",
                prior_type="log_uniform",
                prior_kwargs={"low": 0.01, "high": 0.5},
                unit="dimensionless",
                description="Ring width as fraction of photon-ring radius",
                latex=r"$w_r$",
            ),
            ParameterSpec(
                name="log10_brightness",
                prior_type="uniform",
                prior_kwargs={"low": -4.0, "high": 2.0},
                unit="log10(Jy/μas^2)",
                description="Log10 of peak ring surface brightness",
                latex=r"$\log_{10} I_0$",
            ),
        ]
        if self.include_charge:
            params.insert(
                2,
                ParameterSpec(
                    name="Q_norm",
                    prior_type="uniform",
                    prior_kwargs={"low": 0.0, "high": 0.999},
                    unit="dimensionless",
                    description="Normalized charge Q / Q_max (Kerr-Newman)",
                    latex=r"$Q_N$",
                ),
            )
        return params

    # ── Geometry ──────────────────────────────────────────────────────────────

    def photon_ring_radius_m(self, M_msun: float, a_star: float) -> float:
        """Critical photon orbit impact parameter b_c [m].

        For Schwarzschild: b_c = 3√3 GM/c².
        For Kerr (prograde/retrograde average): approximate formula.
        """
        rg = G * M_msun * M_SUN / C**2  # gravitational radius [m]
        if a_star == 0.0:
            return 3.0 * np.sqrt(3.0) * rg
        # Approximate: photon-ring radius interpolation (Bardeen 1973)
        # r_ph(a*) ≈ 3rg * (1 - 0.0136 * a_star + ...)
        # Use an improved fitting formula (Chan+ 2015):
        b_plus = rg * (3.0 + np.sqrt(9.0 - 8.0 * a_star**2))  # approximate outer photon orbit
        return float(b_plus)

    def shadow_angular_diameter_muas(
        self,
        M_msun: float,
        a_star: float,
        D_L_mpc: float,
    ) -> float:
        """Shadow angular diameter [μas]."""
        b_c = self.photon_ring_radius_m(M_msun, a_star)
        D_L_m = D_L_mpc * MPC_M
        theta_rad = b_c / D_L_m  # half-angle
        return float(2.0 * theta_rad / MUAS_RAD)

    # ── Summary statistics ─────────────────────────────────────────────────────

    def summary_stats(self, params: dict[str, float]) -> dict[str, float]:
        """Return key observable summary statistics.

        Summary stats
        -------------
        theta_d_muas : Shadow angular diameter [μas]
        ring_width_muas : Ring FWHM [μas]
        axial_ratio : Ring ellipticity from inclination (cos i)
        ne : Electron density [cm^{-3}]
        B_Gauss : Magnetic field [Gauss]
        ring_brightness : Peak surface brightness [Jy/μas^2]
        """
        M = params["M"]
        a = params["a_star"]
        D_L = params["D_L"]
        i = params["i"]

        theta_d = self.shadow_angular_diameter_muas(M, a, D_L)
        ring_width = theta_d * params["ring_width_frac"] / 2.0

        return {
            "theta_d_muas": theta_d,
            "ring_width_muas": ring_width,
            "axial_ratio": float(np.abs(np.cos(i))),
            "ne": float(10.0 ** params["log10_ne"]),
            "B_Gauss": float(10.0 ** params["log10_B"]),
            "ring_brightness": float(10.0 ** params["log10_brightness"]),
            "spin_a_star": a,
            "mass_msun": M,
            "dist_mpc": D_L,
        }
