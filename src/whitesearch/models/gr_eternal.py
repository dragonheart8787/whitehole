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


# EHT imaging grid this project analyses, from configs/instruments/eht.yaml ->
# imaging.{fov_muas, n_pixels}, which are also what cli.py's default image
# context is built from.  tests/test_image_forward_model.py asserts these
# against that file, so the YAML stays the source of truth without a model
# doing import-time file IO (same convention as BAND_LOW_HZ/BAND_HIGH_HZ on the
# GW side).
IMAGE_FOV_MUAS = 200.0
IMAGE_N_PIXELS = 128
IMAGE_PIXEL_MUAS = 2.0 * IMAGE_FOV_MUAS / IMAGE_N_PIXELS  # 3.1250 muas

# Smallest ring radius, in pixels, at which the Gaussian annulus is sampled
# well enough that its peak survives for EVERY ring_width_frac the prior allows.
# Measured, not assumed: _gaussian_ring_image() puts the ring at
# exp(-0.5 ((r_grid - r_ring) / (r_ring * ring_width_frac))^2), so a thin ring
# (frac = 0.01) is missed entirely unless a grid point lands close to r_ring.
# Sweeping frac over [0.01, 0.5] the worst case first exceeds half the peak
# brightness at 8.245 pixels; below that the response is not even monotonic in
# r (worst case 0.1217 at 3 px, 0.0003 at 4 px, 0.9749 at 5 px) because it
# depends on whether r_ring happens to be commensurate with the pixel grid.
RING_RADIUS_MIN_PIXELS = 8.245


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

    @classmethod
    def ring_radius_representable_range_muas(
        cls,
        fov_muas: float = IMAGE_FOV_MUAS,
        n_pixels: int = IMAGE_N_PIXELS,
    ) -> tuple[float, float]:
        """Ring radii the image grid can actually represent, in μas.

        Lower bound: ``RING_RADIUS_MIN_PIXELS`` pixels, the point at which the
        annulus peak survives for every ``ring_width_frac`` in the prior.
        Upper bound: the field half-width, beyond which the ring leaves the
        image entirely.  At the shipped 200 μas / 128 px geometry this is
        ``[25.77, 200.0]`` μas -- a window of only 0.89 dex.

        NOT CURRENTLY USED TO SET THE PRIOR, deliberately.  The image
        constrains the ring radius, which depends on ``M`` and ``D_L`` only
        through the ratio ``r ∝ M / D_L`` (numerically
        ``r[μas] = 5.130245e-08 * M[M_sun] / D_L[Mpc]``).  With independent
        log-uniform priors the radius therefore spans
        ``dex(M) + dex(D_L)``: 4.000 + 3.301 = 7.301 dex against a window of
        0.89-1.20 dex, so only 18.29% of prior draws land inside it and 80.98%
        fall below one pixel and produce an exactly-zero image.

        Forcing ~100% representability by narrowing both priors would require
        ``dex(M) + dex(D_L) <= 1.204`` -- for instance a factor of 4 in mass AND
        a factor of 4 in distance.  That is narrower than the sources this
        channel exists to describe: M87* (6.5e9 M_sun at 16.8 Mpc) and Sgr A*
        (4.15e6 M_sun at 0.008178 Mpc) differ by 3.19 dex in mass and 3.31 dex
        in distance, yet both sit at r = 19.7 and 25.8 μas because the two
        co-vary.  Narrowing the marginals cannot express that correlation.

        The structural fix is a reparameterisation onto the ratio the data
        constrains -- the analogue of the burst-delay reparameterisation in
        BOUNCE_PREFLIGHT_AUDIT.md B3-2 -- which is a design decision, not a
        mechanical narrowing.  Recorded in docs/XRAY_IMAGE_PREFLIGHT_AUDIT.md
        X.6 and X.11.1.
        """
        pixel_muas = 2.0 * float(fov_muas) / int(n_pixels)
        return (RING_RADIUS_MIN_PIXELS * pixel_muas, float(fov_muas))

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
