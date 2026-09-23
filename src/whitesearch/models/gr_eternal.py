"""GR Eternal White Hole model.

Represents the time-reversed region of the Schwarzschild/Kerr maximal extension.
Serves as the theoretical baseline rather than the primary observational target:
a GR eternal white hole has no natural astrophysical formation channel, but defines
the simplest geometry for image/shadow comparisons.

Observable: primarily the image channel (shadow angular diameter, photon ring
ellipticity, brightness distribution) and possible quasi-thermal radio continuum.
"""

from __future__ import annotations

import math

import numpy as np

from .base import BaseModel, ParameterSpec
from ..utils.constants import G, C, M_SUN, MPC_M, MUAS_RAD
from ..utils.math_utils import kerr_shadow_radius_rg
from ..utils.targets import ImageTarget, get_target


# EHT imaging grid this project analyses, from configs/instruments/eht.yaml ->
# imaging.{fov_muas, n_pixels}, which are also what cli.py's default image
# context is built from.  tests/test_image_forward_model.py asserts these
# against that file, so the YAML stays the source of truth without a model
# doing import-time file IO (same convention as BAND_LOW_HZ/BAND_HIGH_HZ on the
# GW side).
IMAGE_FOV_MUAS = 50.0
IMAGE_N_PIXELS = 128
IMAGE_PIXEL_MUAS = 2.0 * IMAGE_FOV_MUAS / IMAGE_N_PIXELS  # 0.78125 muas

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

#: Spin at which the Kerr shadow is smallest, i.e. the most demanding case for
#: the representability floor.  Matches the a_star prior's upper edge.
SPIN_MAX = 0.998


def _unit_mass_ring_radius_muas(a_star: float, distance_mpc: float) -> float:
    """Ring radius [muas] of a 1 M_sun source, so radius = M * this.

    ``_shadow_radius_muas`` is linear in mass, so one evaluation inverts it.
    Deliberately calls the simulator's function rather than reproducing the
    formula: this module derives what the *forward model* can represent, and a
    second copy of the expression would be free to drift away from it.
    """
    from ..simulators.image_shadow import _shadow_radius_muas

    return _shadow_radius_muas(1.0, a_star, distance_mpc)


class GREternalWhiteHole(BaseModel):
    """Parametric GR eternal white hole with Schwarzschild/Kerr geometry.

    The shadow angular radius is derived analytically; emission is modelled
    as a thin ring of Gaussian brightness around the photon orbit.

    Parameters
    ----------
    include_charge : bool
        Include the electric charge Q as a free parameter (Kerr-Newman).
        Default False (pure Kerr).
    target : str | ImageTarget | None
        Which source is being analysed ('M87*' or 'SgrA*').  Supplies the
        distance, which this channel treats as a known constant rather than a
        sampled parameter, and the mass prior.  ``None`` is accepted at
        construction so that channel-compatibility checks can instantiate the
        class without committing to a target, but every method that needs the
        distance raises until one is set -- see ``resolved_target``.
    """

    name = "GREternalWhiteHole"
    channel = "image"

    #: The per-target distance/mass constants are mandatory for this model;
    #: ``models.model_for_context`` reads this to know it must supply one.
    requires_target = True

    def __init__(
        self,
        include_charge: bool = False,
        target: str | ImageTarget | None = None,
    ) -> None:
        self.include_charge = include_charge
        self.target = get_target(target) if target is not None else None

    @property
    def resolved_target(self) -> ImageTarget:
        """The analysis target, or raise.

        Fail-closed rather than defaulting to M87*: the two supported targets'
        distances differ by 3.31 dex, so a guessed default would put the ring
        radius off by the same factor.
        """
        if self.target is None:
            raise ValueError(
                f"{self.name} needs to know which target it describes before it "
                "can state a mass prior or a shadow size: the source distance "
                "is a known constant on this channel, not a sampled parameter. "
                "Construct it as get_model('gr_eternal', target='M87*'), or use "
                "models.model_for_context(name, context) to take the target "
                "from the analysis context."
            )
        return self.target

    # ── What the image grid can represent ─────────────────────────────────────

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
        image entirely.  At the shipped 50 μas / 128 px geometry this is
        ``[6.44, 50.0]`` μas, which covers both targets' full mass priors
        (9.07-30.24 μas for M87*, 21.74-31.06 μas for Sgr A*).  The field used
        to be 200 μas, whose 3.125 μas pixels put the floor at 25.77 μas --
        above M87*'s ring entirely.
        """
        pixel_muas = 2.0 * float(fov_muas) / int(n_pixels)
        return (RING_RADIUS_MIN_PIXELS * pixel_muas, float(fov_muas))

    @classmethod
    def mass_representable_range_msun(
        cls,
        target: str | ImageTarget,
        a_star: float = SPIN_MAX,
        fov_muas: float = IMAGE_FOV_MUAS,
        n_pixels: int = IMAGE_N_PIXELS,
    ) -> tuple[float, float]:
        """Masses whose ring the grid can represent, at this target's distance.

        The image constrains the ring radius, which is
        ``r[μas] = 5.130e-08 * M[M_sun] / D_L[Mpc]`` up to a <1% Kerr spin
        correction.  With ``D_L`` fixed by the target this inverts directly, so
        ``ring_radius_representable_range_muas`` becomes a statement about
        ``M`` alone -- which is what makes a per-target mass prior checkable.
        """
        tgt = get_target(target)
        lo_r, hi_r = cls.ring_radius_representable_range_muas(fov_muas, n_pixels)
        unit = _unit_mass_ring_radius_muas(a_star, tgt.distance_mpc)
        return (lo_r / unit, hi_r / unit)

    @classmethod
    def mass_prior_representable_fraction(
        cls,
        target: str | ImageTarget,
        a_star: float = SPIN_MAX,
        fov_muas: float = IMAGE_FOV_MUAS,
        n_pixels: int = IMAGE_N_PIXELS,
    ) -> float:
        """Fraction of the target's log-uniform mass prior the grid represents.

        Analytic rather than sampled: the prior is log-uniform and the window
        is an interval in ``M``, so the fraction is the overlap in dex.
        """
        tgt = get_target(target)
        m_lo, m_hi = cls.mass_representable_range_msun(
            tgt, a_star=a_star, fov_muas=fov_muas, n_pixels=n_pixels
        )
        lo = max(math.log10(tgt.mass_prior_low_msun), math.log10(m_lo))
        hi = min(math.log10(tgt.mass_prior_high_msun), math.log10(m_hi))
        span = math.log10(tgt.mass_prior_high_msun) - math.log10(tgt.mass_prior_low_msun)
        return max(0.0, hi - lo) / span

    @classmethod
    def required_n_pixels_for_prior(
        cls,
        target: str | ImageTarget,
        a_star: float = SPIN_MAX,
        fov_muas: float = IMAGE_FOV_MUAS,
    ) -> int:
        """Smallest ``n_pixels`` at which the whole mass prior is representable.

        Only the prior's *lightest* mass binds: the floor scales as
        ``1 / n_pixels`` while the ceiling (the field half-width) is untouched
        by resolution.
        """
        tgt = get_target(target)
        r_min = tgt.mass_prior_low_msun * _unit_mass_ring_radius_muas(
            a_star, tgt.distance_mpc
        )
        return int(math.ceil(RING_RADIUS_MIN_PIXELS * 2.0 * float(fov_muas) / r_min))

    @classmethod
    def required_fov_muas_for_prior(
        cls,
        target: str | ImageTarget,
        a_star: float = SPIN_MAX,
        n_pixels: int = IMAGE_N_PIXELS,
    ) -> tuple[float, float]:
        """Field half-widths at which the whole mass prior is representable.

        Returns ``(fov_min, fov_max)``.  ``fov_max`` is set by the floor -- the
        pixel size, hence the floor, shrinks with the field, so a *smaller*
        field represents a *smaller* ring.  ``fov_min`` is set by the ceiling:
        the largest ring in the prior still has to fit inside the image.  An
        empty interval means no field half-width works at this ``n_pixels``.
        """
        tgt = get_target(target)
        unit = _unit_mass_ring_radius_muas(a_star, tgt.distance_mpc)
        r_min = tgt.mass_prior_low_msun * unit
        r_max = tgt.mass_prior_high_msun * unit
        fov_max = r_min * int(n_pixels) / (2.0 * RING_RADIUS_MIN_PIXELS)
        return (r_max, fov_max)

    # ── Parameters ────────────────────────────────────────────────────────────

    def parameters(self) -> list[ParameterSpec]:
        tgt = self.resolved_target
        params = [
            ParameterSpec(
                name="M",
                prior_type="log_uniform",
                prior_kwargs={
                    "low": tgt.mass_prior_low_msun,
                    "high": tgt.mass_prior_high_msun,
                },
                unit="M_sun",
                description=(
                    f"BH/WH mass of {tgt.name}; prior brackets the independent "
                    f"published measurements ({tgt.mass_source})"
                ),
                latex=r"$M$",
            ),
            ParameterSpec(
                name="a_star",
                prior_type="uniform",
                prior_kwargs={"low": 0.0, "high": SPIN_MAX},
                unit="dimensionless",
                description="Dimensionless spin |a*| = |J|c / (GM^2)",
                latex=r"$a_*$",
            ),
            # D_L is NOT sampled.  The image constrains the ring radius, which
            # depends on M and D_L only through M / D_L, so a free distance adds
            # a direction the data cannot constrain: with the previous
            # log_uniform(1, 2000) Mpc prior the ring radius spanned 7.30 dex
            # against a representable window of 0.89 dex and 80.98% of draws
            # produced an exactly-zero image (audit X.6).  The distance is now a
            # per-target known constant from utils.targets.
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
            # The amplitude is the ring's INTEGRATED flux, not its peak
            # surface brightness.  Peak brightness used to be the sampled
            # quantity, with the flux derived as F = I0 * G(M, a*, i, w); since
            # G spans 2.264 dex across this prior, the measured flux of the
            # target could not be expressed as a prior on I0, the prior's
            # median total flux came out at 11.3 Jy against M87*'s measured
            # 0.5-1.2 Jy, and 47.8% of prior draws landed above network SNR
            # 1000.  Sampling the flux and inverting for I0 puts the prior on
            # the quantity that is measured and that the short baselines
            # constrain.  Bounds are per target and carry one decade of margin
            # either side of the published flux; see utils.targets and audit
            # X.16.
            ParameterSpec(
                name="log10_total_flux_jy",
                prior_type="uniform",
                prior_kwargs={
                    "low": float(np.log10(tgt.flux_prior_low_jy)),
                    "high": float(np.log10(tgt.flux_prior_high_jy)),
                },
                unit="log10(Jy)",
                description=(
                    f"Log10 of the ring's integrated 230 GHz flux; {tgt.name} "
                    f"is measured at {tgt.flux_low_jy}-{tgt.flux_high_jy} Jy "
                    f"({tgt.flux_source})"
                ),
                latex=r"$\log_{10} F$",
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

    def photon_ring_radius_m(
        self,
        M_msun: float,
        a_star: float,
        inclination_rad: float = 0.0,
    ) -> float:
        """Critical photon orbit impact parameter b_c [m].

        The shared ``kerr_shadow_radius_rg`` fit to the exact Bardeen critical
        curve (0.363% accurate), which is also what the image simulator builds
        its annulus from.

        This method used to return ``rg (3 + sqrt(9 - 8 a^2))`` for a* != 0,
        which is wrong three ways: it is discontinuous at a* = 0 (6 rg against
        the correct 5.196 rg), it makes the shadow *grow* with spin up to
        a* ~ 0.4 when the true shadow shrinks monotonically, and its error runs
        from -16.9% to +13.7%.  Meanwhile the simulator applied a different
        approximation, so this model's reported shadow diameter and the ring it
        actually imaged disagreed by up to 28% at a* = 0.998.
        """
        rg = G * M_msun * M_SUN / C**2  # gravitational radius [m]
        return float(kerr_shadow_radius_rg(a_star, inclination_rad) * rg)

    def shadow_angular_diameter_muas(
        self,
        M_msun: float,
        a_star: float,
        D_L_mpc: float,
        inclination_rad: float = 0.0,
    ) -> float:
        """Shadow angular diameter [μas]."""
        b_c = self.photon_ring_radius_m(M_msun, a_star, inclination_rad)
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
        D_L = self.resolved_target.distance_mpc
        i = params["i"]

        theta_d = self.shadow_angular_diameter_muas(M, a, D_L, inclination_rad=i)
        ring_width = theta_d * params["ring_width_frac"] / 2.0

        return {
            "theta_d_muas": theta_d,
            "ring_width_muas": ring_width,
            "axial_ratio": float(np.abs(np.cos(i))),
            "ne": float(10.0 ** params["log10_ne"]),
            "B_Gauss": float(10.0 ** params["log10_B"]),
            # The sampled amplitude is the integrated flux; the peak surface
            # brightness is a derived quantity and needs the image grid to
            # compute, so it is reported by
            # VisibilityLikelihood.predictive_summary_stats rather than here.
            "total_flux_jy": float(10.0 ** params["log10_total_flux_jy"]),
            "spin_a_star": a,
            "mass_msun": M,
            "dist_mpc": D_L,
        }
