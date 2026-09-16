"""Image/shadow forward simulator for EHT-like VLBI observations.

Implements a simplified Schwarzschild/Kerr shadow with a thin Gaussian
brightness ring, sampled at (u,v) points to produce complex visibilities.

Two emission hypotheses share the geometry and differ in how the ring is lit:

``gr_eternal``
    A free integrated flux (``log10_total_flux_jy``) and a free ring thickness
    (``ring_width_frac``), azimuthally symmetric.  The geometric baseline: the
    ring is as bright and as thick as the data want it to be.  The peak surface
    brightness is derived, ``I0 = F / G``, not sampled -- see
    :func:`build_ring_image` for why round that way.

``bh_accretion``
    Brightness and thickness both follow from one accretion rate
    (``log10_mdot_edd``), and the ring carries an azimuthal brightness
    enhancement at the jet footpoint whose contrast is set by
    ``jet_power_frac``.  Phenomenological in the style of the GW channel's
    ``bh_ringdown`` (a declared scaling law, not first-principles GRMHD), but a
    genuinely different hypothesis: it cannot produce a bright *thin* ring or a
    faint *thick* one, and it is not azimuthally symmetric.

Both are built by :func:`ring_emission_from_params`, which is the single place
either law is written down.  ``VisibilityLikelihood`` evaluates its model by
calling this simulator, and its ``predictive_summary_stats`` calls the same
function, so the injected and the modelled images cannot drift apart -- the
failure mode ``docs/BOUNCE_PREFLIGHT_AUDIT.md`` B.4 records on the GW side.

Optional ehtim / EinsteinPy integration is used when available for more
accurate ray-tracing and interferometric simulation.

Context keys
------------
target        : str — 'M87*' or 'SgrA*'; supplies the source distance, which
                this channel treats as a known constant rather than a sampled
                parameter.  REQUIRED (fail-closed).
uv_coverage   : ndarray, shape (N_baselines, 2) [Gλ] — baseline (u,v) coordinates
beam_fwhm_muas : float — synthesised beam FWHM [μas] (default 20.0)
thermal_noise_jy : float — per-baseline thermal noise [Jy] (default 0.05)
freq_ghz      : float — observing frequency [GHz] (default 230.0)
fov_muas      : float — image FoV half-width [μas] (default 200.0)
n_pixels      : int   — image grid size (default 128)
rng_seed      : int   — random seed
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import BaseSimulator, SimData
from ..utils.constants import G, C, M_SUN, MPC_M, MUAS_RAD
from ..utils.math_utils import kerr_shadow_radius_rg
from ..utils.targets import require_target


# ── bh_accretion emission law ─────────────────────────────────────────────────
# Phenomenological scalings, declared here once and used by both the simulator
# and the likelihood.  They are chosen for physical reasonableness, not fitted:
#
#  * Brightness rises with accretion rate.  Taken linear in log-log with unit
#    index, anchored so that an Eddington-rate flow peaks at 1 Jy/μas^2.
#    CORRECTED: this anchor was 10^2 Jy/μas^2, justified by an integral over a
#    "~250 μas^2 ring".  That arithmetic was wrong -- the Gaussian annulus
#    integrates to 2 pi r0 w sqrt(2 pi), which is 660-1900 μas^2 over the
#    thickness range here -- and it placed the WHOLE prior above reality: the
#    total flux ran 1.87 Jy at the prior's faint edge to 2.4e4 Jy at its bright
#    edge, against the 0.5-1.2 Jy of compact 230 GHz flux measured from M87*
#    (EHT 2019, ApJL 875 L1/L4) and 2.0-2.5 Jy from Sgr A* (EHT 2022, ApJL 930
#    L12).  At 1 Jy/μas^2 the prior spans 0.019-235 Jy total and both targets
#    sit near log10_mdot_edd ~ -3, comfortably inside it.  See audit X.13.5.
#  * Thickness FALLS with accretion rate.  M87* and Sgr A* are radiatively
#    inefficient flows, geometrically thick (H/R ~ 0.4) at low rates, and a
#    flow approaching Eddington collapses towards a thin disc (H/R ~ 0.05).
#    The index -0.18 interpolates between those two endpoints across the prior.
#    This is the qualitative difference from gr_eternal, where brightness and
#    thickness are independent free parameters.
#  * The jet footpoint brightens one azimuthal segment of the ring.  Contrast
#    is linear in jet_power_frac so the parameter cannot be inert anywhere in
#    its prior, and the segment sits at the position angle, i.e. along the
#    projected spin axis, which is where a jet base belongs.
LOG10_I0_EDD = 0.0                 # log10(Jy/μas^2) at mdot = 1 M_Edd
BRIGHTNESS_MDOT_INDEX = 1.0        # d log10 I0 / d log10 mdot
LOG10_W_FRAC_EDD = math.log10(0.05)  # ring thickness / radius at Eddington
W_FRAC_MDOT_INDEX = -0.18          # d log10 (w/r) / d log10 mdot
W_FRAC_MIN = 0.01                  # same physical band gr_eternal's prior spans
W_FRAC_MAX = 0.5
JET_CONTRAST_MAX = 3.0             # peak enhancement at jet_power_frac = 1
JET_FOOTPOINT_SIGMA_RAD = 0.6      # angular half-width of the footpoint (~34 deg)


@dataclass(frozen=True)
class RingEmission:
    """How a parameter vector lights the ring.

    ``emission_model`` is provenance, recorded in the simulator's metadata, so
    a stored SimData says which hypothesis produced it rather than leaving it
    to be inferred from which keys happen to be present.
    """

    emission_model: str
    #: "total_flux" -> `amplitude` is the integrated flux [Jy] and the peak
    #: surface brightness is derived from it; "peak_brightness" -> `amplitude`
    #: IS the peak surface brightness [Jy/muas^2] and the flux is derived.
    normalisation: str
    amplitude: float
    ring_width_frac: float
    asym_amp: float
    asym_azimuth_rad: float


def _require_param(params: dict[str, float], key: str, model: str) -> float:
    """Read a parameter the forward model genuinely uses, or raise.

    Fail-closed rather than ``params.get(key, default)``: a silent default
    makes a parameter that was never supplied look like one that was, which is
    how a model injected on the wrong channel produced a plausible-looking
    signal built entirely out of defaults (docs/RADIO_PREFLIGHT_AUDIT.md R.4).
    """
    if key not in params:
        raise KeyError(
            f"ImageShadowSimulator's {model!r} emission model needs {key!r}, "
            f"which the parameter vector does not carry. Supplied: "
            f"{sorted(params)}."
        )
    return float(params[key])


def ring_emission_from_params(params: dict[str, float]) -> RingEmission:
    """Resolve brightness, thickness and asymmetry from a parameter vector.

    The hypothesis is identified by which brightness parameter is present --
    the same convention the GW simulator uses to tell ``bh_ringdown``'s free
    ``log10_A`` from ``bounce``'s physical ``M / D_L`` amplitude.  Carrying
    both, or neither, is an error rather than a precedence rule.
    """
    has_ring = "log10_total_flux_jy" in params
    has_accretion = "log10_mdot_edd" in params

    if has_ring and has_accretion:
        raise KeyError(
            "Ambiguous image parameter vector: it carries both "
            "'log10_total_flux_jy' (gr_eternal) and 'log10_mdot_edd' "
            "(bh_accretion). These are different emission hypotheses; supply "
            "exactly one."
        )
    if not has_ring and not has_accretion:
        raise KeyError(
            "Image parameter vector carries neither 'log10_total_flux_jy' "
            "(gr_eternal) nor 'log10_mdot_edd' (bh_accretion), so there is no "
            f"way to light the ring. Supplied: {sorted(params)}."
        )

    if has_ring:
        return RingEmission(
            emission_model="gr_eternal",
            normalisation="total_flux",
            amplitude=float(
                10.0 ** _require_param(params, "log10_total_flux_jy", "gr_eternal")
            ),
            ring_width_frac=_require_param(params, "ring_width_frac", "gr_eternal"),
            asym_amp=0.0,
            asym_azimuth_rad=0.0,
        )

    log10_mdot = _require_param(params, "log10_mdot_edd", "bh_accretion")
    jet_frac = _require_param(params, "jet_power_frac", "bh_accretion")
    log10_w = LOG10_W_FRAC_EDD + W_FRAC_MDOT_INDEX * log10_mdot
    return RingEmission(
        emission_model="bh_accretion",
        # NOT reparameterised onto total flux: log10_mdot_edd driving both the
        # brightness AND the thickness is the substance of this hypothesis, and
        # normalising the flux away would delete the coupling that makes it
        # different from gr_eternal.  See audit X.16.4.
        normalisation="peak_brightness",
        amplitude=float(10.0 ** (LOG10_I0_EDD + BRIGHTNESS_MDOT_INDEX * log10_mdot)),
        ring_width_frac=float(np.clip(10.0**log10_w, W_FRAC_MIN, W_FRAC_MAX)),
        asym_amp=float(JET_CONTRAST_MAX * jet_frac),
        # Zero in the ring's own (position-angle-rotated) frame, i.e. the
        # footpoint follows the projected spin axis on sky.  This is what makes
        # position_angle observable even face-on, where the ellipse rotation
        # alone is a no-op.
        asym_azimuth_rad=0.0,
    )


def _shadow_radius_muas(
    M_msun: float,
    a_star: float,
    D_L_mpc: float,
    inclination_rad: float = 0.0,
) -> float:
    """Angular shadow radius [μas].

    Delegates the spin/inclination dependence to
    ``utils.math_utils.kerr_shadow_radius_rg``, a fit to the exact Bardeen
    critical curve accurate to 0.363%.  This function used to apply a local
    ``(1 - 0.0136 a + 0.0038 a^2)`` factor whose spin dependence was ~6x too
    weak, while ``GREternalWhiteHole.photon_ring_radius_m`` applied a different
    and worse one, so the injected ring and the reported shadow diameter
    disagreed by up to 28% at high spin.  Both now call one function.
    """
    rg = G * M_msun * M_SUN / C**2  # gravitational radius [m]
    b_c = kerr_shadow_radius_rg(a_star, inclination_rad) * rg
    D_L_m = D_L_mpc * MPC_M
    return float(b_c / D_L_m / MUAS_RAD)


def _gaussian_ring_image(
    fov_muas: float,
    n_pix: int,
    r_ring_muas: float,
    w_ring_muas: float,
    brightness: float,
    axial_ratio: float = 1.0,
    pos_angle_rad: float = 0.0,
    asym_amp: float = 0.0,
    asym_azimuth_rad: float = 0.0,
    asym_sigma_rad: float = JET_FOOTPOINT_SIGMA_RAD,
) -> NDArray:
    """Generate a 2D Gaussian brightness ring image.

    Returns image in [Jy/μas^2], shape (n_pix, n_pix).
    Origin at centre; x-axis West (RA), y-axis North (Dec).

    ``asym_amp`` adds an azimuthal brightness enhancement -- a jet-footpoint /
    hot-spot segment -- multiplying the ring by
    ``1 + asym_amp * exp(-0.5 (Δφ / asym_sigma_rad)^2)``, where ``Δφ`` is the
    wrapped azimuthal separation from ``asym_azimuth_rad`` measured in the
    ring's own frame (so the segment rotates with ``pos_angle_rad``).
    ``asym_amp = 0`` leaves the axisymmetric ring exactly unchanged, which is
    the gr_eternal case.
    """
    dx = 2.0 * fov_muas / n_pix
    coords = np.linspace(-fov_muas, fov_muas, n_pix)
    xx, yy = np.meshgrid(coords, coords)

    # Rotate coordinates by position angle
    cos_pa = np.cos(pos_angle_rad)
    sin_pa = np.sin(pos_angle_rad)
    xr = xx * cos_pa + yy * sin_pa
    yr = -xx * sin_pa + yy * cos_pa

    # Elliptical ring (axial ratio compresses one axis)
    r_ellipse = np.sqrt(xr**2 + (yr / max(axial_ratio, 0.01)) ** 2)

    # Gaussian annulus
    image = brightness * np.exp(-0.5 * ((r_ellipse - r_ring_muas) / w_ring_muas) ** 2)

    if asym_amp != 0.0:
        azimuth = np.arctan2(yr, xr)
        d_phi = np.angle(np.exp(1j * (azimuth - asym_azimuth_rad)))  # wrap to (-π, π]
        image = image * (
            1.0 + asym_amp * np.exp(-0.5 * (d_phi / asym_sigma_rad) ** 2)
        )
    return image


def _compute_visibilities(
    image: NDArray,
    fov_muas: float,
    uv_coverage: NDArray,
    freq_ghz: float | None = None,
) -> NDArray:
    """Sample the Fourier transform of the image at (u,v) coordinates.

    Two things were wrong here and are fixed together, because fixing either
    one alone leaves the sampler wrong (audit X.13).

    **Baseline units.**  The old code did
    ``uv_rad = uv_coverage * 1e9 * wavelength_m``.  A baseline quoted in Gλ is
    *already* an angular frequency -- fringe cycles per radian -- so multiplying
    by the observing wavelength converts it into the baseline's physical length
    in metres instead.  At 230 GHz that shrank every EHT baseline by a factor
    ``1 / 1.3037e-3 = 767``, putting all of them inside the single FFT cell
    containing the uv origin, where the visibility is just the zero-spacing
    flux.  Model visibilities were therefore equal to the image's total flux on
    every baseline and carried no structural information at all.  ``freq_ghz``
    is consequently no longer used; it is accepted so existing callers keep
    working.

    **Sampling method.**  The old code FFT'd the image onto a regular uv grid
    and bilinearly interpolated onto the requested points.  That grid's spacing
    is ``1 / (2 * fov)``, so the *image* field of view silently sets the *uv*
    accuracy: at the 50 μas field this project images, the spacing is 2.06 Gλ,
    and interpolating EHT baselines across it is wrong by up to 9.0% of the peak
    (measured against the analytic Hankel transform of a Gaussian ring).  A
    direct DFT evaluated at the requested points has no such coupling; the same
    check puts it at 0.76%, which is the pixelisation of the image itself rather
    than the sampler.  It is also cheap, because the transform is separable:
    ~1.2 ms for a 128x128 image at 16 baselines.

    Parameters
    ----------
    image : ndarray, shape (N, N) [Jy/μas^2]
    fov_muas : float — half-width of the image [μas]
    uv_coverage : ndarray, shape (M, 2) — baseline coordinates [Gλ]
    freq_ghz : unused; retained for call compatibility.

    Returns
    -------
    visibilities : complex ndarray, shape (M,) [Jy]
    """
    n_pix = image.shape[0]
    dx_muas = 2.0 * fov_muas / n_pix
    coords_rad = np.linspace(-fov_muas, fov_muas, n_pix) * MUAS_RAD

    uv = np.atleast_2d(np.asarray(uv_coverage, dtype=float))
    # Gλ -> λ, i.e. fringe cycles per radian.  No wavelength factor: see above.
    u = uv[:, 0] * 1.0e9
    v = uv[:, 1] * 1.0e9

    # V(u, v) = ∫∫ I(x, y) exp(-2πi (ux + vy)) dx dy, separable in x and y.
    phase_x = np.exp(-2.0j * np.pi * np.outer(u, coords_rad))  # (M, N)
    phase_y = np.exp(-2.0j * np.pi * np.outer(v, coords_rad))  # (M, N)
    return np.einsum("mx,yx,my->m", phase_x, image, phase_y) * dx_muas**2


class UnrepresentableRingError(ValueError):
    """The image grid cannot represent this ring at all.

    Raised when the ring integrates to zero flux at unit brightness, so there
    is no peak brightness that puts the sampled total flux on it.  In practice
    this is the near-edge-on corner: ``axial_ratio = |cos i|`` collapses the
    annulus towards a line, and a thin ring there becomes a sub-pixel needle
    that falls between grid points.

    ``VisibilityLikelihood`` turns this into ``-inf`` rather than letting it
    abort a sampling run -- which is also what the previous parameterisation
    did numerically, since those geometries imaged to ~zero flux and scored an
    arbitrarily bad likelihood.  It is a forward-model limitation, NOT a
    consequence of sampling the flux; see audit X.16.5.
    """


@dataclass(frozen=True)
class RingImage:
    """A built ring image plus everything derived on the way to it."""

    image: NDArray
    brightness: float              # peak surface brightness actually used [Jy/μas^2]
    total_flux_jy: float           # integrated flux of `image`
    geometry_factor_muas2: float   # G: the flux this geometry gives at I0 = 1
    r_ring_muas: float
    w_ring_muas: float
    axial_ratio: float
    emission: RingEmission


def build_ring_image(params: dict[str, float], context: dict[str, Any]) -> RingImage:
    """Build the source image for a parameter vector.

    The single place the image is constructed, so the simulator, the likelihood
    and the predictive summary statistics cannot drift apart.

    **Why the flux, not the brightness, is the sampled quantity for
    gr_eternal.**  The image's integrated flux is ``F = I0 * G``, where the
    geometry factor ``G`` is the flux the same ring would carry at unit peak
    brightness.  ``G`` depends on ``M``, ``a_star``, ``i`` and
    ``ring_width_frac``, and across this model's prior it spans 2.264 dex
    (5-95%).  Putting the prior on ``I0`` therefore meant that "this source
    emits about 1 Jy" -- the one thing actually known about M87* and Sgr A* --
    could not be stated as a prior at all: the geometry leaked into the flux
    and half the prior mass landed at network SNR above 1000, which is not a
    regime EHT data occupies.  Sampling ``log10_total_flux_jy`` and inverting
    ``I0 = F / G`` puts the prior on the quantity that is measured and that the
    short baselines constrain, and leaves ``I0`` as the derived quantity.
    Same move as the burst-timing reparameterisation in
    docs/BOUNCE_PREFLIGHT_AUDIT.md B3-2.  See audit X.16.

    ``G`` is evaluated numerically on the image grid rather than from the
    analytic annulus integral, so the normalisation absorbs pixelisation and
    the imaged flux equals the sampled flux exactly, not just to within the
    discretisation error.
    """
    fov_muas = float(context.get("fov_muas", 200.0))
    n_pix = int(context.get("n_pixels", 128))
    target = require_target(context)

    M = _require_param(params, "M", "image")
    a = _require_param(params, "a_star", "image")
    i = _require_param(params, "i", "image")
    pos_angle = _require_param(params, "position_angle", "image")
    emission = ring_emission_from_params(params)

    r_ring = _shadow_radius_muas(M, a, target.distance_mpc, inclination_rad=i)
    w_ring = r_ring * emission.ring_width_frac
    axial_ratio = float(np.abs(np.cos(i)))
    pixel_area = (2.0 * fov_muas / n_pix) ** 2

    unit_image = _gaussian_ring_image(
        fov_muas,
        n_pix,
        r_ring,
        w_ring,
        1.0,
        axial_ratio,
        pos_angle,
        asym_amp=emission.asym_amp,
        asym_azimuth_rad=emission.asym_azimuth_rad,
    )
    geometry_factor = float(unit_image.sum() * pixel_area)

    if emission.normalisation == "total_flux":
        if not np.isfinite(geometry_factor) or geometry_factor <= 0.0:
            # The grid cannot represent this ring at all, so there is no
            # brightness that puts the requested flux on it.  Fail closed
            # rather than divide by ~0 and emit an image whose flux is right
            # only because it was forced into one or two stray pixels.
            raise UnrepresentableRingError(
                "Cannot normalise the ring to its sampled total flux: this "
                f"geometry images to {geometry_factor!r} μas^2 at unit "
                f"brightness (r_ring={r_ring:.4g} μas, w_ring={w_ring:.4g} μas, "
                f"axial_ratio={axial_ratio:.4g} on a {n_pix}px/{fov_muas}μas "
                "grid), i.e. it falls between pixels."
            )
        brightness = emission.amplitude / geometry_factor
    elif emission.normalisation == "peak_brightness":
        brightness = emission.amplitude
    else:  # pragma: no cover - guarded by ring_emission_from_params
        raise ValueError(
            f"Unknown ring normalisation {emission.normalisation!r}; "
            "expected 'total_flux' or 'peak_brightness'."
        )

    image = unit_image * brightness
    return RingImage(
        image=image,
        brightness=float(brightness),
        total_flux_jy=float(image.sum() * pixel_area),
        geometry_factor_muas2=geometry_factor,
        r_ring_muas=r_ring,
        w_ring_muas=w_ring,
        axial_ratio=axial_ratio,
        emission=emission,
    )


class ImageShadowSimulator(BaseSimulator):
    """Toy image/shadow forward simulator for EHT-like VLBI observations.

    Produces complex visibilities at requested (u,v) coverage plus
    closure phases and closure amplitudes derived from baseline triangles.
    """

    channel = "image"

    def simulate(
        self,
        params: dict[str, float],
        context: dict[str, Any],
        rng: np.random.Generator | None = None,
    ) -> SimData:
        if rng is None:
            rng = np.random.default_rng(context.get("rng_seed", None))

        # ── Instrument ─────────────────────────────────────────────────────────
        uv_coverage = np.asarray(
            context.get("uv_coverage", _default_eht_uv()),
            dtype=float,
        )
        thermal_noise_jy = float(context.get("thermal_noise_jy", 0.05))
        freq_ghz = float(context.get("freq_ghz", 230.0))
        fov_muas = float(context.get("fov_muas", 200.0))
        n_pix = int(context.get("n_pixels", 128))

        # ── Source ─────────────────────────────────────────────────────────────
        # The distance is a property of the target, not of the sample: it is
        # read from the analysis context and raises when absent.  Sampling it
        # alongside M made the ring radius (which depends only on M / D_L)
        # unidentifiable and unrepresentable -- audit X.6.
        target = require_target(context)
        D_L = target.distance_mpc

        # ── Build image ────────────────────────────────────────────────────────
        built = build_ring_image(params, context)
        image = built.image
        emission = built.emission
        r_ring = built.r_ring_muas
        w_ring = built.w_ring_muas

        # ── Sample visibilities ────────────────────────────────────────────────
        vis_signal = _compute_visibilities(image, fov_muas, uv_coverage, freq_ghz)

        # Add complex Gaussian noise
        noise_re = rng.standard_normal(len(vis_signal)) * thermal_noise_jy
        noise_im = rng.standard_normal(len(vis_signal)) * thermal_noise_jy
        noise = noise_re + 1j * noise_im
        visibilities = vis_signal + noise

        # ── Closure quantities (first triangle) ────────────────────────────────
        closure_phases = _compute_closure_phases(visibilities)

        return SimData(
            channel="image",
            data=visibilities,
            metadata={
                "image": image,
                "uv_coverage": uv_coverage,
                "freq_ghz": freq_ghz,
                "fov_muas": fov_muas,
                "n_pix": n_pix,
                "r_ring_muas": r_ring,
                "w_ring_muas": w_ring,
                "thermal_noise_jy": thermal_noise_jy,
                "closure_phases": closure_phases,
                "vis_signal": vis_signal,
                # Provenance: which target's distance was used, and which
                # emission hypothesis lit the ring.
                "target": target.name,
                "D_L_mpc": D_L,
                "emission_model": emission.emission_model,
                "normalisation": emission.normalisation,
                "total_flux_jy": built.total_flux_jy,
                "geometry_factor_muas2": built.geometry_factor_muas2,
                "ring_width_frac": emission.ring_width_frac,
                "brightness_jy_per_muas2": built.brightness,
                "asym_amp": emission.asym_amp,
            },
            params_true=params,
            noise_realisation=noise,
        )


def _compute_closure_phases(visibilities: NDArray) -> NDArray:
    """Phase closure statistic over sequential baseline triplets.

    KNOWN LIMITATION, recorded rather than silently tolerated: the triplets are
    taken as consecutive entries of the visibility array, and the shipped
    ``_default_eht_uv()`` baselines do not close -- for triplet 0,
    ``(0.5, 0.2) + (0.5, -0.2) = (1.0, 0.0)``, not the third entry
    ``(0.2, 0.5)``.  So this is a phase combination, not a closure phase, and
    it does NOT carry the station-gain invariance closure phases exist for.
    Both the simulator and the likelihood compute it with this same function,
    so it is a self-consistent statistic of the data and does not bias
    inference; it simply is not the robust observable its name claims.  Fixing
    it means deriving the uv coverage from the station positions in
    configs/instruments/eht.yaml so real triangles exist.  See
    docs/XRAY_IMAGE_PREFLIGHT_AUDIT.md X.13.4.

    The wrap below used to be ``closure % (2π) - π``, which maps a zero phase
    closure to -π rather than to 0 -- a constant π offset from the convention
    ``EHTLoader.compute_closure_phases`` uses.  The von Mises likelihood
    compares model against data through this same function, so the offset
    cancelled there, but the values stored in the observation metadata were
    wrong by π.
    """
    n = len(visibilities)
    n_triangles = n // 3
    phases = np.angle(visibilities)
    closure = np.zeros(n_triangles)
    for k in range(n_triangles):
        i, j, l = 3 * k, 3 * k + 1, 3 * k + 2
        closure[k] = phases[i] + phases[j] - phases[l]
    return (closure + np.pi) % (2.0 * np.pi) - np.pi  # wrap to (-π, π]


def _default_eht_uv() -> NDArray:
    """Return a toy EHT-like (u,v) coverage at 230 GHz [Gλ].

    Based on approximate SMTO-SMA-JCMT-IRAM EHT 2017 baselines.
    """
    return np.array([
        [0.5, 0.2], [0.5, -0.2], [0.2, 0.5], [0.2, -0.5],
        [1.0, 0.3], [1.0, -0.3], [0.8, 0.8], [0.8, -0.8],
        [3.2, 0.5], [3.2, -0.5], [2.1, 1.5], [2.1, -1.5],
        [6.5, 1.0], [6.5, -1.0], [4.0, 3.0], [4.0, -3.0],
    ], dtype=float)
