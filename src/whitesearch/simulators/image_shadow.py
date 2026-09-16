"""Image/shadow forward simulator for EHT-like VLBI observations.

Implements a simplified Schwarzschild/Kerr shadow with a thin Gaussian
brightness ring, sampled at (u,v) points to produce complex visibilities.

Two emission hypotheses share the geometry and differ in how the ring is lit:

``gr_eternal``
    A free peak surface brightness (``log10_brightness``) and a free ring
    thickness (``ring_width_frac``), azimuthally symmetric.  The geometric
    baseline: the ring is as bright and as thick as the data want it to be.

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
from ..utils.targets import require_target


# ── bh_accretion emission law ─────────────────────────────────────────────────
# Phenomenological scalings, declared here once and used by both the simulator
# and the likelihood.  They are chosen for physical reasonableness, not fitted:
#
#  * Brightness rises with accretion rate.  Taken linear in log-log with unit
#    index, anchored so that an Eddington-rate flow peaks at 10^2 Jy/μas^2;
#    over the [-5, 0] prior on log10_mdot_edd this spans 10^-3 .. 10^2, which
#    brackets the ~0.5-1 Jy of compact 230 GHz flux actually seen from M87*
#    once integrated over a ~250 μas^2 ring.
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
LOG10_I0_EDD = 2.0                 # log10(Jy/μas^2) at mdot = 1 M_Edd
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
    brightness: float
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
    has_ring = "log10_brightness" in params
    has_accretion = "log10_mdot_edd" in params

    if has_ring and has_accretion:
        raise KeyError(
            "Ambiguous image parameter vector: it carries both "
            "'log10_brightness' (gr_eternal) and 'log10_mdot_edd' "
            "(bh_accretion). These are different emission hypotheses; supply "
            "exactly one."
        )
    if not has_ring and not has_accretion:
        raise KeyError(
            "Image parameter vector carries neither 'log10_brightness' "
            "(gr_eternal) nor 'log10_mdot_edd' (bh_accretion), so there is no "
            f"way to light the ring. Supplied: {sorted(params)}."
        )

    if has_ring:
        return RingEmission(
            emission_model="gr_eternal",
            brightness=float(10.0 ** _require_param(params, "log10_brightness", "gr_eternal")),
            ring_width_frac=_require_param(params, "ring_width_frac", "gr_eternal"),
            asym_amp=0.0,
            asym_azimuth_rad=0.0,
        )

    log10_mdot = _require_param(params, "log10_mdot_edd", "bh_accretion")
    jet_frac = _require_param(params, "jet_power_frac", "bh_accretion")
    log10_w = LOG10_W_FRAC_EDD + W_FRAC_MDOT_INDEX * log10_mdot
    return RingEmission(
        emission_model="bh_accretion",
        brightness=float(10.0 ** (LOG10_I0_EDD + BRIGHTNESS_MDOT_INDEX * log10_mdot)),
        ring_width_frac=float(np.clip(10.0**log10_w, W_FRAC_MIN, W_FRAC_MAX)),
        asym_amp=float(JET_CONTRAST_MAX * jet_frac),
        # Zero in the ring's own (position-angle-rotated) frame, i.e. the
        # footpoint follows the projected spin axis on sky.  This is what makes
        # position_angle observable even face-on, where the ellipse rotation
        # alone is a no-op.
        asym_azimuth_rad=0.0,
    )


def _shadow_radius_muas(M_msun: float, a_star: float, D_L_mpc: float) -> float:
    """Angular shadow radius [μas]."""
    rg = G * M_msun * M_SUN / C**2  # gravitational radius [m]
    # Photon ring approximate: b_c ≈ 3√3 rg for Schwarzschild
    b_c = 3.0 * np.sqrt(3.0) * rg
    # For Kerr: approximate correction (Bardeen 1973; prograde-retrograde avg)
    a = np.clip(np.abs(a_star), 0.0, 0.998)
    b_c_kerr = b_c * (1.0 - 0.0136 * a + 0.0038 * a**2)
    D_L_m = D_L_mpc * MPC_M
    return float(b_c_kerr / D_L_m / MUAS_RAD)


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
    freq_ghz: float,
) -> NDArray:
    """Sample the Fourier transform of the image at (u,v) coordinates.

    Parameters
    ----------
    image : ndarray, shape (N, N) [Jy/pixel]
    fov_muas : float — half-width of the image [μas]
    uv_coverage : ndarray, shape (M, 2) — baseline coordinates [Gλ]
    freq_ghz : float — observing frequency [GHz]

    Returns
    -------
    visibilities : complex ndarray, shape (M,) [Jy]
    """
    n_pix = image.shape[0]
    dx_muas = 2.0 * fov_muas / n_pix

    # FFT of image → uv plane (shift zero-freq to centre)
    image_f = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image))) * dx_muas**2

    # Frequency axis in rad^{-1}
    freqs_pix = np.fft.fftfreq(n_pix, d=dx_muas * MUAS_RAD) / 1.0  # [rad^{-1}]
    freqs_pix = np.fft.fftshift(freqs_pix)

    # Convert uv from Gλ to rad^{-1}
    wavelength_m = C / (freq_ghz * 1e9)
    uv_rad = uv_coverage * 1e9 * wavelength_m  # Gλ → rad^{-1}

    # Bilinear interpolation of the FFT onto requested (u,v) points
    from scipy.interpolate import RegularGridInterpolator

    interp_real = RegularGridInterpolator(
        (freqs_pix, freqs_pix),
        image_f.real,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    interp_imag = RegularGridInterpolator(
        (freqs_pix, freqs_pix),
        image_f.imag,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    vis_real = interp_real(uv_rad[:, ::-1])
    vis_imag = interp_imag(uv_rad[:, ::-1])
    return vis_real + 1j * vis_imag


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

        M = _require_param(params, "M", "image")
        a = _require_param(params, "a_star", "image")
        i = _require_param(params, "i", "image")
        pos_angle = _require_param(params, "position_angle", "image")
        emission = ring_emission_from_params(params)

        r_ring = _shadow_radius_muas(M, a, D_L)
        w_ring = r_ring * emission.ring_width_frac
        axial_ratio = float(np.abs(np.cos(i)))

        # ── Build image ────────────────────────────────────────────────────────
        image = _gaussian_ring_image(
            fov_muas,
            n_pix,
            r_ring,
            w_ring,
            emission.brightness,
            axial_ratio,
            pos_angle,
            asym_amp=emission.asym_amp,
            asym_azimuth_rad=emission.asym_azimuth_rad,
        )

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
                "ring_width_frac": emission.ring_width_frac,
                "brightness_jy_per_muas2": emission.brightness,
                "asym_amp": emission.asym_amp,
            },
            params_true=params,
            noise_realisation=noise,
        )


def _compute_closure_phases(visibilities: NDArray) -> NDArray:
    """Compute closure phases for sequential baseline triplets."""
    n = len(visibilities)
    n_triangles = n // 3
    phases = np.angle(visibilities)
    closure = np.zeros(n_triangles)
    for k in range(n_triangles):
        i, j, l = 3 * k, 3 * k + 1, 3 * k + 2
        closure[k] = phases[i] + phases[j] - phases[l]
    return closure % (2.0 * np.pi) - np.pi  # wrap to (-π, π]


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
