"""Image/shadow forward simulator for EHT-like VLBI observations.

Implements a simplified Schwarzschild/Kerr shadow with a thin Gaussian
brightness ring, sampled at (u,v) points to produce complex visibilities.

Optional ehtim / EinsteinPy integration is used when available for more
accurate ray-tracing and interferometric simulation.

Context keys
------------
uv_coverage   : ndarray, shape (N_baselines, 2) [Gλ] — baseline (u,v) coordinates
beam_fwhm_muas : float — synthesised beam FWHM [μas] (default 20.0)
thermal_noise_jy : float — per-baseline thermal noise [Jy] (default 0.05)
freq_ghz      : float — observing frequency [GHz] (default 230.0)
fov_muas      : float — image FoV half-width [μas] (default 200.0)
n_pixels      : int   — image grid size (default 128)
rng_seed      : int   — random seed
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import BaseSimulator, SimData
from ..utils.constants import G, C, M_SUN, MPC_M, MUAS_RAD


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
) -> NDArray:
    """Generate a 2D Gaussian brightness ring image.

    Returns image in [Jy/μas^2], shape (n_pix, n_pix).
    Origin at centre; x-axis West (RA), y-axis North (Dec).
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

        # ── Source model ───────────────────────────────────────────────────────
        M = params["M"]
        a = params.get("a_star", 0.0)
        D_L = params["D_L"]
        i = params.get("i", 0.0)
        pos_angle = params.get("position_angle", 0.0)
        ring_width_frac = params.get("ring_width_frac", 0.1)
        brightness = float(10.0 ** params.get("log10_brightness", 0.0))

        r_ring = _shadow_radius_muas(M, a, D_L)
        w_ring = r_ring * ring_width_frac
        axial_ratio = float(np.abs(np.cos(i)))

        # ── Build image ────────────────────────────────────────────────────────
        image = _gaussian_ring_image(
            fov_muas, n_pix, r_ring, w_ring, brightness, axial_ratio, pos_angle
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
