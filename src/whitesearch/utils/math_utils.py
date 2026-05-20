"""Mathematical utilities for gravitational wave and signal processing.

Implements noise-weighted inner products, PSD estimation, ringdown waveforms,
and dispersion convolution used across simulators and likelihoods.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import signal as sp_signal


# ── Inner product and whitening ─────────────────────────────────────────────────

def noise_weighted_inner_product(
    a: NDArray,
    b: NDArray,
    psd: NDArray,
    df: float,
) -> complex:
    """Compute the noise-weighted inner product ⟨a|b⟩ in the frequency domain.

    ⟨a|b⟩ = 4 Re[∑_f  ã*(f) b̃(f) / Sn(f) * Δf]

    Parameters
    ----------
    a, b : array_like
        Frequency-domain strain (complex, one-sided, starting at f=0).
    psd : array_like
        One-sided power spectral density Sn(f) [Hz^{-1}].
    df : float
        Frequency bin width [Hz].

    Returns
    -------
    complex
        The complex inner product (real part = noise-weighted overlap).
    """
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    psd = np.asarray(psd, dtype=float)

    integrand = np.conj(a) * b / psd
    return 4.0 * np.sum(integrand) * df


def matched_filter_snr(
    template_f: NDArray,
    data_f: NDArray,
    psd: NDArray,
    df: float,
) -> float:
    """Compute optimal matched filter SNR ρ = ⟨d|h⟩ / √⟨h|h⟩."""
    hh = noise_weighted_inner_product(template_f, template_f, psd, df).real
    dh = noise_weighted_inner_product(data_f, template_f, psd, df).real
    if hh <= 0:
        return 0.0
    return dh / np.sqrt(hh)


def whiten(
    strain: NDArray,
    psd: NDArray,
    dt: float,
    low_freq_cutoff: float = 20.0,
) -> NDArray:
    """Whiten a time-domain strain array using its PSD.

    Parameters
    ----------
    strain : array_like
        Time-domain strain [dimensionless].
    psd : array_like
        One-sided PSD [Hz^{-1}], length N//2+1.
    dt : float
        Sample spacing [s].
    low_freq_cutoff : float
        Zero out frequencies below this value [Hz].

    Returns
    -------
    ndarray
        Whitened time-domain strain (same length as input).
    """
    n = len(strain)
    df = 1.0 / (n * dt)
    freqs = np.fft.rfftfreq(n, d=dt)

    strain_f = np.fft.rfft(strain)

    with np.errstate(divide="ignore", invalid="ignore"):
        white_f = strain_f / np.sqrt(psd / (2.0 * df))
        white_f[freqs < low_freq_cutoff] = 0.0

    return np.fft.irfft(white_f, n=n)


def estimate_psd(
    strain: NDArray,
    sample_rate: float,
    fft_length: float = 4.0,
    overlap: float = 0.5,
    window: str = "hann",
) -> tuple[NDArray, NDArray]:
    """Estimate one-sided PSD via Welch's method.

    Returns
    -------
    freqs : ndarray [Hz]
    psd : ndarray [Hz^{-1}]
    """
    nperseg = int(fft_length * sample_rate)
    noverlap = int(overlap * nperseg)
    freqs, psd = sp_signal.welch(
        strain,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
    )
    return freqs, psd


# ── Ringdown waveforms ───────────────────────────────────────────────────────────

def ringdown_waveform(
    times: NDArray,
    t0: float,
    amplitude: float,
    frequency: float,
    quality: float,
    phase: float = 0.0,
) -> NDArray:
    """Generate a damped sinusoidal ringdown waveform (plus polarization).

    h(t) = A * exp(−π f₀ (t−t₀) / Q) * cos(2π f₀ (t−t₀) + φ),  t ≥ t₀

    Parameters
    ----------
    times : array_like [s]
    t0 : float — onset time [s]
    amplitude : float — peak strain amplitude
    frequency : float — ringdown frequency f₀ [Hz]
    quality : float — quality factor Q (= π f₀ τ)
    phase : float — initial phase [rad]

    Returns
    -------
    ndarray — plus-polarization strain h_+(t)
    """
    times = np.asarray(times, dtype=float)
    h = np.zeros_like(times)
    mask = times >= t0
    dt = times[mask] - t0
    decay = np.exp(-np.pi * frequency * dt / quality)
    h[mask] = amplitude * decay * np.cos(2.0 * np.pi * frequency * dt + phase)
    return h


def kerr_qnm_frequency(mass_msun: float, spin: float) -> tuple[float, float]:
    """Estimate the l=m=2 Kerr QNM frequency and quality factor.

    Uses the fitting formulae from Berti, Cardoso & Will (2006), Table VIII.

    Parameters
    ----------
    mass_msun : float — BH mass in solar masses
    spin : float — dimensionless spin |a*| ∈ [0, 0.998]

    Returns
    -------
    f_qnm : float [Hz]
    q_qnm : float [dimensionless]
    """
    from .constants import G, C, M_SUN, QNM_F1, QNM_F2, QNM_F3, QNM_Q1, QNM_Q2, QNM_Q3

    # Gravitational radius timescale: t_grav = G M / c^3
    t_grav = G * mass_msun * M_SUN / C**3

    # Dimensionless QNM frequency: M ω = F1 + F2*(1-a*)^F3
    m_omega = QNM_F1 + QNM_F2 * (1.0 - np.abs(spin)) ** QNM_F3
    f_qnm = m_omega / (2.0 * np.pi * t_grav)

    # Quality factor: Q = Q1 + Q2*(1-a*)^Q3
    q_qnm = QNM_Q1 + QNM_Q2 * (1.0 - np.abs(spin)) ** QNM_Q3

    return f_qnm, q_qnm


# ── Dispersion and scattering ────────────────────────────────────────────────────

def dm_delay_ms(
    dm: float,
    freq_low_mhz: float,
    freq_ref_mhz: float = float("inf"),
) -> float:
    """Compute DM-induced time delay relative to a reference frequency.

    t_delay [ms] = K_DM * DM [pc/cm^3] * (ν_ref^{-2} − ν_low^{-2}) [MHz]
    """
    from .constants import K_DM

    if np.isinf(freq_ref_mhz):
        return K_DM * dm / freq_low_mhz**2
    return K_DM * dm * (freq_ref_mhz**-2 - freq_low_mhz**-2)


def apply_dm_dispersion(
    times: NDArray,
    intensity: NDArray,
    dm: float,
    freqs_mhz: NDArray,
) -> NDArray:
    """Return a frequency-time (I vs freq, t) array with DM dispersion applied.

    Parameters
    ----------
    times : array [s], length N_t
    intensity : array, shape (N_freq, N_t) — undispersed dynamic spectrum
    dm : float — dispersion measure [pc/cm^3]
    freqs_mhz : array [MHz], length N_freq

    Returns
    -------
    dispersed : ndarray, same shape as intensity
    """
    from .constants import K_DM

    dispersed = np.zeros_like(intensity)
    ref_freq = freqs_mhz.max()
    dt = times[1] - times[0]

    for i, nu in enumerate(freqs_mhz):
        delay_s = K_DM * dm * (nu**-2 - ref_freq**-2) * 1e-3  # ms → s
        shift_bins = int(round(delay_s / dt))
        dispersed[i] = np.roll(intensity[i], shift_bins)

    return dispersed


def scatter_broaden(
    times: NDArray,
    intensity: NDArray,
    tau_sc: float,
) -> NDArray:
    """Convolve an intensity profile with a one-sided exponential scattering kernel.

    I_scattered(t) = I(t) ⊗ exp(−t/τ_sc) * Θ(t)

    Parameters
    ----------
    times : array [s]
    intensity : array — intrinsic intensity profile
    tau_sc : float — scattering time [s]

    Returns
    -------
    ndarray — scatter-broadened profile (same length)
    """
    dt = times[1] - times[0]
    kernel_len = min(10 * int(tau_sc / dt) + 1, len(times))
    kernel_t = np.arange(kernel_len) * dt
    kernel = np.exp(-kernel_t / tau_sc)
    kernel /= kernel.sum()
    return np.convolve(intensity, kernel, mode="same")


# ── Statistics helpers ───────────────────────────────────────────────────────────

def log_sum_exp(log_vals: NDArray) -> float:
    """Numerically stable log(∑ exp(x_i))."""
    a = np.max(log_vals)
    return a + np.log(np.sum(np.exp(log_vals - a)))


def compute_credible_interval(
    samples: NDArray,
    level: float = 0.9,
) -> tuple[float, float]:
    """Return the highest posterior density credible interval."""
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def compute_sbc_rank(
    true_value: float,
    posterior_samples: NDArray,
) -> int:
    """Compute simulation-based calibration rank.

    Returns the number of posterior samples less than the true value.
    Under a well-calibrated posterior this should be uniform over [0, L].
    """
    return int(np.sum(posterior_samples < true_value))
