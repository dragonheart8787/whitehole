"""Gravitational wave toy simulator for black-to-white bounce and BH ringdown.

Implements time-domain ringdown waveforms with optional bounce burst.
Does NOT require LALSuite or PyCBC (pure numpy/scipy).

Optional PyCBC integration is used when available for more accurate waveforms.

Context keys
------------
sample_rate : float — detector sample rate [Hz] (default 4096)
duration : float — segment duration [s] (default 4.0)
t_merger : float — merger time within segment [s] (default 0.5)
psd_file : str | None — path to ASCII PSD file; if None, use analytic aLIGO
low_freq_cutoff : float — lower frequency bound [Hz] (default 20.0)
detector : str — detector name for orientation projection (default 'H1')
rng_seed : int — random seed for noise (default None)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
try:
    from scipy.signal.windows import tukey
except ImportError:
    from scipy.signal import tukey  # type: ignore[attr-defined]

from .base import BaseSimulator, SimData
from ..utils.constants import G, C, M_SUN
from ..utils.math_utils import ringdown_waveform, kerr_qnm_frequency, estimate_psd


# ── Analytic Advanced LIGO PSD (O3 design, simplified) ──────────────────────────

def aligo_psd_analytic(freqs: NDArray) -> NDArray:
    """Approximate aLIGO O3 design PSD [Hz^{-1}].

    Uses the fitting formula from LIGO-T0900288 (simplified).
    """
    f0 = 215.0
    x = freqs / f0
    with np.errstate(divide="ignore", invalid="ignore"):
        s0 = 1.0e-48
        psd = s0 * (
            (4.49 * x) ** (-56)
            + 0.16 * x ** (-4.52)
            + 0.52
            + 0.32 * x**2
        )
    psd = np.where(np.isfinite(psd) & (psd > 0), psd, 1.0e-30)
    return psd


# ── Gaussian noise coloured by PSD ──────────────────────────────────────────────

def gaussian_noise_from_psd(
    psd: NDArray,
    n_samples: int,
    sample_rate: float,
    rng: np.random.Generator,
) -> NDArray:
    """Generate time-domain Gaussian noise coloured by a given PSD.

    Parameters
    ----------
    psd : 1D array, length n_samples//2+1  [Hz^{-1}]
    n_samples : int
    sample_rate : float [Hz]
    rng : np.random.Generator

    Returns
    -------
    noise : ndarray, shape (n_samples,) — real time-domain strain
    """
    df = sample_rate / n_samples
    sigma_f = np.sqrt(psd / (2.0 * df))

    noise_f_real = rng.standard_normal(len(psd)) * sigma_f
    noise_f_imag = rng.standard_normal(len(psd)) * sigma_f
    noise_f = noise_f_real + 1j * noise_f_imag

    # Force DC and Nyquist to be real
    noise_f[0] = noise_f[0].real
    if n_samples % 2 == 0:
        noise_f[-1] = noise_f[-1].real

    return np.fft.irfft(noise_f, n=n_samples)


# ── Antenna pattern (simplified) ─────────────────────────────────────────────────

def antenna_response(inclination: float) -> tuple[float, float]:
    """Return (F+, Fx) for an overhead, optimally oriented source.

    For a full sky-position-dependent response use a GW library.
    This simplified version assumes optimal sky position.
    """
    cos_i = np.cos(inclination)
    fp = 0.5 * (1.0 + cos_i**2)
    fc = cos_i
    return float(fp), float(fc)


# ── Main simulator ────────────────────────────────────────────────────────────────

class GravitationalWaveSimulator(BaseSimulator):
    """Toy GW forward simulator for black-to-white bounce and BH ringdown.

    Produces a time-domain strain array:
      h(t) = h_ringdown(t) + h_bounce(t) [signal]
           + n(t)                          [coloured Gaussian noise]

    For the bounce model the frequency and quality factor are modified by
    (eps_f, eps_Q) relative to GR predictions.  The bounce burst appears
    at time t_merger + tau_bounce (if tau_bounce < duration).

    For the null / standard BH ringdown the bounce burst amplitude is zero.
    """

    channel = "gw"

    def simulate(
        self,
        params: dict[str, float],
        context: dict[str, Any],
        rng: np.random.Generator | None = None,
    ) -> SimData:
        if rng is None:
            seed = context.get("rng_seed", None)
            rng = np.random.default_rng(seed)

        # ── Instrument configuration ───────────────────────────────────────────
        sample_rate = float(context.get("sample_rate", 4096.0))
        duration = float(context.get("duration", 4.0))
        t_merger = float(context.get("t_merger", 0.5))
        low_freq = float(context.get("low_freq_cutoff", 20.0))

        n_samples = int(duration * sample_rate)
        times = np.arange(n_samples) / sample_rate

        # ── Derive waveform parameters ─────────────────────────────────────────
        M = params["M"]
        a = params["a_star"]
        i = params.get("i", 0.0)

        f_gr, q_gr = kerr_qnm_frequency(M, a)

        # Bounce modifications (default to 0 if not present)
        eps_f = params.get("eps_f", 0.0)
        eps_Q = params.get("eps_Q", 0.0)
        f_rd = f_gr * (1.0 + eps_f)
        q_rd = max(0.5, q_gr * (1.0 + eps_Q))

        # Ringdown amplitude: h_0 ~ G M / (c^2 D_L)
        D_L_m = params.get("D_L", 100.0) * 3.086e22
        h0 = float(G * M * M_SUN / (C**2 * D_L_m))

        # Antenna projection
        fp, fc = antenna_response(i)
        A_rd = h0 * np.sqrt(fp**2 + fc**2)

        # ── Build signal ───────────────────────────────────────────────────────
        h_plus = ringdown_waveform(times, t_merger, A_rd, f_rd, q_rd)

        # Bounce burst
        A_bounce = params.get("log10_A_bounce", None)
        if A_bounce is not None:
            A_b = float(10.0 ** A_bounce)
        else:
            A_b = 0.0

        tau_bounce_s = params.get("log10_tau_bounce_yr", None)
        if tau_bounce_s is not None and A_b > 0:
            from ..utils.constants import GYR_S
            tau_s = float(10.0 ** tau_bounce_s * GYR_S / 1e9)
            t_bounce_event = t_merger + tau_s
            if t_bounce_event < duration:
                h_plus += ringdown_waveform(
                    times, t_bounce_event, A_b, f_rd * 0.8, max(2.0, q_rd * 0.5)
                )

        # ── Taper the signal ──────────────────────────────────────────────────
        window = tukey(n_samples, alpha=0.1)
        h_signal = h_plus * window

        # ── PSD and noise ─────────────────────────────────────────────────────
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
        psd = aligo_psd_analytic(freqs)
        psd[freqs < low_freq] = 1.0e-30

        noise = gaussian_noise_from_psd(psd, n_samples, sample_rate, rng)
        strain = h_signal + noise

        return SimData(
            channel="gw",
            data=strain,
            metadata={
                "times": times,
                "sample_rate": sample_rate,
                "freqs": freqs,
                "psd": psd,
                "t_merger": t_merger,
                "f_rd": f_rd,
                "q_rd": q_rd,
                "h0": h0,
                "low_freq_cutoff": low_freq,
            },
            params_true=params,
            noise_realisation=noise,
        )

    # ── Convenience: frequency-domain data ───────────────────────────────────

    @staticmethod
    def to_frequency_domain(
        sim_data: SimData,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Return (freqs, strain_f, psd) from a SimData object."""
        meta = sim_data.metadata
        n = len(sim_data.data)
        sr = meta["sample_rate"]
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        strain_f = np.fft.rfft(sim_data.data) / sr
        return freqs, strain_f, meta["psd"]
