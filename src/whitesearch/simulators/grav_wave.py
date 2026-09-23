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
from ..utils.constants import (
    G,
    C,
    M_SUN,
    MPC_M,
    BOUNCE_BURST_FREQ_FACTOR,
    BOUNCE_BURST_Q_FACTOR,
    BOUNCE_BURST_Q_MIN,
)
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

    Notes
    -----
    Normalised to the gw_units convention h_tilde(f) = rfft(h) * dt with
    E[|h_tilde(f)|^2] = Sn(f) * T / 2 (one-sided), so the per-bin
    contribution to the noise-weighted inner product <n|n> averages 2.
    Since this function returns the time series via irfft (i.e. it fills
    raw rfft coefficients), the per-component sigma is
    sqrt(Sn/(4 df)) / dt.  The previous sqrt(Sn/(2 df)) filled rfft
    coefficients as if they were already dt-scaled, leaving the generated
    noise a factor ~2 dt^2 low in spectral power versus the stated PSD.
    """
    dt = 1.0 / sample_rate
    df = sample_rate / n_samples
    sigma_f = np.sqrt(psd / (4.0 * df)) / dt

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

        f_gr, q_gr = kerr_qnm_frequency(M, a)

        # Bounce modifications (default to 0 if not present)
        eps_f = params.get("eps_f", 0.0)
        eps_Q = params.get("eps_Q", 0.0)
        f_rd = f_gr * (1.0 + eps_f)
        q_rd = max(0.5, q_gr * (1.0 + eps_Q))

        # ── Ringdown amplitude ────────────────────────────────────────────────
        # Models carrying an explicit free amplitude (log10_A, i.e. the
        # phenomenological bh_ringdown model) use it directly, byte-for-byte
        # the same expression as GWLikelihood._build_template()'s bh_ringdown
        # branch.  Deriving the amplitude from M/D_L/i here while the
        # likelihood read 10**log10_A meant the simulator and the likelihood
        # were different forward models, which invalidates SBC.
        # Models that parameterise the amplitude physically (bounce: M, D_L,
        # inclination) keep the distance/antenna path unchanged.
        if "log10_A" in params:
            amplitude_source = "log10_A"
            A_rd = float(10.0 ** params["log10_A"])
            h0 = A_rd
        else:
            amplitude_source = "M_D_L_inclination"
            i = params.get("i", 0.0)
            # utils.constants.MPC_M, not a rounded 3.086e22 literal: the
            # likelihood template uses MPC_M, and the 1.0449e-04 relative
            # difference between them made the injected and modelled
            # amplitudes disagree by that factor.  Found by the burst-timing
            # consistency test below, which compares the two waveforms sample
            # for sample.
            D_L_m = params.get("D_L", 100.0) * MPC_M
            h0 = float(G * M * M_SUN / (C**2 * D_L_m))
            # Plus polarisation only, matching
            # GWLikelihood._build_template()'s A_rd = h0 * 0.5*(1 + cos^2 i)
            # and matching ringdown_waveform(), whose docstring states it
            # returns h_+ .  This previously used sqrt(fp^2 + fc^2), mixing the
            # plus and cross amplitudes into a single real template: the
            # injected amplitude was up to sqrt(2) = 1.4142 times what the
            # likelihood modelled (face-on), which biased the recovered D_L by
            # the same factor.  See docs/BOUNCE_PREFLIGHT_AUDIT.md section B.4.
            fp, _fc = antenna_response(i)
            A_rd = h0 * fp

        # ── Build signal ───────────────────────────────────────────────────────
        h_plus = ringdown_waveform(times, t_merger, A_rd, f_rd, q_rd)

        # Bounce burst
        A_bounce = params.get("log10_A_bounce", None)
        if A_bounce is not None:
            A_b = float(10.0 ** A_bounce)
        else:
            A_b = 0.0

        # Burst delay is measured from the merger in seconds, the same
        # definition GWLikelihood._build_template() uses -- and the same
        # BOUNCE_BURST_* factors -- so the injected burst and the fitted burst
        # are one forward model.  The previous code converted
        # log10_tau_bounce_yr (a cosmological lifetime, in YEARS) to seconds,
        # which put the burst at least 3.156e+04 s after the merger and so
        # never inside the segment.  See docs/BOUNCE_PREFLIGHT_AUDIT.md D.1.
        log10_dt_bounce = params.get("log10_dt_bounce_s", None)
        if log10_dt_bounce is not None and A_b > 0:
            t_bounce_event = t_merger + float(10.0 ** log10_dt_bounce)
            # Same guard as the template's (times[-1], not duration).
            if t_bounce_event < times[-1]:
                h_plus += ringdown_waveform(
                    times,
                    t_bounce_event,
                    A_b,
                    f_rd * BOUNCE_BURST_FREQ_FACTOR,
                    max(BOUNCE_BURST_Q_MIN, BOUNCE_BURST_Q_FACTOR * q_rd),
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
                "A_rd": A_rd,
                "amplitude_source": amplitude_source,
                "low_freq_cutoff": low_freq,
                # Provenance tag, same field dataio/gw_observation.py and
                # dataio/gwosc.py use ("GWOSC", "MOCK_EXPLICIT", "MOCK",
                # "MOCK_FALLBACK").  This path is distinct from all of those:
                # "psd" below is the exact analytic spectrum the noise was
                # generated from, on the same un-windowed basis as the
                # likelihood's rfft, so there is no Welch-vs-rectangular
                # window mismatch for a taper to correct.  GWLikelihood reads
                # this tag to decide; see taper_for_source() there and
                # docs/BOUNCE_PREFLIGHT_AUDIT.md Part H.
                "source": "MOCK_SIMULATOR",
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
