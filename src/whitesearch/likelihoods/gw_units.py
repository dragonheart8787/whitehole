"""GW frequency-domain conventions used by GWLikelihood and diagnostics.

FFT convention: h_tilde(f) = rfft(h) * dt  (one-sided, numpy default)
Inner product: ⟨a|b⟩ = 4 Re[ sum_f conj(a_f) b_f / Sn(f) * df ]
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    from scipy.signal.windows import tukey
except ImportError:  # pragma: no cover - older scipy layout
    from scipy.signal import tukey  # type: ignore[attr-defined]


def time_to_freq(
    strain: NDArray, dt: float, taper_alpha: float = 0.0
) -> tuple[NDArray, NDArray, float]:
    """Return (freqs_hz, strain_f, df).

    Parameters
    ----------
    taper_alpha : float
        Tukey-window fraction applied to the time series before the rfft
        (0.0 = no taper, the historical/default behaviour). A full-segment
        strain array FFT'd with an implicit rectangular window has sidelobe
        leakage that a Welch-estimated PSD (built from short, Hann-windowed
        segments) does not describe, inflating noise-weighted inner products
        near the analysis band edges. Callers on the GW likelihood path pass
        a nonzero value explicitly; this default keeps every other caller
        (mocks, other channels, existing tests) on the old behaviour.
        No power-compensation factor is applied: the taper only touches the
        outer `taper_alpha/2` fraction of samples at each edge, and both the
        data and any template compared against it must be tapered with the
        same alpha (identical windowing cancels in the residual) so a signal
        away from the segment edges is not attenuated.
    """
    strain = np.asarray(strain, dtype=np.float64)
    n = len(strain)
    df = 1.0 / (n * dt)
    freqs = np.fft.rfftfreq(n, d=dt)
    if taper_alpha > 0.0:
        strain = strain * tukey(n, alpha=taper_alpha)
    strain_f = np.fft.rfft(strain) * dt
    return freqs, strain_f, df


def freq_to_time(strain_f: NDArray, n: int, dt: float) -> NDArray:
    """Inverse of time_to_freq (for unit tests)."""
    return np.fft.irfft(strain_f / dt, n=n)


def inner_product_norm(
    strain_f: NDArray,
    template_f: NDArray,
    psd: NDArray,
    df: float,
) -> tuple[float, float, float]:
    """Return (dd, dh, hh) noise-weighted inner products (real parts where applicable)."""
    from ..utils.math_utils import noise_weighted_inner_product

    dd = noise_weighted_inner_product(strain_f, strain_f, psd, df).real
    dh = noise_weighted_inner_product(strain_f, template_f, psd, df).real
    hh = noise_weighted_inner_product(template_f, template_f, psd, df).real
    return float(dd), float(dh), float(hh)
