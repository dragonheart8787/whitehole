"""GW frequency-domain conventions used by GWLikelihood and diagnostics.

FFT convention: h_tilde(f) = rfft(h) * dt  (one-sided, numpy default)
Inner product: ⟨a|b⟩ = 4 Re[ sum_f conj(a_f) b_f / Sn(f) * df ]
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def time_to_freq(strain: NDArray, dt: float) -> tuple[NDArray, NDArray, float]:
    """Return (freqs_hz, strain_f, df)."""
    strain = np.asarray(strain, dtype=np.float64)
    n = len(strain)
    df = 1.0 / (n * dt)
    freqs = np.fft.rfftfreq(n, d=dt)
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
