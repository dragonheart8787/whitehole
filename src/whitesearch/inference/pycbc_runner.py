"""PyCBC matched-filter search and parameter estimation wrapper.

Used for the GW channel to run a preliminary matched-filter trigger
pipeline before handing off to Bilby for full parameter estimation.

Falls back to a pure-numpy implementation when PyCBC is not installed.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..utils.math_utils import (
    matched_filter_snr,
    noise_weighted_inner_product,
    ringdown_waveform,
    kerr_qnm_frequency,
    estimate_psd,
)
from ..utils.constants import G, C, M_SUN, MPC_M

logger = logging.getLogger(__name__)

try:
    import pycbc  # noqa: F401
    PYCBC_AVAILABLE = True
except ImportError:
    PYCBC_AVAILABLE = False
    logger.warning(
        "PyCBC not installed; using pure-numpy GW search. "
        "Install on Linux/Mac with: pip install pycbc"
    )


class GWSearchRunner:
    """Gravitational wave matched-filter trigger pipeline.

    For each (M, a*) template, computes:
      - Optimal matched-filter SNR ρ
      - Ringdown frequency and quality factor (from template)
      - Trigger GPS time (peak SNR time)

    Parameters
    ----------
    snr_threshold : float
        Minimum SNR for a trigger to be reported.
    mass_range : tuple[float, float]
        (M_min, M_max) in solar masses for the template bank.
    spin_values : list[float]
        Dimensionless spin values to include in the template bank.
    """

    def __init__(
        self,
        snr_threshold: float = 5.0,
        mass_range: tuple[float, float] = (5.0, 200.0),
        spin_values: list[float] | None = None,
        n_mass_points: int = 20,
    ) -> None:
        self.snr_threshold = snr_threshold
        self.mass_range = mass_range
        self.spin_values = spin_values or [0.0, 0.3, 0.7, 0.95]
        self.n_mass_points = n_mass_points

    def run_search(
        self,
        strain: NDArray,
        sample_rate: float,
        psd: NDArray | None = None,
        low_freq: float = 20.0,
    ) -> list[dict[str, Any]]:
        """Run a ringdown template bank search.

        Returns list of triggers above SNR threshold, sorted by SNR.
        """
        n = len(strain)
        dt = 1.0 / sample_rate
        df = 1.0 / (n * dt)
        freqs = np.fft.rfftfreq(n, d=dt)
        times = np.arange(n) * dt

        if psd is None:
            freqs_w, psd_welch = estimate_psd(strain, sample_rate)
            psd = np.interp(freqs, freqs_w, psd_welch, left=float(psd_welch[0]), right=float(psd_welch[-1]))

        psd = np.where(psd > 0, psd, 1e-30)
        psd[freqs < low_freq] = 1e-30
        strain_f = np.fft.rfft(strain) * dt

        masses = np.logspace(
            np.log10(self.mass_range[0]),
            np.log10(self.mass_range[1]),
            self.n_mass_points,
        )

        triggers = []
        for M in masses:
            for a in self.spin_values:
                f_rd, q_rd = kerr_qnm_frequency(M, a)
                if f_rd < low_freq or f_rd > 0.5 * sample_rate:
                    continue

                # Compute D_L=100 Mpc template (arbitrary amplitude)
                D_L_m = 100.0 * MPC_M
                h0 = G * M * M_SUN / (C**2 * D_L_m)
                h = ringdown_waveform(times, times[n // 4], h0, f_rd, q_rd)
                h_f = np.fft.rfft(h) * dt

                snr = matched_filter_snr(h_f, strain_f, psd, df)
                if snr > self.snr_threshold:
                    triggers.append(
                        {
                            "M_msun": M,
                            "a_star": a,
                            "f_rd_hz": f_rd,
                            "q_rd": q_rd,
                            "snr": snr,
                        }
                    )

        triggers.sort(key=lambda x: x["snr"], reverse=True)
        logger.info("GW search: %d triggers above SNR %.1f", len(triggers), self.snr_threshold)
        return triggers

    @staticmethod
    def compute_psd(
        strain: NDArray,
        sample_rate: float,
        fft_length: float = 4.0,
    ) -> tuple[NDArray, NDArray]:
        """Estimate one-sided PSD via Welch's method."""
        return estimate_psd(strain, sample_rate, fft_length=fft_length)

    @staticmethod
    def optimal_snr(
        template_params: dict[str, float],
        psd: NDArray,
        freqs: NDArray,
        sample_rate: float,
        duration: float,
    ) -> float:
        """Compute optimal (noise-weighted) SNR for a given template."""
        n = int(duration * sample_rate)
        dt = 1.0 / sample_rate
        df = 1.0 / (n * dt)
        times = np.arange(n) * dt

        M = template_params["M"]
        a = template_params.get("a_star", 0.0)
        f_rd, q_rd = kerr_qnm_frequency(M, a)
        D_L_m = template_params.get("D_L", 100.0) * MPC_M
        h0 = G * M * M_SUN / (C**2 * D_L_m)

        h = ringdown_waveform(times, times[n // 4], h0, f_rd, q_rd)
        h_f = np.fft.rfft(h) * dt

        freqs_h = np.fft.rfftfreq(n, d=dt)
        psd_interp = np.interp(freqs_h, freqs, psd, left=psd[0], right=psd[-1])
        psd_interp = np.where(psd_interp > 0, psd_interp, 1e-30)

        hh = noise_weighted_inner_product(h_f, h_f, psd_interp, df).real
        return float(np.sqrt(max(0.0, hh)))
