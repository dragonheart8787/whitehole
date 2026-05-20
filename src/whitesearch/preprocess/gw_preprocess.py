"""Gravitational wave data preprocessing pipeline.

Steps
-----
1. Load strain + data quality flags
2. Apply notch filters for spectral lines
3. Bandpass filter (20–2048 Hz)
4. Estimate power spectral density (Welch method)
5. Whiten strain
6. Cut off-source windows for background estimation
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt, iirnotch, sosfiltfilt

from ..utils.math_utils import estimate_psd, whiten

logger = logging.getLogger(__name__)

# Known spectral lines in aLIGO (approximate frequencies [Hz])
LIGO_SPECTRAL_LINES = [
    60.0, 120.0, 180.0, 240.0, 300.0,  # power-line harmonics
    500.0, 1000.0,  # calibration lines
    36.7, 37.3,  # violin modes (H1 example)
]


class GWPreprocessor:
    """Preprocessing pipeline for gravitational wave strain data."""

    def __init__(
        self,
        sample_rate: float = 4096.0,
        low_freq: float = 20.0,
        high_freq: float = 1700.0,
        fft_length: float = 4.0,
        notch_lines: list[float] | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.fft_length = fft_length
        self.notch_lines = notch_lines if notch_lines is not None else LIGO_SPECTRAL_LINES

    def prepare_raw(
        self,
        strain: NDArray,
        dq_flags: dict | None = None,
    ) -> dict[str, Any]:
        """Full preprocessing pipeline from raw strain.

        Returns
        -------
        dict with:
            strain_whitened  : ndarray — whitened strain
            strain_bandpass  : ndarray — bandpass-filtered strain (pre-whitening)
            psd              : ndarray — one-sided PSD [Hz^{-1}]
            freqs_psd        : ndarray — PSD frequency axis [Hz]
            sample_rate      : float
            quality          : dict — preprocessing quality metrics
        """
        strain = np.asarray(strain, dtype=np.float64)

        # 1. Check data quality
        quality = self._check_dq(strain, dq_flags)

        # 2. Bandpass filter
        strain_bp = self.bandpass_filter(strain)

        # 3. Notch filter known spectral lines
        strain_notched = self.notch_filter(strain_bp)

        # 4. Estimate PSD
        freqs_psd, psd = estimate_psd(
            strain_notched,
            self.sample_rate,
            fft_length=self.fft_length,
        )

        # 5. Whiten
        psd_interp = self._interpolate_psd(psd, freqs_psd, strain_notched)
        strain_white = whiten(strain_notched, psd_interp, 1.0 / self.sample_rate, self.low_freq)

        quality["rms_white"] = float(np.std(strain_white))
        quality["rms_expected"] = 1.0  # whitened strain should be ~unit RMS
        quality["whitening_ok"] = abs(quality["rms_white"] - 1.0) < 0.3

        return {
            "strain_whitened": strain_white,
            "strain_bandpass": strain_bp,
            "psd": psd_interp,
            "freqs_psd": freqs_psd,
            "sample_rate": self.sample_rate,
            "quality": quality,
        }

    def bandpass_filter(self, strain: NDArray) -> NDArray:
        """Apply a 4th-order Butterworth bandpass filter."""
        nyq = 0.5 * self.sample_rate
        low = self.low_freq / nyq
        high = min(self.high_freq / nyq, 0.999)
        sos = butter(4, [low, high], btype="band", output="sos")
        return sosfiltfilt(sos, strain)

    def notch_filter(self, strain: NDArray) -> NDArray:
        """Apply IIR notch filters at known spectral line frequencies."""
        result = strain.copy()
        for f0 in self.notch_lines:
            if f0 >= 0.5 * self.sample_rate:
                continue
            Q = 30.0
            b, a = iirnotch(f0, Q, fs=self.sample_rate)
            result = np.convolve(result, b, mode="same")
        return result

    @staticmethod
    def make_background(
        strain: NDArray,
        sample_rate: float,
        event_gps: float,
        segment_gps_start: float,
        window_duration: float = 4.0,
        n_background_windows: int = 50,
        time_shift_step: float = 1.0,
    ) -> list[NDArray]:
        """Generate off-source background segments by time-shifting.

        Returns a list of strain arrays, each of length window_duration * sample_rate.
        """
        n_samples_win = int(window_duration * sample_rate)
        event_offset = event_gps - segment_gps_start
        exclusion_start = max(0, event_offset - window_duration)
        exclusion_end = event_offset + window_duration

        n_total = len(strain)
        t_total = n_total / sample_rate

        backgrounds = []
        shift = time_shift_step
        while len(backgrounds) < n_background_windows:
            t_start = shift
            t_end = t_start + window_duration
            if t_start > exclusion_end or t_end < exclusion_start:
                i_start = int(t_start * sample_rate)
                i_end = i_start + n_samples_win
                if i_end <= n_total:
                    backgrounds.append(strain[i_start:i_end])
            shift += time_shift_step
            if shift > t_total - window_duration:
                break

        return backgrounds

    # ── Internal ──────────────────────────────────────────────────────────────

    def _interpolate_psd(
        self,
        psd: NDArray,
        freqs_psd: NDArray,
        strain: NDArray,
    ) -> NDArray:
        """Interpolate PSD to match the strain FFT frequency grid."""
        n = len(strain)
        freqs_full = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)
        psd_interp = np.interp(freqs_full, freqs_psd, psd, left=psd[0], right=psd[-1])
        psd_interp = np.where(psd_interp > 0, psd_interp, np.min(psd[psd > 0]))
        return psd_interp

    @staticmethod
    def _check_dq(strain: NDArray, dq_flags: dict | None) -> dict[str, Any]:
        quality = {
            "n_samples": len(strain),
            "n_nans": int(np.sum(~np.isfinite(strain))),
            "fraction_valid": 1.0,
        }
        if quality["n_nans"] > 0:
            quality["fraction_valid"] = 1.0 - quality["n_nans"] / len(strain)
            logger.warning("%d non-finite samples found in strain.", quality["n_nans"])
        if dq_flags:
            quality["dq_flags"] = dq_flags
        return quality
