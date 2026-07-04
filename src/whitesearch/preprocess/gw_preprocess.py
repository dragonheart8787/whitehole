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
from scipy.signal import butter, sosfilt, iirnotch, sosfiltfilt, tf2sos

from ..utils.math_utils import estimate_psd, whiten

logger = logging.getLogger(__name__)

# Known spectral lines in aLIGO (approximate frequencies [Hz])
LIGO_SPECTRAL_LINES = [
    60.0, 120.0, 180.0, 240.0, 300.0,  # power-line harmonics
    500.0, 1000.0,  # calibration lines
    36.7, 37.3,  # violin modes (H1 example)
]


class PSDEstimationError(Exception):
    """Raised when a PSD cannot be estimated safely (fail-closed).

    In particular: when off-source data is requested but insufficient
    off-source background is available, we refuse to silently fall back
    to an on-source (signal-contaminated) PSD.
    """


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
        *,
        event_gps: float | None = None,
        segment_gps_start: float | None = None,
        background_window_duration: float | None = None,
        min_background_windows: int = 8,
        max_background_windows: int = 50,
    ) -> dict[str, Any]:
        """Full preprocessing pipeline from raw strain.

        Parameters
        ----------
        event_gps, segment_gps_start : float | None
            GPS time of the event and of the start of ``strain``.  When both
            are given, the PSD is estimated from time-shifted *off-source*
            background windows (excluding the on-source window around the
            event) instead of the on-source strain itself, avoiding signal
            self-contamination of the noise estimate.  When either is
            missing, off-source estimation is impossible and the PSD falls
            back to the on-source strain — this is recorded explicitly in
            ``quality`` (never silent).
        background_window_duration : float | None
            Length [s] of each off-source background window.  Defaults to
            ``self.fft_length``.  Automatically shrunk if the available
            strain is too short to yield ``min_background_windows`` windows
            at the requested length.
        min_background_windows : int
            Minimum number of off-source windows required to accept the
            off-source PSD estimate.  If this cannot be met (even after
            shrinking the window), estimation fails closed by raising
            ``PSDEstimationError`` rather than reverting to on-source PSD.
        max_background_windows : int
            Upper bound on the number of off-source windows to average.

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

        # 4. Estimate PSD — prefer off-source to avoid signal self-contamination
        if event_gps is not None and segment_gps_start is not None:
            freqs_psd, psd, psd_quality = self._estimate_off_source_psd(
                strain_notched,
                event_gps=event_gps,
                segment_gps_start=segment_gps_start,
                background_window_duration=background_window_duration,
                min_background_windows=min_background_windows,
                max_background_windows=max_background_windows,
            )
            quality.update(psd_quality)
        else:
            freqs_psd, psd = estimate_psd(
                strain_notched,
                self.sample_rate,
                fft_length=self.fft_length,
            )
            quality["psd_source"] = "on_source"
            quality["n_off_source_windows"] = 0
            quality["psd_source_reason"] = (
                "event_gps/segment_gps_start not provided; cannot identify "
                "on-source window to exclude"
            )
            logger.warning(
                "PSD estimated from on-source strain (event_gps/"
                "segment_gps_start not provided) — this risks signal "
                "self-contamination of the noise-weighted likelihood."
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
            sos = tf2sos(b, a)
            result = sosfiltfilt(sos, result)
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

    def _estimate_off_source_psd(
        self,
        strain: NDArray,
        event_gps: float,
        segment_gps_start: float,
        background_window_duration: float | None,
        min_background_windows: int,
        max_background_windows: int,
    ) -> tuple[NDArray, NDArray, dict[str, Any]]:
        """Estimate the PSD from time-shifted off-source background windows.

        Averages (median) per-window Welch estimates instead of running a
        single Welch estimate over the on-source strain, which would let
        signal power leak into the noise estimate. The window length is
        shrunk (halved) from the requested default when the available
        strain is too short to yield ``min_background_windows`` non-excluded
        windows at that length; if even the shrunk search fails, this fails
        closed via ``PSDEstimationError`` instead of reverting to on-source.
        """
        requested_window = (
            background_window_duration
            if background_window_duration is not None
            else self.fft_length
        )
        total_duration = len(strain) / self.sample_rate

        window_duration = requested_window
        backgrounds: list[NDArray] = []
        for _ in range(12):
            step = max(window_duration / 4.0, 2.0 / self.sample_rate)
            backgrounds = self.make_background(
                strain,
                self.sample_rate,
                event_gps,
                segment_gps_start,
                window_duration=window_duration,
                n_background_windows=max_background_windows,
                time_shift_step=step,
            )
            if len(backgrounds) >= min_background_windows:
                break
            if window_duration * self.sample_rate < 64:
                break
            window_duration /= 2.0

        if len(backgrounds) < min_background_windows:
            raise PSDEstimationError(
                f"Insufficient off-source data for PSD estimation: found only "
                f"{len(backgrounds)} background window(s) (need "
                f"{min_background_windows}) around event_gps={event_gps} in "
                f"{total_duration:.2f}s of strain. Refusing to fall back to "
                "on-source PSD, which would self-contaminate the noise "
                "estimate with signal power. Provide a longer data segment, "
                "or explicitly lower min_background_windows if this is "
                "intentional."
            )

        fft_length_use = min(self.fft_length, window_duration)
        freqs_ref: NDArray | None = None
        psds = []
        for segment in backgrounds:
            freqs_bg, psd_bg = estimate_psd(
                segment, self.sample_rate, fft_length=fft_length_use
            )
            if freqs_ref is None:
                freqs_ref = freqs_bg
            psds.append(psd_bg)

        psd_median = np.median(np.asarray(psds), axis=0)
        quality = {
            "psd_source": "off_source",
            "n_off_source_windows": len(backgrounds),
            "off_source_window_duration_s": window_duration,
        }
        return freqs_ref, psd_median, quality

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
