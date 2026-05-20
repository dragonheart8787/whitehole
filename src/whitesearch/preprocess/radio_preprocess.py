"""Radio / FRB data preprocessing pipeline.

Steps
-----
1. Load dynamic spectrum (frequency × time)
2. Flag RFI (frequency channels and time bins)
3. Dedisperse to a trial DM
4. Bandpass calibration
5. Build on-burst / off-burst windows for background
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..utils.constants import K_DM

logger = logging.getLogger(__name__)


class RadioPreprocessor:
    """Preprocessing pipeline for radio dynamic spectra."""

    def __init__(
        self,
        freq_low_mhz: float = 400.0,
        freq_high_mhz: float = 800.0,
        n_freq_chans: int = 64,
        rfi_threshold_sigma: float = 3.0,
    ) -> None:
        self.freq_low = freq_low_mhz
        self.freq_high = freq_high_mhz
        self.n_freq = n_freq_chans
        self.rfi_sigma = rfi_threshold_sigma
        self.freqs_mhz = np.linspace(freq_low_mhz, freq_high_mhz, n_freq_chans)

    def prepare_raw(
        self,
        dynamic_spectrum: NDArray,
        times_s: NDArray,
        dm_trial: float = 0.0,
    ) -> dict[str, Any]:
        """Full preprocessing pipeline for a raw dynamic spectrum.

        Parameters
        ----------
        dynamic_spectrum : ndarray, shape (n_freq, n_time)
        times_s : array, shape (n_time,) — time axis [s]
        dm_trial : float — trial DM for dedispersion [pc/cm^3]

        Returns
        -------
        dict with:
            ds_clean      : dedispersed, RFI-flagged dynamic spectrum
            bandpass      : per-channel mean intensity (for calibration)
            off_burst_rms : per-channel off-burst RMS (noise level)
            rfi_mask      : boolean mask, True = flagged (bad)
            quality       : dict of quality metrics
        """
        ds = np.asarray(dynamic_spectrum, dtype=np.float64)

        # 1. RFI flagging
        rfi_mask = self.flag_rfi(ds)
        ds_clean = ds.copy()
        ds_clean[rfi_mask] = np.nan

        # 2. Bandpass calibration
        ds_clean = self.calibrate_bandpass(ds_clean)

        # 3. Dedisperse
        if dm_trial > 0:
            ds_clean = self.dedisperse(ds_clean, times_s, dm_trial)

        # 4. Off-burst statistics
        off_rms = np.nanstd(ds_clean, axis=1)

        quality = {
            "fraction_flagged": float(np.mean(rfi_mask)),
            "n_flagged_chans": int(np.sum(np.all(rfi_mask, axis=1))),
        }

        return {
            "ds_clean": ds_clean,
            "times_s": times_s,
            "freqs_mhz": self.freqs_mhz,
            "bandpass": np.nanmean(ds_clean, axis=1),
            "off_burst_rms": off_rms,
            "rfi_mask": rfi_mask,
            "dm_trial": dm_trial,
            "quality": quality,
        }

    def flag_rfi(self, ds: NDArray) -> NDArray:
        """Flag RFI using median absolute deviation per frequency channel.

        Returns boolean mask: True = flagged bad.
        """
        mask = np.zeros(ds.shape, dtype=bool)
        for i_f in range(ds.shape[0]):
            row = ds[i_f]
            med = np.nanmedian(row)
            mad = np.nanmedian(np.abs(row - med)) + 1e-30
            outliers = np.abs(row - med) > self.rfi_sigma * 1.4826 * mad
            mask[i_f, outliers] = True

        # Also flag entire channels with excessive mean power
        chan_means = np.nanmean(np.where(~mask, ds, np.nan), axis=1)
        global_med = np.nanmedian(chan_means)
        global_mad = np.nanmedian(np.abs(chan_means - global_med)) + 1e-30
        bad_chans = np.abs(chan_means - global_med) > 5.0 * 1.4826 * global_mad
        mask[bad_chans, :] = True

        return mask

    def calibrate_bandpass(self, ds: NDArray) -> NDArray:
        """Subtract per-channel median bandpass."""
        bandpass = np.nanmedian(ds, axis=1, keepdims=True)
        with np.errstate(invalid="ignore"):
            calibrated = ds - bandpass
        return calibrated

    def dedisperse(
        self,
        ds: NDArray,
        times_s: NDArray,
        dm: float,
    ) -> NDArray:
        """Shift frequency channels to remove DM dispersion.

        Uses the top channel as the reference (zero delay).
        """
        from ..utils.math_utils import apply_dm_dispersion

        # apply_dm_dispersion needs positive delay applied as roll
        # We call it with the negative of DM to undo the dispersion
        return apply_dm_dispersion(times_s, ds, -dm, self.freqs_mhz)

    @staticmethod
    def make_background_windows(
        ds: NDArray,
        times_s: NDArray,
        burst_time_s: float,
        window_width_s: float = 0.1,
        n_windows: int = 50,
        rng: np.random.Generator | None = None,
    ) -> list[NDArray]:
        """Extract off-burst windows from the dynamic spectrum."""
        if rng is None:
            rng = np.random.default_rng(42)

        dt = times_s[1] - times_s[0]
        win_bins = int(window_width_s / dt)
        exclusion_bins = int(window_width_s * 2 / dt)
        burst_idx = int(np.argmin(np.abs(times_s - burst_time_s)))

        windows = []
        for _ in range(n_windows * 5):
            start = rng.integers(0, len(times_s) - win_bins)
            end = start + win_bins
            if abs(start - burst_idx) < exclusion_bins or abs(end - burst_idx) < exclusion_bins:
                continue
            windows.append(ds[:, start:end])
            if len(windows) >= n_windows:
                break
        return windows

    @staticmethod
    def compute_dm_structure_function(
        ds: NDArray,
        times_s: NDArray,
        freqs_mhz: NDArray,
        dm_range: tuple[float, float] = (0.0, 3000.0),
        n_dm_trials: int = 1000,
    ) -> dict[str, NDArray]:
        """Search for a DM via the structure function method.

        Returns DM trial values and corresponding peak S/N.
        """
        dms = np.linspace(dm_range[0], dm_range[1], n_dm_trials)
        snr = np.zeros(n_dm_trials)

        for i, dm in enumerate(dms):
            from ..utils.math_utils import apply_dm_dispersion

            ds_dd = apply_dm_dispersion(times_s, ds.copy(), -dm, freqs_mhz)
            profile = np.nansum(ds_dd, axis=0)
            rms = np.nanstd(profile)
            if rms > 0:
                snr[i] = np.nanmax(profile) / rms

        best_dm = dms[np.argmax(snr)]
        return {"dm_trials": dms, "snr": snr, "best_dm": best_dm}
