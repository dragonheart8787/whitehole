"""Unit tests for GWPreprocessor PSD estimation (on-source vs off-source)."""

from __future__ import annotations

import numpy as np
import pytest

from whitesearch.preprocess.gw_preprocess import GWPreprocessor, PSDEstimationError
from whitesearch.utils.math_utils import estimate_psd


@pytest.fixture
def rng():
    return np.random.default_rng(1)


class TestNotchFilter:
    def test_rms_preserved_on_white_noise(self, rng):
        """Notch filters remove narrow lines only; broadband RMS must stay at
        the input's order of magnitude (the old FIR-convolve bug blew it up
        by a factor of ~8000)."""
        sr = 4096.0
        n = int(32.0 * sr)
        strain = rng.standard_normal(n) * 1e-21

        prep = GWPreprocessor(sample_rate=sr)
        notched = prep.notch_filter(strain)

        ratio = np.std(notched) / np.std(strain)
        assert 0.5 < ratio < 2.0

    def test_power_suppressed_at_notch_lines(self, rng):
        """The filter must actually attenuate the notch frequencies (not be
        a no-op) while leaving nearby frequencies essentially untouched."""
        sr = 4096.0
        n = int(32.0 * sr)
        strain = rng.standard_normal(n) * 1e-21

        prep = GWPreprocessor(sample_rate=sr)
        notched = prep.notch_filter(strain)

        freqs, psd_in = estimate_psd(strain, sr, fft_length=4.0)
        _, psd_out = estimate_psd(notched, sr, fft_length=4.0)

        for f0 in (60.0, 120.0, 500.0):
            i_line = int(np.argmin(np.abs(freqs - f0)))
            i_ref = int(np.argmin(np.abs(freqs - (f0 + 20.0))))
            assert psd_out[i_line] < 0.01 * psd_in[i_line]
            assert psd_out[i_ref] > 0.5 * psd_in[i_ref]


class TestOffSourcePSD:
    def test_off_source_used_when_location_given(self, rng):
        """With event_gps/segment_gps_start supplied, PSD must come off-source."""
        sr = 4096.0
        n = int(32.0 * sr)
        strain = rng.standard_normal(n) * 1e-21

        prep = GWPreprocessor(sample_rate=sr)
        out = prep.prepare_raw(strain, event_gps=16.0, segment_gps_start=0.0)

        q = out["quality"]
        assert q["psd_source"] == "off_source"
        assert q["n_off_source_windows"] >= 8
        # 32s of data around a centered event comfortably fits the default
        # (fft_length) window without needing to shrink.
        assert q["off_source_window_duration_s"] == prep.fft_length

    def test_on_source_psd_is_self_contaminated_by_on_source_burst(self, rng):
        """A loud burst inside the on-source window must not leak into the
        off-source PSD estimate, but does contaminate the naive on-source one.

        The 100 Hz burst frequency deliberately avoids the default notch
        lines, so the standard preprocessing chain leaves it intact.
        """
        sr = 4096.0
        duration = 32.0
        n = int(duration * sr)
        t = np.arange(n) / sr
        strain = rng.standard_normal(n) * 1e-21

        event_gps = 16.0
        gps_start = 0.0
        f_burst = 100.0
        burst_amp = 1e-21 * 200
        on_source_mask = (t > 15.0) & (t < 17.0)
        strain[on_source_mask] += burst_amp * np.sin(2 * np.pi * f_burst * t[on_source_mask])

        prep = GWPreprocessor(sample_rate=sr)
        out_on = prep.prepare_raw(strain.copy())
        out_off = prep.prepare_raw(
            strain.copy(), event_gps=event_gps, segment_gps_start=gps_start
        )

        assert out_on["quality"]["psd_source"] == "on_source"
        assert out_off["quality"]["psd_source"] == "off_source"

        freqs_full = np.fft.rfftfreq(n, d=1.0 / sr)
        idx = int(np.argmin(np.abs(freqs_full - f_burst)))

        psd_on_at_burst = out_on["psd"][idx]
        psd_off_at_burst = out_off["psd"][idx]

        # The on-source (self-contaminated) PSD absorbs the burst power and
        # is orders of magnitude higher at the burst frequency than the
        # off-source estimate, which never sees the burst.
        assert psd_on_at_burst > 1000 * psd_off_at_burst

    def test_fails_closed_when_off_source_data_insufficient(self, rng):
        """A segment with (almost) no off-source room must raise, not
        silently fall back to on-source PSD."""
        sr = 4096.0
        strain = rng.standard_normal(64) * 1e-21

        prep = GWPreprocessor(sample_rate=sr)
        with pytest.raises(PSDEstimationError):
            prep.prepare_raw(
                strain,
                event_gps=0.0078,
                segment_gps_start=0.0,
                min_background_windows=8,
            )

    def test_on_source_fallback_explicitly_marked_when_no_location(self, rng):
        """Omitting event_gps/segment_gps_start keeps the old on-source
        behaviour, but it must show up explicitly in quality, never silently."""
        sr = 4096.0
        n = int(2.0 * sr)
        strain = rng.standard_normal(n) * 1e-21

        prep = GWPreprocessor(sample_rate=sr)
        out = prep.prepare_raw(strain)

        q = out["quality"]
        assert q["psd_source"] == "on_source"
        assert q["n_off_source_windows"] == 0
        assert "psd_source_reason" in q

    def test_short_mock_segment_still_gets_off_source_psd(self, rng):
        """Regression guard: short mock-style segments (as used across the
        test suite, e.g. 1-4s) must still produce a valid off-source PSD by
        adaptively shrinking the background window, not fail closed."""
        sr = 4096.0
        n = int(1.0 * sr)
        strain = rng.standard_normal(n) * 1e-21

        prep = GWPreprocessor(sample_rate=sr)
        out = prep.prepare_raw(strain, event_gps=0.5, segment_gps_start=0.0)

        q = out["quality"]
        assert q["psd_source"] == "off_source"
        assert q["n_off_source_windows"] >= 8
        assert q["off_source_window_duration_s"] < prep.fft_length
