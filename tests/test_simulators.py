"""Unit tests for forward simulators."""

from __future__ import annotations

import numpy as np
import pytest

from whitesearch.simulators import (
    GravitationalWaveSimulator,
    EMBurstSimulator,
    XRayLightCurveSimulator,
    ImageShadowSimulator,
    SimData,
)


class TestGWSimulator:
    def test_output_shape(self, gw_simulator, bounce_params, gw_context):
        sd = gw_simulator.simulate(bounce_params, gw_context)
        n_expected = int(gw_context["sample_rate"] * gw_context["duration"])
        assert len(sd.data) == n_expected

    def test_returns_simdata(self, gw_simulator, bounce_params, gw_context):
        sd = gw_simulator.simulate(bounce_params, gw_context)
        assert isinstance(sd, SimData)
        assert sd.channel == "gw"

    def test_metadata_keys(self, gw_simulator, bounce_params, gw_context):
        sd = gw_simulator.simulate(bounce_params, gw_context)
        for key in ["times", "sample_rate", "psd", "f_rd", "q_rd"]:
            assert key in sd.metadata

    def test_finite_output(self, gw_simulator, bounce_params, gw_context):
        sd = gw_simulator.simulate(bounce_params, gw_context)
        assert np.all(np.isfinite(sd.data))

    def test_deterministic_seed(self, gw_simulator, bounce_params, gw_context):
        ctx = dict(gw_context)
        ctx["rng_seed"] = 7
        sd1 = gw_simulator.simulate(bounce_params, ctx)
        sd2 = gw_simulator.simulate(bounce_params, ctx)
        np.testing.assert_array_equal(sd1.data, sd2.data)

    def test_signal_only_zero_noise(self, gw_simulator, bounce_params, gw_context):
        sd_full = gw_simulator.simulate(bounce_params, gw_context, rng=np.random.default_rng(0))
        sd_signal = gw_simulator.signal_only(bounce_params, gw_context)
        # Signal-only should have smaller RMS than signal+noise
        assert np.std(sd_signal.data) <= np.std(sd_full.data) + 1e-30

    def test_psd_positive(self, gw_simulator, bounce_params, gw_context):
        sd = gw_simulator.simulate(bounce_params, gw_context)
        psd = sd.metadata["psd"]
        assert np.all(psd >= 0)

    def test_params_true_stored(self, gw_simulator, bounce_params, gw_context):
        sd = gw_simulator.simulate(bounce_params, gw_context)
        assert sd.params_true["M"] == bounce_params["M"]

    def test_noise_normalization_matches_gw_units_convention(self):
        """Generated noise must satisfy E[|rfft(n)*dt|^2] = Sn*T/2, i.e. the
        per-bin contribution to the noise-weighted inner product averages 2
        (the old normalisation was a factor ~2*dt^2 low)."""
        from whitesearch.simulators.grav_wave import gaussian_noise_from_psd
        from whitesearch.likelihoods.gw_units import time_to_freq

        sr = 4096.0
        n = int(16.0 * sr)
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        psd = np.full(len(freqs), 1e-40)
        rng = np.random.default_rng(0)

        acc = np.zeros(len(freqs))
        n_real = 10
        for _ in range(n_real):
            noise = gaussian_noise_from_psd(psd, n, sr, rng)
            _, n_f, df = time_to_freq(noise, 1.0 / sr)
            acc += 4.0 * np.abs(n_f) ** 2 / psd * df
        per_bin = acc / n_real
        mean = float(per_bin[10:-10].mean())
        assert 1.8 < mean < 2.2


class TestEMBurstSimulator:
    def test_output_shape(self, radio_simulator, pbh_params, radio_context):
        sd = radio_simulator.simulate(pbh_params, radio_context)
        n_freq = radio_context["n_freq_chans"]
        n_time = radio_context["n_time_bins"]
        assert sd.data.shape == (n_freq, n_time)

    def test_returns_simdata(self, radio_simulator, pbh_params, radio_context):
        sd = radio_simulator.simulate(pbh_params, radio_context)
        assert isinstance(sd, SimData)
        assert sd.channel == "radio"

    def test_metadata_has_dm(self, radio_simulator, pbh_params, radio_context):
        sd = radio_simulator.simulate(pbh_params, radio_context)
        assert "dm" in sd.metadata

    def test_finite_output(self, radio_simulator, pbh_params, radio_context):
        sd = radio_simulator.simulate(pbh_params, radio_context)
        assert np.all(np.isfinite(sd.data))

    def test_deterministic_seed(self, radio_simulator, pbh_params, radio_context):
        ctx = dict(radio_context)
        ctx["rng_seed"] = 3
        sd1 = radio_simulator.simulate(pbh_params, ctx)
        sd2 = radio_simulator.simulate(pbh_params, ctx)
        np.testing.assert_array_equal(sd1.data, sd2.data)


class TestXRayLightCurveSimulator:
    def test_output_integer_counts(self, pbh_params):
        sim = XRayLightCurveSimulator()
        context = {
            "area_cm2": 500.0,
            "bg_rate_cps": 0.2,
            "duration_s": 50.0,
            "dt_s": 1.0,
            "rng_seed": 5,
        }
        params = {**pbh_params, "log10_fluence_erg_cm2": -7.0, "log10_T90_s": 1.0}
        sd = sim.simulate(params, context)
        assert np.all(sd.data >= 0)
        assert sd.data.dtype in (np.int64, np.int32, int, np.intp)

    def test_returns_simdata(self, pbh_params):
        sim = XRayLightCurveSimulator()
        context = {"area_cm2": 500.0, "bg_rate_cps": 0.2, "duration_s": 20.0, "dt_s": 1.0}
        params = {**pbh_params, "log10_fluence_erg_cm2": -7.0, "log10_T90_s": 0.5}
        sd = sim.simulate(params, context)
        assert isinstance(sd, SimData)
        assert sd.channel == "xray"


class TestImageShadowSimulator:
    def test_output_complex(self, image_simulator, gr_params, image_context):
        sd = image_simulator.simulate(gr_params, image_context)
        assert np.iscomplexobj(sd.data)

    def test_metadata_has_image(self, image_simulator, gr_params, image_context):
        sd = image_simulator.simulate(gr_params, image_context)
        assert "image" in sd.metadata
        assert sd.metadata["image"].shape == (
            image_context["n_pixels"],
            image_context["n_pixels"],
        )

    def test_closure_phases_in_range(self, image_simulator, gr_params, image_context):
        sd = image_simulator.simulate(gr_params, image_context)
        cp = sd.metadata["closure_phases"]
        assert np.all(np.abs(cp) <= np.pi + 1e-6)

    def test_shadow_radius_positive(self, image_simulator, gr_params, image_context):
        sd = image_simulator.simulate(gr_params, image_context)
        assert sd.metadata["r_ring_muas"] > 0
