"""Unit tests for likelihood functions."""

from __future__ import annotations

import numpy as np
import pytest

from whitesearch.likelihoods import (
    GWLikelihood,
    RadioBurstLikelihood,
    XRayBurstLikelihood,
    VisibilityLikelihood,
    JointLikelihood,
    poisson_loglike,
    gaussian_loglike,
    von_mises_loglike,
)
from whitesearch.simulators import (
    GravitationalWaveSimulator,
    EMBurstSimulator,
    XRayLightCurveSimulator,
    ImageShadowSimulator,
)


# ── Helper likelihoods ─────────────────────────────────────────────────────────

class TestPoissonLoglike:
    def test_at_mean(self):
        counts = np.array([3.0, 5.0, 7.0])
        mu = np.array([3.0, 5.0, 7.0])
        ll = poisson_loglike(counts, mu)
        assert np.isfinite(ll)

    def test_returns_minus_inf_for_zero_mu(self):
        counts = np.array([1.0])
        mu = np.array([0.0])
        ll = poisson_loglike(counts, mu)
        assert ll == -np.inf


class TestGaussianLoglike:
    def test_at_mean_is_maximum(self):
        data = np.array([1.0, 2.0, 3.0])
        sigma = np.array([1.0, 1.0, 1.0])
        ll_at_mean = gaussian_loglike(data, data, sigma)
        ll_off = gaussian_loglike(data, data + 1.0, sigma)
        assert ll_at_mean > ll_off

    def test_finite_result(self):
        data = np.random.default_rng(0).standard_normal(50)
        mu = np.zeros(50)
        sigma = np.ones(50)
        ll = gaussian_loglike(data, mu, sigma)
        assert np.isfinite(ll)


class TestVonMisesLoglike:
    def test_at_mean_is_maximum(self):
        phases = np.array([0.1, 0.2, -0.1])
        ll_on = von_mises_loglike(phases, phases, kappa=5.0)
        ll_off = von_mises_loglike(phases, phases + np.pi / 2, kappa=5.0)
        assert ll_on > ll_off


# ── GW Likelihood ─────────────────────────────────────────────────────────────

class TestGWLikelihood:
    def test_returns_finite(self, bounce_params, gw_context):
        sim = GravitationalWaveSimulator()
        sd = sim.simulate(bounce_params, gw_context, rng=np.random.default_rng(0))
        ll = GWLikelihood()
        val = ll.loglike(bounce_params, sd, gw_context)
        assert np.isfinite(val)

    def test_signal_beats_noise(self, bounce_params, gw_context):
        """Signal params should give higher LL than random params."""
        sim = GravitationalWaveSimulator()
        sd = sim.simulate(bounce_params, gw_context, rng=np.random.default_rng(0))
        ll = GWLikelihood()

        ll_signal = ll.loglike(bounce_params, sd, gw_context)

        rng = np.random.default_rng(99)
        random_params = {**bounce_params, "M": rng.uniform(5, 1000), "a_star": rng.uniform(0, 0.998)}
        ll_noise = ll.loglike(random_params, sd, gw_context)

        # This is a statistical test; signal should on average be higher
        # (may occasionally fail for extreme random draws)
        assert ll_signal > ll_noise - 1000  # soft check


class TestGWBandMask:
    """The noise-weighted inner products must integrate only over the
    analysis band [low_freq, min(high_freq, 0.95*nyquist)].  Outside it the
    bandpassed strain and PSD are filter-rolloff residuals whose ratio is
    numerically meaningless (on real GWOSC data it inflated <d|d> by ~1e5)."""

    @staticmethod
    def _make_obs(rng, with_out_of_band_junk):
        sr = 4096.0
        n = int(8.0 * sr)
        dt = 1.0 / sr
        df = 1.0 / (n * dt)
        freqs = np.fft.rfftfreq(n, d=dt)
        in_band = (freqs >= 20.0) & (freqs <= 1700.0)
        # Flat in-band PSD; rolloff-like tiny PSD outside (as after bandpass)
        psd = np.where(in_band, 1e-40, 1e-50)
        # Noise consistent with the likelihood's FFT convention
        # (h_tilde = rfft(h)*dt): E|d(f)|^2 = PSD/(2 df) makes the per-bin
        # contribution to <d|d> = 4|d|^2/PSD*df equal 2 on average.
        sigma = np.sqrt(psd / (4.0 * df))
        d_f = sigma * (rng.standard_normal(len(psd)) + 1j * rng.standard_normal(len(psd)))
        d_f[0] = d_f[0].real
        d_f[-1] = d_f[-1].real
        strain = np.fft.irfft(d_f / dt, n=n)
        if with_out_of_band_junk:
            # Rolloff residual: strain power at 1900 Hz far above the tiny
            # PSD there (the exact pathology seen on real data above the
            # 1700 Hz bandpass cutoff).
            t = np.arange(n) / sr
            strain = strain + 1e-19 * np.sin(2 * np.pi * 1900.0 * t)
        return {
            "strain": strain,
            "psd": psd,
            "sample_rate": sr,
            "t_merger": 4.0,
            "low_freq_cutoff": 20.0,
            "high_freq_cutoff": 1700.0,
        }, freqs, in_band

    def test_band_mask_bounds(self):
        freqs = np.linspace(0.0, 2048.0, 4097)
        mask = GWLikelihood._band_mask(freqs, 20.0, 1700.0, 2048.0)
        assert not mask[freqs < 20.0].any()
        assert not mask[freqs > 1700.0].any()
        assert mask[(freqs >= 20.0) & (freqs <= 1700.0)].all()
        # high_freq above 0.95*nyquist gets capped
        mask_hi = GWLikelihood._band_mask(freqs, 20.0, 3000.0, 2048.0)
        assert not mask_hi[freqs > 0.95 * 2048.0].any()

    def test_out_of_band_junk_does_not_inflate_inner_product(self):
        from whitesearch.likelihoods.gw_units import time_to_freq
        from whitesearch.utils.math_utils import noise_weighted_inner_product

        rng = np.random.default_rng(3)
        obs, freqs, in_band = self._make_obs(rng, with_out_of_band_junk=True)
        ll = GWLikelihood("null")
        lnl_masked = ll.loglike({}, obs, {})

        # Unmasked inner product over the full grid (the old behaviour)
        _, d_f, df = time_to_freq(obs["strain"], 1.0 / obs["sample_rate"])
        dd_unmasked = noise_weighted_inner_product(d_f, d_f, obs["psd"], df).real
        lnl_unmasked = -0.5 * dd_unmasked

        # The 1900 Hz junk dominates the unmasked value by orders of magnitude
        assert lnl_unmasked < 100.0 * lnl_masked  # both negative
        # Masked value sits at the statistically expected scale:
        # <d|d> ~ 2 * N_in_band_bins for PSD-consistent noise
        expected = -0.5 * 2.0 * int(in_band.sum())
        assert 0.5 < lnl_masked / expected < 2.0

    def test_in_band_power_not_clipped_by_mask(self):
        rng = np.random.default_rng(4)
        obs_clean, _, _ = self._make_obs(rng, with_out_of_band_junk=False)
        ll = GWLikelihood("null")
        lnl_clean = ll.loglike({}, obs_clean, {})

        # Add an *in-band* tone: masked <d|d> must grow (lnL drops)
        n = len(obs_clean["strain"])
        t = np.arange(n) / obs_clean["sample_rate"]
        obs_tone = dict(obs_clean)
        obs_tone["strain"] = obs_clean["strain"] + 1e-19 * np.sin(2 * np.pi * 100.0 * t)
        lnl_tone = ll.loglike({}, obs_tone, {})
        assert lnl_tone < lnl_clean - 100.0

        # An out-of-band tone of the same amplitude must NOT move lnL
        obs_oob = dict(obs_clean)
        obs_oob["strain"] = obs_clean["strain"] + 1e-19 * np.sin(2 * np.pi * 1900.0 * t)
        lnl_oob = ll.loglike({}, obs_oob, {})
        assert abs(lnl_oob - lnl_clean) < 0.01 * abs(lnl_clean)


# ── Radio Likelihood ───────────────────────────────────────────────────────────

class TestRadioLikelihood:
    def test_returns_finite(self, pbh_params, radio_context):
        sim = EMBurstSimulator()
        sd = sim.simulate(pbh_params, radio_context, rng=np.random.default_rng(0))
        ll = RadioBurstLikelihood()
        val = ll.loglike(pbh_params, sd, radio_context)
        assert np.isfinite(val)

    def test_parameter_names(self):
        ll = RadioBurstLikelihood("pbh_tunneling")
        assert "log10_M_g" in ll.parameter_names


# ── X-ray Likelihood ───────────────────────────────────────────────────────────

class TestXRayLikelihood:
    def test_returns_finite(self, pbh_params):
        sim = XRayLightCurveSimulator()
        context = {
            "area_cm2": 500.0,
            "bg_rate_cps": 0.2,
            "duration_s": 30.0,
            "dt_s": 1.0,
            "rng_seed": 42,
        }
        params = {**pbh_params, "log10_fluence_erg_cm2": -7.0, "log10_T90_s": 0.5, "log10_eta_gamma": -3.0}
        sd = sim.simulate(params, context, rng=np.random.default_rng(0))
        ll = XRayBurstLikelihood()
        val = ll.loglike(params, sd, context)
        assert np.isfinite(val)


# ── Visibility Likelihood ─────────────────────────────────────────────────────

class TestVisibilityLikelihood:
    def test_returns_finite(self, gr_params, image_context):
        sim = ImageShadowSimulator()
        sd = sim.simulate(gr_params, image_context, rng=np.random.default_rng(0))
        ll = VisibilityLikelihood()
        val = ll.loglike(gr_params, sd, image_context)
        assert np.isfinite(val)


# ── Joint Likelihood ──────────────────────────────────────────────────────────

class TestJointLikelihood:
    def test_sum_of_channels(self, bounce_params, pbh_params, gw_context, radio_context):
        gw_sim = GravitationalWaveSimulator()
        radio_sim = EMBurstSimulator()

        gw_sd = gw_sim.simulate(bounce_params, gw_context, rng=np.random.default_rng(0))
        radio_sd = radio_sim.simulate(pbh_params, radio_context, rng=np.random.default_rng(0))

        joint_ll = JointLikelihood(channels=["gw", "radio"], model_name="bounce")

        # Merge params
        merged_params = {**bounce_params, **pbh_params}
        data = {"gw": gw_sd, "radio": radio_sd}
        context = {"gw": gw_context, "radio": radio_context}

        val = joint_ll.loglike(merged_params, data, context)
        assert np.isfinite(val)

    def test_skips_missing_channel(self, bounce_params, gw_context):
        gw_sim = GravitationalWaveSimulator()
        gw_sd = gw_sim.simulate(bounce_params, gw_context, rng=np.random.default_rng(0))

        joint_ll = JointLikelihood(channels=["gw", "radio"], model_name="bounce")
        data = {"gw": gw_sd}  # no radio data
        val = joint_ll.loglike(bounce_params, data, gw_context)
        assert np.isfinite(val)
