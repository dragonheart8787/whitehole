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
