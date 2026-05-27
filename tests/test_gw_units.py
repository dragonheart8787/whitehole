"""Tests for GW FFT conventions and likelihood finiteness."""

from __future__ import annotations

import numpy as np
import pytest

from whitesearch.dataio.gw_observation import prepare_gw_from_simdata
from whitesearch.likelihoods.gw_likelihood import GWLikelihood
from whitesearch.likelihoods.gw_units import time_to_freq
from whitesearch.models import BlackToWhiteBounce, NullHypothesis, StandardBHRingdown
from whitesearch.simulators import GravitationalWaveSimulator


@pytest.fixture
def gw_obs_dict(bounce_params, gw_context):
    sim = GravitationalWaveSimulator()
    d = sim.simulate(bounce_params, gw_context, rng=np.random.default_rng(0))
    return prepare_gw_from_simdata(d, reference_amplitude=False)


@pytest.fixture
def gw_context_dict(gw_context):
    return gw_context


def test_time_to_freq_roundtrip():
    rng = np.random.default_rng(0)
    h = rng.standard_normal(4096)
    dt = 1.0 / 4096
    _, hf, _ = time_to_freq(h, dt)
    h2 = np.fft.irfft(hf / dt, n=len(h))
    assert np.allclose(h, h2, atol=1e-10)


def test_three_models_finite_lnL(gw_obs_dict, gw_context_dict):
    data = gw_obs_dict
    ctx = gw_context_dict
    null_ll = GWLikelihood("null").loglike({}, data, ctx)
    assert np.isfinite(null_ll)

    bounce = BlackToWhiteBounce()
    theta_b = bounce.sample_prior(np.random.default_rng(1))
    ll_b = GWLikelihood("bounce").loglike(theta_b, data, ctx)
    assert np.isfinite(ll_b)

    bh = StandardBHRingdown()
    theta_bh = bh.sample_prior(np.random.default_rng(2))
    ll_bh = GWLikelihood("bh_ringdown").loglike(theta_bh, data, ctx)
    assert np.isfinite(ll_bh)


def test_bh_ringdown_uses_log10_A(gw_obs_dict, gw_context_dict):
    data = gw_obs_dict
    ctx = gw_context_dict
    ll = GWLikelihood("bh_ringdown")
    theta = {"M": 30.0, "a_star": 0.6, "log10_A": -22.0, "D_L": 400.0, "i": 0.5}
    l1 = ll.loglike(theta, data, ctx)
    theta2 = dict(theta)
    theta2["log10_A"] = -20.0
    l2 = ll.loglike(theta2, data, ctx)
    assert l1 != l2
