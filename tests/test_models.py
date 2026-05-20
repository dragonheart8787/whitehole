"""Unit tests for model parameter schemas and prior sampling."""

from __future__ import annotations

import numpy as np
import pytest

from whitesearch.models import (
    BlackToWhiteBounce,
    PBHTunnelingWhiteHole,
    GREternalWhiteHole,
    NullHypothesis,
    MagnetarFlare,
    StandardBHRingdown,
    BHAccretion,
    get_model,
    MODEL_REGISTRY,
)
from whitesearch.models.base import ParameterSpec


# ── ParameterSpec ─────────────────────────────────────────────────────────────

class TestParameterSpec:
    def test_uniform_sample_in_range(self, rng):
        spec = ParameterSpec("x", "uniform", prior_kwargs={"low": 1.0, "high": 5.0})
        for _ in range(100):
            v = spec.sample(rng)
            assert 1.0 <= v <= 5.0

    def test_log_uniform_positive(self, rng):
        spec = ParameterSpec("x", "log_uniform", prior_kwargs={"low": 0.01, "high": 100.0})
        for _ in range(100):
            v = spec.sample(rng)
            assert 0.01 <= v <= 100.0

    def test_cos_uniform_in_range(self, rng):
        spec = ParameterSpec("x", "cos_uniform", prior_kwargs={})
        for _ in range(100):
            v = spec.sample(rng)
            assert 0.0 <= v <= np.pi

    def test_volume_uniform_in_range(self, rng):
        spec = ParameterSpec("x", "volume_uniform", prior_kwargs={"low": 1.0, "high": 100.0})
        for _ in range(100):
            v = spec.sample(rng)
            assert 1.0 <= v <= 100.0

    def test_discrete_uniform_in_values(self, rng):
        spec = ParameterSpec("x", "discrete_uniform", prior_kwargs={"values": [4, 5]})
        for _ in range(50):
            v = spec.sample(rng)
            assert v in (4, 5)

    def test_log_prior_outside_support(self):
        spec = ParameterSpec("x", "uniform", prior_kwargs={"low": 0.0, "high": 1.0})
        assert spec.log_prior(-0.1) == -np.inf
        assert spec.log_prior(1.1) == -np.inf

    def test_log_prior_normalised(self, rng):
        spec = ParameterSpec("x", "uniform", prior_kwargs={"low": 0.0, "high": 2.0})
        # ∫ p(x) dx = 1 → log p = log(1/2)
        assert abs(spec.log_prior(0.5) - np.log(0.5)) < 1e-10


# ── Bounce model ──────────────────────────────────────────────────────────────

class TestBlackToWhiteBounce:
    def test_parameter_names_present(self, bounce_model):
        names = bounce_model.parameter_names
        required = ["M", "a_star", "eps_f", "eps_Q", "D_L", "i"]
        for r in required:
            assert r in names, f"Missing parameter: {r}"

    def test_sample_prior_in_support(self, bounce_model, rng):
        for _ in range(20):
            params = bounce_model.sample_prior(rng)
            assert 5.0 <= params["M"] <= 1000.0
            assert 0.0 <= params["a_star"] <= 0.998
            assert -0.3 <= params["eps_f"] <= 0.3

    def test_log_prior_finite(self, bounce_model, bounce_params):
        lp = bounce_model.log_prior(bounce_params)
        assert np.isfinite(lp)

    def test_summary_stats_keys(self, bounce_model, bounce_params):
        stats = bounce_model.summary_stats(bounce_params)
        assert "f_gr_hz" in stats
        assert "q_gr" in stats
        assert "f_mod_hz" in stats
        assert "delta_f_hz" in stats
        assert "tau_bounce_s" in stats

    def test_frequency_positive(self, bounce_model, bounce_params):
        stats = bounce_model.summary_stats(bounce_params)
        assert stats["f_gr_hz"] > 0
        assert stats["f_mod_hz"] > 0

    def test_eps_f_modifies_frequency(self, bounce_model, bounce_params):
        stats0 = bounce_model.summary_stats({**bounce_params, "eps_f": 0.0})
        statsP = bounce_model.summary_stats({**bounce_params, "eps_f": 0.1})
        assert abs(statsP["f_mod_hz"] - stats0["f_mod_hz"] * 1.1) < 1.0


# ── PBH tunneling model ────────────────────────────────────────────────────────

class TestPBHTunnelingWhiteHole:
    def test_parameter_names(self, pbh_model):
        assert "log10_M_g" in pbh_model.parameter_names
        assert "log10_f_pbh" in pbh_model.parameter_names

    def test_sample_prior(self, pbh_model, rng):
        params = pbh_model.sample_prior(rng)
        assert 13.0 <= params["log10_M_g"] <= 16.0
        assert -10.0 <= params["log10_f_pbh"] <= 0.0

    def test_summary_stats(self, pbh_model, pbh_params):
        stats = pbh_model.summary_stats(pbh_params)
        assert "DM_total" in stats
        assert "fluence_jy_ms" in stats
        assert stats["DM_total"] > 0
        assert stats["fluence_jy_ms"] > 0

    def test_dm_increases_with_z(self, pbh_model, pbh_params):
        stats_lo = pbh_model.summary_stats({**pbh_params, "z": 0.1})
        stats_hi = pbh_model.summary_stats({**pbh_params, "z": 1.0})
        assert stats_hi["DM_total"] > stats_lo["DM_total"]


# ── GR Eternal White Hole ─────────────────────────────────────────────────────

class TestGREternalWhiteHole:
    def test_shadow_diameter_m87_order(self, gr_model, gr_params):
        stats = gr_model.summary_stats(gr_params)
        # M87*: ~40 μas
        assert 10.0 < stats["theta_d_muas"] < 200.0

    def test_axial_ratio_range(self, gr_model, gr_params):
        stats = gr_model.summary_stats(gr_params)
        assert 0.0 <= stats["axial_ratio"] <= 1.0


# ── Alternative models ────────────────────────────────────────────────────────

class TestAlternativeModels:
    def test_null_no_params(self, null_model):
        assert len(null_model.parameters()) == 0
        assert null_model.summary_stats({}) == {}

    def test_magnetar_has_dm(self, magnetar_model):
        assert "DM" in magnetar_model.parameter_names

    def test_bh_ringdown_eps_zero(self, bh_ringdown_model):
        params = {
            "M": 60.0, "a_star": 0.6,
            "log10_A": -22.0, "D_L": 100.0, "i": 0.3,
        }
        stats = bh_ringdown_model.summary_stats(params)
        assert stats["delta_f_hz"] == 0.0
        assert stats["delta_Q"] == 0.0


# ── Registry ──────────────────────────────────────────────────────────────────

def test_model_registry():
    assert "bounce" in MODEL_REGISTRY
    assert "pbh_tunneling" in MODEL_REGISTRY
    assert "gr_eternal" in MODEL_REGISTRY
    assert "null" in MODEL_REGISTRY


def test_get_model():
    m = get_model("bounce")
    assert isinstance(m, BlackToWhiteBounce)


def test_get_model_unknown():
    with pytest.raises(KeyError):
        get_model("does_not_exist")
