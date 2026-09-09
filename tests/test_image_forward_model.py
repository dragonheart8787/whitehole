"""image/VLBI channel forward-model alignment and fail-closed behaviour.

Locks in the work on docs/XRAY_IMAGE_PREFLIGHT_AUDIT.md X.4 (null could not be
fit on this channel), X.8.1 (the data's own uv_coverage was never forwarded to
the simulator) and X.8.2 (the closure-phase term vanished silently when the key
was absent), and records the state of X.6 (ring-radius representability), which
is deliberately NOT fixed here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from whitesearch.inference import BilbyRunner
from whitesearch.likelihoods import VisibilityLikelihood
from whitesearch.models import get_model
from whitesearch.models.gr_eternal import (
    IMAGE_FOV_MUAS,
    IMAGE_N_PIXELS,
    IMAGE_PIXEL_MUAS,
    GREternalWhiteHole,
)
from whitesearch.simulators import get_simulator
from whitesearch.simulators.image_shadow import (
    _default_eht_uv,
    _shadow_radius_muas,
)

CTX = {"thermal_noise_jy": 0.05, "freq_ghz": 230.0,
       "fov_muas": IMAGE_FOV_MUAS, "n_pixels": IMAGE_N_PIXELS, "rng_seed": 7}


@pytest.fixture(scope="module")
def observation():
    """A draw whose ring the grid can actually represent.

    M and D_L are pinned rather than sampled: 80.98% of this model's prior puts
    the ring below one pixel, where the image is exactly zero and every test
    below would pass or fail for the wrong reason (audit X.6).
    """
    model = get_model("gr_eternal")
    theta = model.sample_prior(np.random.default_rng(3))
    theta = {**theta, "M": 6.5e9, "D_L": 5.0, "ring_width_frac": 0.2}
    assert _shadow_radius_muas(theta["M"], theta["a_star"], theta["D_L"]) > 50.0
    data = get_simulator("image").simulate(
        theta, CTX, rng=np.random.default_rng(11)
    )
    return theta, data


class TestUvCoverageComesFromTheData:
    """X.8.1: model visibilities must be built on the data's own baselines."""

    def test_data_uv_coverage_is_used_not_the_default(self, observation):
        theta, data = observation
        shifted = np.asarray(_default_eht_uv(), dtype=float) * 1.7
        moved = dict(data.metadata)
        moved["uv_coverage"] = shifted
        obs = {"visibilities": np.asarray(data.data), **moved}

        like = VisibilityLikelihood(use_closure_phases=False)
        ll_shifted = like.loglike(theta, obs, CTX)

        default_obs = dict(obs)
        default_obs["uv_coverage"] = np.asarray(_default_eht_uv(), dtype=float)
        ll_default = like.loglike(theta, default_obs, CTX)

        # Same observed visibilities, different baselines for the model: if the
        # coverage were ignored these would be identical.
        assert ll_shifted != ll_default

    def test_context_is_not_mutated(self, observation):
        theta, data = observation
        ctx = dict(CTX)
        VisibilityLikelihood(use_closure_phases=False).loglike(theta, data, ctx)
        assert "uv_coverage" not in ctx

    def test_missing_uv_coverage_raises(self, observation):
        theta, data = observation
        obs = {k: v for k, v in data.metadata.items() if k != "uv_coverage"}
        obs["visibilities"] = np.asarray(data.data)
        with pytest.raises(KeyError, match="uv_coverage"):
            VisibilityLikelihood(use_closure_phases=False).loglike(theta, obs, CTX)

    def test_context_may_supply_it_when_metadata_does_not(self, observation):
        theta, data = observation
        obs = {k: v for k, v in data.metadata.items() if k != "uv_coverage"}
        obs["visibilities"] = np.asarray(data.data)
        ctx = {**CTX, "uv_coverage": np.asarray(_default_eht_uv(), dtype=float)}
        assert np.isfinite(
            VisibilityLikelihood(use_closure_phases=False).loglike(theta, obs, ctx)
        )


class TestClosurePhasesAreExplicit:
    """X.8.2: no silent degradation to amplitude-only."""

    def test_missing_closure_phases_raises_when_requested(self, observation):
        theta, data = observation
        obs = {k: v for k, v in data.metadata.items() if k != "closure_phases"}
        obs["visibilities"] = np.asarray(data.data)
        with pytest.raises(KeyError, match="closure_phases"):
            VisibilityLikelihood(use_closure_phases=True).loglike(theta, obs, CTX)

    def test_amplitude_only_must_be_declared(self, observation):
        theta, data = observation
        obs = {k: v for k, v in data.metadata.items() if k != "closure_phases"}
        obs["visibilities"] = np.asarray(data.data)
        like = VisibilityLikelihood(use_closure_phases=False)
        assert np.isfinite(like.loglike(theta, obs, CTX))
        assert like.last_closure_config["used_closure_phase"] is False
        assert "amplitude-only" in like.last_closure_config["reason"]

    def test_use_records_how_many_were_used(self, observation):
        theta, data = observation
        like = VisibilityLikelihood(use_closure_phases=True)
        like.loglike(theta, data, CTX)
        assert like.last_closure_config["used_closure_phase"] is True
        assert like.last_closure_config["n_closure_phases"] > 0

    def test_the_term_actually_changes_the_answer(self, observation):
        theta, data = observation
        with_cp = VisibilityLikelihood(use_closure_phases=True).loglike(theta, data, CTX)
        without = VisibilityLikelihood(use_closure_phases=False).loglike(theta, data, CTX)
        assert with_cp != without


class TestSampledDimension:
    """X.4: the null hypothesis can be fit on this channel again."""

    def test_null_has_an_empty_parameter_list(self):
        assert VisibilityLikelihood("null").parameter_names == []

    def test_null_builds_priors(self):
        eff = BilbyRunner.effective_parameter_names(
            get_model("null"), VisibilityLikelihood("null")
        )
        assert eff == []

    def test_gr_eternal_dimension_is_unchanged(self):
        eff = BilbyRunner.effective_parameter_names(
            get_model("gr_eternal"), VisibilityLikelihood("gr_eternal")
        )
        assert eff == ["M", "a_star", "D_L", "i", "position_angle",
                       "ring_width_frac", "log10_brightness"]

    def test_bh_accretion_still_cannot_build_priors(self):
        """NOT fixed here, and recorded as such.

        BHAccretion declares log10_mdot_edd / jet_power_frac and none of
        position_angle / ring_width_frac / log10_brightness, so wiring it up
        means deciding how an accretion model maps onto a geometric ring - a
        modelling decision, not a mechanical one.  See
        docs/XRAY_IMAGE_PREFLIGHT_AUDIT.md X.11.4.
        """
        with pytest.raises(ValueError, match="does not declare"):
            BilbyRunner.effective_parameter_names(
                get_model("bh_accretion"), VisibilityLikelihood("bh_accretion")
            )


class TestRingRepresentability:
    """X.6: derivation is in code; the prior is deliberately NOT narrowed."""

    def test_grid_constants_match_eht_yaml(self):
        cfg = yaml.safe_load(
            (Path(__file__).resolve().parents[1]
             / "configs" / "instruments" / "eht.yaml").read_text()
        )
        assert cfg["imaging"]["fov_muas"] == IMAGE_FOV_MUAS
        assert cfg["imaging"]["n_pixels"] == IMAGE_N_PIXELS
        assert IMAGE_PIXEL_MUAS == pytest.approx(3.125)

    def test_window_is_derived_from_the_grid(self):
        lo, hi = GREternalWhiteHole.ring_radius_representable_range_muas()
        assert lo == pytest.approx(8.245 * IMAGE_PIXEL_MUAS)
        assert hi == pytest.approx(IMAGE_FOV_MUAS)
        # halving the pixel size halves the floor
        lo2, _ = GREternalWhiteHole.ring_radius_representable_range_muas(
            fov_muas=IMAGE_FOV_MUAS, n_pixels=2 * IMAGE_N_PIXELS
        )
        assert lo2 == pytest.approx(lo / 2.0)

    def test_the_floor_lands_on_top_of_the_real_targets(self):
        """KNOWN DEFICIENCY: the floor is not comfortably below the targets.

        The two sources this channel exists to describe sit either side of it:
        M87* at 19.67 muas is BELOW the 25.77 muas floor, Sgr A* at 25.80 muas
        is barely above.  So the shipped grid cannot faithfully represent a thin
        ring at M87*'s scale.  Recorded, not fixed -- see
        docs/XRAY_IMAGE_PREFLIGHT_AUDIT.md X.11.1.
        """
        lo, hi = GREternalWhiteHole.ring_radius_representable_range_muas()
        r_m87 = _shadow_radius_muas(6.5e9, 0.9, 16.8)
        r_sgra = _shadow_radius_muas(4.15e6, 0.9, 0.008178)
        assert r_m87 == pytest.approx(19.667, abs=0.01)
        assert r_sgra == pytest.approx(25.795, abs=0.01)
        assert r_m87 < lo, "M87* sits below the representability floor"
        assert lo < r_sgra < hi

    def test_prior_representability_is_still_poor(self):
        """KNOWN DEFICIENCY, recorded rather than silently tolerated.

        Not a target to preserve: if a reparameterisation onto the M/D_L ratio
        lands, this test SHOULD fail and be updated deliberately.
        """
        model = get_model("gr_eternal")
        r = np.array([
            _shadow_radius_muas(
                (th := model.sample_prior(np.random.default_rng(80000 + k)))["M"],
                th["a_star"], th["D_L"])
            for k in range(4000)
        ])
        inside = ((r >= IMAGE_PIXEL_MUAS) & (r <= IMAGE_FOV_MUAS)).mean()
        assert 0.15 < inside < 0.25
        assert (r < IMAGE_PIXEL_MUAS).mean() > 0.75
