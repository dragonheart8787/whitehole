"""image channel: per-target distance (I-2) and the bh_accretion model (I-1).

I-2 -- the source distance is a per-target known constant, taken from the
observation metadata or the analysis context (fail-closed, no default), and
``D_L`` has left the sampled parameter vector.  The mass prior is now stated
per target, from published measurements.

I-1 -- ``bh_accretion`` is implemented as a genuinely different emission
hypothesis: one accretion rate sets both ring brightness and ring thickness,
and a jet footpoint breaks the ring's azimuthal symmetry.

Both of the KNOWN DEFICIENCY tests this file originally carried have since
been resolved and the assertions inverted:

* ``TestMassPriorRepresentability`` recorded 13.29% / 52.39% of each target's
  mass prior being representable on a 200 μas field.  Sizing the field to the
  source (50 μas / 128 px) makes both 100%.
* ``TestEveryAccretionParameterIsActive`` recorded ``position_angle`` moving
  the image by ~100% and the visibilities by 4e-4, because the baseline unit
  conversion put every EHT baseline on the uv origin.  With that fixed the
  visibility response is 0.392 and every declared parameter is active.
  The uv sampler itself is tested in tests/test_image_uv_sampling.py.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from whitesearch.dataio import EHTLoader
from whitesearch.inference import BilbyRunner
from whitesearch.likelihoods import VisibilityLikelihood
from whitesearch.models import get_model, model_for_context
from whitesearch.models.gr_eternal import (
    IMAGE_FOV_MUAS,
    IMAGE_N_PIXELS,
    GREternalWhiteHole,
)
from whitesearch.simulators.image_shadow import (
    ImageShadowSimulator,
    _shadow_radius_muas,
    ring_emission_from_params,
)
from whitesearch.utils.targets import (
    EHT_TARGETS,
    require_target,
    resolve_target_name,
    target_distance_mpc,
)

#: A context whose grid can actually represent M87*'s ring, so that activity
#: measurements are about the parameters and not about a sub-pixel ring.  NOT
#: the shipped grid: at 200 μas / 128 px the representability floor (25.77 μas)
#: sits above M87*'s ring (19.7 μas).  See TestMassPriorRepresentability.
CTX = {
    "target": "M87*",
    "fov_muas": 50.0,
    "n_pixels": 128,
    "freq_ghz": 230.0,
    "thermal_noise_jy": 0.05,
    "rng_seed": 7,
}

ACCRETION_TRUTH = {
    "M": 6.5e9,
    "a_star": 0.5,
    "i": 0.9,
    "position_angle": 0.7,
    "log10_mdot_edd": -2.5,
    "jet_power_frac": 0.3,
}

#: A gr_eternal vector whose ring has the SAME radius, thickness and peak
#: brightness as ACCRETION_TRUTH, so any difference between the two is the
#: azimuthal structure rather than a mismatch of scale.
def _matched_ring_truth() -> dict[str, float]:
    """gr_eternal parameters giving the SAME ring as ACCRETION_TRUTH.

    Matched on radius, thickness and total flux.  gr_eternal now samples the
    integrated flux, so the match is made there: build the accretion image,
    read its flux, and hand that to gr_eternal.  Any remaining difference
    between the two images is the jet footpoint, not a difference of scale.
    """
    accretion = ImageShadowSimulator().simulate(
        ACCRETION_TRUTH, CTX, rng=np.random.default_rng(0)
    )
    return {
        "M": 6.5e9,
        "a_star": 0.5,
        "i": 0.9,
        "position_angle": 0.7,
        "ring_width_frac": ring_emission_from_params(ACCRETION_TRUTH).ring_width_frac,
        "log10_total_flux_jy": float(np.log10(accretion.metadata["total_flux_jy"])),
    }


MATCHED_RING_TRUTH = _matched_ring_truth()


def _sim(params, ctx=CTX, seed=0):
    return ImageShadowSimulator().simulate(
        params, ctx, rng=np.random.default_rng(seed)
    )


# ── I-2: the target, and nothing guessed ──────────────────────────────────────


class TestTargetResolution:
    def test_known_spellings_resolve(self):
        assert resolve_target_name("M87") == "M87*"
        assert resolve_target_name("m87*") == "M87*"
        assert resolve_target_name("Sgr A*") == "SgrA*"
        assert resolve_target_name("sgra") == "SgrA*"

    def test_unknown_target_raises_rather_than_guessing(self):
        with pytest.raises(ValueError, match="Unknown image-channel target"):
            resolve_target_name("Cygnus A")

    def test_metadata_wins_over_context(self):
        """Same precedence as GWLikelihood._parse_data's per-event quantities."""
        assert require_target({"target": "SgrA*"}, {"target": "M87*"}).name == "SgrA*"

    def test_context_is_used_when_metadata_is_silent(self):
        assert require_target({"freq_ghz": 230.0}, {"target": "M87*"}).name == "M87*"

    def test_no_target_anywhere_raises(self):
        with pytest.raises(KeyError, match="which target it is analysing"):
            require_target({"freq_ghz": 230.0}, {"n_pixels": 128})


class TestDistancesAreKnownConstants:
    def test_published_values(self):
        assert EHT_TARGETS["M87*"].distance_mpc == pytest.approx(16.8)
        assert EHT_TARGETS["SgrA*"].distance_mpc == pytest.approx(0.008178)
        assert target_distance_mpc({"target": "M87"}) == pytest.approx(16.8)

    def test_every_target_cites_its_sources(self):
        for tgt in EHT_TARGETS.values():
            assert tgt.distance_source.strip()
            assert tgt.mass_source.strip()

    def test_distance_is_not_a_sampled_parameter_of_either_model(self):
        for name in ("gr_eternal", "bh_accretion"):
            model = get_model(name, target="M87*")
            assert "D_L" not in model.parameter_names
        assert "D_L" not in VisibilityLikelihood("gr_eternal").parameter_names
        assert "D_L" not in VisibilityLikelihood("bh_accretion").parameter_names

    def test_the_two_targets_really_do_differ_enough_to_matter(self):
        """Why there is no default: guessing costs 3.31 dex."""
        ratio = EHT_TARGETS["M87*"].distance_mpc / EHT_TARGETS["SgrA*"].distance_mpc
        assert np.log10(ratio) == pytest.approx(3.313, abs=0.01)


class TestFailClosedWithoutATarget:
    def test_simulator_raises(self):
        ctx = {k: v for k, v in CTX.items() if k != "target"}
        with pytest.raises(KeyError, match="which target it is analysing"):
            _sim(ACCRETION_TRUTH, ctx)

    def test_likelihood_raises_when_neither_data_nor_context_has_one(self):
        data = _sim(ACCRETION_TRUTH)
        obs = {k: v for k, v in data.metadata.items() if k != "target"}
        obs["visibilities"] = np.asarray(data.data)
        ctx = {k: v for k, v in CTX.items() if k != "target"}
        with pytest.raises(KeyError, match="which target it is analysing"):
            VisibilityLikelihood("bh_accretion").loglike(ACCRETION_TRUTH, obs, ctx)

    def test_likelihood_takes_it_from_the_observation(self):
        """The observation carries its own target; the context need not repeat it."""
        data = _sim(ACCRETION_TRUTH)
        ctx = {k: v for k, v in CTX.items() if k != "target"}
        assert data.metadata["target"] == "M87*"
        assert np.isfinite(
            VisibilityLikelihood("bh_accretion").loglike(ACCRETION_TRUTH, data, ctx)
        )

    @pytest.mark.parametrize("name", ["gr_eternal", "bh_accretion"])
    def test_model_raises_before_stating_a_prior(self, name):
        with pytest.raises(ValueError, match="which target it describes"):
            get_model(name).parameters()

    @pytest.mark.parametrize("name", ["gr_eternal", "bh_accretion"])
    def test_model_for_context_supplies_it(self, name):
        model = model_for_context(name, CTX)
        assert model.resolved_target.name == "M87*"

    @pytest.mark.parametrize("name", ["gr_eternal", "bh_accretion"])
    def test_model_for_context_raises_on_a_targetless_context(self, name):
        with pytest.raises(KeyError, match="which target it is analysing"):
            model_for_context(name, {"n_pixels": 128})

    def test_models_that_do_not_need_a_target_are_unaffected(self):
        assert model_for_context("null", {}).parameter_names == []
        assert model_for_context("bounce", {}).parameter_names


class TestEHTLoaderDeclaresItsTarget:
    def test_mock_path_sets_it_explicitly(self):
        loader = EHTLoader(cache_dir=tempfile.mkdtemp())
        for raw, canonical in (("M87", "M87*"), ("SgrA", "SgrA*")):
            record = loader.load_uvfits(raw, 2017, "LO")
            assert record["target"] == canonical

    def test_the_provenance_key_is_left_alone(self):
        """'source' keeps its existing (colliding) meaning; 'target' is new."""
        loader = EHTLoader(cache_dir=tempfile.mkdtemp())
        record = loader.load_uvfits("M87", 2017, "LO")
        assert record["source"] == "M87"
        assert record["target"] == "M87*"

    def test_an_unknown_source_raises_rather_than_mocking_something(self):
        loader = EHTLoader(cache_dir=tempfile.mkdtemp())
        with pytest.raises(ValueError, match="Unknown image-channel target"):
            loader.load_uvfits("Cygnus A", 2017, "LO")


# ── I-2: what the fixed distance bought, and what it did not ─────────────────


class TestMassPriorRepresentability:
    """Re-derivation of X.6 now that only the mass prior contributes."""

    def test_the_prior_side_of_the_problem_is_gone(self):
        """7.301 dex of ring-radius spread became 0.523 / 0.155 dex."""
        for name, expected in (("M87*", 0.5229), ("SgrA*", 0.1549)):
            tgt = EHT_TARGETS[name]
            span = np.log10(tgt.mass_prior_high_msun / tgt.mass_prior_low_msun)
            assert span == pytest.approx(expected, abs=0.001)
            assert span < 1.204, "the old M+D_L prior spanned 7.301 dex"

    def test_the_true_mass_is_inside_its_prior(self):
        for tgt in EHT_TARGETS.values():
            assert tgt.mass_prior_low_msun < tgt.mass_msun < tgt.mass_prior_high_msun

    def test_representable_fraction_at_the_shipped_grid(self):
        """RESOLVED: 100% for both targets, on the grid the project ships.

        This assertion used to read 13.29% (M87*) and 52.39% (Sgr A*) at a
        200 μas field.  The prior was never the problem -- it is unchanged, and
        is still set from the published mass measurements rather than from what
        any grid can draw.  The field was ~10x the size of the sources.
        """
        for name in EHT_TARGETS:
            assert GREternalWhiteHole.mass_prior_representable_fraction(
                name
            ) == pytest.approx(1.0), name

    def test_the_old_field_is_what_failed_not_the_prior(self):
        """The same prior, scored on the field this project used to ship."""
        m87 = GREternalWhiteHole.mass_prior_representable_fraction(
            "M87*", fov_muas=200.0
        )
        sgra = GREternalWhiteHole.mass_prior_representable_fraction(
            "SgrA*", fov_muas=200.0
        )
        assert m87 < 0.1 and sgra < 0.4
        # And cli.py's old second copy of the grid was worse still: at 64 px
        # over a 200 μas field the floor is 51.53 μas and nothing is
        # representable at either target.
        for name in EHT_TARGETS:
            assert GREternalWhiteHole.mass_prior_representable_fraction(
                name, fov_muas=200.0, n_pixels=64
            ) == 0.0, name

    def test_the_grid_requirement_is_derived_not_asserted(self):
        """The floor scales as 1 / n_pixels, so the requirement inverts."""
        assert GREternalWhiteHole.required_n_pixels_for_prior("M87*") == 97
        assert GREternalWhiteHole.required_n_pixels_for_prior("SgrA*") == 41
        assert IMAGE_N_PIXELS > 97, "the shipped grid clears the harder target"
        for name in EHT_TARGETS:
            n_req = GREternalWhiteHole.required_n_pixels_for_prior(name)
            assert GREternalWhiteHole.mass_prior_representable_fraction(
                name, n_pixels=n_req
            ) == pytest.approx(1.0)
            # One pixel coarser and the faintest mass in the prior drops out.
            assert GREternalWhiteHole.mass_prior_representable_fraction(
                name, n_pixels=n_req - 1
            ) < 1.0

    def test_the_shipped_field_is_inside_the_usable_window(self):
        """``required_fov_muas_for_prior`` brackets the shipped choice."""
        for name in EHT_TARGETS:
            fov_min, fov_max = GREternalWhiteHole.required_fov_muas_for_prior(name)
            assert fov_min < IMAGE_FOV_MUAS < fov_max, name
        assert IMAGE_FOV_MUAS == 50.0

    def test_the_window_inverts_the_forward_model_not_a_copy_of_it(self):
        lo_r, hi_r = GREternalWhiteHole.ring_radius_representable_range_muas()
        for name in EHT_TARGETS:
            tgt = EHT_TARGETS[name]
            m_lo, m_hi = GREternalWhiteHole.mass_representable_range_msun(
                name, a_star=0.0
            )
            assert _shadow_radius_muas(m_lo, 0.0, tgt.distance_mpc) == pytest.approx(lo_r)
            assert _shadow_radius_muas(m_hi, 0.0, tgt.distance_mpc) == pytest.approx(hi_r)

    def test_no_prior_draw_produces_an_empty_image_on_a_working_grid(self):
        """80.98% of the old prior imaged exactly zero; none of this one does."""
        model = get_model("gr_eternal", target="M87*")
        for k in range(50):
            theta = model.sample_prior(np.random.default_rng(900 + k))
            image = _sim(theta).metadata["image"]
            assert image.max() > 0.0


# ── I-1: bh_accretion as a distinct hypothesis ───────────────────────────────


class TestAccretionEmissionLaw:
    def test_one_rate_sets_both_brightness_and_thickness(self):
        """The substance of the hypothesis: they are not independent."""
        lo = ring_emission_from_params({**ACCRETION_TRUTH, "log10_mdot_edd": -5.0})
        hi = ring_emission_from_params({**ACCRETION_TRUTH, "log10_mdot_edd": 0.0})
        assert lo.normalisation == hi.normalisation == "peak_brightness"
        assert hi.amplitude > lo.amplitude
        # A radiatively inefficient flow is thick; approaching Eddington it thins.
        assert hi.ring_width_frac < lo.ring_width_frac
        assert lo.ring_width_frac == pytest.approx(0.397, abs=0.005)
        assert hi.ring_width_frac == pytest.approx(0.05, abs=0.001)

    def test_thickness_stays_inside_the_physical_band(self):
        for log10_mdot in np.linspace(-5.0, 0.0, 21):
            em = ring_emission_from_params(
                {**ACCRETION_TRUTH, "log10_mdot_edd": float(log10_mdot)}
            )
            assert 0.01 <= em.ring_width_frac <= 0.5

    def test_jet_fraction_sets_the_azimuthal_contrast(self):
        faint = ring_emission_from_params({**ACCRETION_TRUTH, "jet_power_frac": 0.01})
        full = ring_emission_from_params({**ACCRETION_TRUTH, "jet_power_frac": 1.0})
        assert faint.asym_amp == pytest.approx(0.03)
        assert full.asym_amp == pytest.approx(3.0)

    def test_gr_eternal_keeps_an_axisymmetric_ring(self):
        emission = ring_emission_from_params(MATCHED_RING_TRUTH)
        assert emission.asym_amp == 0.0
        assert emission.normalisation == "total_flux"

    def test_a_vector_carrying_both_hypotheses_raises(self):
        with pytest.raises(KeyError, match="Ambiguous"):
            ring_emission_from_params({**ACCRETION_TRUTH, "log10_total_flux_jy": 0.0})

    def test_a_vector_carrying_neither_raises(self):
        with pytest.raises(KeyError, match="neither"):
            ring_emission_from_params({"M": 6.5e9, "a_star": 0.5})

    def test_a_half_specified_vector_raises_instead_of_defaulting(self):
        partial = {k: v for k, v in ACCRETION_TRUTH.items() if k != "jet_power_frac"}
        with pytest.raises(KeyError, match="jet_power_frac"):
            ring_emission_from_params(partial)

    def test_the_simulator_records_which_hypothesis_it_used(self):
        assert _sim(ACCRETION_TRUTH).metadata["emission_model"] == "bh_accretion"
        assert _sim(MATCHED_RING_TRUTH).metadata["emission_model"] == "gr_eternal"


class TestSimulatorAndLikelihoodShareTheFormula:
    """Forward-model alignment: one copy of each expression, not two."""

    @pytest.mark.parametrize(
        "truth", [ACCRETION_TRUTH, MATCHED_RING_TRUTH], ids=["accretion", "ring"]
    )
    def test_predictive_stats_match_the_simulated_image(self, truth):
        data = _sim(truth)
        stats = VisibilityLikelihood().predictive_summary_stats(truth, CTX)
        assert stats["ring_width_muas"] == pytest.approx(data.metadata["w_ring_muas"])
        assert stats["brightness"] == pytest.approx(
            data.metadata["brightness_jy_per_muas2"]
        )
        assert stats["total_flux_jy"] == pytest.approx(data.metadata["total_flux_jy"])
        assert stats["theta_d_muas"] == pytest.approx(
            2.0 * data.metadata["r_ring_muas"]
        )
        assert stats["emission_model"] == data.metadata["emission_model"]

    def test_the_likelihood_uses_the_targets_distance_not_the_samples(self):
        """There is no D_L in theta to disagree with; the target supplies it."""
        data = _sim(ACCRETION_TRUTH)
        assert data.metadata["D_L_mpc"] == EHT_TARGETS["M87*"].distance_mpc

    def test_model_summary_stats_agree_with_the_simulator(self):
        model = get_model("bh_accretion", target="M87*")
        stats = model.summary_stats(ACCRETION_TRUTH)
        data = _sim(ACCRETION_TRUTH)
        # bh_accretion still samples a peak brightness, so its summary stat and
        # the simulator's brightness agree directly.
        assert stats["ring_brightness"] == pytest.approx(
            data.metadata["brightness_jy_per_muas2"]
        )
        assert stats["jet_contrast"] == pytest.approx(1.0 + data.metadata["asym_amp"])


class TestEveryAccretionParameterIsActive:
    """Per-parameter perturbation: no declared parameter may be inert.

    ``bh_accretion``'s parameter list was chosen for physical reasonableness
    rather than copied from ``gr_eternal``, so this is the check that the
    choice is honest.
    """

    PERTURBED = {
        "M": 6.5e9 * 1.05,
        "a_star": 0.9,
        "i": 1.3,
        "position_angle": 1.9,
        "log10_mdot_edd": -2.2,
        "jet_power_frac": 0.9,
    }

    @pytest.fixture(scope="class")
    def baseline(self):
        data = _sim(ACCRETION_TRUTH)
        like = VisibilityLikelihood("bh_accretion")
        return data, like, like.loglike(ACCRETION_TRUTH, data, CTX)

    def test_the_perturbation_covers_every_declared_parameter(self):
        declared = BilbyRunner.effective_parameter_names(
            get_model("bh_accretion", target="M87*"),
            VisibilityLikelihood("bh_accretion"),
        )
        assert sorted(declared) == sorted(self.PERTURBED)

    @pytest.mark.parametrize("name", sorted(PERTURBED))
    def test_the_image_responds(self, name):
        base = _sim(ACCRETION_TRUTH).metadata["image"]
        moved = _sim({**ACCRETION_TRUTH, name: self.PERTURBED[name]}).metadata["image"]
        rel = np.abs(moved - base).max() / base.max()
        assert rel > 1e-3, f"{name} leaves the image unchanged"

    @pytest.mark.parametrize("name", sorted(PERTURBED))
    def test_the_visibilities_and_the_likelihood_respond(self, name, baseline):
        """All six, position_angle included since the uv sampler was fixed."""
        data, like, ll0 = baseline
        theta = {**ACCRETION_TRUTH, name: self.PERTURBED[name]}
        base_vis = np.abs(_sim(ACCRETION_TRUTH).data)
        moved_vis = np.abs(_sim(theta).data)
        rel = np.abs(moved_vis - base_vis).max() / base_vis.max()
        assert rel > 1e-3, f"{name} does not move the visibilities"
        assert abs(like.loglike(theta, data, CTX) - ll0) > 1.0

    def test_position_angle_is_no_longer_inert(self, baseline):
        """RESOLVED.  This test used to assert the opposite and explain why.

        ``_compute_visibilities`` converted baselines from Gλ to rad^-1 as
        ``uv * 1e9 * wavelength_m``, which is the conversion to physical
        baseline length in metres; a baseline in Gλ is already an angular
        frequency.  Every EHT baseline landed within 0.4% of one FFT cell of
        the uv origin, where the visibility is just the total flux -- and
        rotating an image cannot change its total flux.

        Measured before and after, same truth, same perturbation
        (position_angle 0.7 -> 1.9 rad):

            max|dV| / max|V|        9.6e-4  ->  0.570   (594x)
            max|d phase|            8.7e-4  ->  2.80 rad  (3200x)

        See docs/XRAY_IMAGE_PREFLIGHT_AUDIT.md X.13.1.
        """
        data, like, ll0 = baseline
        theta = {**ACCRETION_TRUTH, "position_angle": self.PERTURBED["position_angle"]}

        base_vis = _sim(ACCRETION_TRUTH).metadata["vis_signal"]
        moved_vis = _sim(theta).metadata["vis_signal"]
        rel = np.abs(moved_vis - base_vis).max() / np.abs(base_vis).max()
        assert rel > 0.3, "rotation must move the visibilities, not just the image"

        d_phase = np.abs(np.angle(moved_vis / base_vis)).max()
        assert d_phase > 1.0, "and it must move the phases by order a radian"

        assert abs(like.loglike(theta, data, CTX) - ll0) > 10.0

    def test_the_visibilities_are_not_all_the_total_flux(self, baseline):
        """The signature of the old bug: |V| identical on every baseline."""
        data, _like, _ll0 = baseline
        signal = np.abs(data.metadata["vis_signal"])
        flux = data.metadata["image"].sum() * (
            2.0 * CTX["fov_muas"] / CTX["n_pixels"]
        ) ** 2
        assert signal.max() / signal.min() > 2.0
        assert signal.min() < 0.9 * flux

class TestAccretionIsNotGREternalRelabelled:
    """I-1's closing check, with the magnitude reported."""

    def test_the_images_differ_at_matched_ring_geometry(self):
        """Same radius, thickness AND total flux -- only the jet differs.

        This became a stricter comparison when gr_eternal started sampling the
        integrated flux: the two images now carry identical flux, so the
        difference is purely how that flux is distributed in azimuth.  Matching
        on peak brightness instead (the old parameterisation) let the accretion
        image carry ~2x the flux, which inflated the difference to 89.8%.
        """
        ring_data, accretion_data = _sim(MATCHED_RING_TRUTH), _sim(ACCRETION_TRUTH)
        assert ring_data.metadata["w_ring_muas"] == pytest.approx(
            accretion_data.metadata["w_ring_muas"]
        )
        assert ring_data.metadata["total_flux_jy"] == pytest.approx(
            accretion_data.metadata["total_flux_jy"]
        )
        rel = (
            np.abs(accretion_data.metadata["image"] - ring_data.metadata["image"]).max()
            / ring_data.metadata["image"].max()
        )
        assert rel == pytest.approx(0.500, abs=0.02), "measured 50.0%"

    def test_only_the_accretion_image_is_azimuthally_asymmetric(self):
        """Compare each image with itself rotated by 180 degrees."""
        for truth, asymmetric in ((MATCHED_RING_TRUTH, False), (ACCRETION_TRUTH, True)):
            image = _sim(truth).metadata["image"]
            flipped = image[::-1, ::-1]
            rel = np.abs(image - flipped).max() / image.max()
            assert bool(rel > 0.3) is asymmetric, truth.get("log10_mdot_edd")

    def test_the_likelihoods_disagree_sharply_on_the_same_data(self):
        """lnL of each hypothesis at its own truth, against accretion data."""
        data = _sim(ACCRETION_TRUTH)
        ll_accretion = VisibilityLikelihood("bh_accretion").loglike(
            ACCRETION_TRUTH, data, CTX
        )
        ll_ring = VisibilityLikelihood("gr_eternal").loglike(
            MATCHED_RING_TRUTH, data, CTX
        )
        # Flux-matched, so this separation is the azimuthal structure alone and
        # not a brightness mismatch; it was > 1e5 when the two carried
        # different fluxes.  ln BF = 17.8 is still far past this project's
        # ln BF > 5 publication gate.
        assert ll_accretion > ll_ring
        assert ll_accretion - ll_ring > 10.0

    def test_the_two_models_do_not_share_a_parameter_vector(self):
        ring = VisibilityLikelihood("gr_eternal").parameter_names
        accretion = VisibilityLikelihood("bh_accretion").parameter_names
        assert ring != accretion
        assert set(ring) & set(accretion) == {"M", "a_star", "i", "position_angle"}
        assert set(ring) - set(accretion) == {"ring_width_frac", "log10_total_flux_jy"}
        assert set(accretion) - set(ring) == {"log10_mdot_edd", "jet_power_frac"}
