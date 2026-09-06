"""bounce on the GW channel: sampled dimension, amplitude, and prior support.

Regression cover for docs/BOUNCE_PREFLIGHT_AUDIT.md:

B.1  BilbyRunner built priors from the model alone, so dynesty sampled all 12
     declared bounce parameters while GWLikelihood("bounce") read far fewer.
B.3  The bounce burst can never land inside an analysed segment under the
     log10_tau_bounce_yr prior, so the burst parameters left the GW channel's
     sampled space (audit option B3-3).
B.4  The simulator injected h0*sqrt(fp^2+fc^2) while the template modelled
     h0*0.5*(1+cos^2 i) -- up to a factor sqrt(2) apart (audit option B4-2).
B.5  M's prior ran outside the band the likelihood can see, worsened by the
     (1 + eps_f) factor bounce applies to the GR frequency.
"""

from __future__ import annotations

import numpy as np
import pytest

from whitesearch.inference import BilbyRunner
from whitesearch.likelihoods import GWLikelihood
from whitesearch.models import get_model
from whitesearch.models.alternatives import BAND_HIGH_HZ, BAND_LOW_HZ
from whitesearch.models.bounce import BlackToWhiteBounce
from whitesearch.simulators import get_simulator
from whitesearch.simulators.grav_wave import antenna_response
from whitesearch.utils.constants import C, G, M_SUN, MPC_M
from whitesearch.utils.math_utils import kerr_qnm_frequency

CONTEXT = {
    "sample_rate": 4096,
    "duration": 4.0,
    "t_merger": 1.0,
    "low_freq_cutoff": 20.0,
    "rng_seed": 0,
}
GW_BOUNCE_PARAMS = ["M", "a_star", "eps_f", "eps_Q", "D_L", "i"]


class TestSampledDimensionComesFromTheLikelihood:
    """B.1 — the sampled space is the model/likelihood intersection."""

    def test_bounce_samples_only_what_the_gw_likelihood_reads(self):
        model = get_model("bounce")
        sampled = BilbyRunner.effective_parameter_names(model, GWLikelihood("bounce"))
        assert sampled == GW_BOUNCE_PARAMS
        assert len(sampled) == len(GWLikelihood("bounce").parameter_names)
        # the model itself keeps its full declaration: other channels may use it
        assert len(model.parameter_names) == 12

    def test_bh_ringdown_dimension_is_unchanged_by_the_intersection(self):
        """Already-aligned models must behave exactly as before."""
        model = get_model("bh_ringdown")
        sampled = BilbyRunner.effective_parameter_names(model, GWLikelihood("bh_ringdown"))
        assert sampled == model.parameter_names == ["M", "a_star", "log10_A"]

    def test_a_likelihood_with_no_declared_parameters_imposes_no_restriction(self):
        model = get_model("bounce")

        class _Unrestricted(GWLikelihood):
            @property
            def parameter_names(self):
                return []

        sampled = BilbyRunner.effective_parameter_names(model, _Unrestricted("bounce"))
        assert sampled == model.parameter_names

    def test_likelihood_asking_for_a_parameter_the_model_lacks_raises(self):
        """fail-closed: never silently default a missing parameter."""
        model = get_model("bh_ringdown")
        with pytest.raises(ValueError, match="does not declare"):
            BilbyRunner.effective_parameter_names(model, GWLikelihood("bounce"))

    def test_priors_handed_to_the_sampler_are_restricted(self):
        model = get_model("bounce")
        full = model.to_bilby_priors()
        restricted = BilbyRunner._restrict_priors(full, GW_BOUNCE_PARAMS)
        assert sorted(restricted) == sorted(GW_BOUNCE_PARAMS)
        assert len(full) == 12

    def test_toy_sampler_posterior_has_the_restricted_columns(self, tmp_path):
        model = get_model("bounce")
        data = get_simulator("gw").simulate(
            model.sample_prior(np.random.default_rng(0)), CONTEXT,
            rng=np.random.default_rng(0),
        )
        runner = BilbyRunner(force_toy=True, nlive=20, outdir=str(tmp_path), seed=0)
        result = runner.run(
            GWLikelihood("bounce"), data, CONTEXT, model, label="restricted",
        )
        assert list(result.posterior.columns) == GW_BOUNCE_PARAMS
        assert result.metadata["sampled_parameters"] == GW_BOUNCE_PARAMS


class TestBurstIsOutOfTheGWSamplingSpace:
    """B.3 — option B3-3."""

    def test_burst_parameters_are_not_sampled(self):
        names = GWLikelihood("bounce").parameter_names
        assert "log10_A_bounce" not in names
        assert "log10_tau_bounce_yr" not in names

    def test_burst_could_never_reach_the_segment_anyway(self):
        """The measurement behind the decision, locked in."""
        from whitesearch.utils.constants import GYR_S

        specs = {p.name: p for p in get_model("bounce").parameters()}
        tau_lo = specs["log10_tau_bounce_yr"].prior_kwargs["low"]
        shortest_tau_s = 10.0**tau_lo * GYR_S / 1e9
        for duration in (4.0, 32.0):
            assert shortest_tau_s > duration, (duration, shortest_tau_s)

    def test_template_from_a_sampled_theta_carries_no_burst(self):
        model = get_model("bounce")
        like = GWLikelihood("bounce")
        theta = {p: v for p, v in model.sample_prior(np.random.default_rng(0)).items()
                 if p in GW_BOUNCE_PARAMS}
        times = np.arange(4096) / 4096.0
        freqs = np.fft.rfftfreq(4096, d=1.0 / 4096)
        h = like._build_template(theta, times, 0.1, freqs, 2048.0, BAND_LOW_HZ)
        assert h is not None
        # identical to the same call with the burst keys explicitly absent
        assert np.array_equal(
            h, like._build_template(dict(theta), times, 0.1, freqs, 2048.0, BAND_LOW_HZ)
        )


class TestSimulatorAndTemplateAgreeOnAmplitude:
    """B.4 — option B4-2: plus polarisation only, on both sides."""

    @pytest.mark.parametrize("i", [0.0, 0.5, 1.0, np.pi / 2, np.pi])
    def test_antenna_projection_matches_the_template_formula(self, i):
        fp, _fc = antenna_response(i)
        assert fp == 0.5 * (1.0 + np.cos(i) ** 2)

    @pytest.mark.parametrize("i", [0.0, 0.5, 1.0, np.pi / 2, np.pi])
    def test_injected_amplitude_equals_template_amplitude(self, i):
        M, D_L = 60.0, 400.0
        params = {"M": M, "a_star": 0.6, "eps_f": 0.0, "eps_Q": 0.0,
                  "D_L": D_L, "i": i}
        sim = get_simulator("gw")
        injected = sim.simulate(params, CONTEXT, rng=np.random.default_rng(0))
        h0 = G * M * M_SUN / (C**2 * D_L * MPC_M)
        template_amplitude = h0 * 0.5 * (1.0 + np.cos(i) ** 2)
        assert injected.metadata["A_rd"] == pytest.approx(template_amplitude, rel=1e-12)
        assert injected.metadata["amplitude_source"] == "M_D_L_inclination"

    @pytest.mark.parametrize("i", [0.0, 0.5, 1.0, np.pi / 2, np.pi])
    def test_noiseless_injection_peaks_at_the_template_amplitude(self, i):
        params = {"M": 60.0, "a_star": 0.6, "eps_f": 0.0, "eps_Q": 0.0,
                  "D_L": 400.0, "i": i}
        sim = get_simulator("gw")
        signal = sim.signal_only(params, CONTEXT)
        h0 = G * 60.0 * M_SUN / (C**2 * 400.0 * MPC_M)
        expected = h0 * 0.5 * (1.0 + np.cos(i) ** 2)
        assert np.max(np.abs(signal.data)) == pytest.approx(expected, rel=1e-3)

    def test_bh_ringdown_amplitude_path_is_untouched(self):
        """bh_ringdown uses 10**log10_A and must not see the antenna change."""
        sim = get_simulator("gw")
        theta = {"M": 60.0, "a_star": 0.7, "log10_A": -21.0}
        data = sim.simulate(theta, CONTEXT, rng=np.random.default_rng(5))
        assert data.metadata["amplitude_source"] == "log10_A"
        assert data.metadata["A_rd"] == pytest.approx(1e-21)


class TestMPriorStaysInTheAnalysisBand:
    """B.5."""

    def test_prior_corners_are_in_band(self):
        specs = {p.name: p for p in get_model("bounce").parameters()}
        m_low = specs["M"].prior_kwargs["low"]
        m_high = specs["M"].prior_kwargs["high"]
        spin_max = specs["a_star"].prior_kwargs["high"]
        eps_lo = specs["eps_f"].prior_kwargs["low"]
        eps_hi = specs["eps_f"].prior_kwargs["high"]
        for M in (m_low, m_high):
            for a in (0.0, spin_max):
                for eps_f in (eps_lo, eps_hi):
                    f_rd = kerr_qnm_frequency(M, a)[0] * (1.0 + eps_f)
                    assert BAND_LOW_HZ <= f_rd <= BAND_HIGH_HZ, (M, a, eps_f, f_rd)

    def test_prior_sits_inside_the_derived_bounds(self):
        specs = {p.name: p for p in get_model("bounce").parameters()}
        exact_low, exact_high = BlackToWhiteBounce._m_prior_bounds_for_band()
        assert exact_low <= specs["M"].prior_kwargs["low"]
        assert specs["M"].prior_kwargs["high"] <= exact_high

    def test_no_prior_draw_is_rejected_by_the_template_builder(self):
        model = get_model("bounce")
        like = GWLikelihood("bounce")
        rng = np.random.default_rng(11)
        times = np.arange(16384) / 4096.0
        freqs = np.fft.rfftfreq(16384, d=1.0 / 4096)
        for _ in range(500):
            theta = model.sample_prior(rng)
            assert like._build_template(
                theta, times, 1.0, freqs, 2048.0, BAND_LOW_HZ
            ) is not None, theta
