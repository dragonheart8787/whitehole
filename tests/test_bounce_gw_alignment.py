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
from whitesearch.simulators.grav_wave import antenna_response, tukey
from whitesearch.utils.constants import C, G, M_SUN, MPC_M
from whitesearch.utils.math_utils import kerr_qnm_frequency

CONTEXT = {
    "sample_rate": 4096,
    "duration": 4.0,
    "t_merger": 1.0,
    "low_freq_cutoff": 20.0,
    "rng_seed": 0,
}
GW_BOUNCE_PARAMS = ["M", "a_star", "eps_f", "eps_Q", "log10_A_bounce", "log10_dt_bounce_s", "D_L", "i"]


class TestSampledDimensionComesFromTheLikelihood:
    """B.1 — the sampled space is the model/likelihood intersection."""

    def test_bounce_samples_only_what_the_gw_likelihood_reads(self):
        model = get_model("bounce")
        sampled = BilbyRunner.effective_parameter_names(model, GWLikelihood("bounce"))
        assert sorted(sampled) == sorted(GW_BOUNCE_PARAMS)
        assert len(sampled) == len(GWLikelihood("bounce").parameter_names) == 8
        # the model itself keeps its full declaration: other channels may use it
        assert len(model.parameter_names) == 13

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
        assert len(full) == 13

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
        assert sorted(result.posterior.columns) == sorted(GW_BOUNCE_PARAMS)
        assert sorted(result.metadata["sampled_parameters"]) == sorted(GW_BOUNCE_PARAMS)


class TestBurstIsBackInTheGWSamplingSpace:
    """B3-2 — the burst delay is re-parameterised and inferred again.

    This class replaces TestBurstIsOutOfTheGWSamplingSpace, which locked in the
    B3-3 state where log10_A_bounce and log10_tau_bounce_yr were removed from
    the GW sampling space because no prior draw could put the burst inside the
    segment.
    """

    def test_burst_parameters_are_sampled_again(self):
        names = GWLikelihood("bounce").parameter_names
        assert "log10_A_bounce" in names
        assert "log10_dt_bounce_s" in names

    def test_the_cosmological_lifetime_is_still_not_sampled(self):
        """log10_tau_bounce_yr is a different quantity, kept on the model."""
        assert "log10_tau_bounce_yr" not in GWLikelihood("bounce").parameter_names
        assert "log10_tau_bounce_yr" in get_model("bounce").parameter_names

    def test_prior_draws_now_land_inside_the_segment(self):
        """The reverse of the B.3 measurement: prior mass ~1.0, not 0.0000."""
        model = get_model("bounce")
        rng = np.random.default_rng(5)
        duration = CONTEXT["duration"]
        t_merger = CONTEXT["t_merger"]
        last_sample_time = duration - 1.0 / CONTEXT["sample_rate"]
        inside = 0
        n = 5000
        for _ in range(n):
            theta = model.sample_prior(rng)
            if t_merger + 10.0 ** theta["log10_dt_bounce_s"] < last_sample_time:
                inside += 1
        assert inside == n, f"only {inside}/{n} prior draws put the burst in segment"

    def test_declared_prior_sits_inside_the_derived_bounds(self):
        specs = {p.name: p for p in get_model("bounce").parameters()}
        lo, hi = BlackToWhiteBounce._dt_bounce_prior_bounds()
        assert lo <= specs["log10_dt_bounce_s"].prior_kwargs["low"]
        assert specs["log10_dt_bounce_s"].prior_kwargs["high"] <= hi

    def test_segment_constants_match_the_shipped_run_config(self):
        import yaml

        from whitesearch.models.bounce import (
            SEGMENT_DURATION_S, SEGMENT_SAMPLE_RATE_HZ, SEGMENT_T_MERGER_S,
        )

        inst = yaml.safe_load(
            open("configs/runs/gw_run.yaml", encoding="utf-8")
        )["instrument"]
        assert SEGMENT_SAMPLE_RATE_HZ == inst["sample_rate"]
        assert SEGMENT_DURATION_S == inst["duration"]
        assert SEGMENT_T_MERGER_S == inst["t_merger"]

    def test_simulator_and_template_place_the_burst_at_the_same_time(self):
        """Forward-model consistency on the burst timing."""
        model = get_model("bounce")
        like = GWLikelihood("bounce")
        sr = CONTEXT["sample_rate"]
        n = int(CONTEXT["duration"] * sr)
        times = np.arange(n) / sr
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        rng = np.random.default_rng(17)
        for _ in range(5):
            theta = model.sample_prior(rng)
            theta["log10_A_bounce"] = -19.0  # loud enough to locate
            expected_t = CONTEXT["t_merger"] + 10.0 ** theta["log10_dt_bounce_s"]

            signal = get_simulator("gw").signal_only(theta, CONTEXT)
            template = like._build_template(
                theta, times, CONTEXT["t_merger"], freqs, sr / 2.0, BAND_LOW_HZ
            )
            assert template is not None
            # The simulator tapers the injected signal with a Tukey window
            # before adding noise; the template is untapered (loglike() tapers
            # strain and template together).  Applying the same window makes
            # the two waveforms directly comparable, which is the real
            # forward-model consistency claim: identical amplitudes AND
            # identical burst timing, sample for sample.
            # signal_only() recovers the signal as (signal + noise) - noise, so
            # samples far below the noise scale cancel to exactly zero; compare
            # with an absolute floor well under the burst amplitude rather than
            # demanding bit equality in that numerical dust.
            window = tukey(len(times), alpha=0.1)
            atol = 1e-6 * 10.0 ** theta["log10_A_bounce"]
            assert np.allclose(signal.data, template * window, rtol=1e-6, atol=atol)
            # and the onset really is where the parameterisation says
            burst_index = int(round(expected_t * sr))
            assert np.abs(template[burst_index]) == pytest.approx(
                10.0 ** theta["log10_A_bounce"], rel=1e-6
            )


class TestBurstFrequencyIsBandChecked:
    """B.5 — 0.8*f_rd must be rejected when it leaves the band."""

    def test_burst_below_the_cutoff_is_rejected(self):
        like = GWLikelihood("bounce")
        sr = CONTEXT["sample_rate"]
        n = int(CONTEXT["duration"] * sr)
        times = np.arange(n) / sr
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        # f_rd = 22 Hz is in band, but 0.8*22 = 17.6 Hz is not
        theta = {"M": 1.0, "a_star": 0.0, "eps_f": 0.0, "eps_Q": 0.0,
                 "log10_A_bounce": -21.0, "log10_dt_bounce_s": -1.0,
                 "D_L": 400.0, "i": 0.5}
        k = kerr_qnm_frequency(1.0, 0.0)[0]
        theta["M"] = k / 22.0
        assert BAND_LOW_HZ <= kerr_qnm_frequency(theta["M"], 0.0)[0] <= BAND_HIGH_HZ
        assert like._build_template(
            theta, times, CONTEXT["t_merger"], freqs, sr / 2.0, BAND_LOW_HZ
        ) is None

    def test_burst_inside_the_band_is_accepted(self):
        like = GWLikelihood("bounce")
        sr = CONTEXT["sample_rate"]
        n = int(CONTEXT["duration"] * sr)
        times = np.arange(n) / sr
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        k = kerr_qnm_frequency(1.0, 0.0)[0]
        theta = {"M": k / 200.0, "a_star": 0.0, "eps_f": 0.0, "eps_Q": 0.0,
                 "log10_A_bounce": -21.0, "log10_dt_bounce_s": -1.0,
                 "D_L": 400.0, "i": 0.5}
        assert like._build_template(
            theta, times, CONTEXT["t_merger"], freqs, sr / 2.0, BAND_LOW_HZ
        ) is not None

    def test_bh_ringdown_is_not_burst_band_checked(self):
        """The extra check is bounce-only; bh_ringdown has no burst."""
        like = GWLikelihood("bh_ringdown")
        sr = CONTEXT["sample_rate"]
        n = int(CONTEXT["duration"] * sr)
        times = np.arange(n) / sr
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        k = kerr_qnm_frequency(1.0, 0.0)[0]
        theta = {"M": k / 22.0, "a_star": 0.0, "log10_A": -21.0}
        assert like._build_template(
            theta, times, CONTEXT["t_merger"], freqs, sr / 2.0, BAND_LOW_HZ
        ) is not None


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
