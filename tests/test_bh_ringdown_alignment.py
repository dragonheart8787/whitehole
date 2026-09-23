"""bh_ringdown: model spec, likelihood and mock simulator must be one model.

Regression cover for findings (B) and (D) of
docs/BH_RINGDOWN_SBC_COVERAGE_REPORT.md:

(B) the model declared D_L and i, which GWLikelihood._build_template() never
    read (dlnL was bit-for-bit 0 in them), while the mock simulator derived the
    ringdown amplitude from M/D_L/i and never read log10_A (changing log10_A
    left the simulated data byte-identical).  SBC assumes data are drawn from
    the same p(theta)p(d|theta) the likelihood evaluates; they were not.

(D) M's prior spanned f_QNM in [11.9, 6505] Hz while the likelihood can only
    see [low_freq_cutoff, min(high_freq_cutoff, 0.95*nyquist)], so 17% of prior
    draws were unrepresentable and M's coverage was biased.
"""

from __future__ import annotations

import numpy as np
import pytest

from whitesearch.likelihoods import GWLikelihood
from whitesearch.models import get_model
from whitesearch.models.alternatives import BAND_HIGH_HZ, BAND_LOW_HZ, StandardBHRingdown
from whitesearch.simulators import get_simulator
from whitesearch.utils.math_utils import kerr_qnm_frequency

CONTEXT = {
    "sample_rate": 4096,
    "duration": 4.0,
    "t_merger": 1.0,
    "low_freq_cutoff": 20.0,
    "rng_seed": 0,
}


class TestParameterVectorAlignment:
    def test_model_declares_exactly_what_the_template_uses(self):
        model = get_model("bh_ringdown")
        assert model.parameter_names == ["M", "a_star", "log10_A"]
        assert GWLikelihood("bh_ringdown").parameter_names == model.parameter_names

    def test_no_dead_parameters_left(self):
        """D_L and i are gone: the template never read them."""
        model = get_model("bh_ringdown")
        assert "D_L" not in model.parameter_names
        assert "i" not in model.parameter_names

    def test_summary_stats_needs_only_the_declared_parameters(self):
        model = get_model("bh_ringdown")
        theta = model.sample_prior(np.random.default_rng(0))
        stats = model.summary_stats(theta)
        assert stats["delta_f_hz"] == 0.0
        assert stats["delta_Q"] == 0.0
        assert stats["A"] == pytest.approx(10.0 ** theta["log10_A"])


class TestSimulatorLikelihoodAmplitudeAgreement:
    def test_simulator_uses_log10_A_for_the_injected_amplitude(self):
        """The forward model that makes the data must be the one the likelihood fits."""
        sim = get_simulator("gw")
        theta = {"M": 60.0, "a_star": 0.7, "log10_A": -21.0}
        data = sim.simulate(theta, CONTEXT, rng=np.random.default_rng(5))
        assert data.metadata["amplitude_source"] == "log10_A"
        assert data.metadata["A_rd"] == pytest.approx(1e-21)

        signal = data.data - data.noise_realisation
        # The ringdown peaks at t_merger, well inside the Tukey unit-gain
        # region, so the peak of the injected signal is the requested amplitude.
        assert np.max(np.abs(signal)) == pytest.approx(1e-21, rel=1e-3)

    def test_changing_log10_A_changes_the_simulated_data(self):
        sim = get_simulator("gw")
        base = {"M": 60.0, "a_star": 0.7}
        d1 = sim.simulate({**base, "log10_A": -24.0}, CONTEXT, rng=np.random.default_rng(5))
        d2 = sim.simulate({**base, "log10_A": -18.0}, CONTEXT, rng=np.random.default_rng(5))
        assert np.max(np.abs(d1.data - d2.data)) > 0.0

    def test_bounce_keeps_the_physical_distance_amplitude_path(self):
        """The M/D_L/inclination path must be untouched for models that use it."""
        sim = get_simulator("gw")
        bounce = get_model("bounce")
        theta = bounce.sample_prior(np.random.default_rng(3))
        assert "log10_A" not in theta
        data = sim.simulate(theta, CONTEXT, rng=np.random.default_rng(3))
        assert data.metadata["amplitude_source"] == "M_D_L_inclination"

        far = sim.simulate({**theta, "D_L": theta["D_L"] * 10.0}, CONTEXT,
                           rng=np.random.default_rng(3))
        assert far.metadata["h0"] == pytest.approx(data.metadata["h0"] / 10.0, rel=1e-9)


class TestMPriorStaysInTheAnalysisBand:
    def test_prior_edges_are_in_band_for_every_spin(self):
        model = get_model("bh_ringdown")
        specs = {p.name: p for p in model.parameters()}
        m_low = specs["M"].prior_kwargs["low"]
        m_high = specs["M"].prior_kwargs["high"]
        spin_max = specs["a_star"].prior_kwargs["high"]

        for M in (m_low, m_high):
            for a in (0.0, spin_max):
                f_rd, _ = kerr_qnm_frequency(M, a)
                assert BAND_LOW_HZ <= f_rd <= BAND_HIGH_HZ, (M, a, f_rd)

    def test_prior_sits_inside_the_analytically_derived_bounds(self):
        specs = {p.name: p for p in get_model("bh_ringdown").parameters()}
        exact_low, exact_high = StandardBHRingdown._m_prior_bounds_for_band()
        assert exact_low <= specs["M"].prior_kwargs["low"]
        assert specs["M"].prior_kwargs["high"] <= exact_high

    def test_no_prior_draw_is_rejected_by_the_template_builder(self):
        """Every prior draw must produce a template the likelihood can evaluate."""
        model = get_model("bh_ringdown")
        like = GWLikelihood("bh_ringdown")
        rng = np.random.default_rng(7)
        times = np.arange(int(CONTEXT["duration"] * CONTEXT["sample_rate"]))
        times = times / CONTEXT["sample_rate"]
        freqs = np.fft.rfftfreq(len(times), d=1.0 / CONTEXT["sample_rate"])
        nyquist = CONTEXT["sample_rate"] / 2.0

        for _ in range(500):
            theta = model.sample_prior(rng)
            template = like._build_template(
                theta, times, CONTEXT["t_merger"], freqs, nyquist, BAND_LOW_HZ
            )
            assert template is not None, theta

    def test_band_constants_match_the_shipped_instrument_config(self):
        import yaml

        cfg = yaml.safe_load(open("configs/instruments/ligo.yaml", encoding="utf-8"))
        pre = cfg["preprocessing"]
        assert BAND_LOW_HZ == pre["low_freq_cutoff"]
        assert BAND_HIGH_HZ == pre["high_freq_cutoff"]
