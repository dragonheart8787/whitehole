"""Integration tests for the inference pipeline.

Uses the toy sampler (bilby not required) for fast CI execution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from whitesearch.inference import BilbyRunner, InferenceResult, compute_bayes_factor
from whitesearch.likelihoods import GWLikelihood, RadioBurstLikelihood
from whitesearch.models import BlackToWhiteBounce, StandardBHRingdown, PBHTunnelingWhiteHole
from whitesearch.simulators import GravitationalWaveSimulator, EMBurstSimulator


@pytest.fixture
def gw_sim_data(bounce_params, gw_context):
    sim = GravitationalWaveSimulator()
    return sim.simulate(bounce_params, gw_context, rng=np.random.default_rng(42))


@pytest.fixture
def radio_sim_data(pbh_params, radio_context):
    sim = EMBurstSimulator()
    return sim.simulate(pbh_params, radio_context, rng=np.random.default_rng(42))


@pytest.fixture
def toy_runner(tmp_path):
    return BilbyRunner(
        force_toy=True,
        nlive=50,
        outdir=str(tmp_path / "bilby"),
        seed=42,
    )


class TestInferenceResult:
    def test_credible_intervals_keys(self):
        posterior = pd.DataFrame({"M": np.random.default_rng(0).standard_normal(100) + 60.0})
        result = InferenceResult(
            log_evidence=-10.0,
            log_evidence_err=0.1,
            posterior=posterior,
            log_likelihood_samples=np.full(100, -10.0),
        )
        ci = result.credible_intervals()
        assert "M" in ci
        lo, hi = ci["M"]
        assert lo < hi

    def test_median_params(self):
        posterior = pd.DataFrame({"M": [60.0, 61.0, 59.0]})
        result = InferenceResult(
            log_evidence=-5.0,
            log_evidence_err=0.05,
            posterior=posterior,
            log_likelihood_samples=np.full(3, -5.0),
        )
        med = result.median_params()
        assert "M" in med
        assert abs(med["M"] - 60.0) < 2.0


class TestBilbyRunner:
    def test_toy_sampler_returns_result(
        self, bounce_params, gw_context, gw_sim_data, tmp_path
    ):
        runner = BilbyRunner(force_toy=True, nlive=50, outdir=str(tmp_path), seed=0)
        model = BlackToWhiteBounce()
        ll = GWLikelihood()

        result = runner.run(ll, gw_sim_data, gw_context, model, label="test_gw")
        assert isinstance(result, InferenceResult)
        assert np.isfinite(result.log_evidence)
        assert len(result.posterior) > 0

    def test_posterior_has_correct_columns(
        self, bounce_params, gw_context, gw_sim_data, tmp_path
    ):
        runner = BilbyRunner(force_toy=True, nlive=50, outdir=str(tmp_path), seed=0)
        model = BlackToWhiteBounce()
        ll = GWLikelihood()

        result = runner.run(ll, gw_sim_data, gw_context, model, label="test_cols")
        for param in model.parameter_names:
            assert param in result.posterior.columns

    def test_compare_models_returns_dataframe(
        self, bounce_params, gw_context, gw_sim_data, tmp_path
    ):
        runner = BilbyRunner(force_toy=True, nlive=50, outdir=str(tmp_path), seed=0)
        bounce = BlackToWhiteBounce()
        std_bh = StandardBHRingdown()
        ll_bounce = GWLikelihood("bounce")
        ll_bh = GWLikelihood("bh_ringdown")

        r_bounce = runner.run(ll_bounce, gw_sim_data, gw_context, bounce, label="bounce")
        r_bh = runner.run(ll_bh, gw_sim_data, gw_context, std_bh, label="bh")

        results = {"bounce": r_bounce, "bh_ringdown": r_bh}
        df = runner.compare_models(results, reference="bh_ringdown")
        assert "ln_BF" in df.columns
        assert "interpretation" in df.columns or "BF_interpretation" in df.columns


class TestBayesFactor:
    def test_compute_returns_dict(
        self, bounce_params, gw_context, gw_sim_data, tmp_path
    ):
        runner = BilbyRunner(force_toy=True, nlive=50, outdir=str(tmp_path), seed=0)
        model = BlackToWhiteBounce()
        null_model = StandardBHRingdown()
        ll = GWLikelihood()

        r_wh = runner.run(ll, gw_sim_data, gw_context, model, label="wh")
        r_null = runner.run(ll, gw_sim_data, gw_context, null_model, label="null")

        result = compute_bayes_factor(
            {"bounce": r_wh, "null": r_null},
            wh_model="bounce",
            null_model="null",
        )
        assert "ln_BF_vs_null" in result
        assert "gate_internal_passed" in result
        assert "gate_publication_passed" in result
        assert isinstance(result["gate_internal_passed"], bool)

    def test_interpretation_strings(self):
        from whitesearch.inference.evidence import interpret_ln_bf
        assert "not worth" in interpret_ln_bf(0.5)
        assert "positive" in interpret_ln_bf(1.5)
        assert "strong" in interpret_ln_bf(3.5)
        assert "very strong" in interpret_ln_bf(6.0)


class TestValidationPipeline:
    """Smoke test the validation modules end-to-end."""

    def test_injection_recovery_smoke(
        self, bounce_params, gw_context, tmp_path
    ):
        from whitesearch.validation import InjectionRecovery

        runner = BilbyRunner(force_toy=True, nlive=20, outdir=str(tmp_path), seed=0)
        sim = GravitationalWaveSimulator()
        model = BlackToWhiteBounce()
        ll = GWLikelihood()

        ir = InjectionRecovery(simulator=sim, runner=runner, n_injections=3, rng_seed=0)
        result = ir.run_injections(model, ll, gw_context, save_dir=tmp_path / "injections")

        assert len(result.theta_true) == 3
        assert len(result.posteriors) == 3
        assert len(result.evidences) == 3

    def test_sbc_smoke(self, gw_context, tmp_path):
        from whitesearch.validation import SBCRunner

        runner = BilbyRunner(force_toy=True, nlive=20, outdir=str(tmp_path), seed=0)
        sim = GravitationalWaveSimulator()
        model = BlackToWhiteBounce()
        ll = GWLikelihood()

        sbc = SBCRunner(n_simulations=3, n_posterior_samples=20, rng_seed=0)
        result = sbc.run(model, sim, ll, runner, gw_context)

        assert isinstance(result.ranks, dict)
        assert "M" in result.ranks
