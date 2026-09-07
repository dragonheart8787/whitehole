"""InjectionRecoveryResult must not score coverage for unsampled parameters.

Regression cover for docs/BOUNCE_PREFLIGHT_AUDIT.md section C.6: the interval
lookup used ``ci.get(p, (-np.inf, np.inf))``, so a parameter absent from the
posterior had every true value fall inside its "interval" and was reported as
coverage = 1.0 with no warning.  A model may legitimately declare parameters
the channel's likelihood never reads -- BlackToWhiteBounce declares 12 while
GWLikelihood("bounce") reads 6 -- and coverage for those is undefined, not 1.0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from whitesearch.validation import InjectionRecoveryResult

N = 10


def _result(
    *,
    truths: list[dict[str, float]],
    intervals: list[dict[str, tuple[float, float]]],
    posteriors: list[pd.DataFrame],
) -> InjectionRecoveryResult:
    return InjectionRecoveryResult(
        theta_true=truths,
        posteriors=posteriors,
        evidences=[0.0] * len(truths),
        evidence_errs=[0.0] * len(truths),
        credible_intervals=intervals,
    )


@pytest.fixture
def seven_of_ten_covered():
    """M is covered by 7 of 10 injections; `dead` is never sampled."""
    truths, intervals, posteriors = [], [], []
    for i in range(N):
        truths.append({"M": 50.0, "dead": 7.0})
        # the first 7 intervals bracket the truth, the last 3 do not
        intervals.append({"M": (40.0, 60.0) if i < 7 else (10.0, 20.0)})
        posteriors.append(pd.DataFrame({"M": np.linspace(40.0, 60.0, 101)}))
    return _result(truths=truths, intervals=intervals, posteriors=posteriors)


class TestUnsampledParametersAreNotScored:
    def test_unsampled_parameter_is_excluded_not_reported_as_one(
        self, seven_of_ten_covered
    ):
        result = seven_of_ten_covered
        assert "dead" not in result.coverage
        assert result.unsampled_parameters == ["dead"]
        assert result.coverage.get("dead") != 1.0

    def test_querying_an_unsampled_parameter_raises(self, seven_of_ten_covered):
        with pytest.raises(KeyError, match="never sampled"):
            seven_of_ten_covered.coverage_of("dead")

    def test_querying_an_uninjected_parameter_raises(self, seven_of_ten_covered):
        with pytest.raises(KeyError, match="Unknown parameter"):
            seven_of_ten_covered.coverage_of("not_a_parameter")

    def test_partially_sampled_parameter_is_excluded_and_raises(self):
        """No coverage over an inconsistent denominator either."""
        truths = [{"M": 50.0, "sometimes": 1.0} for _ in range(N)]
        intervals = [
            {"M": (40.0, 60.0), **({"sometimes": (0.0, 2.0)} if i < 4 else {})}
            for i in range(N)
        ]
        posteriors = [pd.DataFrame({"M": np.linspace(40.0, 60.0, 101)}) for _ in range(N)]
        result = _result(truths=truths, intervals=intervals, posteriors=posteriors)

        assert result.partially_sampled_parameters == ["sometimes"]
        assert "sometimes" not in result.coverage
        with pytest.raises(KeyError, match="only some injections"):
            result.coverage_of("sometimes")

    def test_warning_is_logged_for_excluded_parameters(self, caplog):
        truths = [{"M": 50.0, "dead": 7.0} for _ in range(N)]
        intervals = [{"M": (40.0, 60.0)} for _ in range(N)]
        posteriors = [pd.DataFrame({"M": np.linspace(40.0, 60.0, 101)}) for _ in range(N)]
        with caplog.at_level("WARNING"):
            _result(truths=truths, intervals=intervals, posteriors=posteriors)
        assert "Coverage undefined" in caplog.text
        assert "dead" in caplog.text


class TestSampledParameterBehaviourIsUnchanged:
    """Regression: the numbers for genuinely sampled parameters must not move."""

    def test_coverage_fraction_is_hits_over_n(self, seven_of_ten_covered):
        assert seven_of_ten_covered.coverage["M"] == pytest.approx(0.7)
        assert seven_of_ten_covered.coverage_of("M") == pytest.approx(0.7)

    def test_sbc_ranks_are_still_collected(self, seven_of_ten_covered):
        ranks = seven_of_ten_covered.sbc_ranks["M"]
        assert len(ranks) == N
        # truth 50.0 sits at the midpoint of linspace(40, 60, 101)
        assert set(ranks) == {50}

    def test_summary_lists_only_the_scored_parameters(self, seven_of_ten_covered):
        df = seven_of_ten_covered.summary()
        assert list(df["parameter"]) == ["M"]
        assert df["coverage_90pct"].iloc[0] == pytest.approx(0.7)
        assert bool(df["coverage_ok"].iloc[0]) is False  # 0.7 < 0.8
        assert df["sbc_n"].iloc[0] == N

    def test_fully_covered_case_still_reports_one(self):
        truths = [{"M": 50.0} for _ in range(N)]
        intervals = [{"M": (40.0, 60.0)} for _ in range(N)]
        posteriors = [pd.DataFrame({"M": np.linspace(40.0, 60.0, 101)}) for _ in range(N)]
        result = _result(truths=truths, intervals=intervals, posteriors=posteriors)
        assert result.coverage_of("M") == pytest.approx(1.0)
        assert result.unsampled_parameters == []

    def test_empty_campaign_is_still_handled(self):
        result = _result(truths=[], intervals=[], posteriors=[])
        assert result.coverage == {}
        assert result.unsampled_parameters == []


class TestRealCampaignShape:
    """The bounce case that motivated the fix, end to end through the runner."""

    def test_declared_but_unsampled_model_parameters_are_not_scored(self, tmp_path):
        from whitesearch.inference import BilbyRunner
        from whitesearch.likelihoods import GWLikelihood
        from whitesearch.models import get_model
        from whitesearch.simulators import get_simulator
        from whitesearch.validation import InjectionRecovery

        context = {"sample_rate": 4096, "duration": 4.0, "t_merger": 1.0,
                   "low_freq_cutoff": 20.0, "rng_seed": 0}
        model = get_model("bounce")
        runner = BilbyRunner(force_toy=True, nlive=20, outdir=str(tmp_path), seed=0)
        ir = InjectionRecovery(
            simulator=get_simulator("gw"), runner=runner, n_injections=2, rng_seed=0
        )
        result = ir.run_injections(model, GWLikelihood("bounce"), context)

        assert len(model.parameter_names) == 12
        assert sorted(result.coverage) == sorted(
            ["M", "a_star", "eps_f", "eps_Q", "D_L", "i"]
        )
        assert sorted(result.unsampled_parameters) == sorted(
            ["log10_tau_bounce_yr", "log10_ell_q", "p_lifetime",
             "log10_A_bounce", "eta_r", "eta_gamma"]
        )
        assert result.metadata["n_failed"] == 0
