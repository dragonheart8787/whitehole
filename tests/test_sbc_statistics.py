"""SBCResult's uniformity verdict must be deterministic.

Regression cover for the tooling finding in
docs/BH_RINGDOWN_SBC_COVERAGE_REPORT.md (E1): _compute_uniformity() drew a
fresh, unseeded np.random.uniform() comparison sample and ran ks_2samp against
it, so the same ranks produced a different p-value on every call
(0.628 / 0.628 / 0.328 / 0.112 / 0.866 were observed on one rank set) and
`calibrated` was a coin flip near the 0.05 threshold.
"""

from __future__ import annotations

import numpy as np
import pytest

from whitesearch.validation.sbc import SBCResult

L = 100
N_SIM = 200


def _result(ranks: list[int]) -> SBCResult:
    return SBCResult(ranks={"M": list(ranks)}, n_posterior_samples=L, n_simulations=N_SIM)


@pytest.fixture
def uniform_ranks() -> list[int]:
    return [int(v) for v in np.random.default_rng(0).integers(0, L + 1, N_SIM)]


def test_pvalue_is_identical_across_repeated_calls(uniform_ranks):
    pvalues = [_result(uniform_ranks).uniformity_pvalues["M"] for _ in range(5)]
    assert len(set(pvalues)) == 1, pvalues
    verdicts = [_result(uniform_ranks).calibrated["M"] for _ in range(5)]
    assert len(set(verdicts)) == 1, verdicts


def test_pvalue_does_not_depend_on_the_global_numpy_rng(uniform_ranks):
    np.random.seed(1)
    first = _result(uniform_ranks).uniformity_pvalues["M"]
    np.random.seed(999)
    [np.random.uniform() for _ in range(37)]
    assert _result(uniform_ranks).uniformity_pvalues["M"] == first


def test_uniform_ranks_pass_and_degenerate_ranks_fail(uniform_ranks):
    assert _result(uniform_ranks).calibrated["M"] is True

    # Every truth below every posterior sample: the failure mode seen for
    # inclination before the cos_uniform -> Sine prior fix.
    piled_up = [L] * N_SIM
    degenerate = _result(piled_up)
    assert degenerate.calibrated["M"] is False
    assert degenerate.uniformity_pvalues["M"] < 1e-6


def test_ranks_are_normalised_against_the_rank_denominator():
    """Ranks live in {0..L}: L+1 outcomes, so the divisor must be L+1."""
    from scipy import stats

    ranks = [int(v) for v in np.random.default_rng(3).integers(0, L + 1, N_SIM)]
    expected = float(stats.kstest((np.array(ranks) + 0.5) / (L + 1.0), "uniform").pvalue)
    assert _result(ranks).uniformity_pvalues["M"] == pytest.approx(expected)


def test_too_few_simulations_reports_nan_and_not_calibrated():
    result = SBCResult(ranks={"M": [1, 2, 3]}, n_posterior_samples=L, n_simulations=3)
    assert np.isnan(result.uniformity_pvalues["M"])
    assert result.calibrated["M"] is False
