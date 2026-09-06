"""ParameterSpec.sample() and ParameterSpec.to_bilby_prior() must agree.

Regression cover for the prior-mapping bug family found by the bh_ringdown
SBC run (docs/BH_RINGDOWN_SBC_COVERAGE_REPORT.md, finding (A)): ``cos_uniform``
drew arccos(U(-1,1)) on [0, pi] but was translated to bilby's ``Cosine``, which
lives on [-pi/2, +pi/2] with a different density, so half the SBC truths fell
outside the sampler's support.  ``half_normal`` and ``beta`` had no case at all
and fell through to ``Uniform(0, 1)``.  ``discrete_uniform`` returned
``DeltaFunction(values[0])``, pinning bounce's ``p_lifetime`` to 4 instead of
sampling {4, 5}; it now maps to ``bilby.core.prior.DiscreteValues``.

The parametrised test below covers *every* prior type in the ``PriorType``
Literal, so a newly added type cannot be introduced with a wrong (or missing)
bilby translation without a red test.
"""

from __future__ import annotations

import typing

import numpy as np
import pytest
from scipy import stats

from whitesearch.models.base import ParameterSpec, PriorType

pytest.importorskip("bilby")

N_SAMPLES = 20_000
# Two-sample KS between 20k draws from each side.  0.001 keeps false alarms
# negligible while still catching support/shape errors, which show up at
# p < 1e-10 rather than marginally.
KS_P_MIN = 1e-3

# One representative spec per supported prior type.
SPECS: dict[str, dict] = {
    "uniform": {"low": -3.0, "high": 7.0},
    "log_uniform": {"low": 1e-2, "high": 1e3},
    "normal": {"mean": -1.6, "std": 1.2},
    "half_normal": {"sigma": 2.0},
    "cos_uniform": {},
    "volume_uniform": {"low": 10.0, "high": 10_000.0},
    "beta": {"a": 2.0, "b": 5.0},
    "discrete_uniform": {"values": [4, 5]},
}


def test_specs_cover_every_declared_prior_type():
    """If PriorType grows, this file must grow with it."""
    assert set(SPECS) == set(typing.get_args(PriorType))


def _spec(prior_type: str) -> ParameterSpec:
    return ParameterSpec(
        name=f"p_{prior_type}", prior_type=prior_type, prior_kwargs=SPECS[prior_type]
    )


@pytest.mark.parametrize("prior_type", list(SPECS))
def test_bilby_prior_matches_sample_prior(prior_type):
    """Empirical distributions from both code paths must be indistinguishable."""
    spec = _spec(prior_type)

    rng = np.random.default_rng(20260903)
    ours = np.array([spec.sample(rng) for _ in range(N_SAMPLES)], dtype=float)

    np.random.seed(20260903)  # bilby priors sample from the numpy global RNG
    theirs = np.asarray(spec.to_bilby_prior().sample(N_SAMPLES), dtype=float)

    assert np.isfinite(ours).all()
    assert np.isfinite(theirs).all()

    if prior_type == "discrete_uniform":
        # KS is the wrong instrument for a finite support: compare the value
        # sets and their frequencies with a chi-square test instead.
        expected = sorted(float(v) for v in SPECS[prior_type]["values"])
        assert sorted(np.unique(ours).tolist()) == expected
        assert sorted(np.unique(theirs).tolist()) == expected
        observed = np.array([(theirs == v).sum() for v in expected], dtype=float)
        expected_counts = np.array([(ours == v).sum() for v in expected], dtype=float)
        _, pvalue = stats.chisquare(observed, f_exp=expected_counts)
    else:
        _, pvalue = stats.ks_2samp(ours, theirs)

    assert pvalue > KS_P_MIN, (
        f"{prior_type}: sample_prior and bilby prior disagree "
        f"(p={pvalue:.3g}); ours [{ours.min():.4g}, {ours.max():.4g}] "
        f"vs bilby [{theirs.min():.4g}, {theirs.max():.4g}]"
    )


def test_cos_uniform_maps_to_sine_over_zero_to_pi():
    """The specific mapping that broke inclination calibration."""
    prior = _spec("cos_uniform").to_bilby_prior()
    assert type(prior).__name__ == "Sine"
    assert prior.minimum == pytest.approx(0.0)
    assert prior.maximum == pytest.approx(np.pi)


def test_unknown_prior_type_raises_instead_of_silent_uniform():
    """fail-closed: no silent Uniform(0, 1) substitution."""
    spec = ParameterSpec(name="mystery", prior_type="not_a_real_prior")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="No bilby prior mapping"):
        spec.to_bilby_prior()


def test_every_model_builds_bilby_priors():
    """to_bilby_priors() must not raise for any registered model."""
    from whitesearch.models import MODEL_REGISTRY, get_model

    for name in MODEL_REGISTRY:
        model = get_model(name)
        priors = model.to_bilby_priors()
        assert set(priors) == set(model.parameter_names), name


@pytest.mark.parametrize("values", [[4, 5], [2, 7, 11], [-3, 0, 1, 9]])
def test_discrete_uniform_covers_arbitrary_offset_value_sets(values):
    """Not just {0..n-1}, and not just two values."""
    spec = ParameterSpec(
        name="discrete", prior_type="discrete_uniform", prior_kwargs={"values": values}
    )
    prior = spec.to_bilby_prior()
    np.random.seed(20260906)
    drawn = np.asarray(prior.sample(40_000), dtype=float)
    assert sorted(np.unique(drawn).tolist()) == sorted(float(v) for v in values)
    counts = np.array([(drawn == v).sum() for v in values], dtype=float)
    _, pvalue = stats.chisquare(counts)
    assert pvalue > KS_P_MIN, f"{values}: not uniform over the value set (p={pvalue:.3g})"
    # The prior must also score the values it claims to support, so nested
    # sampling sees a real density rather than a point mass.
    for v in values:
        assert prior.prob(v) == pytest.approx(1.0 / len(values))
    assert prior.prob(max(values) + 1) == 0.0


def test_discrete_uniform_is_not_a_point_mass():
    """Regression: this used to be DeltaFunction(values[0])."""
    prior = ParameterSpec(
        name="p_lifetime", prior_type="discrete_uniform", prior_kwargs={"values": [4, 5]}
    ).to_bilby_prior()
    assert type(prior).__name__ != "DeltaFunction"
    np.random.seed(0)
    assert len(np.unique(prior.sample(2_000))) == 2


def test_empty_discrete_uniform_raises():
    spec = ParameterSpec(
        name="empty", prior_type="discrete_uniform", prior_kwargs={"values": []}
    )
    with pytest.raises(ValueError, match="no values"):
        spec.to_bilby_prior()
