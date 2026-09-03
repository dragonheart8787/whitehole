"""ParameterSpec.sample() and ParameterSpec.to_bilby_prior() must agree.

Regression cover for the prior-mapping bug family found by the bh_ringdown
SBC run (docs/BH_RINGDOWN_SBC_COVERAGE_REPORT.md, finding (A)): ``cos_uniform``
drew arccos(U(-1,1)) on [0, pi] but was translated to bilby's ``Cosine``, which
lives on [-pi/2, +pi/2] with a different density, so half the SBC truths fell
outside the sampler's support.  ``half_normal`` and ``beta`` had no case at all
and fell through to ``Uniform(0, 1)``.

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


@pytest.mark.parametrize(
    "prior_type",
    [
        pytest.param(
            "discrete_uniform",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "KNOWN MISMATCH: sample() draws uniformly from `values`, but "
                    "to_bilby_prior() returns DeltaFunction(values[0]), pinning the "
                    "sampler to a single value. bilby 2.8's Categorical only covers "
                    "0..n-1, so an offset set such as bounce's p_lifetime=[4, 5] has "
                    "no faithful equivalent; raising instead would take bounce's "
                    "dynesty path offline. Left as an explicit decision -- if this "
                    "xfail starts passing, the mapping was fixed and this marker "
                    "should be removed."
                ),
            ),
        )
        if pt == "discrete_uniform"
        else pt
        for pt in SPECS
    ],
)
def test_bilby_prior_matches_sample_prior(prior_type):
    """Empirical distributions from both code paths must be indistinguishable."""
    spec = _spec(prior_type)

    rng = np.random.default_rng(20260903)
    ours = np.array([spec.sample(rng) for _ in range(N_SAMPLES)], dtype=float)

    np.random.seed(20260903)  # bilby priors sample from the numpy global RNG
    theirs = np.asarray(spec.to_bilby_prior().sample(N_SAMPLES), dtype=float)

    assert np.isfinite(ours).all()
    assert np.isfinite(theirs).all()

    _, pvalue = stats.ks_2samp(ours, theirs)
    assert pvalue > KS_P_MIN, (
        f"{prior_type}: sample_prior and bilby prior disagree "
        f"(KS p={pvalue:.3g}); ours [{ours.min():.4g}, {ours.max():.4g}] "
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
