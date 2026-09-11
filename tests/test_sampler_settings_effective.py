"""The sampler settings this project passes must actually take effect.

Two settings were found to be silently discarded (docs/BOUNCE_PREFLIGHT_AUDIT.md
K.2): ``walks`` under ``sample="rwalk"``, and a caller-supplied ``maxcall``.
These tests check EFFECT, not the presence of a key in a kwargs dict - the very
thing that made both failures invisible.  bilby's own kwargs dict reports
``walks: 100`` on an rwalk run while ignoring it completely, so a key-presence
assertion would have passed throughout.
"""

from __future__ import annotations

import time

import pytest

from whitesearch.inference.bilby_runner import (
    BilbyRunner,
    SamplingBudgetExceeded,
    _BudgetGuard,
)

bilby = pytest.importorskip("bilby")


def _dynesty(tmp_path, **kwargs):
    """A bilby Dynesty set up far enough to report its chain length, not run."""
    class _L(bilby.core.likelihood.Likelihood):
        def __init__(self):
            super().__init__(parameters={"x": None})

        def log_likelihood(self) -> float:
            return -0.5 * float(self.parameters["x"]) ** 2

    priors = bilby.core.prior.PriorDict(
        {"x": bilby.core.prior.Uniform(-1, 1, "x")}
    )
    from bilby.core.sampler.dynesty import Dynesty

    return Dynesty(
        likelihood=_L(), priors=priors, outdir=str(tmp_path), label="t",
        nlive=10, skip_import_verification=True, **kwargs
    )


class TestChainLengthKnob:
    """`nact` is live under sample='rwalk'; `walks` is not."""

    def test_nact_reaches_the_internal_sampler(self, tmp_path):
        for nact in (2, 8):
            internal = _dynesty(tmp_path, sample="rwalk", nact=nact).sampler_init_kwargs["sample"]
            assert type(internal).__name__ == "AcceptanceTrackingRWalk"
            assert internal.nact == nact

    def test_walks_is_ignored_under_rwalk(self, tmp_path):
        """The regression this locks: walks=128 behaved exactly like walks=32.

        Note the trap this test exists to expose: the internal sampler DOES
        carry a ``walks`` attribute, so asserting it exists proves nothing.  Its
        value is dynesty's own baseline 25 no matter what the caller passed.
        """
        a = _dynesty(tmp_path, sample="rwalk", walks=32).sampler_init_kwargs["sample"]
        b = _dynesty(tmp_path, sample="rwalk", walks=128).sampler_init_kwargs["sample"]
        assert a.nact == b.nact == 2
        assert a.walks == b.walks
        assert a.walks not in (32, 128), "the passed value never reaches the sampler"

    def test_walks_is_live_under_acceptance_walk(self, tmp_path):
        """Same key, different sample method, genuinely different behaviour."""
        internal = _dynesty(
            tmp_path, sample="acceptance-walk", walks=128
        ).sampler_init_kwargs["sample"]
        assert internal.walks == 128

    def test_project_default_uses_the_live_knob(self):
        kw = BilbyRunner.DEFAULT_DYNESTY_KWARGS
        assert kw["sample"] == "rwalk"
        assert kw["nact"] == 2, "must stay at bilby's default so behaviour is unchanged"
        assert "walks" not in kw, "walks does nothing under sample='rwalk'"

    def test_runner_forwards_nact(self):
        assert BilbyRunner(nact=8).sampler_kwargs["nact"] == 8


class TestBudgetGuard:
    """An independent budget, because dynesty's maxcall is overwritten by bilby."""

    def test_inactive_guard_does_not_wrap(self):
        guard = _BudgetGuard()
        assert not guard.active
        fn = lambda: 1
        assert guard.guard(fn) is fn

    def test_call_budget_stops_a_runaway(self):
        guard = _BudgetGuard(max_calls=5).start()
        slow = guard.guard(lambda: 0.0)
        for _ in range(5):
            slow()
        with pytest.raises(SamplingBudgetExceeded, match="likelihood-call budget"):
            slow()
        assert guard.calls == 6

    def test_wall_clock_budget_stops_a_stuck_run(self):
        """A mock 'stuck' likelihood: each call takes a long simulated time."""
        clock = {"t": 0.0}

        def fake_time() -> float:
            return clock["t"]

        guard = _BudgetGuard(max_seconds=10.0, time_fn=fake_time).start()

        def stuck() -> float:
            clock["t"] += 4.0          # every evaluation burns 4 simulated seconds
            return -1.0

        guarded = guard.guard(stuck)
        guarded()   # t=4
        guarded()   # t=8
        guarded()   # t=12, but the check runs BEFORE the call, so still inside
        with pytest.raises(SamplingBudgetExceeded, match="wall-clock budget"):
            guarded()
        assert guard.elapsed > 10.0

    def test_real_clock_terminates_promptly(self):
        """Same logic against the real clock, bounded well under a second."""
        guard = _BudgetGuard(max_seconds=0.05).start()
        guarded = guard.guard(lambda: time.sleep(0.01))
        t0 = time.monotonic()
        with pytest.raises(SamplingBudgetExceeded):
            for _ in range(1000):
                guarded()
        assert time.monotonic() - t0 < 2.0

    def test_neither_budget_never_raises(self):
        guard = _BudgetGuard().start()
        guarded = guard.guard(lambda: 1.0)
        for _ in range(1000):
            assert guarded() == 1.0

    def test_message_names_both_counters(self):
        guard = _BudgetGuard(max_calls=1).start()
        guarded = guard.guard(lambda: 1.0)
        guarded()
        with pytest.raises(SamplingBudgetExceeded) as exc:
            guarded()
        assert "calls" in str(exc.value) and "s" in str(exc.value)


class TestRunnerAppliesTheBudget:
    def test_wrapped_likelihood_raises_when_the_budget_is_spent(self):
        from whitesearch.likelihoods import GWLikelihood
        from whitesearch.models import get_model

        runner = BilbyRunner(max_likelihood_calls=3)
        wrapped = runner._wrap_likelihood(
            GWLikelihood("bh_ringdown"), {"psd": None}, {}
        )
        wrapped.parameters = {"M": 60.0, "a_star": 0.6, "log10_A": -21.0}
        for _ in range(3):
            wrapped.log_likelihood()
        with pytest.raises(SamplingBudgetExceeded):
            wrapped.log_likelihood()

    def test_no_budget_means_no_wrapping_overhead(self):
        from whitesearch.likelihoods import GWLikelihood

        runner = BilbyRunner()
        runner._wrap_likelihood(GWLikelihood("bh_ringdown"), {"psd": None}, {})
        assert not runner.last_budget_guard.active
