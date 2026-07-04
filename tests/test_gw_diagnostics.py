"""Unit tests for GW diagnostics (frac_finite scale independence)."""

from __future__ import annotations

import numpy as np
import pytest

from whitesearch.dataio.gw_observation import prepare_gw_from_simdata
from whitesearch.simulators import GravitationalWaveSimulator
from whitesearch.validation.gw_diagnostics import run_gw_diagnostics


@pytest.fixture
def mock_obs(bounce_params, gw_context):
    sim = GravitationalWaveSimulator()
    sd = sim.simulate(bounce_params, gw_context, rng=np.random.default_rng(0))
    return prepare_gw_from_simdata(sd)


class TestFracFinite:
    def test_counts_accepted_draws_on_mock_data(self, mock_obs, gw_context):
        diag = run_gw_diagnostics(
            mock_obs, gw_context, ["bh_ringdown", "null"], n_prior_draws=50, seed=0,
        )
        bh = diag["models"]["bh_ringdown"]
        assert bh["frac_finite"] > 0.0
        assert bh["median_lnL"] is not None

    def test_scale_free_rule_survives_large_negative_lnl(self, mock_obs, gw_context):
        """Draws whose lnL is hugely negative but genuine (template accepted)
        must still count as finite.  The old absolute -1e5 cutoff classified
        every draw on real-data scales (lnL ~ -1e6 and below) as non-finite."""
        scaled = dict(mock_obs)
        # Inflate <d|d> so every lnL falls far below the old -1e5 cutoff
        scaled["strain"] = np.asarray(mock_obs["strain"]) * 1e3
        diag = run_gw_diagnostics(
            scaled, gw_context, ["bh_ringdown"], n_prior_draws=50, seed=0,
        )
        bh = diag["models"]["bh_ringdown"]
        assert diag["null_lnL"] < -1e5
        assert bh["frac_finite"] > 0.0

    def test_explicit_threshold_still_available(self, mock_obs, gw_context):
        diag = run_gw_diagnostics(
            mock_obs, gw_context, ["bh_ringdown"], n_prior_draws=50, seed=0,
            finite_threshold=np.inf,
        )
        assert diag["models"]["bh_ringdown"]["frac_finite"] == 0.0
