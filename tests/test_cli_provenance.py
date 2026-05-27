"""Tests for CLI provenance and fail-closed data loading."""

from __future__ import annotations

import pytest

from whitesearch.dataio.loader import load_observation_data
from whitesearch.dataio.provenance import DataLoadError


def test_mock_explicit_provenance():
    data, prov = load_observation_data(
        "mock", "gw", inject_model="bounce", seed=0,
        context={"sample_rate": 4096, "duration": 1.0, "t_merger": 0.5, "rng_seed": 0},
    )
    assert prov.actual_source == "MOCK_EXPLICIT"
    assert not prov.fallback_used
    assert prov.inject_model == "bounce"
    assert "psd" in data
    assert "strain_rms_used" in data


def test_inject_model_defaults_to_fit_model_via_loader():
    _, prov = load_observation_data(
        "mock", "radio", inject_model="bh_ringdown", seed=1,
        context={
            "freq_low_mhz": 400, "freq_high_mhz": 800, "n_freq_chans": 8,
            "n_time_bins": 32, "rng_seed": 1,
        },
    )
    assert prov.inject_model == "bh_ringdown"


def test_heasarc_fail_closed():
    with pytest.raises(DataLoadError):
        load_observation_data(
            "heasarc", "gw", inject_model="bounce", seed=0,
            context={}, allow_mock_fallback=False,
        )


def test_heasarc_with_fallback_flag():
    _, prov = load_observation_data(
        "heasarc", "gw", inject_model="bounce", seed=0,
        context={"sample_rate": 4096, "duration": 1.0, "t_merger": 0.5},
        allow_mock_fallback=True,
    )
    assert prov.fallback_used
    assert prov.actual_source == "MOCK_FALLBACK"


def test_cli_entry_point_importable():
    from whitesearch.cli import main
    assert callable(main)
