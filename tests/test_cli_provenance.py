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


def test_model_seed_offset_stable_across_hash_randomization():
    """The per-model seed offset must not depend on PYTHONHASHSEED (the old
    hash()-based offset changed between processes for the same --seed)."""
    import os
    import subprocess
    import sys
    import zlib

    code = (
        "from whitesearch.cli import _model_seed_offset; "
        "print(_model_seed_offset('bh_ringdown'))"
    )
    outputs = set()
    for hashseed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hashseed}
        r = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env,
        )
        assert r.returncode == 0, r.stderr
        outputs.add(r.stdout.strip())
    assert len(outputs) == 1
    assert int(outputs.pop()) == zlib.crc32(b"bh_ringdown") % 1000


def test_run_single_fit_persists_sampler_provenance(tmp_path, monkeypatch):
    """The saved *_metadata.json must carry the runner's actual-vs-requested
    sampler kwargs and any bound fallback, so fallbacks are auditable from
    the artifact itself, not only from logs."""
    import json

    import pandas as pd

    from whitesearch.inference.bilby_runner import BILBY_AVAILABLE

    if not BILBY_AVAILABLE:
        pytest.skip("bilby not installed")

    import whitesearch.inference.bilby_runner as bilby_runner_mod

    def fake_run_sampler(*args, **kwargs):
        if kwargs.get("bound") == "live":
            raise RuntimeError("ellipsoid update failed for bound=live")

        class FakeResult:
            log_evidence = -5.0
            log_evidence_err = 0.1
            posterior = pd.DataFrame(
                {"M": [60.0], "log_likelihood": [-1.0]}
            )

        return FakeResult()

    monkeypatch.setattr(bilby_runner_mod.bilby, "run_sampler", fake_run_sampler)

    from whitesearch.cli import _run_single_fit

    ctx = {"sample_rate": 4096, "duration": 1.0, "t_merger": 0.5, "rng_seed": 0}
    obs, prov = load_observation_data(
        "mock", "gw", inject_model="bh_ringdown", seed=0, context=ctx,
    )
    _run_single_fit(
        "bh_ringdown", "gw", obs, ctx, prov, "bh_ringdown",
        nlive=10, outdir=str(tmp_path), seed=0, resume=False,
        event=None, data="mock", label_suffix="bh_ringdown",
        likelihood_mode="full",
    )

    meta = json.loads((tmp_path / "bh_ringdown_metadata.json").read_text())
    assert meta["bound_fallback_occurred"] is True
    assert meta["bound_fallback_from"] == "live"
    assert meta["bound_fallback_to"] == "multi"
    assert meta["sampler_kwargs"]["bound"] == "multi"
    assert meta["requested_sampler_kwargs"]["bound"] == "live"
