"""Slow GWOSC dynesty smoke tests (opt-in with pytest -m slow)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


@pytest.fixture
def gwosc_available():
    try:
        from whitesearch.dataio.gwosc import GWPY_AVAILABLE
    except ImportError:
        pytest.skip("gwosc module unavailable")
    if not GWPY_AVAILABLE:
        pytest.skip("gwpy not installed")


@pytest.mark.skipif(
    os.environ.get("WHITESEARCH_RUN_GWOSC_DYNESTY") != "1",
    reason="Set WHITESEARCH_RUN_GWOSC_DYNESTY=1 to run (slow, network)",
)
def test_gwosc_bh_ringdown_fit_finite(gwosc_available, tmp_path):
    from whitesearch.inference.bilby_runner import BILBY_AVAILABLE

    if not BILBY_AVAILABLE:
        pytest.skip("bilby not installed")

    os.environ.pop("WHITESEARCH_FORCE_TOY", None)
    out = tmp_path / "bh"
    import subprocess
    import sys

    r = subprocess.run(
        [
            sys.executable, "-m", "whitesearch", "fit",
            "--model", "bh_ringdown",
            "--data", "gwosc",
            "--event", "GW150914",
            "--channel", "gw",
            "--nlive", "15",
            "--likelihood-mode", "mf",
            "--outdir", str(out),
            "--no-resume",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    diag = json.loads((out / "diagnostics.json").read_text())
    assert diag["models"]["bh_ringdown"]["frac_finite"] > 0.5
