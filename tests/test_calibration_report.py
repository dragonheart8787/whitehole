"""Smoke tests for fixed-layout calibration reports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from whitesearch.validation.calibration_report import (
    CALIBRATION_ARTIFACTS,
    generate_calibration_report,
)


@pytest.fixture
def calib_dir(tmp_path):
    root, report = generate_calibration_report(
        tmp_path / "calib",
        model="bounce",
        channel="gw",
        data_source="mock",
        n_injections=2,
        n_sbc=3,
        n_ppc=10,
        nlive=20,
        seed=0,
        force_toy=True,
        profile="quick",
    )
    return root, report


def test_calibration_artifact_contract(calib_dir):
    root, report = calib_dir
    for rel in (
        CALIBRATION_ARTIFACTS["index"],
        CALIBRATION_ARTIFACTS["report"],
        CALIBRATION_ARTIFACTS["config"],
        CALIBRATION_ARTIFACTS["diagnostics"],
        CALIBRATION_ARTIFACTS["coverage_csv"],
        CALIBRATION_ARTIFACTS["coverage_png"],
        CALIBRATION_ARTIFACTS["prior_audit"],
        CALIBRATION_ARTIFACTS["sbc_summary"],
        CALIBRATION_ARTIFACTS["ppc_summary"],
        CALIBRATION_ARTIFACTS["mock_strain_csv"],
        CALIBRATION_ARTIFACTS["mock_lnz_csv"],
        CALIBRATION_ARTIFACTS["mock_diag_mock"],
    ):
        assert (root / rel).is_file(), f"missing {rel}"

    parsed = json.loads((root / "report.json").read_text(encoding="utf-8"))
    assert parsed["overall"] in ("PASS", "FAIL")
    assert "sections" in parsed
    assert (root / "index.md").read_text(encoding="utf-8").strip()


def test_sbc_per_param_plots(calib_dir):
    root, _ = calib_dir
    sbc_dir = root / "sbc"
    per_param = list(sbc_dir.glob("rank_hist_*.png"))
    assert len(per_param) >= 1


def test_ppc_per_stat_plots(calib_dir):
    root, _ = calib_dir
    ppc_dir = root / "ppc"
    per_stat = list(ppc_dir.glob("ppc_*.png"))
    assert len(per_stat) >= 1
