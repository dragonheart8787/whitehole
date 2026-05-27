"""Prepare GW strain for likelihood evaluation (shared mock + GWOSC path)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..preprocess.gw_preprocess import GWPreprocessor


def prepare_gw_observation(
    strain: np.ndarray,
    sample_rate: float,
    *,
    event_gps: float | None = None,
    gps_start: float | None = None,
    dq_flags: dict | None = None,
    reference_amplitude: bool = False,
    target_rms: float = 1e-20,
    source: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bandpass, estimate PSD, optionally reference-calibrate amplitude.

    Parameters
    ----------
    reference_amplitude : bool
        If True, scale bandpass strain so RMS matches ``target_rms`` and scale PSD by
        scale**2.  Recorded explicitly in metadata (never silent).
    """
    strain = np.asarray(strain, dtype=np.float64)
    sr = float(sample_rate)
    strain_rms_raw = float(np.std(strain))

    prep = GWPreprocessor(sample_rate=sr)
    proc = prep.prepare_raw(strain, dq_flags)

    strain_bp = np.asarray(proc["strain_bandpass"], dtype=np.float64)
    psd = np.asarray(proc["psd"], dtype=np.float64)
    strain_rms_bp = float(np.std(strain_bp))

    amplitude_scale = 1.0
    if reference_amplitude and strain_rms_bp > 0:
        amplitude_scale = target_rms / strain_rms_bp
        strain_bp = strain_bp * amplitude_scale
        psd = psd * (amplitude_scale ** 2)

    n = len(strain_bp)
    if event_gps is not None and gps_start is not None:
        t_merger = float(event_gps) - float(gps_start)
    else:
        t_merger = 0.5 * n / sr

    out: dict[str, Any] = {
        "strain": strain_bp,
        "strain_raw": strain,
        "strain_whitened": proc["strain_whitened"],
        "strain_bandpass": proc["strain_bandpass"],
        "psd": psd,
        "sample_rate": sr,
        "t_merger": t_merger,
        "strain_rms_raw": strain_rms_raw,
        "strain_rms_bp": strain_rms_bp,
        "strain_rms_used": float(np.std(strain_bp)),
        "amplitude_scale_applied": amplitude_scale,
        "reference_amplitude": reference_amplitude,
        "preprocess_quality": proc["quality"],
    }
    if source is not None:
        out["source"] = source
    if extra:
        out.update(extra)
    return out


def prepare_gw_from_simdata(sim_data: Any, *, reference_amplitude: bool = False) -> dict[str, Any]:
    """Convert simulator SimData to GW observation dict with unified preprocessing."""
    meta = sim_data.metadata if hasattr(sim_data, "metadata") else {}
    strain = np.asarray(sim_data.data if hasattr(sim_data, "data") else sim_data["strain"])
    sr = float(meta.get("sample_rate", 4096.0))
    t_merger = float(meta.get("t_merger", 0.5 * len(strain) / sr))
    gps_start = t_merger - 0.5 * len(strain) / sr
    return prepare_gw_observation(
        strain,
        sr,
        event_gps=t_merger,
        gps_start=gps_start,
        reference_amplitude=reference_amplitude,
        source="MOCK_EXPLICIT",
    )


def prepare_gw_from_record(record: dict[str, Any], *, reference_amplitude: bool = False) -> dict[str, Any]:
    """Prepare GW observation from GWOSC loader record."""
    return prepare_gw_observation(
        record["strain"],
        float(record["sample_rate"]),
        event_gps=record.get("event_gps"),
        gps_start=record.get("gps_start"),
        dq_flags=record.get("dq_flags"),
        reference_amplitude=reference_amplitude,
        source=record.get("source", "GWOSC"),
        extra={k: v for k, v in record.items() if k not in ("strain",)},
    )
