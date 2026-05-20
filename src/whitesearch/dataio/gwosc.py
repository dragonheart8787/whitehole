"""GWOSC (Gravitational-Wave Open Science Center) data interface.

Fetches public strain data via GWPy (which uses the GWOSC HTTP API).
Falls back gracefully when GWPy is not installed.

Typical usage
-------------
>>> loader = GWOSCLoader()
>>> data = loader.load_event("GW150914", detectors=["H1", "L1"])
>>> segment = loader.load_segment("H1", gps_start=1126259462, gps_end=1126259462+4)
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .provenance import DataLoadError

logger = logging.getLogger(__name__)

try:
    from gwpy.timeseries import TimeSeries
    from gwpy.segments import DataQualityFlag
    GWPY_AVAILABLE = True
except ImportError:
    GWPY_AVAILABLE = False
    logger.warning(
        "GWPy not installed; GWOSC data loading will use mock/cached data. "
        "Install with: pip install gwpy"
    )


class GWOSCLoader:
    """Interface for downloading and caching GWOSC public strain data.

    Parameters
    ----------
    cache_dir : Path | str | None
        Directory for caching downloaded files.  Defaults to ``./artifacts/gwosc``.
    """

    # Curated subset validated in CI — not full O1–O4a catalogue coverage
    KNOWN_EVENTS = {
        "GW150914": {"detectors": ["H1", "L1"], "gps": 1126259462.4, "duration": 32},
        "GW151226": {"detectors": ["H1", "L1"], "gps": 1135136350.6, "duration": 32},
        "GW170814": {"detectors": ["H1", "L1", "V1"], "gps": 1186741861.5, "duration": 32},
        "GW200105": {"detectors": ["H1", "L1"], "gps": 1262276512.0, "duration": 32},
    }

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        allow_mock_fallback: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else Path("artifacts/gwosc")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.allow_mock_fallback = allow_mock_fallback

    @classmethod
    def list_validated_events(cls) -> list[str]:
        """Return event names in the curated validation whitelist."""
        return list(cls.KNOWN_EVENTS.keys())

    # ── Event loading ─────────────────────────────────────────────────────────

    def load_event(
        self,
        event_name: str,
        detectors: list[str] | None = None,
        duration: float = 32.0,
        sample_rate: float = 4096.0,
    ) -> dict[str, dict[str, Any]]:
        """Load strain and metadata for a named GW event.

        Returns
        -------
        dict[detector → dict]
            Each value contains 'strain' [ndarray], 'times' [ndarray],
            'sample_rate', 'gps_start', 'checksum', 'dq_flags'.
        """
        if event_name not in self.KNOWN_EVENTS:
            raise ValueError(
                f"Event {event_name!r} not in known list. "
                f"Available: {list(self.KNOWN_EVENTS)}"
            )
        meta = self.KNOWN_EVENTS[event_name]
        detectors = detectors or meta["detectors"]
        gps_center = meta["gps"]
        gps_start = gps_center - duration / 2
        gps_end = gps_center + duration / 2

        result = {}
        for det in detectors:
            result[det] = self._load_strain(
                det, gps_start, gps_end, sample_rate, event_name
            )
        return result

    def load_segment(
        self,
        detector: str,
        gps_start: float,
        gps_end: float,
        sample_rate: float = 4096.0,
        label: str = "custom",
    ) -> dict[str, Any]:
        """Load an arbitrary GPS time segment for a detector."""
        return self._load_strain(detector, gps_start, gps_end, sample_rate, label)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_strain(
        self,
        detector: str,
        gps_start: float,
        gps_end: float,
        sample_rate: float,
        label: str,
    ) -> dict[str, Any]:
        cache_path = self._cache_path(detector, gps_start, gps_end, sample_rate)

        if cache_path.exists():
            logger.info("Loading from cache: %s", cache_path)
            return self._load_cache(cache_path)

        if not GWPY_AVAILABLE:
            reason = "GWPy not installed"
            if not self.allow_mock_fallback:
                raise DataLoadError(
                    f"Cannot load GWOSC strain for {detector}: {reason}. "
                    "Install gwpy or pass allow_mock_fallback=True (CLI: --allow-mock-fallback).",
                    requested_source="GWOSC",
                    reason=reason,
                )
            logger.warning("GWPy unavailable; MOCK_FALLBACK for %s.", detector)
            return self._mock_strain_fallback(
                detector, gps_start, gps_end, sample_rate, reason=reason
            )

        try:
            logger.info("Downloading %s %s–%s from GWOSC …", detector, gps_start, gps_end)
            ts = TimeSeries.fetch_open_data(
                detector,
                gps_start,
                gps_end,
                sample_rate=sample_rate,
                verbose=False,
            )
            strain = ts.value.astype(np.float64)
            times = ts.times.value.astype(np.float64)
            dq_flags = self._get_dq_flags(detector, gps_start, gps_end)
            checksum = self._checksum(strain)

            record = {
                "strain": strain,
                "times": times,
                "sample_rate": sample_rate,
                "gps_start": gps_start,
                "gps_end": gps_end,
                "detector": detector,
                "checksum": checksum,
                "dq_flags": dq_flags,
                "source": "GWOSC",
            }
            self._save_cache(cache_path, record)
            return record

        except Exception as exc:
            reason = f"GWOSC download failed: {exc}"
            if not self.allow_mock_fallback:
                raise DataLoadError(
                    reason,
                    requested_source="GWOSC",
                    reason=str(exc),
                ) from exc
            logger.error("%s; using MOCK_FALLBACK.", reason)
            return self._mock_strain_fallback(
                detector, gps_start, gps_end, sample_rate, reason=str(exc)
            )

    def _get_dq_flags(
        self,
        detector: str,
        gps_start: float,
        gps_end: float,
    ) -> dict[str, list]:
        """Fetch data quality flags for the segment."""
        if not GWPY_AVAILABLE:
            return {"science_mode": [], "notes": "GWPy not available"}
        try:
            flag = DataQualityFlag.query_dqsegdb2(
                f"{detector}:DATA",
                gps_start,
                gps_end,
            )
            return {
                "active_segments": [(s.start, s.end) for s in flag.active],
                "flag_name": flag.name,
            }
        except Exception as exc:
            logger.warning("DQ flag fetch failed: %s", exc)
            return {"error": str(exc)}

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _cache_path(
        self, detector: str, gps_start: float, gps_end: float, sr: float
    ) -> Path:
        tag = f"{detector}_{gps_start:.1f}_{gps_end:.1f}_{sr:.0f}"
        return self.cache_dir / f"{tag}.npz"

    @staticmethod
    def _checksum(arr: NDArray) -> str:
        return hashlib.sha256(arr.tobytes()).hexdigest()[:16]

    def _save_cache(self, path: Path, record: dict[str, Any]) -> None:
        np.savez_compressed(
            path,
            strain=record["strain"],
            times=record["times"],
            metadata=json.dumps(
                {k: v for k, v in record.items() if k not in ("strain", "times")}
            ),
        )
        logger.info("Cached to %s", path)

    def _load_cache(self, path: Path) -> dict[str, Any]:
        with np.load(path, allow_pickle=False) as f:
            record = {
                "strain": f["strain"],
                "times": f["times"],
            }
            meta = json.loads(str(f["metadata"]))
            record.update(meta)
        return record

    @staticmethod
    def _mock_strain(
        detector: str,
        gps_start: float,
        gps_end: float,
        sample_rate: float,
    ) -> dict[str, Any]:
        """Return Gaussian white noise (internal/testing only)."""
        record = GWOSCLoader._mock_strain_fallback(
            detector, gps_start, gps_end, sample_rate, reason="internal mock"
        )
        record["source"] = "MOCK"
        return record

    @staticmethod
    def _mock_strain_fallback(
        detector: str,
        gps_start: float,
        gps_end: float,
        sample_rate: float,
        reason: str,
    ) -> dict[str, Any]:
        """Mock strain tagged as MOCK_FALLBACK — must not be used silently."""
        n = int((gps_end - gps_start) * sample_rate)
        rng = np.random.default_rng(int(gps_start) % (2**32))
        strain = rng.standard_normal(n) * 1e-21
        times = np.linspace(gps_start, gps_end, n)
        return {
            "strain": strain,
            "times": times,
            "sample_rate": sample_rate,
            "gps_start": gps_start,
            "gps_end": gps_end,
            "detector": detector,
            "checksum": "mock_fallback",
            "dq_flags": {},
            "source": "MOCK_FALLBACK",
            "requested_source": "GWOSC",
            "fallback_reason": reason,
        }
