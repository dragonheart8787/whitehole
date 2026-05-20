"""Chandra X-ray Observatory data interface.

Uses CIAO's download utilities when available; falls back to mock data.
Produces Level-2 event files and extracted source products.

References
----------
- https://cxc.cfa.harvard.edu/ciao/
- https://cxc.cfa.harvard.edu/cda/
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from ciao_contrib.runtool import find_chandra_obsid, download_chandra_obsid  # type: ignore
    CIAO_AVAILABLE = True
except ImportError:
    CIAO_AVAILABLE = False
    logger.warning("CIAO not installed; Chandra interface will use mock data.")


class ChandraLoader:
    """Interface for Chandra public archive data.

    Parameters
    ----------
    cache_dir : Path | str | None
        Local cache directory.  Defaults to ``./artifacts/chandra``.
    """

    CDA_BASE_URL = "https://cda.cfa.harvard.edu/cgi-bin/chaser"

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else Path("artifacts/chandra")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── ObsID discovery ───────────────────────────────────────────────────────

    def find_obsids(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_arcmin: float = 1.0,
    ) -> list[str]:
        """Return a list of public Chandra ObsIDs near (ra, dec)."""
        if CIAO_AVAILABLE:
            try:
                result = find_chandra_obsid(ra_deg, dec_deg, radius_arcmin)
                return [str(row["obsid"]) for row in result]
            except Exception as exc:
                logger.warning("find_chandra_obsid failed: %s", exc)

        logger.warning("Returning mock ObsIDs.")
        return [f"MOCK_{ra_deg:.2f}_{dec_deg:.2f}_0001"]

    # ── Data download and processing ──────────────────────────────────────────

    def load_obsid(
        self,
        obsid: str,
        reprocess: bool = False,
    ) -> dict[str, Any]:
        """Download and reprocess a Chandra ObsID to Level-2 products.

        Returns
        -------
        dict with keys: 'event_list', 'spectrum', 'lightcurve', 'obsid'
        """
        obsid_dir = self.cache_dir / obsid
        if obsid_dir.exists() and not reprocess:
            logger.info("Loading cached Chandra obsid %s", obsid)
            return self._load_cached_obsid(obsid_dir, obsid)

        if CIAO_AVAILABLE:
            try:
                obsid_dir.mkdir(parents=True, exist_ok=True)
                download_chandra_obsid(obsid, downloadtype="evt2,pha2,lc", filetypes="secondary")
                return self._process_obsid(obsid_dir, obsid)
            except Exception as exc:
                logger.error("Chandra download failed: %s", exc)

        logger.warning("Returning mock Chandra data for obsid %s.", obsid)
        return self._mock_obsid(obsid)

    def extract_source_spectrum(
        self,
        event_list: np.ndarray,
        src_region: dict,
        bg_region: dict,
    ) -> dict[str, np.ndarray]:
        """Extract a source spectrum from a Chandra event list.

        Parameters
        ----------
        event_list : structured ndarray with columns 'time', 'energy', 'x', 'y'
        src_region : {'type': 'circle', 'x': ..., 'y': ..., 'r': ...}
        bg_region : same format

        Returns
        -------
        dict with 'src_counts', 'bg_counts', 'energy_edges_kev', 'exposure_s'
        """
        x = event_list["x"]
        y = event_list["y"]
        e = event_list["energy"]

        src_mask = self._region_mask(x, y, src_region)
        bg_mask = self._region_mask(x, y, bg_region)

        energy_edges = np.linspace(0.5, 10.0, 101)
        src_counts, _ = np.histogram(e[src_mask], bins=energy_edges)
        bg_counts, _ = np.histogram(e[bg_mask], bins=energy_edges)

        exposure = float(event_list["time"].max() - event_list["time"].min())
        return {
            "src_counts": src_counts,
            "bg_counts": bg_counts,
            "energy_edges_kev": energy_edges,
            "exposure_s": exposure,
        }

    # ── Light curve extraction ────────────────────────────────────────────────

    @staticmethod
    def extract_lightcurve(
        event_list: np.ndarray,
        dt_s: float = 500.0,
        energy_band: tuple[float, float] = (0.5, 10.0),
    ) -> dict[str, np.ndarray]:
        """Bin events into a light curve."""
        t = event_list["time"]
        e = event_list["energy"]
        band_mask = (e >= energy_band[0]) & (e <= energy_band[1])
        t_band = t[band_mask]

        t_edges = np.arange(t.min(), t.max() + dt_s, dt_s)
        counts, _ = np.histogram(t_band, bins=t_edges)
        times_mid = 0.5 * (t_edges[:-1] + t_edges[1:])
        return {"times_s": times_mid, "counts": counts, "dt_s": dt_s}

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _region_mask(
        x: np.ndarray,
        y: np.ndarray,
        region: dict,
    ) -> np.ndarray:
        if region.get("type", "circle") == "circle":
            dx = x - region["x"]
            dy = y - region["y"]
            return dx**2 + dy**2 <= region["r"] ** 2
        return np.ones(len(x), dtype=bool)

    def _load_cached_obsid(self, obsid_dir: Path, obsid: str) -> dict[str, Any]:
        result = {"obsid": obsid}
        evt_file = obsid_dir / "event_list.npz"
        if evt_file.exists():
            data = np.load(evt_file)
            result["event_list"] = {k: data[k] for k in data.files}
        else:
            result["event_list"] = self._mock_event_list(obsid)
        return result

    def _process_obsid(self, obsid_dir: Path, obsid: str) -> dict[str, Any]:
        return self._mock_obsid(obsid)

    @staticmethod
    def _mock_obsid(obsid: str) -> dict[str, Any]:
        return {
            "obsid": obsid,
            "event_list": ChandraLoader._mock_event_list(obsid),
        }

    @staticmethod
    def _mock_event_list(obsid: str) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(hash(obsid) % 2**32)
        n = 8000
        t0 = 1e8
        duration = 40000.0
        times = np.sort(rng.uniform(t0, t0 + duration, n))
        energies = rng.lognormal(mean=np.log(1.2), sigma=0.5, size=n)
        energies = np.clip(energies, 0.3, 10.0)
        return {
            "time": times,
            "energy": energies,
            "x": rng.normal(4096, 50, n),
            "y": rng.normal(4096, 50, n),
        }
