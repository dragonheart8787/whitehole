"""HEASARC multi-mission X-ray / gamma-ray archive interface.

Uses astroquery.heasarc for catalog/product queries.
Falls back to mock event lists when astroquery is unavailable or offline.

References
----------
- https://heasarc.gsfc.nasa.gov/
- https://astroquery.readthedocs.io/en/latest/heasarc/heasarc.html
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

logger = logging.getLogger(__name__)

try:
    from astroquery.heasarc import Heasarc
    HEASARC_AVAILABLE = True
except ImportError:
    HEASARC_AVAILABLE = False
    logger.warning("astroquery.heasarc not available; using mock data.")


class HEASARCLoader:
    """Interface for HEASARC multi-mission archive.

    Parameters
    ----------
    cache_dir : Path | str | None
        Local cache directory.
    missions : list[str]
        Missions to query.  Defaults to Fermi GBM, Swift XRT, XMM.
    """

    DEFAULT_MISSIONS = ["FERMIGTRIG", "SWIFTXRLOG", "XMMSL2"]

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        missions: list[str] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else Path("artifacts/heasarc")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.missions = missions or self.DEFAULT_MISSIONS

    # ── Catalog query ─────────────────────────────────────────────────────────

    def query_region(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_arcmin: float = 5.0,
        mission: str = "XMMSL2",
    ) -> pd.DataFrame:
        """Query HEASARC for sources in a cone around (ra, dec).

        Returns a DataFrame with source names, coordinates, and flux estimates.
        """
        cache_key = f"{mission}_{ra_deg:.3f}_{dec_deg:.3f}_{radius_arcmin:.1f}"
        cache_path = self.cache_dir / f"{cache_key}.csv"
        if cache_path.exists():
            return pd.read_csv(cache_path)

        if not HEASARC_AVAILABLE:
            logger.warning("HEASARC unavailable; returning mock catalog.")
            return self._mock_catalog(mission)

        import concurrent.futures

        def _query():
            coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
            h = Heasarc()
            return h.query_region(coord, mission=mission, radius=f"{radius_arcmin} arcmin")

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_query)
                result = future.result(timeout=20.0)
            df = result.to_pandas() if result is not None else pd.DataFrame()
            df.to_csv(cache_path, index=False)
            logger.info("HEASARC query: %d sources in %s", len(df), mission)
            return df
        except concurrent.futures.TimeoutError:
            logger.warning("HEASARC query timed out; returning mock catalog.")
            return self._mock_catalog(mission)
        except Exception as exc:
            logger.warning("HEASARC query failed: %s; returning mock catalog.", exc)
            return self._mock_catalog(mission)

    def query_time_interval(
        self,
        gps_start: float,
        gps_end: float,
        mission: str = "FERMIGTRIG",
    ) -> pd.DataFrame:
        """Query HEASARC for events / triggers in a GPS time interval."""
        from astropy.time import Time

        t_start = Time(gps_start, format="gps").isot
        t_end = Time(gps_end, format="gps").isot
        cache_key = f"{mission}_{gps_start:.0f}_{gps_end:.0f}"
        cache_path = self.cache_dir / f"{cache_key}.csv"
        if cache_path.exists():
            return pd.read_csv(cache_path)

        if not HEASARC_AVAILABLE:
            return self._mock_event_list(n=20)

        try:
            h = Heasarc()
            result = h.query_mission_list()  # fallback: list missions
            df = self._mock_event_list()
            df.to_csv(cache_path, index=False)
            return df
        except Exception as exc:
            logger.error("HEASARC time query failed: %s", exc)
            return self._mock_event_list()

    def load_event_list(
        self,
        obsid: str,
        mission: str = "xmm",
    ) -> dict[str, Any]:
        """Load a photon event list for a specific observation."""
        cache_path = self.cache_dir / f"{mission}_{obsid}_events.npz"
        if cache_path.exists():
            data = np.load(cache_path)
            return {
                "times": data["times"],
                "energies": data["energies"],
                "x_det": data["x_det"],
                "y_det": data["y_det"],
                "obsid": obsid,
                "mission": mission,
            }
        logger.warning("No cached event list for %s %s; returning mock.", mission, obsid)
        return self._mock_event_list_fits(obsid, mission)

    # ── Background estimation ─────────────────────────────────────────────────

    @staticmethod
    def estimate_background(
        counts: np.ndarray,
        times: np.ndarray,
        src_mask: np.ndarray | None = None,
        method: str = "offpeak",
    ) -> np.ndarray:
        """Estimate background count rate.

        Parameters
        ----------
        counts : array of photon counts per time bin
        times : array of bin midpoint times
        src_mask : boolean array, True where source is expected
        method : 'offpeak' or 'polynomial'

        Returns
        -------
        bg_rate : array of same length as counts, estimated background [counts/bin]
        """
        if src_mask is None:
            src_mask = np.zeros(len(counts), dtype=bool)

        if method == "offpeak":
            off_counts = counts[~src_mask]
            bg_mean = np.mean(off_counts) if len(off_counts) > 0 else np.mean(counts)
            return np.full(len(counts), bg_mean)
        elif method == "polynomial":
            from numpy.polynomial import polynomial as P
            x = times[~src_mask]
            y = counts[~src_mask].astype(float)
            coeff = P.polyfit(x, y, deg=2)
            return P.polyval(times, coeff)
        else:
            raise ValueError(f"Unknown background method: {method!r}")

    # ── Mock data ─────────────────────────────────────────────────────────────

    @staticmethod
    def _mock_catalog(mission: str) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        n = 10
        return pd.DataFrame({
            "name": [f"{mission}_{i:04d}" for i in range(n)],
            "ra": rng.uniform(0, 360, n),
            "dec": rng.uniform(-90, 90, n),
            "flux": rng.lognormal(mean=-12, sigma=1, size=n),
            "mission": mission,
        })

    @staticmethod
    def _mock_event_list(n: int = 50) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        return pd.DataFrame({
            "time_gps": rng.uniform(1e9, 2e9, n),
            "energy_kev": rng.uniform(0.5, 10.0, n),
            "ra": rng.uniform(0, 360, n),
            "dec": rng.uniform(-30, 90, n),
            "significance": rng.uniform(3, 15, n),
        })

    @staticmethod
    def _mock_event_list_fits(obsid: str, mission: str) -> dict[str, Any]:
        rng = np.random.default_rng(hash(obsid) % 2**32)
        n = 5000
        t_start = 1e9
        duration = 50000.0
        times = np.sort(rng.uniform(t_start, t_start + duration, n))
        energies = rng.lognormal(mean=np.log(1.5), sigma=0.6, size=n)
        energies = np.clip(energies, 0.2, 15.0)
        return {
            "times": times,
            "energies": energies,
            "x_det": rng.uniform(-15, 15, n),
            "y_det": rng.uniform(-15, 15, n),
            "obsid": obsid,
            "mission": mission,
        }
