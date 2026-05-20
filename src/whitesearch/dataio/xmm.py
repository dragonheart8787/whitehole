"""XMM-Newton Science Archive interface.

Uses astroquery.esa.xmm_newton when available for PPS/ODF downloads.
Falls back to mock data for development and CI testing.

References
----------
- https://www.cosmos.esa.int/web/xmm-newton/xsa
- https://astroquery.readthedocs.io/en/latest/esa/xmm_newton/xmm_newton.html
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from astroquery.esa.xmm_newton import XMMNewton  # type: ignore
    XMM_AVAILABLE = True
except ImportError:
    XMM_AVAILABLE = False
    logger.warning("astroquery.esa.xmm_newton not available; using mock data.")


class XMMLoader:
    """Interface for XMM-Newton Science Archive (XSA).

    Parameters
    ----------
    cache_dir : Path | str | None
        Local cache directory.
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        auto_download: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        auto_download : bool
            If True, attempt to download data from XSA when not in cache.
            Default False to avoid hanging on network unavailability.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("artifacts/xmm")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.auto_download = auto_download

    # ── Observation query ─────────────────────────────────────────────────────

    def query_region(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_arcmin: float = 5.0,
    ) -> pd.DataFrame:
        """Query XSA for public observations near a sky position."""
        if XMM_AVAILABLE and self.auto_download:
            import concurrent.futures

            def _query():
                return XMMNewton.query_region(
                    f"{ra_deg} {dec_deg}",
                    radius=radius_arcmin / 60.0,
                )

            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_query)
                    result = future.result(timeout=15.0)
                return result.to_pandas() if result is not None else pd.DataFrame()
            except Exception as exc:
                logger.warning("XMM query_region failed: %s; using mock data.", exc)
        return self._mock_observation_table()

    def get_observation(
        self,
        observation_id: str,
        level: str = "PPS",
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        """Download PPS/ODF for a given XMM observation ID.

        Parameters
        ----------
        observation_id : str — e.g. '0111970101'
        level : 'PPS' (pipeline products) or 'ODF' (raw data)
        timeout_s : float — network timeout in seconds (default 30)
        """
        cache_path = self.cache_dir / observation_id
        if cache_path.exists():
            logger.info("Loading cached XMM obsid %s", observation_id)
            return self._load_cached(cache_path, observation_id)

        if not self.auto_download:
            logger.info("auto_download=False; returning mock XMM data for %s.", observation_id)
            return self._mock_obsid_data(observation_id)

        if XMM_AVAILABLE:
            import concurrent.futures
            def _download():
                XMMNewton.download_data(observation_id, level=level, filename=str(cache_path))

            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_download)
                    future.result(timeout=timeout_s)
                return self._load_cached(cache_path, observation_id)
            except concurrent.futures.TimeoutError:
                logger.warning("XMM download timed out for %s; using mock data.", observation_id)
            except Exception as exc:
                logger.warning("XMM download failed for %s: %s; using mock data.", observation_id, exc)

        logger.info("Returning mock XMM data for %s.", observation_id)
        return self._mock_obsid_data(observation_id)

    # ── Spectrum / light curve extraction ─────────────────────────────────────

    @staticmethod
    def extract_pps_lightcurve(
        obsid_data: dict[str, Any],
        instrument: str = "EPIC-PN",
        energy_band_kev: tuple[float, float] = (0.5, 10.0),
        dt_s: float = 200.0,
    ) -> dict[str, np.ndarray]:
        """Extract or build a light curve from PPS data.

        Applies soft-proton flare filtering (GTI selection).
        """
        events = obsid_data.get("events", {})
        times = events.get("time", np.array([]))
        energies = events.get("energy_kev", np.array([]))

        if len(times) == 0:
            return {"times_s": np.array([]), "counts": np.array([]), "dt_s": dt_s}

        # GTI filter (mock: remove top 10% background rate intervals)
        gti_mask = XMMLoader._apply_gti_filter(times, energies)
        t_gti = times[gti_mask]
        e_gti = energies[gti_mask]

        band_mask = (e_gti >= energy_band_kev[0]) & (e_gti <= energy_band_kev[1])
        t_band = t_gti[band_mask]

        t_edges = np.arange(t_band.min(), t_band.max() + dt_s, dt_s)
        counts, _ = np.histogram(t_band, bins=t_edges)
        times_mid = 0.5 * (t_edges[:-1] + t_edges[1:])

        return {
            "times_s": times_mid,
            "counts": counts,
            "dt_s": dt_s,
            "instrument": instrument,
            "gti_fraction": float(np.sum(gti_mask)) / max(len(gti_mask), 1),
        }

    @staticmethod
    def extract_spectrum(
        obsid_data: dict[str, Any],
        src_mask: np.ndarray | None = None,
        bg_mask: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Extract source and background spectra from PPS events."""
        events = obsid_data.get("events", {})
        energies = events.get("energy_kev", np.array([]))

        energy_edges = np.linspace(0.3, 12.0, 100)
        if len(energies) == 0:
            n_bins = len(energy_edges) - 1
            return {
                "src_counts": np.zeros(n_bins),
                "bg_counts": np.zeros(n_bins),
                "energy_edges_kev": energy_edges,
            }

        src_e = energies[src_mask] if src_mask is not None else energies
        bg_e = energies[bg_mask] if bg_mask is not None else np.array([])

        src_counts, _ = np.histogram(src_e, bins=energy_edges)
        bg_counts, _ = np.histogram(bg_e, bins=energy_edges)
        return {
            "src_counts": src_counts,
            "bg_counts": bg_counts,
            "energy_edges_kev": energy_edges,
        }

    # ── GTI filtering ─────────────────────────────────────────────────────────

    @staticmethod
    def _apply_gti_filter(
        times: np.ndarray,
        energies: np.ndarray,
        high_e_kev: float = 10.0,
        threshold_factor: float = 1.5,
        dt_s: float = 100.0,
    ) -> np.ndarray:
        """Simple soft-proton flare filter: remove bins where high-E rate > threshold.

        This mimics the standard XMM SAS GTI selection procedure.
        """
        if len(times) == 0:
            return np.ones(0, dtype=bool)

        high_e_mask = energies > high_e_kev
        t_min, t_max = times.min(), times.max()
        bins = np.arange(t_min, t_max + dt_s, dt_s)
        rate_counts, _ = np.histogram(times[high_e_mask], bins=bins)
        median_rate = np.median(rate_counts) + 1e-10
        good_bins = rate_counts < threshold_factor * median_rate

        gti_mask = np.zeros(len(times), dtype=bool)
        for i, ok in enumerate(good_bins):
            if ok:
                in_bin = (times >= bins[i]) & (times < bins[i + 1])
                gti_mask[in_bin] = True
        return gti_mask

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_cached(self, path: Path, obsid: str) -> dict[str, Any]:
        evt_file = path / "events.npz"
        if evt_file.exists():
            data = np.load(evt_file)
            return {"obsid": obsid, "events": {k: data[k] for k in data.files}}
        return self._mock_obsid_data(obsid)

    @staticmethod
    def _mock_obsid_data(obsid: str) -> dict[str, Any]:
        rng = np.random.default_rng(hash(obsid) % 2**32)
        n = 12000
        t0 = 5e8
        duration = 80000.0
        times = np.sort(rng.uniform(t0, t0 + duration, n))
        energies = rng.lognormal(mean=np.log(1.5), sigma=0.6, size=n)
        energies = np.clip(energies, 0.2, 15.0)
        return {
            "obsid": obsid,
            "events": {
                "time": times,
                "energy_kev": energies,
                "x": rng.normal(0, 10, n),
                "y": rng.normal(0, 10, n),
            },
        }

    @staticmethod
    def _mock_observation_table() -> pd.DataFrame:
        return pd.DataFrame({
            "observation_id": ["0111970101", "0149780101", "0300210401"],
            "ra": [10.68, 83.82, 187.71],
            "dec": [41.27, -5.39, 12.39],
            "duration_s": [50000, 30000, 100000],
            "instrument": ["EPIC-PN", "EPIC-PN", "EPIC-PN"],
        })
