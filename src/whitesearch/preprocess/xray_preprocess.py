"""X-ray / gamma-ray data preprocessing pipeline.

Steps
-----
1. Load photon event list (time, energy, sky position)
2. Apply GTI (good time interval) filtering
3. Barycentric correction
4. Source / background region extraction
5. Build light curve and spectrum
6. Build background model
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class XRayPreprocessor:
    """Preprocessing pipeline for X-ray photon event lists."""

    def __init__(
        self,
        energy_band_kev: tuple[float, float] = (0.5, 10.0),
        dt_lc_s: float = 200.0,
        bg_method: str = "offpeak",
    ) -> None:
        self.e_low, self.e_high = energy_band_kev
        self.dt_lc = dt_lc_s
        self.bg_method = bg_method

    def prepare_raw(
        self,
        event_list: dict[str, NDArray],
        gti_segments: list[tuple[float, float]] | None = None,
        src_region: dict | None = None,
        bg_region: dict | None = None,
    ) -> dict[str, Any]:
        """Full preprocessing pipeline for X-ray event list data.

        Parameters
        ----------
        event_list : dict with 'time', 'energy', 'x', 'y' arrays
        gti_segments : list of (t_start, t_stop) tuples; None = use all
        src_region : dict {'type': 'circle', 'x': ..., 'y': ..., 'r': ...}
        bg_region : same format

        Returns
        -------
        dict with lightcurve, spectrum, background model, quality info
        """
        times = np.asarray(event_list["time"], dtype=np.float64)
        energies = np.asarray(event_list.get("energy", event_list.get("energy_kev", [])), dtype=np.float64)
        x = np.asarray(event_list.get("x", np.zeros_like(times)), dtype=np.float64)
        y = np.asarray(event_list.get("y", np.zeros_like(times)), dtype=np.float64)

        # 1. GTI filtering
        if gti_segments is not None:
            gti_mask = self._gti_filter(times, gti_segments)
        else:
            gti_mask = np.ones(len(times), dtype=bool)

        t_gti = times[gti_mask]
        e_gti = energies[gti_mask]
        x_gti = x[gti_mask]
        y_gti = y[gti_mask]

        # 2. Energy band selection
        band_mask = (e_gti >= self.e_low) & (e_gti <= self.e_high)

        # 3. Region selection
        if src_region is not None:
            src_mask = self._region_mask(x_gti, y_gti, src_region) & band_mask
            bg_mask = self._region_mask(x_gti, y_gti, bg_region) & band_mask if bg_region else ~self._region_mask(x_gti, y_gti, src_region) & band_mask
        else:
            src_mask = band_mask
            bg_mask = np.zeros(len(t_gti), dtype=bool)

        # 4. Light curve
        lc = self._make_lightcurve(t_gti[src_mask], t_gti)
        bg_lc = self._make_lightcurve(t_gti[bg_mask], t_gti)

        # 5. Background model
        bg_model = self._estimate_background(lc["counts"], lc["times"], bg_lc)

        # 6. Spectrum
        spectrum = self._make_spectrum(e_gti[src_mask], e_gti[bg_mask])

        quality = {
            "n_events_total": len(times),
            "n_events_gti": int(np.sum(gti_mask)),
            "n_events_src_band": int(np.sum(src_mask)),
            "gti_fraction": float(np.sum(gti_mask)) / max(len(times), 1),
        }

        return {
            "lightcurve": lc,
            "background_lc": bg_lc,
            "background_model": bg_model,
            "spectrum": spectrum,
            "quality": quality,
        }

    def _make_lightcurve(
        self,
        times_src: NDArray,
        times_all: NDArray,
    ) -> dict[str, NDArray]:
        if len(times_src) == 0 or len(times_all) == 0:
            return {"times": np.array([]), "counts": np.array([]), "dt_s": self.dt_lc}
        t_edges = np.arange(times_all.min(), times_all.max() + self.dt_lc, self.dt_lc)
        counts, _ = np.histogram(times_src, bins=t_edges)
        times_mid = 0.5 * (t_edges[:-1] + t_edges[1:])
        return {"times": times_mid, "counts": counts, "dt_s": self.dt_lc}

    @staticmethod
    def _make_spectrum(
        src_energies: NDArray,
        bg_energies: NDArray,
        n_bins: int = 50,
    ) -> dict[str, NDArray]:
        e_min, e_max = 0.3, 12.0
        edges = np.logspace(np.log10(e_min), np.log10(e_max), n_bins + 1)
        src_counts, _ = np.histogram(src_energies, bins=edges)
        bg_counts, _ = np.histogram(bg_energies, bins=edges)
        return {
            "energy_edges_kev": edges,
            "src_counts": src_counts,
            "bg_counts": bg_counts,
        }

    def _estimate_background(
        self,
        counts: NDArray,
        times: NDArray,
        bg_lc: dict,
    ) -> NDArray:
        """Estimate background rate using off-peak or polynomial method."""
        if len(counts) == 0:
            return np.array([])

        if self.bg_method == "offpeak":
            if len(bg_lc["counts"]) > 0:
                bg_rate = np.mean(bg_lc["counts"]) / self.dt_lc
            else:
                bg_rate = np.median(counts)
            return np.full(len(counts), bg_rate * self.dt_lc)

        elif self.bg_method == "polynomial":
            if len(times) < 4:
                return np.full(len(counts), np.median(counts))
            from numpy.polynomial import polynomial as P
            coeff = P.polyfit(times, counts.astype(float), deg=2)
            return P.polyval(times, coeff)

        return np.full(len(counts), np.median(counts))

    @staticmethod
    def _gti_filter(
        times: NDArray,
        gti_segments: list[tuple[float, float]],
    ) -> NDArray:
        mask = np.zeros(len(times), dtype=bool)
        for t_start, t_stop in gti_segments:
            mask |= (times >= t_start) & (times <= t_stop)
        return mask

    @staticmethod
    def _region_mask(
        x: NDArray,
        y: NDArray,
        region: dict,
    ) -> NDArray:
        if region.get("type", "circle") == "circle":
            dx = x - region.get("x", 0.0)
            dy = y - region.get("y", 0.0)
            return dx**2 + dy**2 <= region.get("r", 10.0) ** 2
        return np.ones(len(x), dtype=bool)

    @staticmethod
    def barycentric_correction(
        times_s: NDArray,
        ra_deg: float,
        dec_deg: float,
        observatory_lat_deg: float = 0.0,
        observatory_lon_deg: float = 0.0,
    ) -> NDArray:
        """Apply a simple barycentric time correction (Earth-frame → solar barycenter).

        For a rigorous correction, use HEASOFT's barycorr or CIAO's axbary tool.
        This is a simplified placeholder.
        """
        try:
            from astropy.coordinates import SkyCoord, EarthLocation
            from astropy.time import Time
            import astropy.units as u_astropy

            loc = EarthLocation(
                lat=observatory_lat_deg * u_astropy.deg,
                lon=observatory_lon_deg * u_astropy.deg,
                height=0.0 * u_astropy.m,
            )
            t = Time(times_s, format="unix", scale="tt", location=loc)
            coord = SkyCoord(ra=ra_deg, dec=dec_deg, unit="deg")
            bary_corr = t.light_travel_time(coord, "barycentric").to(u_astropy.s).value
            return times_s + bary_corr
        except Exception as exc:
            logger.warning("Barycentric correction failed: %s. Returning uncorrected times.", exc)
            return times_s
