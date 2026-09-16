"""Event Horizon Telescope (EHT) data interface — Phase 2.

Handles the 2017 EHT L1 public data release (M87* and Sgr A*).
Data is available from CyVerse / ALMA Science Portal.

References
----------
- EHT Collaboration (2019), ApJL 875 L1-L6 (M87* papers)
- EHT Data Release: https://eventhorizontelescope.org/for-scientists/data
- ehtim library: https://github.com/achael/eht-imaging
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils.targets import resolve_target_name

logger = logging.getLogger(__name__)

try:
    import ehtim as eh  # type: ignore
    EHTIM_AVAILABLE = True
except ImportError:
    EHTIM_AVAILABLE = False
    logger.warning(
        "ehtim not installed; EHT interface will use mock visibility data. "
        "Install with: pip install ehtim"
    )

# Public EHT 2017 data URLs (illustrative — real data requires registration)
EHT_DATA_URLS = {
    "M87_2017_LO": "https://eventhorizontelescope.org/files/eht/files/SR1_M87_2017_095_lo.uvfits",
    "M87_2017_HI": "https://eventhorizontelescope.org/files/eht/files/SR1_M87_2017_095_hi.uvfits",
}

EHT_2017_PARAMS = {
    "M87": {
        "freq_ghz": 230.0,
        "bandwidth_ghz": 2.0,
        "fov_muas": 200.0,
        "stations": ["ALMA", "APEX", "JCMT", "LMT", "SMA", "SMTO", "SPT", "IRAM30"],
    },
}


class EHTLoader:
    """Interface for EHT public data products.

    Phase 2 component — intended for use after the GW and radio pipelines
    are validated.

    Parameters
    ----------
    cache_dir : Path | str | None
        Local cache directory.
    """

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else Path("artifacts/eht")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Data loading ──────────────────────────────────────────────────────────

    def load_uvfits(
        self,
        source: str = "M87",
        year: int = 2017,
        band: str = "LO",
    ) -> dict[str, Any]:
        """Load EHT L1 UVFITS data.

        Returns a dict with visibilities, (u,v) coordinates, baseline metadata,
        frequency information, and closure quantities.

        Parameters
        ----------
        source : str — 'M87' or 'SgrA'
        year : int — observation year (2017, 2018, 2021)
        band : str — 'LO' or 'HI'

        Notes
        -----
        The returned dict carries a ``target`` key with the canonical name
        ('M87*' / 'SgrA*').  The image forward model needs it: it holds the
        source distance fixed rather than sampling it, and reads that distance
        from the target.  Unknown source names raise here rather than producing
        a record no downstream analysis can interpret.

        The pre-existing ``source`` key also holds the target name, which
        collides with the provenance meaning ``source`` carries on the GW and
        radio paths ('GWOSC', 'MOCK_EXPLICIT', ...).  It is left in place for
        backwards compatibility; ``target`` is the key the forward model reads.
        """
        target = resolve_target_name(source)
        key = f"{source}_{year}_{band}"
        cache_path = self.cache_dir / f"{key}.npz"
        if cache_path.exists():
            return self._load_cache(cache_path)

        uvfits_path = self.cache_dir / f"{key}.uvfits"
        if EHTIM_AVAILABLE and uvfits_path.exists():
            return self._load_ehtim(uvfits_path, source, target)

        logger.warning(
            "EHT L1 data for %s not found locally. "
            "Download from https://eventhorizontelescope.org/for-scientists/data . "
            "Returning mock visibility data.",
            key,
        )
        return self._mock_eht_data(source, year, band, target)

    def load_from_file(self, filepath: str | Path, target: str) -> dict[str, Any]:
        """Load from a local UVFITS or HDF5 file.

        ``target`` is required, not inferred from the file: the forward model
        turns it into the source distance, and a file that says nothing about
        which source it holds cannot be analysed without being told.
        """
        canonical = resolve_target_name(target)
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"EHT data file not found: {path}")
        if EHTIM_AVAILABLE:
            return self._load_ehtim(path, source=target, target=canonical)
        return self._parse_uvfits_minimal(path, canonical)

    # ── Closure quantities ────────────────────────────────────────────────────

    @staticmethod
    def compute_closure_phases(
        visibilities: np.ndarray,
        baseline_ids: np.ndarray,
        station_list: list[str],
    ) -> dict[str, np.ndarray]:
        """Compute closure phases for all independent triangles.

        Closure phase φ_{ijk} = arg(V_{ij}) + arg(V_{jk}) − arg(V_{ik})
        is insensitive to station-based gain errors.

        Returns dict with 'triangles' (list of tuples) and 'closure_phases' (array).
        """
        n_stations = len(station_list)
        triangles = []
        phases = []

        phase_array = np.angle(visibilities)
        for i in range(n_stations - 2):
            for j in range(i + 1, n_stations - 1):
                for k in range(j + 1, n_stations):
                    cp = phase_array[i] + phase_array[j] - phase_array[k]
                    cp = (cp + np.pi) % (2 * np.pi) - np.pi  # wrap
                    triangles.append((station_list[i], station_list[j], station_list[k]))
                    phases.append(cp)

        return {
            "triangles": triangles,
            "closure_phases": np.array(phases),
        }

    @staticmethod
    def compute_closure_amplitudes(
        visibilities: np.ndarray,
    ) -> np.ndarray:
        """Compute closure amplitudes for sequential quadrangles."""
        n = len(visibilities)
        amps = []
        abs_vis = np.abs(visibilities)
        for k in range(n // 4):
            i, j, l, m = 4 * k, 4 * k + 1, 4 * k + 2, 4 * k + 3
            with np.errstate(divide="ignore"):
                ca = (abs_vis[i] * abs_vis[l]) / (abs_vis[j] * abs_vis[m] + 1e-30)
            amps.append(float(ca))
        return np.array(amps)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_ehtim(self, path: Path, source: str, target: str) -> dict[str, Any]:
        try:
            obs = eh.obsdata.load_uvfits(str(path))
            u = obs.data["u"]
            v = obs.data["v"]
            vis = obs.data["vis"]
            sigma = obs.data["sigma"]
            return {
                "u": u,
                "v": v,
                "visibilities": vis,
                "sigma": sigma,
                "freq_ghz": obs.rf / 1e9,
                "source": source,
                "target": target,
                "stations": obs.tarr["site"].tolist(),
                "uv_coverage": np.column_stack([u, v]),
            }
        except Exception as exc:
            logger.error("ehtim load failed: %s", exc)
            return self._mock_eht_data(source, 2017, "LO", target)

    def _load_cache(self, path: Path) -> dict[str, Any]:
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    @staticmethod
    def _mock_eht_data(
        source: str, year: int, band: str, target: str | None = None
    ) -> dict[str, Any]:
        """Return synthetic EHT-like visibility data for testing.

        ``target`` is resolved from ``source`` when not given, and set
        explicitly on the record: mock data goes through the same forward model
        as real data, so it needs the same per-target distance and must fail
        the same way when the source is not a known target.
        """
        canonical = resolve_target_name(target if target is not None else source)
        from ..simulators.image_shadow import _default_eht_uv

        rng = np.random.default_rng(hash(f"{source}{year}{band}") % 2**32)
        uv = _default_eht_uv()
        n_baselines = len(uv)

        # Mock M87*-like ring: shadow ~ 40 μas
        ring_size_muas = 40.0 if canonical == "M87*" else 50.0
        from ..utils.constants import MUAS_RAD
        ring_rad = ring_size_muas * MUAS_RAD

        freq_hz = 230e9
        from ..utils.constants import C
        wavelength_m = C / freq_hz
        uv_rad = uv * 1e9 * wavelength_m

        vis_amp = np.sinc(uv_rad[:, 0] * ring_rad) * np.sinc(uv_rad[:, 1] * ring_rad)
        vis_phase = rng.uniform(-np.pi, np.pi, n_baselines)
        visibilities = vis_amp * np.exp(1j * vis_phase)
        sigma = rng.uniform(0.02, 0.1, n_baselines)
        noise = rng.standard_normal(n_baselines) + 1j * rng.standard_normal(n_baselines)
        visibilities += sigma * noise

        return {
            "u": uv[:, 0],
            "v": uv[:, 1],
            "visibilities": visibilities,
            "sigma": sigma,
            "freq_ghz": 230.0,
            "source": source,
            "target": canonical,
            "stations": ["ALMA", "SMA", "JCMT", "SMTO", "LMT", "IRAM30", "SPT", "APEX"],
            "uv_coverage": uv,
        }

    @staticmethod
    def _parse_uvfits_minimal(path: Path, target: str) -> dict[str, Any]:
        """Minimal UVFITS parser (fallback without ehtim)."""
        try:
            from astropy.io import fits
            with fits.open(path) as hdul:
                data = hdul[0].data
                return {
                    "u": data["UU"] if "UU" in data.dtype.names else np.array([]),
                    "v": data["VV"] if "VV" in data.dtype.names else np.array([]),
                    "source": "custom",
                    "target": target,
                }
        except Exception as exc:
            logger.error("Minimal UVFITS parse failed: %s", exc)
            return {}
