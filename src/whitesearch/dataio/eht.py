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
from typing import Any, Sequence

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


#: Where the EHT instrument configuration lives, relative to the repo root.
EHT_CONFIG_PATH = Path("configs/instruments/eht.yaml")


#: Mean Earth radius [m].  The station array is built on a sphere rather than
#: on a full geodetic ellipsoid: the closure relation this exists to guarantee
#: is a property of the BASELINE ALGEBRA (u is a linear map of the baseline
#: vector, and vectors close by construction), not of the Earth model, so the
#: extra precision would buy nothing for it.
EARTH_RADIUS_M = 6_371_000.0

#: Declination used to project the default array, in degrees.  M87*'s, from
#: ``configs/instruments/eht.yaml``.  The projection is linear in the baseline
#: vector for ANY fixed (hour angle, declination), so this choice sets where
#: the baselines land in the uv plane but cannot break closure.
DEFAULT_DEC_DEG = 12.391
DEFAULT_HOUR_ANGLE_HR = 0.0


def eht_station_config() -> tuple[list[str], np.ndarray]:
    """Station names and geocentric XYZ [m], from the instrument config.

    Reads ``configs/instruments/eht.yaml`` -- the same single source of truth
    the imaging grid comes from -- so the array is not a second in-code copy
    that can drift from it (the failure mode audit X.13.2 records).
    """
    import yaml

    for base in (Path.cwd(), Path(__file__).resolve().parents[3]):
        path = base / EHT_CONFIG_PATH
        if not path.exists():
            continue
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        stations = cfg.get("stations")
        if not stations:
            raise KeyError(f"{path} has no 'stations' block")
        names = sorted(stations)
        lat = np.radians([float(stations[n]["latitude_deg"]) for n in names])
        lon = np.radians([float(stations[n]["longitude_deg"]) for n in names])
        xyz = EARTH_RADIUS_M * np.column_stack([
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        ])
        return names, xyz
    raise FileNotFoundError(
        f"EHT instrument config {EHT_CONFIG_PATH} not found; it is the single "
        "source of truth for the station array."
    )


def eht_station_uv(
    dec_deg: float = DEFAULT_DEC_DEG,
    hour_angle_hr: float = DEFAULT_HOUR_ANGLE_HR,
    freq_ghz: float = 230.0,
) -> tuple[np.ndarray, list[tuple[int, int]], list[str]]:
    """Derive (u, v) [Gλ] from the STATION ARRAY, one baseline per station pair.

    Returns ``(uv, pairs, station_names)`` where ``pairs[b] = (i, j)`` names the
    two stations of baseline ``b``.

    This is what makes closure phases real (decision I-5).  The previous
    ``_default_eht_uv()`` listed baselines directly, so no three of them formed
    a triangle: ``u_ij + u_jk != u_ik``, and the "closure phase" computed from
    them carried none of the station-gain invariance the observable exists for.
    Here ``B_ij = X_i - X_j`` by construction, and the projection

        u = ( sin H  Bx + cos H  By) / λ
        v = (-sin δ cos H  Bx + sin δ sin H  By + cos δ  Bz) / λ

    is LINEAR in ``B``, so ``u_ij + u_jk = u_ik`` holds exactly, for every
    triangle, at machine precision.

    KNOWN SIMPLIFICATION, recorded rather than silently tolerated: this is a
    snapshot at a single hour angle with no elevation/visibility cut, so some
    pairs would not in reality see the source simultaneously.  That affects
    which baselines exist, not whether they close.
    """
    names, xyz = eht_station_config()
    wavelength_m = 299_792_458.0 / (float(freq_ghz) * 1e9)
    dec = np.radians(float(dec_deg))
    ha = np.radians(float(hour_angle_hr) * 15.0)

    pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]
    b = np.array([xyz[i] - xyz[j] for i, j in pairs])
    u = (np.sin(ha) * b[:, 0] + np.cos(ha) * b[:, 1]) / wavelength_m
    v = (
        -np.sin(dec) * np.cos(ha) * b[:, 0]
        + np.sin(dec) * np.sin(ha) * b[:, 1]
        + np.cos(dec) * b[:, 2]
    ) / wavelength_m
    return np.column_stack([u, v]) / 1e9, pairs, names


def independent_triangles(n_stations: int) -> list[tuple[int, int, int]]:
    """Independent closure triangles: every triple that contains station 0.

    For N stations there are C(N,3) triangles but only (N-1)(N-2)/2 independent
    ones; fixing a reference station picks exactly that many, so the likelihood
    does not count the same phase information several times.
    """
    return [(0, j, k)
            for j in range(1, n_stations)
            for k in range(j + 1, n_stations)]


def baseline_index(pairs: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    """Map a station pair to its baseline row (both orderings)."""
    idx: dict[tuple[int, int], int] = {}
    for b, (i, j) in enumerate(pairs):
        idx[(i, j)] = b
        idx[(j, i)] = b
    return idx


def eht_imaging_config() -> dict[str, Any]:
    """Imaging grid and noise level from ``configs/instruments/eht.yaml``.

    One reader, so the field of view and pixel count are not written down twice.
    ``cli.py`` used to carry its own ``n_pixels: 64`` next to the YAML's 128;
    at 64 px the representability floor is 51.53 μas and *neither* target's
    ring can be represented at all, so the two copies did not merely differ,
    they disagreed about whether the channel worked.

    Fail-closed on a missing or malformed file rather than falling back to
    in-code numbers: a silent fallback is how the two copies drifted apart.
    """
    import yaml

    for base in (Path.cwd(), Path(__file__).resolve().parents[3]):
        path = base / EHT_CONFIG_PATH
        if path.exists():
            cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            imaging = cfg.get("imaging")
            if not imaging:
                raise KeyError(f"{path} has no 'imaging' block")
            return {
                "fov_muas": float(imaging["fov_muas"]),
                "n_pixels": int(imaging["n_pixels"]),
                "thermal_noise_jy": float(imaging["thermal_noise_jy"]),
                "freq_ghz": float(cfg["observing_frequency_ghz"]),
            }
    raise FileNotFoundError(
        f"EHT instrument config {EHT_CONFIG_PATH} not found; it is the single "
        "source of truth for the image channel's imaging grid."
    )


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
        baseline_pairs: Sequence[tuple[int, int]],
        station_list: list[str],
    ) -> dict[str, np.ndarray]:
        """Closure phases over independent station triangles (decision I-5).

        φ_ijk = arg(V_ij) + arg(V_jk) − arg(V_ik), which is invariant under any
        per-station complex gain: g_i contributes +arg g_i − arg g_j to V_ij and
        the three contributions cancel around the triangle.

        FIXED IN I-5: this used to index ``visibilities`` by STATION number
        (``phase_array[i]`` for station ``i``), but visibilities are per
        BASELINE, so it read three unrelated baselines and the result was not a
        closure phase at all.  ``baseline_pairs[b] = (i, j)`` now says which two
        stations baseline ``b`` joins, and the lookup goes through that.

        Returns ``{'triangles': [(name_i, name_j, name_k), ...],
        'closure_phases': array}``.
        """
        index = baseline_index([(int(i), int(j)) for i, j in baseline_pairs])
        phase_array = np.angle(visibilities)
        triangles, phases = [], []
        for i, j, k in independent_triangles(len(station_list)):
            cp = (phase_array[index[(i, j)]]
                  + phase_array[index[(j, k)]]
                  - phase_array[index[(i, k)]])
            triangles.append((station_list[i], station_list[j], station_list[k]))
            phases.append((cp + np.pi) % (2 * np.pi) - np.pi)
        return {"triangles": triangles, "closure_phases": np.array(phases)}

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

        rng = np.random.default_rng(hash(f"{source}{year}{band}") % 2**32)
        # Decision I-5: the mock record now carries a real station array, so the
        # closure phases computed from it are closure phases.
        uv, station_pairs, station_names = eht_station_uv()
        n_baselines = len(uv)

        # Measured ring DIAMETERS: 42 muas for M87* (EHT 2019, ApJL 875 L1) and
        # 51.8 muas for Sgr A* (EHT 2022, ApJL 930 L12).
        ring_diameter_muas = 42.0 if canonical == "M87*" else 51.8
        from ..utils.constants import MUAS_RAD
        ring_radius_rad = 0.5 * ring_diameter_muas * MUAS_RAD

        # Gλ -> λ (fringe cycles per radian).  This used to multiply by the
        # observing wavelength as well, which converts a baseline in Gλ into its
        # physical length in metres and shrank every baseline by 767x at
        # 230 GHz; the identical error was in
        # simulators.image_shadow._compute_visibilities, so the two agreed with
        # each other and no forward-model consistency check could see it.
        # See docs/XRAY_IMAGE_PREFLIGHT_AUDIT.md X.13.
        uv_lambda = uv * 1e9
        uv_radial = np.hypot(uv_lambda[:, 0], uv_lambda[:, 1])

        # Thin circular ring of radius r: V(u) = F J0(2 pi r u).  Previously a
        # separable product of two sincs, which is the transform of a rectangle,
        # not of the ring this record claims to hold.  That only became visible
        # once the baselines stopped landing on the uv origin, where every
        # transform equals the zero-spacing flux.
        from scipy.special import j0
        total_flux_jy = 1.0  # M87* compact flux at 230 GHz is ~0.5-1.2 Jy
        vis_amp = total_flux_jy * j0(2.0 * np.pi * ring_radius_rad * uv_radial)
        vis_phase = rng.uniform(-np.pi, np.pi, n_baselines)
        visibilities = vis_amp * np.exp(1j * vis_phase)
        sigma = rng.uniform(0.02, 0.1, n_baselines)
        noise = rng.standard_normal(n_baselines) + 1j * rng.standard_normal(n_baselines)
        visibilities += sigma * noise

        return {
            "u": uv[:, 0],
            "v": uv[:, 1],
            "uv_coverage": uv,
            "station_pairs": station_pairs,
            "stations": station_names,
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
