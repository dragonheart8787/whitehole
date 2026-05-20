"""CHIME/FRB open data interface.

Downloads and parses the CHIME/FRB Catalog (CSV/FITS) and injection data.
Uses the ``cfod`` package when available, otherwise falls back to direct
HTTP downloads of the public catalog files.

References
----------
- CHIME/FRB Collaboration (2021), ApJS 257 59 (Catalog 1)
- https://www.chime-frb.ca/catalog
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import cfod  # type: ignore
    CFOD_AVAILABLE = True
except ImportError:
    CFOD_AVAILABLE = False
    logger.warning(
        "cfod not installed; CHIME/FRB catalog will be fetched via HTTP. "
        "Install with: pip install cfod"
    )

CATALOG1_URL = (
    "https://raw.githubusercontent.com/CHIMEFRB/"
    "catalog1/main/data/catalog1.csv"
)


class CHIMEFRBLoader:
    """Interface for loading CHIME/FRB public catalog and injection data.

    Parameters
    ----------
    cache_dir : Path | str | None
        Local cache directory.  Defaults to ``./artifacts/chime``.
    """

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else Path("artifacts/chime")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Catalog ───────────────────────────────────────────────────────────────

    def load_catalog(self, version: str = "1") -> pd.DataFrame:
        """Load the public CHIME/FRB catalog as a DataFrame.

        Columns include: ``tns_name``, ``dm``, ``dm_exc_ne2001``, ``sn``,
        ``fluence``, ``width_observed``, ``scattering_time``, ``bc_width``,
        ``ra``, ``dec``, ``is_repeater``, etc.

        Parameters
        ----------
        version : str
            Catalog version, currently only ``"1"`` is publicly available.
        """
        if version != "1":
            raise NotImplementedError("Only Catalog 1 is currently supported.")

        cache_path = self.cache_dir / "catalog1.csv"
        if not cache_path.exists():
            self._download_catalog(cache_path)

        df = pd.read_csv(cache_path)
        df = self._clean_catalog(df)
        logger.info("Loaded CHIME/FRB Catalog 1: %d events", len(df))
        return df

    def load_injection_data(self) -> pd.DataFrame:
        """Load the CHIME/FRB injection calibration data.

        Used for modelling selection effects and survey completeness.
        Falls back to a synthetic injection set when real data is unavailable.
        """
        inj_path = self.cache_dir / "injections.csv"
        if inj_path.exists():
            return pd.read_csv(inj_path)

        if CFOD_AVAILABLE:
            try:
                df = cfod.catalog.read_injections()
                df.to_csv(inj_path, index=False)
                return df
            except Exception as exc:
                logger.warning("cfod injection load failed: %s", exc)

        logger.warning("Returning synthetic injection data as placeholder.")
        return self._synthetic_injections()

    # ── Subsampling helpers ───────────────────────────────────────────────────

    @staticmethod
    def non_repeaters(df: pd.DataFrame) -> pd.DataFrame:
        """Return only non-repeating FRBs."""
        return df[~df.get("is_repeater", pd.Series(False, index=df.index))]

    @staticmethod
    def high_snr(df: pd.DataFrame, min_sn: float = 10.0) -> pd.DataFrame:
        """Select bursts with S/N ≥ min_sn."""
        return df[df["sn"] >= min_sn]

    @staticmethod
    def build_background_windows(
        df: pd.DataFrame,
        window_width_s: float = 1.0,
        n_windows: int = 100,
        rng: np.random.Generator | None = None,
    ) -> pd.DataFrame:
        """Generate off-burst control windows from the catalog.

        Each control window is a random time offset that avoids
        the actual burst arrival times.
        """
        if rng is None:
            rng = np.random.default_rng(42)
        offsets = rng.uniform(10.0, 3600.0, size=n_windows)
        control = []
        for i, off in enumerate(offsets):
            control.append({"window_id": i, "time_offset_s": off, "is_signal": False})
        return pd.DataFrame(control)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _download_catalog(self, path: Path) -> None:
        logger.info("Downloading CHIME/FRB Catalog 1 from %s …", CATALOG1_URL)
        try:
            urllib.request.urlretrieve(CATALOG1_URL, path)
            checksum = self._checksum(path)
            (path.parent / "catalog1.sha256").write_text(checksum)
            logger.info("Saved catalog to %s (sha256: %s)", path, checksum)
        except Exception as exc:
            logger.error("Catalog download failed: %s. Using synthetic catalog.", exc)
            self._synthetic_catalog().to_csv(path, index=False)

    @staticmethod
    def _checksum(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _clean_catalog(df: pd.DataFrame) -> pd.DataFrame:
        """Standardise column names and fill missing values."""
        col_map = {
            "tns_name": "name",
            "dm_fitb": "dm",
            "snr": "sn",
            "fluence": "fluence_jy_ms",
        }
        for old, new in col_map.items():
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})
        numeric_cols = ["dm", "sn", "fluence_jy_ms", "width_observed"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.reset_index(drop=True)

    @staticmethod
    def _synthetic_catalog() -> pd.DataFrame:
        """Generate a synthetic CHIME/FRB-like catalog for testing."""
        rng = np.random.default_rng(0)
        n = 500
        return pd.DataFrame({
            "name": [f"FRB_SYNTH_{i:04d}" for i in range(n)],
            "dm": rng.lognormal(mean=np.log(500), sigma=0.8, size=n),
            "sn": np.exp(rng.uniform(np.log(8), np.log(100), n)),
            "fluence_jy_ms": rng.lognormal(mean=np.log(5), sigma=1.2, size=n),
            "width_observed": rng.lognormal(mean=np.log(1.5), sigma=0.8, size=n),
            "scattering_time": rng.lognormal(mean=np.log(0.5), sigma=1.5, size=n),
            "is_repeater": rng.random(n) < 0.05,
            "ra": rng.uniform(0, 360, n),
            "dec": rng.uniform(-30, 90, n),
            "source": "SYNTHETIC",
        })

    @staticmethod
    def _synthetic_injections() -> pd.DataFrame:
        """Generate synthetic injection events for completeness modelling."""
        rng = np.random.default_rng(1)
        n = 2000
        fluence_true = rng.lognormal(mean=np.log(3), sigma=1.5, size=n)
        detected = fluence_true > rng.lognormal(mean=np.log(1.5), sigma=0.3, size=n)
        return pd.DataFrame({
            "fluence_injected_jy_ms": fluence_true,
            "dm_injected": rng.lognormal(mean=np.log(400), sigma=0.7, size=n),
            "width_injected_ms": rng.lognormal(mean=np.log(1.5), sigma=0.8, size=n),
            "detected": detected.astype(int),
        })
