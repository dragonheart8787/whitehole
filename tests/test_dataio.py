"""Unit tests for data IO interfaces (using mock data paths)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from whitesearch.dataio import (
    GWOSCLoader,
    CHIMEFRBLoader,
    HEASARCLoader,
    ChandraLoader,
    XMMLoader,
    EHTLoader,
)


@pytest.fixture
def tmp_cache(tmp_path):
    return tmp_path


class TestGWOSCLoader:
    def test_load_known_event_returns_dict(self, tmp_cache):
        loader = GWOSCLoader(cache_dir=tmp_cache / "gwosc")
        result = loader.load_event("GW150914")
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_strain_is_array(self, tmp_cache):
        loader = GWOSCLoader(cache_dir=tmp_cache / "gwosc")
        result = loader.load_event("GW150914", detectors=["H1"])
        assert "H1" in result
        assert isinstance(result["H1"]["strain"], np.ndarray)
        assert len(result["H1"]["strain"]) > 0

    def test_unknown_event_raises(self, tmp_cache):
        loader = GWOSCLoader(cache_dir=tmp_cache / "gwosc")
        with pytest.raises(ValueError):
            loader.load_event("GW_DOES_NOT_EXIST")

    def test_segment_load_returns_dict(self, tmp_cache):
        loader = GWOSCLoader(cache_dir=tmp_cache / "gwosc")
        result = loader.load_segment("H1", gps_start=1126259462.0, gps_end=1126259462.0 + 4.0)
        assert "strain" in result
        assert "times" in result

    def test_checksum_is_string(self, tmp_cache):
        loader = GWOSCLoader(cache_dir=tmp_cache / "gwosc")
        result = loader.load_event("GW150914", detectors=["H1"])
        assert isinstance(result["H1"]["checksum"], str)


class TestCHIMEFRBLoader:
    def test_load_catalog_returns_dataframe(self, tmp_cache):
        loader = CHIMEFRBLoader(cache_dir=tmp_cache / "chime")
        df = loader.load_catalog()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_catalog_has_dm_column(self, tmp_cache):
        loader = CHIMEFRBLoader(cache_dir=tmp_cache / "chime")
        df = loader.load_catalog()
        assert "dm" in df.columns or "DM" in df.columns.str.upper().tolist()

    def test_injection_data_returns_dataframe(self, tmp_cache):
        loader = CHIMEFRBLoader(cache_dir=tmp_cache / "chime")
        df = loader.load_injection_data()
        assert isinstance(df, pd.DataFrame)

    def test_non_repeaters(self, tmp_cache):
        loader = CHIMEFRBLoader(cache_dir=tmp_cache / "chime")
        df = loader.load_catalog()
        nr = loader.non_repeaters(df)
        assert isinstance(nr, pd.DataFrame)


class TestHEASARCLoader:
    def test_query_region_returns_dataframe(self, tmp_cache):
        loader = HEASARCLoader(cache_dir=tmp_cache / "heasarc")
        df = loader.query_region(83.82, -5.39, radius_arcmin=5.0)
        assert isinstance(df, pd.DataFrame)

    def test_load_event_list_returns_dict(self, tmp_cache):
        loader = HEASARCLoader(cache_dir=tmp_cache / "heasarc")
        result = loader.load_event_list("0111970101", mission="xmm")
        assert "times" in result
        assert isinstance(result["times"], np.ndarray)

    def test_background_estimation(self):
        counts = np.array([2.0, 3.0, 50.0, 4.0, 2.0])
        times = np.arange(5.0)
        bg = HEASARCLoader.estimate_background(counts, times)
        assert len(bg) == len(counts)
        assert np.all(bg >= 0)


class TestChandraLoader:
    def test_load_obsid_returns_dict(self, tmp_cache):
        loader = ChandraLoader(cache_dir=tmp_cache / "chandra")
        result = loader.load_obsid("MOCK_001")
        assert "event_list" in result

    def test_extract_lightcurve(self, tmp_cache):
        loader = ChandraLoader(cache_dir=tmp_cache / "chandra")
        result = loader.load_obsid("MOCK_001")
        evt = result["event_list"]
        lc = loader.extract_lightcurve(evt)
        assert "times_s" in lc
        assert "counts" in lc
        assert len(lc["times_s"]) > 0


class TestXMMLoader:
    def test_query_region_returns_dataframe(self, tmp_cache):
        loader = XMMLoader(cache_dir=tmp_cache / "xmm")
        df = loader.query_region(187.7, 12.4)
        assert isinstance(df, pd.DataFrame)

    def test_get_observation_returns_dict(self, tmp_cache):
        loader = XMMLoader(cache_dir=tmp_cache / "xmm")
        result = loader.get_observation("0111970101")
        assert "events" in result

    def test_extract_lightcurve(self, tmp_cache):
        loader = XMMLoader(cache_dir=tmp_cache / "xmm")
        result = loader.get_observation("0111970101")
        lc = loader.extract_pps_lightcurve(result)
        assert "times_s" in lc
        assert "counts" in lc


class TestEHTLoader:
    def test_load_uvfits_returns_dict(self, tmp_cache):
        loader = EHTLoader(cache_dir=tmp_cache / "eht")
        result = loader.load_uvfits("M87", 2017, "LO")
        assert "visibilities" in result
        assert "u" in result
        assert "v" in result

    def test_visibilities_complex(self, tmp_cache):
        loader = EHTLoader(cache_dir=tmp_cache / "eht")
        result = loader.load_uvfits("M87", 2017, "LO")
        vis = result["visibilities"]
        assert np.iscomplexobj(vis)

    def test_closure_amplitudes_positive(self, tmp_cache):
        loader = EHTLoader(cache_dir=tmp_cache / "eht")
        result = loader.load_uvfits("M87", 2017, "LO")
        ca = loader.compute_closure_amplitudes(result["visibilities"])
        assert np.all(ca >= 0)
