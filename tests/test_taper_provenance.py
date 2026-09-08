"""Taper applies only where the window mismatch it corrects actually exists.

Background: TAPER_ALPHA=0.1 corrects a full-segment rectangular rfft analysed
against a Welch-estimated PSD.  Raw simulator output carries the exact analytic
PSD its noise was generated from, on the same un-windowed basis as the rfft, so
there is no mismatch there and tapering instead smears the steep seismic wall of
aligo_psd_analytic() across the band.  See docs/BOUNCE_PREFLIGHT_AUDIT.md Part H.
"""

from __future__ import annotations

import numpy as np
import pytest

from whitesearch.dataio.gw_observation import prepare_gw_from_simdata
from whitesearch.likelihoods import GWLikelihood
from whitesearch.likelihoods.gw_likelihood import (
    TAPER_ALPHA,
    TAPER_ALPHA_NONE,
    TAPER_REASON_GENERATIVE_PSD,
    TAPER_REASON_UNKNOWN_SOURCE,
    TAPER_REASON_WELCH_PSD,
    taper_for_source,
)
from whitesearch.likelihoods.gw_units import time_to_freq
from whitesearch.models import get_model
from whitesearch.simulators import get_simulator
from whitesearch.utils.math_utils import noise_weighted_inner_product

CTX = {
    "sample_rate": 4096.0,
    "duration": 4.0,
    "t_merger": 1.0,
    "low_freq_cutoff": 20.0,
    "rng_seed": 0,
}
SEED = 20260907


@pytest.fixture(scope="module")
def sim_data():
    model = get_model("bounce")
    sim = get_simulator("gw")
    rng = np.random.default_rng(SEED)
    theta = model.sample_prior(rng)
    data = sim.simulate(theta, {**CTX, "rng_seed": SEED}, rng=rng)
    return theta, data


class TestTaperForSource:
    def test_gwosc_keeps_the_validated_taper(self):
        """REGRESSION LOCK: the real-data path must not change.

        TAPER_ALPHA=0.1 was validated on GW150914/GW170814; nothing in the
        provenance branch may move it.
        """
        alpha, reason = taper_for_source("GWOSC")
        assert alpha == TAPER_ALPHA == 0.1
        assert reason == TAPER_REASON_WELCH_PSD

    def test_mock_simulator_skips_the_taper(self):
        alpha, reason = taper_for_source("MOCK_SIMULATOR")
        assert alpha == TAPER_ALPHA_NONE == 0.0
        assert reason == TAPER_REASON_GENERATIVE_PSD

    @pytest.mark.parametrize("source", ["MOCK_EXPLICIT", "MOCK", "MOCK_FALLBACK"])
    def test_preprocessed_sources_keep_the_taper(self, source):
        """Mock strain that went through GWPreprocessor carries a Welch PSD,
        so it has the same mismatch as real data and keeps the taper."""
        alpha, reason = taper_for_source(source)
        assert alpha == TAPER_ALPHA
        assert reason == TAPER_REASON_WELCH_PSD

    def test_absent_source_defaults_to_the_real_data_convention(self):
        """Skipping the taper must be opted into, never inherited by omission."""
        alpha, reason = taper_for_source(None)
        assert alpha == TAPER_ALPHA
        assert reason == TAPER_REASON_UNKNOWN_SOURCE

    def test_unrecognised_source_keeps_the_taper(self):
        alpha, _ = taper_for_source("SOMETHING_NEW")
        assert alpha == TAPER_ALPHA


class TestSimulatorProvenanceTag:
    def test_simulator_tags_its_output(self, sim_data):
        _theta, data = sim_data
        assert data.metadata["source"] == "MOCK_SIMULATOR"

    def test_preprocessing_retags_as_mock_explicit(self, sim_data):
        """prepare_gw_from_simdata replaces the PSD with a Welch estimate, so
        it must also replace the provenance tag."""
        _theta, data = sim_data
        obs = prepare_gw_from_simdata(data)
        assert obs["source"] == "MOCK_EXPLICIT"


class TestLikelihoodRecordsTheChoice:
    def test_taper_config_reports_source_alpha_and_reason(self, sim_data):
        _theta, data = sim_data
        cfg = GWLikelihood("bounce").taper_config(data)
        assert cfg == {
            "data_source": "MOCK_SIMULATOR",
            "taper_alpha_used": 0.0,
            "taper_alpha_reason": TAPER_REASON_GENERATIVE_PSD,
        }

    def test_loglike_records_what_it_used(self, sim_data):
        theta, data = sim_data
        like = GWLikelihood("bounce", use_full_likelihood=True)
        assert like.last_taper_config == {}
        like.loglike(theta, data, CTX)
        assert like.last_taper_config["taper_alpha_used"] == 0.0
        assert like.last_taper_config["data_source"] == "MOCK_SIMULATOR"
        assert like.last_taper_config["taper_alpha_reason"] == (
            TAPER_REASON_GENERATIVE_PSD
        )

    def test_gwosc_tagged_data_records_the_taper(self, sim_data):
        _theta, data = sim_data
        record = {
            "strain": np.asarray(data.data),
            "psd": np.asarray(data.metadata["psd"]),
            "sample_rate": CTX["sample_rate"],
            "t_merger": CTX["t_merger"],
            "source": "GWOSC",
        }
        cfg = GWLikelihood("bounce").taper_config(record)
        assert cfg["taper_alpha_used"] == 0.1
        assert cfg["data_source"] == "GWOSC"

    def test_null_loglike_also_records_the_choice(self, sim_data):
        _theta, data = sim_data
        like = GWLikelihood("null")
        like.loglike({}, data, CTX)
        assert like.last_taper_config["taper_alpha_used"] == 0.0


class TestScaleIsRestoredOnTheMockPath:
    def test_per_bin_inner_product_is_order_two(self, sim_data):
        """The defect this branch fixes, measured end to end.

        With the taper the mock path gave per-bin <n|n> of 5.8e6-1.3e8 against
        a theory value of 2.0 (BOUNCE_PREFLIGHT_AUDIT.md H.5); without it the
        noise is diagonal by construction.
        """
        _theta, data = sim_data
        like = GWLikelihood("bounce")
        alpha = like.taper_config(data)["taper_alpha_used"]
        noise = np.asarray(data.noise_realisation, dtype=float)
        psd = np.asarray(data.metadata["psd"], dtype=float)
        freqs, noise_f, df = time_to_freq(
            noise, 1.0 / CTX["sample_rate"], taper_alpha=alpha
        )
        band = (freqs >= 20.0) & (freqs <= 1700.0)
        nn = float(
            noise_weighted_inner_product(
                noise_f[band], noise_f[band], psd[band], df
            ).real
        )
        assert 1.0 < nn / int(band.sum()) < 4.0

    def test_null_loglike_is_no_longer_astronomically_negative(self, sim_data):
        _theta, data = sim_data
        ll = GWLikelihood("null").loglike({}, data, CTX)
        assert -1e5 < ll < 0.0
