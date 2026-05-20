"""Pytest fixtures shared across the test suite."""

from __future__ import annotations

import numpy as np
import pytest

from whitesearch.models import (
    BlackToWhiteBounce,
    PBHTunnelingWhiteHole,
    GREternalWhiteHole,
    NullHypothesis,
    MagnetarFlare,
    StandardBHRingdown,
)
from whitesearch.simulators import (
    GravitationalWaveSimulator,
    EMBurstSimulator,
    ImageShadowSimulator,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# ── Models ────────────────────────────────────────────────────────────────────

@pytest.fixture
def bounce_model():
    return BlackToWhiteBounce()


@pytest.fixture
def pbh_model():
    return PBHTunnelingWhiteHole()


@pytest.fixture
def gr_model():
    return GREternalWhiteHole()


@pytest.fixture
def null_model():
    return NullHypothesis()


@pytest.fixture
def magnetar_model():
    return MagnetarFlare()


@pytest.fixture
def bh_ringdown_model():
    return StandardBHRingdown()


# ── Canonical parameter dicts ─────────────────────────────────────────────────

@pytest.fixture
def bounce_params():
    return {
        "M": 60.0,
        "a_star": 0.6,
        "log10_tau_bounce_yr": 5.0,
        "log10_ell_q": 3.0,
        "p_lifetime": 4,
        "eps_f": 0.05,
        "eps_Q": -0.1,
        "log10_A_bounce": -22.0,
        "D_L": 500.0,
        "i": 0.4,
        "eta_r": 1e-5,
        "eta_gamma": 1e-6,
    }


@pytest.fixture
def pbh_params():
    return {
        "log10_M_g": 14.5,
        "log10_f_pbh": -3.0,
        "log10_k_tunnel": 0.5,
        "log10_eta_r": -4.0,
        "log10_eta_gamma": -5.0,
        "z": 0.5,
        "DM_host": 80.0,
        "log10_W_int_ms": 0.5,
        "log10_tau_sc_ms": -0.5,
        "spectral_index": -1.5,
    }


@pytest.fixture
def gr_params():
    return {
        "M": 6.5e9,
        "a_star": 0.5,
        "D_L": 16.8,
        "i": 1.1,
        "position_angle": 0.8,
        "log10_ne": -3.0,
        "log10_B": -5.0,
        "ring_width_frac": 0.1,
        "log10_brightness": 0.5,
    }


# ── Simulators ────────────────────────────────────────────────────────────────

@pytest.fixture
def gw_simulator():
    return GravitationalWaveSimulator()


@pytest.fixture
def radio_simulator():
    return EMBurstSimulator()


@pytest.fixture
def image_simulator():
    return ImageShadowSimulator()


# ── Contexts ──────────────────────────────────────────────────────────────────

@pytest.fixture
def gw_context():
    return {
        "sample_rate": 4096,
        "duration": 2.0,
        "t_merger": 0.5,
        "low_freq_cutoff": 20.0,
        "rng_seed": 42,
    }


@pytest.fixture
def radio_context():
    return {
        "freq_low_mhz": 400.0,
        "freq_high_mhz": 800.0,
        "n_freq_chans": 16,
        "n_time_bins": 256,
        "t_start_s": 0.05,
        "t_end_s": 0.15,
        "tsys_jy": 1000.0,
        "t_samp_ms": 0.1,
        "rng_seed": 42,
    }


@pytest.fixture
def image_context():
    return {
        "fov_muas": 200.0,
        "n_pixels": 32,
        "freq_ghz": 230.0,
        "thermal_noise_jy": 0.05,
        "rng_seed": 42,
    }
