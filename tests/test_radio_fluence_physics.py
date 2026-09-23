"""Physics contract for PBHTunnelingWhiteHole.burst_fluence_jy_ms().

Locks in the three corrections from docs/RADIO_PREFLIGHT_AUDIT.md R.15:
the dead width terms are gone, the bandwidth comes from the CHIME band this
project analyses rather than a hardcoded 1 GHz, and the observed-bandwidth
(1 + z) factor is present.  These are factor-level corrections; the test at the
end asserts they stay factor-level.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from whitesearch.models.pbh_tunneling import (
    CHIME_BAND_HIGH_MHZ,
    CHIME_BAND_LOW_MHZ,
    RADIO_BANDWIDTH_HZ,
    PBHTunnelingWhiteHole,
)
from whitesearch.utils.constants import C, JY, MPC_M

MODEL = PBHTunnelingWhiteHole()

# The three reference points used in the R.13 diagnostic.
OPTIMISTIC = {"log10_M_g": 16.0, "log10_eta_r": 0.0, "z": 1e-4}
MEDIANISH = {"log10_M_g": 14.5, "log10_eta_r": -5.0, "z": 0.022}
WORKED = {"log10_M_g": 15.0, "log10_eta_r": -3.0, "z": 0.1}


def _legacy_fluence(params):
    """The expression as it stood at 824392e: 1 GHz, no (1 + z)."""
    M_kg = (10.0 ** params["log10_M_g"]) * 1e-3
    eta_r = 10.0 ** params["log10_eta_r"]
    D_L_m = PBHTunnelingWhiteHole._dl_mpc(params["z"]) * MPC_M
    return (eta_r * M_kg * C**2 / (4.0 * np.pi * D_L_m**2 * 1.0e9)) / (JY * 1e-3)


class TestBandwidthComesFromTheInstrumentConfig:
    def test_constants_match_chime_yaml(self):
        """configs/instruments/chime.yaml stays the source of truth.

        The constants are module-level so a physics model does no import-time
        file IO; this test is what stops them drifting from the YAML.
        """
        cfg = yaml.safe_load(
            (Path(__file__).resolve().parents[1]
             / "configs" / "instruments" / "chime.yaml").read_text()
        )
        assert cfg["observing"]["freq_low_mhz"] == CHIME_BAND_LOW_MHZ
        assert cfg["observing"]["freq_high_mhz"] == CHIME_BAND_HIGH_MHZ

    def test_bandwidth_is_the_band_width(self):
        assert RADIO_BANDWIDTH_HZ == pytest.approx(4.0e8)
        assert RADIO_BANDWIDTH_HZ == pytest.approx(
            (CHIME_BAND_HIGH_MHZ - CHIME_BAND_LOW_MHZ) * 1e6
        )

    def test_bandwidth_is_overridable_and_inverse(self):
        a = MODEL.burst_fluence_jy_ms(WORKED, delta_nu_hz=4.0e8)
        b = MODEL.burst_fluence_jy_ms(WORKED, delta_nu_hz=8.0e8)
        assert a / b == pytest.approx(2.0)


class TestRedshiftFactor:
    def test_luminosity_distance_already_carries_its_own_factor(self):
        """The (1 + z) added to the fluence is NOT the distance one."""
        for z in (0.01, 0.1, 1.0, 5.0):
            d_c = PBHTunnelingWhiteHole._dl_mpc(z) / (1.0 + z)
            # comoving distance is monotonic and strictly below D_L for z > 0
            assert 0.0 < d_c < PBHTunnelingWhiteHole._dl_mpc(z)

    def test_fluence_carries_exactly_one_extra_redshift_factor(self):
        """Ratio to the legacy expression must be exactly 2.5 * (1 + z)."""
        for params in (OPTIMISTIC, MEDIANISH, WORKED,
                       {"log10_M_g": 15.0, "log10_eta_r": -3.0, "z": 5.0}):
            ratio = MODEL.burst_fluence_jy_ms(params) / _legacy_fluence(params)
            assert ratio == pytest.approx(2.5 * (1.0 + params["z"]), rel=1e-12)

    def test_matches_the_standard_frb_energetics_relation(self):
        """E = 4 pi D_L^2 F_nu delta_nu / (1 + z), inverted."""
        p = WORKED
        f_jy_ms = MODEL.burst_fluence_jy_ms(p)
        f_si = f_jy_ms * JY * 1e-3          # J m^-2 Hz^-1
        D_L_m = PBHTunnelingWhiteHole._dl_mpc(p["z"]) * MPC_M
        e_recovered = (4.0 * np.pi * D_L_m**2 * f_si * RADIO_BANDWIDTH_HZ
                       / (1.0 + p["z"]))
        e_expected = (10.0 ** p["log10_eta_r"]) * (10.0 ** p["log10_M_g"]) * 1e-3 * C**2
        assert e_recovered == pytest.approx(e_expected, rel=1e-12)


class TestWidthIsNotUsed:
    def test_fluence_is_independent_of_the_burst_width(self):
        """A fluence is time-integrated; the caller divides by the width."""
        base = MODEL.burst_fluence_jy_ms(WORKED)
        for w, tau in ((-1.0, -2.0), (1.0, 0.0), (3.0, 2.0)):
            assert MODEL.burst_fluence_jy_ms(
                {**WORKED, "log10_W_int_ms": w, "log10_tau_sc_ms": tau}
            ) == pytest.approx(base, rel=1e-15)

    def test_width_parameters_are_not_required(self):
        MODEL.burst_fluence_jy_ms(dict(WORKED))  # no width keys at all

    def test_simulator_key_contract_dropped_the_width_terms(self):
        from whitesearch.simulators.em_burst import EMBurstSimulator

        assert EMBurstSimulator._PBH_FLUENCE_KEYS == (
            "log10_M_g", "log10_eta_r", "z"
        )


class TestCorrectionsStayFactorLevel:
    """The R.6 conclusion must survive: this is a factor fix, not a decade fix."""

    def test_total_change_is_under_two_decades_across_the_prior(self):
        from whitesearch.models import get_model

        model = get_model("pbh_tunneling")
        ratios = []
        for k in range(2000):
            th = model.sample_prior(np.random.default_rng(90000 + k))
            ratios.append(MODEL.burst_fluence_jy_ms(th) / _legacy_fluence(th))
        ratios = np.array(ratios)
        assert ratios.min() >= 2.5
        assert np.log10(ratios.max()) < 2.0
        assert np.median(ratios) < 10.0

    def test_optimistic_corner_still_above_and_median_still_far_below_chime(self):
        """CHIME completeness threshold band is 0.4-7 Jy ms."""
        assert MODEL.burst_fluence_jy_ms(OPTIMISTIC) > 7.0
        assert MODEL.burst_fluence_jy_ms(MEDIANISH) < 0.4 * 1e-4
