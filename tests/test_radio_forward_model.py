"""Radio channel: every sampled parameter must reach the data and the lnL.

Locks in the fixes for docs/RADIO_PREFLIGHT_AUDIT.md R.3 (pbh_tunneling's
fluence path was never connected), R.4 (parameter-name mismatches silently
absorbed by params.get defaults) and R.5 (RadioBurstLikelihood handed every
unlisted model the magnetar parameter list).
"""

from __future__ import annotations

import numpy as np
import pytest

from whitesearch.inference import BilbyRunner
from whitesearch.likelihoods import RadioBurstLikelihood
from whitesearch.models import check_model_channel, get_model
from whitesearch.models.pbh_tunneling import PBHTunnelingWhiteHole
from whitesearch.simulators import get_simulator
from whitesearch.simulators.em_burst import EMBurstSimulator

CTX = {
    "freq_low_mhz": 400.0, "freq_high_mhz": 800.0, "n_freq_chans": 16,
    "t_start_s": 0.1, "t_end_s": 0.5, "n_time_bins": 256,
    "tsys_jy": 1000.0, "t_samp_ms": 0.1, "rng_seed": 7,
}


def _simulate(theta):
    return get_simulator("radio").simulate(
        theta, CTX, rng=np.random.default_rng(11)
    )


class TestParameterNamesAreShared:
    """R.4: the name a model declares is the name the simulator reads."""

    @pytest.mark.parametrize(
        "model_name,param",
        [
            ("magnetar", "log10_W_int_ms"),
            ("magnetar", "DM"),
            ("magnetar", "spectral_index"),
            ("pbh_tunneling", "log10_W_int_ms"),
            ("pbh_tunneling", "spectral_index"),
            ("grb_frb", "spectral_index"),
            ("grb_frb", "DM"),
        ],
    )
    def test_model_declares_the_name(self, model_name, param):
        assert param in get_model(model_name).parameter_names

    @pytest.mark.parametrize("stale", ["log10_W_ms", "spectral_index_radio"])
    def test_stale_names_are_gone(self, stale):
        for model_name in ("magnetar", "grb_frb", "pbh_tunneling"):
            assert stale not in get_model(model_name).parameter_names


class TestSimulatorFailsClosed:
    """R.4: a missing key raises instead of silently becoming a default."""

    @pytest.mark.parametrize(
        "missing", ["log10_W_int_ms", "log10_tau_sc_ms", "spectral_index"]
    )
    def test_missing_burst_parameter_raises(self, missing):
        theta = get_model("magnetar").sample_prior(np.random.default_rng(3))
        theta.pop(missing)
        with pytest.raises(KeyError, match=missing):
            _simulate(theta)

    def test_missing_dispersion_measure_raises(self):
        theta = get_model("magnetar").sample_prior(np.random.default_rng(3))
        theta.pop("DM")
        with pytest.raises(KeyError, match="dispersion measure"):
            _simulate(theta)

    def test_missing_fluence_route_raises(self):
        theta = get_model("magnetar").sample_prior(np.random.default_rng(3))
        theta.pop("log10_fluence_jy_ms")
        with pytest.raises(KeyError, match="fluence"):
            _simulate(theta)


class TestEverySampledParameterIsLive:
    """R.3/R.4: perturbing a sampled parameter moves the data and the lnL.

    The magnetar model is the one with enough signal for the lnL response to
    sit well clear of float noise; pbh_tunneling's own prior produces a burst
    far below the noise (audit R.6), so it is checked on the data only.
    """

    @pytest.mark.parametrize(
        "param", ["log10_fluence_jy_ms", "log10_W_int_ms", "DM",
                  "log10_tau_sc_ms", "spectral_index"]
    )
    def test_magnetar_parameter_moves_data_and_loglike(self, param):
        model, like = get_model("magnetar"), RadioBurstLikelihood("magnetar")
        theta = model.sample_prior(np.random.default_rng(5))
        base = _simulate(theta)
        ll0 = like.loglike(theta, base, CTX)

        spec = {p.name: p for p in model.parameters()}[param]
        kw = spec.prior_kwargs
        if spec.prior_type == "log_uniform":
            bumped = float(theta[param]) * 2.0
        elif "low" in kw:
            bumped = float(theta[param]) + 0.2 * (kw["high"] - kw["low"])
        else:
            bumped = float(theta[param]) + 0.5
        moved = {**theta, param: bumped}

        assert np.abs(np.asarray(_simulate(moved).data)
                      - np.asarray(base.data)).max() > 0.0
        assert abs(like.loglike(moved, base, CTX) - ll0) > 1e-6

    @pytest.mark.parametrize("param", ["log10_M_g", "log10_eta_r", "z"])
    def test_pbh_fluence_parameters_move_the_noiseless_signal(self, param):
        """These were exactly dead before R.3; they now reach the waveform.

        Checked on the noiseless signal rather than the strain, because at the
        fluence this model's own prior implies the burst sits far below the
        receiver noise (audit R.6, asserted separately below).  The width is
        pinned to a resolvable 10 ms so the test measures the fluence wiring
        and not the time-grid sampling of a sub-bin pulse.
        """
        theta = get_model("pbh_tunneling").sample_prior(np.random.default_rng(5))
        theta = {**theta, "log10_W_int_ms": 1.0, "log10_tau_sc_ms": 0.0}

        def signal(th):
            d = _simulate(th)
            return np.asarray(d.data) - np.asarray(d.noise_realisation)

        base = signal(theta)
        moved = signal({**theta, param: float(theta[param]) + 0.5})
        assert np.abs(base).max() > 0.0
        assert np.abs(moved - base).max() > 0.0

    def test_pbh_burst_is_far_below_the_receiver_noise(self):
        """Audit R.6, locked in as a measured fact rather than tolerated silently.

        Not a defect in the wiring: the fluence the model derives is simply
        many orders of magnitude below what this instrument configuration can
        register.  If a later change to the prior or the instrument setup makes
        the burst visible, this test fails and should be updated deliberately.
        """
        theta = get_model("pbh_tunneling").sample_prior(np.random.default_rng(5))
        theta = {**theta, "log10_W_int_ms": 1.0, "log10_tau_sc_ms": 0.0}
        d = _simulate(theta)
        sig = np.asarray(d.data) - np.asarray(d.noise_realisation)
        assert np.abs(sig).max() > 0.0
        assert np.abs(sig).max() < 1e-6 * float(d.metadata["sigma_noise_jy"])


class TestPBHFluenceIsDerived:
    """R.3: the simulator uses the model's own fluence, not a hardcoded 1.0."""

    def test_simulated_fluence_matches_the_model_derivation(self):
        model = get_model("pbh_tunneling")
        for seed in (1, 2, 3):
            theta = model.sample_prior(np.random.default_rng(seed))
            used = _simulate(theta).metadata["fluence_jy_ms"]
            assert used == pytest.approx(
                PBHTunnelingWhiteHole().burst_fluence_jy_ms(theta), rel=1e-12
            )

    def test_fluence_is_no_longer_pinned_at_one(self):
        model = get_model("pbh_tunneling")
        vals = [
            _simulate(model.sample_prior(np.random.default_rng(s)))
            .metadata["fluence_jy_ms"]
            for s in range(6)
        ]
        assert len(set(vals)) == len(vals)
        assert not any(v == 1.0 for v in vals)

    def test_direct_fluence_still_wins_for_models_that_declare_it(self):
        theta = get_model("magnetar").sample_prior(np.random.default_rng(4))
        used = _simulate(theta).metadata["fluence_jy_ms"]
        assert used == pytest.approx(10.0 ** theta["log10_fluence_jy_ms"])


class TestSampledDimension:
    """R.3/R.5: each model gets its own list, and it is buildable."""

    def test_pbh_rate_parameters_are_not_sampled(self):
        names = RadioBurstLikelihood("pbh_tunneling").parameter_names
        assert "log10_f_pbh" not in names
        assert "log10_k_tunnel" not in names
        # still declared by the model, just not inferred from one burst
        assert "log10_f_pbh" in get_model("pbh_tunneling").parameter_names

    def test_pbh_effective_dimension(self):
        eff = BilbyRunner.effective_parameter_names(
            get_model("pbh_tunneling"), RadioBurstLikelihood("pbh_tunneling")
        )
        assert eff == ["log10_M_g", "log10_eta_r", "z", "DM_host",
                       "log10_W_int_ms", "log10_tau_sc_ms", "spectral_index"]

    def test_grb_frb_no_longer_raises(self):
        """R.5 regression lock: this used to raise ValueError."""
        eff = BilbyRunner.effective_parameter_names(
            get_model("grb_frb"), RadioBurstLikelihood("grb_frb")
        )
        assert eff == ["log10_fluence_jy_ms", "spectral_index", "DM"]

    def test_magnetar_effective_dimension(self):
        eff = BilbyRunner.effective_parameter_names(
            get_model("magnetar"), RadioBurstLikelihood("magnetar")
        )
        assert eff == ["log10_fluence_jy_ms", "log10_W_int_ms", "DM",
                       "log10_tau_sc_ms", "spectral_index"]

    def test_unknown_model_raises_instead_of_inheriting_magnetar(self):
        with pytest.raises(ValueError, match="no parameter list"):
            _ = RadioBurstLikelihood("bh_ringdown").parameter_names


class TestGRBRemainsUnsimulable:
    """Documented gap, NOT a fix: GRBAfterglowFRB declares no burst width.

    Priors can now be built (R.5), but the model declares neither
    log10_W_int_ms nor log10_tau_sc_ms, so EMBurstSimulator cannot produce a
    dynamic spectrum for it.  Before the R.4 change this was masked by default
    values; the failure is now explicit and names the missing parameter.  See
    docs/RADIO_PREFLIGHT_AUDIT.md R.11.
    """

    def test_simulating_grb_frb_raises_naming_the_missing_width(self):
        theta = get_model("grb_frb").sample_prior(np.random.default_rng(2))
        with pytest.raises(KeyError, match="log10_W_int_ms"):
            _simulate(theta)


class TestFluenceKeyContract:
    def test_pbh_fluence_keys_are_all_declared_by_the_model(self):
        declared = set(get_model("pbh_tunneling").parameter_names)
        assert set(EMBurstSimulator._PBH_FLUENCE_KEYS) <= declared


class TestChannelCompatibilityIsCheckedOnBothSides:
    """R.12.2: the injection side now uses the same table as the fit side."""

    def test_matching_channel_passes(self):
        for model_name, channel in (("magnetar", "radio"), ("bh_ringdown", "gw"),
                                    ("pbh_tunneling", "radio"),
                                    ("pbh_tunneling", "xray"),
                                    ("null", "radio"), ("null", "gw")):
            check_model_channel(model_name, channel)

    @pytest.mark.parametrize(
        "model_name,channel",
        [("bh_ringdown", "radio"), ("magnetar", "gw"),
         ("bh_accretion", "radio"), ("magnetar", "image")],
    )
    def test_mismatched_channel_raises(self, model_name, channel):
        with pytest.raises(ValueError, match="native channel"):
            check_model_channel(model_name, channel)

    def test_unknown_data_channel_accepts_nothing(self):
        with pytest.raises(ValueError, match="unknown data channel"):
            check_model_channel("magnetar", "neutrino")

    def test_loader_rejects_before_simulating(self):
        from whitesearch.dataio.loader import load_observation_data

        with pytest.raises(ValueError, match="native channel"):
            load_observation_data(
                "mock", "radio", inject_model="bh_ringdown", seed=1,
                context={"n_freq_chans": 8, "n_time_bins": 32, "rng_seed": 1},
            )
