"""Tests for GW FFT conventions and likelihood finiteness."""

from __future__ import annotations

import numpy as np
import pytest

from whitesearch.dataio.gw_observation import prepare_gw_from_simdata
from whitesearch.likelihoods.gw_likelihood import TAPER_ALPHA, GWLikelihood
from whitesearch.likelihoods.gw_units import inner_product_norm, time_to_freq
from whitesearch.models import BlackToWhiteBounce, NullHypothesis, StandardBHRingdown
from whitesearch.simulators import GravitationalWaveSimulator
from whitesearch.utils.math_utils import estimate_psd, matched_filter_snr, ringdown_waveform


@pytest.fixture
def gw_obs_dict(bounce_params, gw_context):
    sim = GravitationalWaveSimulator()
    d = sim.simulate(bounce_params, gw_context, rng=np.random.default_rng(0))
    return prepare_gw_from_simdata(d, reference_amplitude=False)


@pytest.fixture
def gw_context_dict(gw_context):
    return gw_context


def test_time_to_freq_roundtrip():
    rng = np.random.default_rng(0)
    h = rng.standard_normal(4096)
    dt = 1.0 / 4096
    _, hf, _ = time_to_freq(h, dt)
    h2 = np.fft.irfft(hf / dt, n=len(h))
    assert np.allclose(h, h2, atol=1e-10)


def test_three_models_finite_lnL(gw_obs_dict, gw_context_dict):
    data = gw_obs_dict
    ctx = gw_context_dict
    null_ll = GWLikelihood("null").loglike({}, data, ctx)
    assert np.isfinite(null_ll)

    bounce = BlackToWhiteBounce()
    theta_b = bounce.sample_prior(np.random.default_rng(1))
    ll_b = GWLikelihood("bounce").loglike(theta_b, data, ctx)
    assert np.isfinite(ll_b)

    bh = StandardBHRingdown()
    theta_bh = bh.sample_prior(np.random.default_rng(2))
    ll_bh = GWLikelihood("bh_ringdown").loglike(theta_bh, data, ctx)
    assert np.isfinite(ll_bh)


def test_bh_ringdown_uses_log10_A(gw_obs_dict, gw_context_dict):
    data = gw_obs_dict
    ctx = gw_context_dict
    ll = GWLikelihood("bh_ringdown")
    theta = {"M": 30.0, "a_star": 0.6, "log10_A": -22.0, "D_L": 400.0, "i": 0.5}
    l1 = ll.loglike(theta, data, ctx)
    theta2 = dict(theta)
    theta2["log10_A"] = -20.0
    l2 = ll.loglike(theta2, data, ctx)
    assert l1 != l2


# ── Taper (spectral-leakage) tests ──────────────────────────────────────────

class TestTimeToFreqTaper:
    """time_to_freq's taper_alpha must default to the historical (untapered)
    behaviour and only change output when explicitly requested."""

    def test_default_alpha_is_untapered(self):
        rng = np.random.default_rng(0)
        h = rng.standard_normal(4096)
        dt = 1.0 / 4096
        _, hf_default, _ = time_to_freq(h, dt)
        _, hf_explicit_zero, _ = time_to_freq(h, dt, taper_alpha=0.0)
        assert np.array_equal(hf_default, hf_explicit_zero)

    def test_nonzero_alpha_changes_output(self):
        rng = np.random.default_rng(0)
        h = rng.standard_normal(4096)
        dt = 1.0 / 4096
        _, hf_untapered, _ = time_to_freq(h, dt)
        _, hf_tapered, _ = time_to_freq(h, dt, taper_alpha=0.1)
        assert not np.allclose(hf_untapered, hf_tapered)


class TestTaperMaskedDD:
    """Synthetic-PSD reproduction of the real-data leakage pathology: a PSD
    estimated via Welch/Hann from one noise realisation, applied to the
    full-segment (rectangular-window) FFT of an *independent* realisation
    drawn from the same true PSD. Without a taper this measurably inflates
    the masked per-bin <d|d> above its theoretical mean of 2; taper_alpha
    matching GWLikelihood.TAPER_ALPHA should bring it closer to 2."""

    SR = 4096.0
    DUR = 32.0
    S0 = 1.0e-46

    @classmethod
    def _make_noise(cls, rng, n, dt, df):
        psd = np.full(n // 2 + 1, cls.S0)
        sigma_f = np.sqrt(psd / (4.0 * df)) / dt
        nf = (rng.standard_normal(len(psd)) + 1j * rng.standard_normal(len(psd))) * sigma_f
        nf[0] = nf[0].real
        if n % 2 == 0:
            nf[-1] = nf[-1].real
        return np.fft.irfft(nf, n=n)

    def test_taper_moves_masked_dd_toward_theoretical_mean(self):
        sr, dur = self.SR, self.DUR
        dt = 1.0 / sr
        n = int(dur * sr)
        df = sr / n
        freqs = np.fft.rfftfreq(n, d=dt)
        mask = (freqs >= 50.0) & (freqs <= 200.0)

        n_trials = 12
        mean_untapered = []
        mean_tapered = []
        for trial in range(n_trials):
            rng_psd = np.random.default_rng(9000 + trial)
            rng_on = np.random.default_rng(9500 + trial)

            off_source = self._make_noise(rng_psd, n, dt, df)
            freqs_w, psd_welch = estimate_psd(off_source, sr, fft_length=4.0, overlap=0.5, window="hann")
            psd_interp = np.interp(freqs, freqs_w, psd_welch)

            on_source = self._make_noise(rng_on, n, dt, df)

            _, d_f_u, df_u = time_to_freq(on_source, dt, taper_alpha=0.0)
            per_bin_u = 4.0 * (np.conj(d_f_u[mask]) * d_f_u[mask]).real / psd_interp[mask] * df_u
            mean_untapered.append(np.mean(per_bin_u))

            _, d_f_t, df_t = time_to_freq(on_source, dt, taper_alpha=TAPER_ALPHA)
            per_bin_t = 4.0 * (np.conj(d_f_t[mask]) * d_f_t[mask]).real / psd_interp[mask] * df_t
            mean_tapered.append(np.mean(per_bin_t))

        mean_untapered = float(np.mean(mean_untapered))
        mean_tapered = float(np.mean(mean_tapered))

        # The tapered estimate must land closer to the theoretical value 2.0.
        assert abs(mean_tapered - 2.0) < abs(mean_untapered - 2.0)
        # And stay within a broad sanity band (not overcorrected to ~0).
        assert 1.0 < mean_tapered < 3.0


class TestTaperSignalRecovery:
    """Regression test: tapering the GW-likelihood FFT path must not gut
    sensitivity to a ringdown centred well away from the segment edges
    (as GW150914-like events are, with t_merger ~ mid-segment)."""

    def test_centered_ringdown_snr_barely_changes_with_taper(self):
        sr = 4096.0
        dur = 32.0
        dt = 1.0 / sr
        n = int(dur * sr)
        df = sr / n
        freqs = np.fft.rfftfreq(n, d=dt)
        t_merger = dur / 2.0

        psd = np.full_like(freqs, 1.0e-46)
        band = (freqs >= 20.0) & (freqs <= 0.95 * sr / 2)

        times = np.arange(n) * dt
        h_signal = ringdown_waveform(times, t_merger, 3.0e-22, 250.0, 8.0)

        rng = np.random.default_rng(42)
        sigma_f = np.sqrt(psd / (4.0 * df)) / dt
        nf = (rng.standard_normal(len(psd)) + 1j * rng.standard_normal(len(psd))) * sigma_f
        nf[0] = nf[0].real
        nf[-1] = nf[-1].real
        noise = np.fft.irfft(nf, n=n)
        strain = h_signal + noise

        _, strain_f_u, df_u = time_to_freq(strain, dt, taper_alpha=0.0)
        _, h_f_u, _ = time_to_freq(h_signal, dt, taper_alpha=0.0)
        snr_untapered = matched_filter_snr(h_f_u[band], strain_f_u[band], psd[band], df_u)

        _, strain_f_t, df_t = time_to_freq(strain, dt, taper_alpha=TAPER_ALPHA)
        _, h_f_t, _ = time_to_freq(h_signal, dt, taper_alpha=TAPER_ALPHA)
        snr_tapered = matched_filter_snr(h_f_t[band], strain_f_t[band], psd[band], df_t)

        # A signal parked at mid-segment sits in the taper's unity-gain
        # region, so SNR should move by well under 1% (measured ~0.001%).
        rel_change = abs(snr_tapered - snr_untapered) / snr_untapered
        assert rel_change < 0.01
