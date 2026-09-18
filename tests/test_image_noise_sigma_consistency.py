"""The sigma the likelihood divides by must be the sigma the simulator used.

A global scale mismatch between the assumed and the actual noise is invisible
to every forward-model consistency check -- both sides stay self-consistent --
and it biases EVERY parameter's interval by the same factor, which is exactly
the signature of the residual under-coverage in docs/XRAY_IMAGE_PREFLIGHT_AUDIT
X.20 (six parameters uniformly narrow).  These tests pin the convention that
audit X.23 verified numerically, because a silent change to either side would
reopen a candidate that has already been ruled out.

The convention: ImageShadowSimulator draws Re and Im noise INDEPENDENTLY, each
with sd = thermal_noise_jy.  VisibilityLikelihood applies that same scalar to
the AMPLITUDE |V|.  That is correct because the projection of complex noise
onto the signal direction has sd = thermal_noise_jy -- no sqrt(2).
"""

from __future__ import annotations

import numpy as np
import pytest

from whitesearch.cli import _default_context
from whitesearch.likelihoods import VisibilityLikelihood
from whitesearch.models import model_for_context
from whitesearch.simulators import get_simulator


@pytest.fixture(scope="module")
def image_data():
    ctx = {**_default_context("image"), "target": "M87*", "rng_seed": 700029}
    model = model_for_context("gr_eternal", ctx)
    theta = model.sample_prior(np.random.default_rng(700029))
    data = get_simulator("image").simulate(
        theta, ctx, rng=np.random.default_rng(700030)
    )
    return ctx, data


class TestOneSigmaNotTwo:
    """There is a single noise number, not two independently defined ones."""

    def test_likelihood_reads_the_simulator_s_own_recorded_sigma(self, image_data):
        ctx, data = image_data
        # The likelihood resolves metadata-first, so it reads the value the
        # simulator RECORDED, not an independent re-derivation of it.
        assert data.metadata["thermal_noise_jy"] == ctx["thermal_noise_jy"]

    def test_the_simulator_emits_no_per_baseline_sigma_override(self, image_data):
        """`meta['sigma']`, if present, silently replaces the scalar.

        ImageShadowSimulator does not set it, so the SBC campaigns used the
        scalar.  EHTLoader._mock_eht_data DOES set it, with the same
        per-component convention -- this test exists so that a change making
        the simulator emit `sigma` in some other convention is not silent.
        """
        _, data = image_data
        assert "sigma" not in data.metadata

    def test_model_and_observation_cover_the_same_baselines(self, image_data):
        """loglike truncates to min(len(obs), len(model)); nothing may be lost."""
        _, data = image_data
        assert len(data.data) == len(data.metadata["vis_signal"])
        assert len(data.data) == len(data.metadata["uv_coverage"])


class TestAmplitudeNoiseScale:
    """The assumed sigma against the actual scatter of |V| -- measured, not assumed."""

    @staticmethod
    def _amplitude_sd(nu: np.ndarray, sigma: float, n: int = 20000) -> np.ndarray:
        rng = np.random.default_rng(4242)
        amp = np.abs(
            nu[None, :]
            + rng.standard_normal((n, len(nu))) * sigma
            + 1j * rng.standard_normal((n, len(nu))) * sigma
        )
        return amp.std(axis=0, ddof=1)

    def test_high_snr_amplitude_sd_equals_the_assumed_sigma(self):
        """The load-bearing check: no missing or spurious sqrt(2) or 2.

        A sqrt(2) error either way would land this ratio at 0.707 or 1.414.
        """
        sigma = 0.05
        nu = np.full(8, 200.0 * sigma)  # per-baseline SNR 200: Rician -> Gaussian
        ratio = self._amplitude_sd(nu, sigma) / sigma
        assert np.allclose(ratio, 1.0, atol=0.03), ratio
        assert not np.any(np.isclose(ratio, np.sqrt(2.0), atol=0.05))
        assert not np.any(np.isclose(ratio, 1.0 / np.sqrt(2.0), atol=0.05))

    def test_assumed_sigma_is_never_an_underestimate(self):
        """Rician sd is bounded above by sigma, so intervals can only be WIDE.

        This fixes the SIGN of the only discrepancy that does exist: the
        likelihood over-states the amplitude noise at low per-baseline SNR,
        which widens posteriors.  It therefore cannot explain an
        under-coverage, and a future reader should not re-propose it.
        """
        sigma = 0.05
        nu = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0]) * sigma
        ratio = self._amplitude_sd(nu, sigma) / sigma
        assert np.all(ratio <= 1.0 + 0.03), ratio
        assert ratio[0] < 0.75, "zero-signal limit is Rayleigh, sd = 0.655 sigma"
        assert ratio[-1] > 0.97, "high-SNR limit must recover sigma exactly"


class TestLikelihoodUsesThatSigma:
    def test_loglike_scales_with_the_recorded_sigma(self, image_data):
        """Doubling the recorded sigma must move loglike the Gaussian way.

        Guards against the sigma being read but then ignored -- the failure
        mode that made `walks` look configured while doing nothing (K.2.1).
        """
        ctx, data = image_data
        like = VisibilityLikelihood("gr_eternal", use_closure_phases=False)
        theta = dict(data.params_true)

        ll1 = like.loglike(theta, data, ctx)
        meta2 = dict(data.metadata)
        meta2["thermal_noise_jy"] = 2.0 * float(data.metadata["thermal_noise_jy"])
        data2 = type(data)(
            channel=data.channel, data=data.data, metadata=meta2,
            params_true=data.params_true,
            noise_realisation=data.noise_realisation,
        )
        ll2 = like.loglike(theta, data2, ctx)

        n = len(data.data)
        sigma = float(data.metadata["thermal_noise_jy"])
        chi2 = np.sum(
            ((np.abs(data.data) - np.abs(data.metadata["vis_signal"])) / sigma) ** 2
        )
        # ll = -0.5*chi2 - n*log(sigma*sqrt(2pi)); doubling sigma quarters chi2
        # and adds n*log(2) to the normalisation term.
        expected = (-0.5 * chi2 / 4.0 - n * np.log(2.0)) - (-0.5 * chi2)
        assert ll2 - ll1 == pytest.approx(expected, rel=1e-9)
