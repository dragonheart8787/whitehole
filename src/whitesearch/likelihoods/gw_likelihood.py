"""Gravitational wave likelihood for the black-to-white bounce model.

Implements a frequency-domain Gaussian noise likelihood:

  log L = −½ ⟨d − h(θ) | d − h(θ)⟩  + constant
         = −2 Re[ ∑_f |d̃(f) − h̃(f,θ)|² / Sn(f) Δf ]

plus a matched-filter SNR log-likelihood approximation for the bounce burst.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import BaseLikelihood, gaussian_loglike
from ..utils.math_utils import (
    noise_weighted_inner_product,
    ringdown_waveform,
    kerr_qnm_frequency,
)
from ..utils.constants import G, C, M_SUN, MPC_M


class GWLikelihood(BaseLikelihood):
    """Frequency-domain GW likelihood for black-to-white bounce / BH ringdown.

    Parameters
    ----------
    model_name : str
        'bounce' or 'bh_ringdown'.
    use_full_likelihood : bool
        If True, use the full ⟨d−h|d−h⟩ inner product.
        If False, use the matched-filter SNR approximation (faster).
    """

    def __init__(
        self,
        model_name: str = "bounce",
        use_full_likelihood: bool = True,
    ) -> None:
        self.model_name = model_name
        self.use_full = use_full_likelihood
        self._param_names: list[str] | None = None

    @property
    def parameter_names(self) -> list[str]:
        if self.model_name == "bounce":
            return [
                "M", "a_star", "eps_f", "eps_Q",
                "log10_A_bounce", "log10_tau_bounce_yr",
                "D_L", "i",
            ]
        return ["M", "a_star", "log10_A", "D_L", "i"]

    def loglike(
        self,
        theta: dict[str, float],
        data: Any,
        context: dict[str, Any],
    ) -> float:
        """Compute log p(d | θ, model) in the frequency domain.

        Parameters
        ----------
        theta : dict
            Current parameter point.
        data : SimData or dict
            Observed / simulated data.  Must have 'data' (strain), 'metadata'
            with 'psd', 'sample_rate', 'freqs'.
        context : dict
            Observation configuration.

        Returns
        -------
        float — log-likelihood
        """
        try:
            strain = np.asarray(
                data.data if hasattr(data, "data") else data["strain"],
                dtype=np.float64,
            )
            meta = data.metadata if hasattr(data, "metadata") else data
            psd = np.asarray(meta["psd"], dtype=np.float64)
            sample_rate = float(meta.get("sample_rate", 4096.0))
            t_merger = float(meta.get("t_merger", 0.5))
        except (KeyError, AttributeError) as exc:
            raise ValueError(f"GWLikelihood: malformed data: {exc}") from exc

        n = len(strain)
        dt = 1.0 / sample_rate
        df = 1.0 / (n * dt)
        freqs = np.fft.rfftfreq(n, d=dt)

        # Build template waveform
        times = np.arange(n) * dt
        h_template = self._build_template(theta, times, t_merger)
        h_template_f = np.fft.rfft(h_template) * dt

        # Data FFT
        strain_f = np.fft.rfft(strain) * dt

        if self.use_full:
            ll = self._full_inner_product_loglike(strain_f, h_template_f, psd, df)
        else:
            ll = self._mf_snr_loglike(strain_f, h_template_f, psd, df)

        return ll

    # ── Template builder ──────────────────────────────────────────────────────

    def _build_template(
        self,
        theta: dict[str, float],
        times: NDArray,
        t_merger: float,
    ) -> NDArray:
        M = theta["M"]
        a = theta["a_star"]
        i = theta.get("i", 0.0)

        f_gr, q_gr = kerr_qnm_frequency(M, a)
        eps_f = theta.get("eps_f", 0.0)
        eps_Q = theta.get("eps_Q", 0.0)
        f_rd = f_gr * (1.0 + eps_f)
        q_rd = max(0.5, q_gr * (1.0 + eps_Q))

        D_L_m = theta.get("D_L", 100.0) * MPC_M
        h0 = G * M * M_SUN / (C**2 * D_L_m)

        cos_i = np.cos(i)
        A_rd = h0 * 0.5 * (1.0 + cos_i**2)  # plus polarisation

        h = ringdown_waveform(times, t_merger, A_rd, f_rd, q_rd)

        # Bounce burst
        if "log10_A_bounce" in theta and "log10_tau_bounce_yr" in theta:
            A_b = float(10.0 ** theta["log10_A_bounce"])
            from ..utils.constants import GYR_S
            tau_s = float(10.0 ** theta["log10_tau_bounce_yr"] * GYR_S / 1e9)
            t_bounce = t_merger + tau_s
            if t_bounce < times[-1]:
                h += ringdown_waveform(
                    times, t_bounce, A_b, f_rd * 0.8, max(2.0, q_rd * 0.5)
                )
        return h

    # ── Inner products ────────────────────────────────────────────────────────

    @staticmethod
    def _full_inner_product_loglike(
        strain_f: NDArray,
        template_f: NDArray,
        psd: NDArray,
        df: float,
    ) -> float:
        """Full matched-filter log-likelihood: −½ ⟨d−h|d−h⟩."""
        residual_f = strain_f - template_f
        inner = noise_weighted_inner_product(residual_f, residual_f, psd, df)
        return float(-0.5 * inner.real)

    @staticmethod
    def _mf_snr_loglike(
        strain_f: NDArray,
        template_f: NDArray,
        psd: NDArray,
        df: float,
    ) -> float:
        """Approximation: ρ_opt² − ½ ρ_opt²  (maximised over time, phase)."""
        dh = noise_weighted_inner_product(strain_f, template_f, psd, df).real
        hh = noise_weighted_inner_product(template_f, template_f, psd, df).real
        if hh <= 0:
            return -np.inf
        return float(dh - 0.5 * hh)

    # ── Predictive checks ─────────────────────────────────────────────────────

    def predictive_summary_stats(
        self,
        theta: dict[str, float],
        context: dict[str, Any],
    ) -> dict[str, float]:
        """Compute observable summary statistics for PPC.

        Returns ringdown frequency, quality factor, SNR.
        """
        M = theta["M"]
        a = theta["a_star"]
        eps_f = theta.get("eps_f", 0.0)
        eps_Q = theta.get("eps_Q", 0.0)
        f_gr, q_gr = kerr_qnm_frequency(M, a)
        return {
            "f_rd_hz": f_gr * (1.0 + eps_f),
            "q_rd": q_gr * (1.0 + eps_Q),
            "delta_f_hz": f_gr * eps_f,
            "delta_Q": q_gr * eps_Q,
        }
