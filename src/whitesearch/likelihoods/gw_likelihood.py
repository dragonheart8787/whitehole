"""Gravitational wave likelihood for the black-to-white bounce model.

Implements a frequency-domain Gaussian noise likelihood:

  log L = −½ ⟨d − h(θ) | d − h(θ)⟩  + constant
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import BaseLikelihood
from .gw_units import inner_product_norm, time_to_freq
from ..utils.math_utils import (
    noise_weighted_inner_product,
    ringdown_waveform,
    kerr_qnm_frequency,
)
from ..utils.constants import G, C, M_SUN, MPC_M

# Soft floor instead of -inf for dynesty stability.  Must sit BELOW any
# genuine in-band log-likelihood so template-rejected points rank worst:
# on real GWOSC data (GW150914 H1, 32 s, band-masked) the null lnL is
# ~-5.8e7 and the worst prior-draw template lnL observed is ~-1.9e9.  The
# previous -1e6 floor sat ABOVE all genuine values on that scale, so
# rejected templates formed the global-maximum plateau and dynesty
# terminated on it, returning ln Z = floor + ln(plateau prior mass).
LL_MIN = -1e12
HH_MIN = 1e-30


class GWLikelihood(BaseLikelihood):
    """Frequency-domain GW likelihood for bounce / BH ringdown / null."""

    def __init__(
        self,
        model_name: str = "bounce",
        use_full_likelihood: bool = True,
        ll_min: float = LL_MIN,
    ) -> None:
        self.model_name = model_name
        self.use_full = use_full_likelihood
        self.ll_min = ll_min

    @property
    def parameter_names(self) -> list[str]:
        if self.model_name == "null":
            return []
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
        if self.model_name == "null":
            return self._null_loglike(data, context)

        try:
            strain, meta, sample_rate, t_merger, low_freq, high_freq = self._parse_data(
                data, context
            )
        except (KeyError, AttributeError, ValueError) as exc:
            return self.ll_min

        n = len(strain)
        dt = 1.0 / sample_rate
        freqs, strain_f, df = time_to_freq(strain, dt)
        nyquist = sample_rate / 2.0

        times = np.arange(n) * dt
        h_template = self._build_template(theta, times, t_merger, freqs, nyquist, low_freq)
        if h_template is None:
            return self.ll_min

        _, h_template_f, _ = time_to_freq(h_template, dt)

        band = self._band_mask(freqs, low_freq, high_freq, nyquist)
        if not np.any(band):
            return self.ll_min
        strain_fb = strain_f[band]
        h_template_fb = h_template_f[band]
        psd_b = meta["psd"][band]

        if self.use_full:
            ll = self._full_inner_product_loglike(strain_fb, h_template_fb, psd_b, df)
        else:
            ll = self._mf_snr_loglike(strain_fb, h_template_fb, psd_b, df)

        if not np.isfinite(ll):
            return self.ll_min
        return float(ll)

    def _parse_data(
        self, data: Any, context: dict[str, Any]
    ) -> tuple[NDArray, dict, float, float, float, float]:
        strain = np.asarray(
            data.data if hasattr(data, "data") else data["strain"],
            dtype=np.float64,
        )
        meta = data.metadata if hasattr(data, "metadata") else data
        psd = np.asarray(meta["psd"], dtype=np.float64)
        sample_rate = float(meta.get("sample_rate", 4096.0))
        t_merger = float(meta.get("t_merger", context.get("t_merger", 0.5)))
        low_freq = float(meta.get("low_freq_cutoff", context.get("low_freq_cutoff", 20.0)))
        high_freq = float(
            meta.get("high_freq_cutoff", context.get("high_freq_cutoff", 1700.0))
        )
        meta = dict(meta) if isinstance(meta, dict) else {"psd": psd}
        meta["psd"] = psd
        meta["sample_rate"] = sample_rate
        meta["low_freq_cutoff"] = low_freq
        meta["high_freq_cutoff"] = high_freq
        return strain, meta, sample_rate, t_merger, low_freq, high_freq

    @staticmethod
    def _band_mask(
        freqs: NDArray,
        low_freq: float,
        high_freq: float,
        nyquist: float,
    ) -> NDArray:
        """Boolean mask restricting inner products to the analysis band.

        Outside [low_freq, min(high_freq, 0.95*nyquist)] the bandpassed
        strain and the PSD are both filter-rolloff residuals; their ratio is
        numerically meaningless and (on real data) dominates ⟨d|d⟩ by orders
        of magnitude if included.
        """
        f_hi = min(high_freq, 0.95 * nyquist)
        return (freqs >= low_freq) & (freqs <= f_hi)

    def _null_loglike(self, data: Any, context: dict[str, Any] | None = None) -> float:
        strain, meta, sample_rate, _, low_freq, high_freq = self._parse_data(
            data, context or {}
        )
        freqs, strain_f, df = time_to_freq(strain, 1.0 / sample_rate)
        band = self._band_mask(freqs, low_freq, high_freq, sample_rate / 2.0)
        if not np.any(band):
            return self.ll_min
        inner = noise_weighted_inner_product(
            strain_f[band], strain_f[band], meta["psd"][band], df
        )
        ll = float(-0.5 * inner.real)
        return ll if np.isfinite(ll) else self.ll_min

    def _build_template(
        self,
        theta: dict[str, float],
        times: NDArray,
        t_merger: float,
        freqs: NDArray,
        nyquist: float,
        low_freq: float,
    ) -> NDArray | None:
        M = theta.get("M")
        a = theta.get("a_star")
        if M is None or a is None:
            return None

        try:
            f_gr, q_gr = kerr_qnm_frequency(float(M), float(a))
        except (ValueError, ZeroDivisionError):
            return None

        if self.model_name == "bh_ringdown":
            eps_f, eps_Q = 0.0, 0.0
            log10_a = theta.get("log10_A")
            if log10_a is None:
                return None
            A_rd = float(10.0 ** log10_a)
        else:
            eps_f = theta.get("eps_f", 0.0)
            eps_Q = theta.get("eps_Q", 0.0)
            D_L_m = theta.get("D_L", 100.0) * MPC_M
            h0 = G * float(M) * M_SUN / (C**2 * D_L_m)
            i = theta.get("i", 0.0)
            cos_i = np.cos(i)
            A_rd = h0 * 0.5 * (1.0 + cos_i**2)

        f_rd = f_gr * (1.0 + eps_f)
        q_rd = max(0.5, q_gr * (1.0 + eps_Q))

        if f_rd < low_freq or f_rd > 0.95 * nyquist:
            return None

        h = ringdown_waveform(times, t_merger, A_rd, f_rd, q_rd)

        if self.model_name == "bounce":
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

    @staticmethod
    def _full_inner_product_loglike(
        strain_f: NDArray,
        template_f: NDArray,
        psd: NDArray,
        df: float,
    ) -> float:
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
        _, dh, hh = inner_product_norm(strain_f, template_f, psd, df)
        if hh < HH_MIN:
            return LL_MIN
        return float(dh - 0.5 * hh)

    def predictive_summary_stats(
        self,
        theta: dict[str, float],
        context: dict[str, Any],
    ) -> dict[str, float]:
        M = theta.get("M", 30.0)
        a = theta.get("a_star", 0.5)
        eps_f = theta.get("eps_f", 0.0)
        eps_Q = theta.get("eps_Q", 0.0)
        f_gr, q_gr = kerr_qnm_frequency(M, a)
        return {
            "f_rd_hz": f_gr * (1.0 + eps_f),
            "q_rd": q_gr * (1.0 + eps_Q),
            "delta_f_hz": f_gr * eps_f,
            "delta_Q": q_gr * eps_Q,
        }
