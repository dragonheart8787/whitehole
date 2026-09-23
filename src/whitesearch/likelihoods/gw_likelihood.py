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
from ..utils.constants import (
    G,
    C,
    M_SUN,
    MPC_M,
    BOUNCE_BURST_FREQ_FACTOR,
    BOUNCE_BURST_Q_FACTOR,
    BOUNCE_BURST_Q_MIN,
)

# Soft floor instead of -inf for dynesty stability.  Must sit BELOW any
# genuine in-band log-likelihood so template-rejected points rank worst:
# on real GWOSC data (GW150914 H1, 32 s, band-masked) the null lnL is
# ~-5.8e7 and the worst prior-draw template lnL observed is ~-1.9e9.  The
# previous -1e6 floor sat ABOVE all genuine values on that scale, so
# rejected templates formed the global-maximum plateau and dynesty
# terminated on it, returning ln Z = floor + ln(plateau prior mass).
LL_MIN = -1e12
HH_MIN = 1e-30

# Tukey taper fraction applied before every rfft on the GW likelihood path
# (strain AND template, so the residual d-h sees identical windowing).
# alpha=0.1 matches the taper already used when constructing mock injected
# signals in simulators/grav_wave.py, keeping the mock and real-data paths
# on the same convention; it also sits at the low end of the 0.1-0.25 range
# common in LIGO/Virgo analyses, tapering only the outer 5% of samples at
# each edge of the 32 s segment so a merger centred well away from the edges
# (t_merger ~ mid-segment) is essentially untouched. See time_to_freq() in
# gw_units.py for why no additional power-compensation factor is applied.
TAPER_ALPHA = 0.1

# Taper is a correction for ONE specific defect: a full-segment rectangular-window
# rfft analysed against a PSD estimated by Welch (short, Hann-windowed segments).
# The two disagree about spectral leakage, and on real strain that mismatch
# inflates noise-weighted inner products near the band edges.  Data whose PSD is
# the exact analytic spectrum its noise was generated from has no such mismatch:
# the mock simulator fills rfft bins directly and irffts them, so at
# taper_alpha = 0 its spectrum is diagonal by construction (measured per-bin
# <n|n> = 1.96-2.05 against a theory value of 2.0).  Applying the taper there
# CREATES the problem it was meant to fix: the time-domain window convolves in
# frequency and smears the 20-60 Hz seismic wall of aligo_psd_analytic() -- which
# spans ~20 decades inside the analysis band -- across 100-1700 Hz, inflating
# per-bin <n|n> to 5.8e6-1.3e8 and displacing the likelihood maximum of a loud
# injection by 100+ carrier periods.  See docs/BOUNCE_PREFLIGHT_AUDIT.md Part H.
#
# So the taper follows the data's PSD provenance, not a global constant.  The
# discriminator is the existing ``source`` provenance field carried in the GW
# observation metadata (dataio/gw_observation.py, dataio/gwosc.py).
TAPER_ALPHA_NONE = 0.0

# Sources whose PSD is the generative spectrum on the same un-windowed basis as
# the analysis rfft.  Everything else -- GWOSC strain, and mock strain that has
# been through GWPreprocessor's bandpass/notch/Welch pipeline (source
# MOCK_EXPLICIT, MOCK, MOCK_FALLBACK) -- carries a Welch-estimated PSD and needs
# the taper exactly as before.  Measured for MOCK_EXPLICIT: per-bin <d|d> is
# 3.4e9 at taper 0.0 versus 9.9 at taper 0.1, so preprocessed mock belongs with
# real data, not with the raw simulator.
NO_TAPER_SOURCES = frozenset({"MOCK_SIMULATOR"})

# Recorded verbatim in the run metadata so the choice is never invisible.
TAPER_REASON_GENERATIVE_PSD = (
    "source_psd_is_generative_no_window_mismatch"
)
TAPER_REASON_WELCH_PSD = "source_psd_is_welch_estimated_window_mismatch"
TAPER_REASON_UNKNOWN_SOURCE = "source_absent_default_to_real_data_convention"


def taper_for_source(source: str | None) -> tuple[float, str]:
    """Return ``(taper_alpha, reason)`` for a GW observation's ``source`` tag.

    An absent or unrecognised ``source`` keeps the real-data convention
    (``TAPER_ALPHA``): every caller that predates this branch is unaffected,
    and a new data path has to opt in explicitly to skip the taper rather
    than inheriting the skip by omission.
    """
    if source is None:
        return TAPER_ALPHA, TAPER_REASON_UNKNOWN_SOURCE
    if str(source) in NO_TAPER_SOURCES:
        return TAPER_ALPHA_NONE, TAPER_REASON_GENERATIVE_PSD
    return TAPER_ALPHA, TAPER_REASON_WELCH_PSD


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
        # Provenance of the most recent evaluation's taper choice.  Populated
        # by every loglike() call and copied into run metadata by the
        # inference runners, so "this run skipped the taper because the data
        # was mock" is always visible in the record.
        self.last_taper_config: dict[str, Any] = {}

    @property
    def parameter_names(self) -> list[str]:
        if self.model_name == "null":
            return []
        if self.model_name == "bounce":
            # log10_dt_bounce_s (the burst delay measured from the merger) and
            # log10_A_bounce are both sampled.  log10_tau_bounce_yr -- the
            # cosmological BH lifetime -- is NOT: it is a different physical
            # quantity that no observable strain segment constrains.  See
            # docs/BOUNCE_PREFLIGHT_AUDIT.md sections B.3 and D.1.
            return [
                "M", "a_star", "eps_f", "eps_Q",
                "log10_A_bounce", "log10_dt_bounce_s",
                "D_L", "i",
            ]
        # bh_ringdown (and any other non-bounce GW model routed here) is a
        # phenomenological ringdown: the template amplitude is the free
        # log10_A used by _build_template() below.  D_L and i used to be
        # listed here but never entered the bh_ringdown template, so the
        # sampler explored two parameters the likelihood was flat in.
        return ["M", "a_star", "log10_A"]

    def taper_config(self, data: Any) -> dict[str, Any]:
        """Resolve the taper for ``data`` from its ``source`` provenance tag.

        Returns the audit record written into run metadata:
        ``data_source``, ``taper_alpha_used``, ``taper_alpha_reason``.
        """
        meta = data.metadata if hasattr(data, "metadata") else data
        source = None
        if isinstance(meta, dict):
            source = meta.get("source")
        alpha, reason = taper_for_source(source)
        return {
            "data_source": source,
            "taper_alpha_used": float(alpha),
            "taper_alpha_reason": reason,
        }

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

        taper_cfg = self.taper_config(data)
        self.last_taper_config = taper_cfg
        taper_alpha = taper_cfg["taper_alpha_used"]

        n = len(strain)
        dt = 1.0 / sample_rate
        freqs, strain_f, df = time_to_freq(strain, dt, taper_alpha=taper_alpha)
        nyquist = sample_rate / 2.0

        times = np.arange(n) * dt
        h_template = self._build_template(theta, times, t_merger, freqs, nyquist, low_freq)
        if h_template is None:
            return self.ll_min

        # Same taper as the strain above: the residual strain_f - template_f
        # must see identical windowing or the mismatch shows up as spurious
        # power at the band edges.
        _, h_template_f, _ = time_to_freq(h_template, dt, taper_alpha=taper_alpha)

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
        taper_cfg = self.taper_config(data)
        self.last_taper_config = taper_cfg
        freqs, strain_f, df = time_to_freq(
            strain, 1.0 / sample_rate, taper_alpha=taper_cfg["taper_alpha_used"]
        )
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

        # The bounce burst sits at BOUNCE_BURST_FREQ_FACTOR * f_rd, so a
        # ringdown inside the band can still put its burst below the low-
        # frequency cutoff.  Reject those the same way, instead of leaving a
        # blind spot where the template carries a component the inner product
        # never sees.  See docs/BOUNCE_PREFLIGHT_AUDIT.md section B.5.
        f_burst = f_rd * BOUNCE_BURST_FREQ_FACTOR
        if self.model_name == "bounce" and (
            f_burst < low_freq or f_burst > 0.95 * nyquist
        ):
            return None

        h = ringdown_waveform(times, t_merger, A_rd, f_rd, q_rd)

        if self.model_name == "bounce":
            # Burst delay is measured from the merger in seconds
            # (log10_dt_bounce_s), not converted from the cosmological
            # lifetime log10_tau_bounce_yr.  Under the old parameterisation the
            # shortest prior draw put the burst 3.156e+04 s after the merger,
            # i.e. never inside a 4 s or 32 s segment (prior mass 0.0000), so
            # the burst was a dead component.  See
            # docs/BOUNCE_PREFLIGHT_AUDIT.md sections B.3 and D.1.
            if "log10_A_bounce" in theta and "log10_dt_bounce_s" in theta:
                A_b = float(10.0 ** theta["log10_A_bounce"])
                dt_bounce = float(10.0 ** theta["log10_dt_bounce_s"])
                t_bounce = t_merger + dt_bounce
                if t_bounce >= times[-1]:
                    # fail closed: a burst outside the segment is a template
                    # this data cannot constrain, not a silent pure ringdown.
                    return None
                h = h + ringdown_waveform(
                    times,
                    t_bounce,
                    A_b,
                    f_burst,
                    max(BOUNCE_BURST_Q_MIN, BOUNCE_BURST_Q_FACTOR * q_rd),
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
