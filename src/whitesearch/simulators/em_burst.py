"""Electromagnetic burst toy simulator for PBH tunneling and magnetar flares.

Models a coherent radio burst (FRB-like) with:
  - Gaussian intrinsic pulse profile
  - DM dispersion (frequency-dependent time delay)
  - Scattering broadening (one-sided exponential convolution)
  - Band-averaged Gaussian receiver noise

Also models a multi-channel X-ray / gamma-ray light curve for high-energy channel.

Context keys (radio)
--------------------
freq_low_mhz  : float — low end of band [MHz] (default 400)
freq_high_mhz : float — high end of band [MHz] (default 800)
n_freq_chans  : int   — number of frequency channels (default 64)
t_start_s     : float — time before burst onset [s] (default 0.1)
t_end_s       : float — time after burst onset [s] (default 0.5)
n_time_bins   : int   — number of time bins (default 2048)
tsys_jy       : float — system temperature in Jy equivalent (default 1000.0)
bandwidth_mhz : float — channel bandwidth [MHz] (derived)
t_samp_ms     : float — sample time [ms] (default 0.1)
rng_seed      : int   — random seed

Context keys (xray)
-------------------
e_low_kev   : float — low energy bound [keV] (default 0.5)
e_high_kev  : float — high energy bound [keV] (default 10.0)
area_cm2    : float — effective area [cm^2] (default 1000)
bg_rate_cps : float — background count rate [counts/s] (default 0.5)
duration_s  : float — light curve duration [s] (default 100.0)
dt_s        : float — time bin width [s] (default 1.0)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import BaseSimulator, SimData
from ..utils.constants import K_DM, JY
from ..utils.math_utils import apply_dm_dispersion, scatter_broaden


class EMBurstSimulator(BaseSimulator):
    """Toy forward simulator for coherent radio burst (FRB-like) signals.

    Produces a frequency-time dynamic spectrum:
      S(ν, t) = F_signal(ν, t) + N(ν, t)

    where F_signal includes intrinsic pulse, DM dispersion, and scattering.
    """

    channel = "radio"

    def simulate(
        self,
        params: dict[str, float],
        context: dict[str, Any],
        rng: np.random.Generator | None = None,
    ) -> SimData:
        if rng is None:
            rng = np.random.default_rng(context.get("rng_seed", None))

        # ── Instrument configuration ───────────────────────────────────────────
        freq_low = float(context.get("freq_low_mhz", 400.0))
        freq_high = float(context.get("freq_high_mhz", 800.0))
        n_freq = int(context.get("n_freq_chans", 64))
        t_start = float(context.get("t_start_s", 0.1))
        t_end = float(context.get("t_end_s", 0.5))
        n_time = int(context.get("n_time_bins", 2048))
        tsys_jy = float(context.get("tsys_jy", 1000.0))
        t_samp_ms = float(context.get("t_samp_ms", 0.1))

        freqs_mhz = np.linspace(freq_low, freq_high, n_freq)
        times_s = np.linspace(-t_start, t_end, n_time)
        dt_s = times_s[1] - times_s[0]
        delta_nu_mhz = (freq_high - freq_low) / n_freq

        # ── Extract burst parameters ───────────────────────────────────────────
        W_int_ms = float(10.0 ** params.get("log10_W_int_ms", 1.0))
        tau_sc_ms_1ghz = float(10.0 ** params.get("log10_tau_sc_ms", 0.0))
        alpha = float(params.get("spectral_index", -1.5))

        # Build DM (sum of contributions)
        dm = self._get_dm(params)

        # Peak flux at reference frequency (1 GHz or band centre)
        nu_ref_mhz = 0.5 * (freq_low + freq_high)
        fluence_jy_ms = self._get_fluence(params)
        W_obs_ms = np.sqrt(W_int_ms**2 + tau_sc_ms_1ghz**2)
        F_peak_jy = fluence_jy_ms / max(W_obs_ms, 1e-6)

        # ── Build noiseless dynamic spectrum ──────────────────────────────────
        # Frequency-dependent peak flux with power-law spectrum
        F_nu = F_peak_jy * (freqs_mhz / nu_ref_mhz) ** alpha  # shape (n_freq,)

        # Intrinsic Gaussian pulse profile (time axis centred at 0)
        sigma_t_s = (W_int_ms * 1e-3) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        pulse = np.exp(-0.5 * (times_s / sigma_t_s) ** 2)  # shape (n_time,)

        # Signal = outer product of spectrum × time profile
        signal = F_nu[:, np.newaxis] * pulse[np.newaxis, :]  # (n_freq, n_time)

        # Apply DM dispersion
        if dm > 0:
            signal = apply_dm_dispersion(times_s, signal, dm, freqs_mhz)

        # Apply scattering broadening (frequency-dependent: τ ∝ ν^{-4})
        for i_f, nu in enumerate(freqs_mhz):
            tau_sc_ms = tau_sc_ms_1ghz * (nu / 1000.0) ** (-4.0)
            if tau_sc_ms > 0.01:
                tau_sc_s = tau_sc_ms * 1e-3
                signal[i_f] = scatter_broaden(times_s, signal[i_f], tau_sc_s)

        # ── Receiver noise ────────────────────────────────────────────────────
        # σ_noise = T_sys / sqrt(δν * t_samp)
        # Here T_sys is given in equivalent Jy units.
        delta_nu_hz = delta_nu_mhz * 1e6
        t_samp_s = t_samp_ms * 1e-3
        sigma_noise = tsys_jy / np.sqrt(delta_nu_hz * t_samp_s)
        noise = rng.standard_normal(signal.shape) * sigma_noise

        data = signal + noise

        return SimData(
            channel="radio",
            data=data,
            metadata={
                "freqs_mhz": freqs_mhz,
                "times_s": times_s,
                "dm": dm,
                "W_int_ms": W_int_ms,
                "W_obs_ms": W_obs_ms,
                "tau_sc_ms_1ghz": tau_sc_ms_1ghz,
                "fluence_jy_ms": fluence_jy_ms,
                "sigma_noise_jy": sigma_noise,
                "spectral_index": alpha,
            },
            params_true=params,
            noise_realisation=noise,
        )

    @staticmethod
    def _get_dm(params: dict[str, float]) -> float:
        if "DM_total" in params:
            return float(params["DM_total"])
        dm_mw = 100.0
        dm_igm = params.get("z", 0.0) * 855.0
        dm_host = params.get("DM_host", 50.0)
        return dm_mw + dm_igm + dm_host

    @staticmethod
    def _get_fluence(params: dict[str, float]) -> float:
        if "fluence_jy_ms" in params:
            return float(params["fluence_jy_ms"])
        return float(10.0 ** params.get("log10_fluence_jy_ms", 0.0))

    # ── Band-averaged pulse ────────────────────────────────────────────────────

    @staticmethod
    def band_averaged_profile(sim_data: SimData) -> NDArray:
        """Return the band-averaged time profile [Jy] from a dynamic spectrum."""
        return np.mean(sim_data.data, axis=0)


class XRayLightCurveSimulator(BaseSimulator):
    """Poisson-process X-ray light curve simulator for PBH WH and GRB alternatives.

    Models source photon counts as:
      C_i ~ Poisson(μ_i)
      μ_i = A * F_model(t_i) * Δt + B * Δt

    where F_model is a power-law decay or Gaussian burst, A is the collecting
    area, and B is the background rate.
    """

    channel = "xray"

    def simulate(
        self,
        params: dict[str, float],
        context: dict[str, Any],
        rng: np.random.Generator | None = None,
    ) -> SimData:
        if rng is None:
            rng = np.random.default_rng(context.get("rng_seed", None))

        e_low = float(context.get("e_low_kev", 0.5))
        e_high = float(context.get("e_high_kev", 10.0))
        area_cm2 = float(context.get("area_cm2", 1000.0))
        bg_cps = float(context.get("bg_rate_cps", 0.5))
        duration_s = float(context.get("duration_s", 100.0))
        dt_s = float(context.get("dt_s", 1.0))

        n_bins = int(duration_s / dt_s)
        times_s = (np.arange(n_bins) + 0.5) * dt_s
        t_peak = duration_s * 0.2  # burst at 20% of segment

        # Source model: Gaussian burst
        log10_fluence_erg = params.get("log10_fluence_erg_cm2", -7.0)
        fluence_erg_cm2 = 10.0 ** log10_fluence_erg
        log10_T90 = params.get("log10_T90_s", 0.0)
        T90_s = 10.0 ** log10_T90

        sigma_t = T90_s / (2.0 * np.sqrt(2.0 * np.log(20.0)))
        from ..utils.constants import KEV_J, ERG_J
        mean_energy_kev = 0.5 * (e_low + e_high)
        mean_energy_erg = mean_energy_kev * KEV_J / ERG_J

        photon_fluence = fluence_erg_cm2 / mean_energy_erg / (e_high - e_low)
        flux_profile = (
            photon_fluence
            / (sigma_t * np.sqrt(2.0 * np.pi))
            * np.exp(-0.5 * ((times_s - t_peak) / sigma_t) ** 2)
        )

        mu_signal = flux_profile * area_cm2 * dt_s
        mu_bg = bg_cps * dt_s
        mu_total = mu_signal + mu_bg

        counts = rng.poisson(mu_total)
        noise = rng.poisson(np.full(n_bins, mu_bg))

        return SimData(
            channel="xray",
            data=counts,
            metadata={
                "times_s": times_s,
                "dt_s": dt_s,
                "mu_signal": mu_signal,
                "mu_bg": np.full(n_bins, mu_bg),
                "area_cm2": area_cm2,
                "e_low_kev": e_low,
                "e_high_kev": e_high,
            },
            params_true=params,
            noise_realisation=noise,
        )
