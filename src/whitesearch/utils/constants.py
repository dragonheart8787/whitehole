"""Physical constants and astrophysical unit conversions used throughout WhiteSearch.

All values in SI unless otherwise noted. Sources: CODATA 2018, IAU 2012/2015.
"""

from __future__ import annotations

# ── Fundamental physical constants (SI) ────────────────────────────────────────
G = 6.67430e-11          # m^3 kg^{-1} s^{-2}  — Newton's constant
C = 2.99792458e8         # m s^{-1}             — speed of light
HBAR = 1.054571817e-34   # J s                  — reduced Planck constant
K_B = 1.380649e-23       # J K^{-1}             — Boltzmann constant

# ── Planck units ────────────────────────────────────────────────────────────────
M_PLANCK = 2.176434e-8   # kg
L_PLANCK = 1.616255e-35  # m
T_PLANCK = 5.391247e-44  # s
E_PLANCK = 1.956e9       # J

# ── Astrophysical mass scales ───────────────────────────────────────────────────
M_SUN = 1.98892e30       # kg — solar mass
M_EARTH = 5.9722e24      # kg — Earth mass

# ── Distance / angle conversions ───────────────────────────────────────────────
PC_M = 3.085677581e16    # m pc^{-1}
KPC_M = 3.085677581e19   # m kpc^{-1}
MPC_M = 3.085677581e22   # m Mpc^{-1}
GPC_M = 3.085677581e25   # m Gpc^{-1}

AU_M = 1.495978707e11    # m AU^{-1}
LY_M = 9.4607304725808e15  # m ly^{-1}

ARCSEC_RAD = 4.84813681e-6   # rad arcsec^{-1}
MAS_RAD = 4.84813681e-9      # rad mas^{-1}
MUAS_RAD = 4.84813681e-12    # rad μas^{-1}

# ── Time conversions ────────────────────────────────────────────────────────────
YR_S = 3.15576e7         # s yr^{-1}
MYR_S = 3.15576e13       # s Myr^{-1}
GYR_S = 3.15576e16       # s Gyr^{-1}

# ── Flux / luminosity ────────────────────────────────────────────────────────────
JY = 1.0e-26             # W m^{-2} Hz^{-1} — 1 Jansky
ERG_J = 1.0e-7           # J erg^{-1}
L_SUN = 3.828e26         # W — solar luminosity

# ── Gravitational wave ───────────────────────────────────────────────────────────
# Schwarzschild l=m=2 QNM fit coefficients (Echeverria 1989)
#   f_QNM [Hz] = F_QNM_COEFF * (M_sun/M)
#   τ_QNM [s]  = T_QNM_COEFF * (M/M_sun)
F_QNM_SCHW = 1.207e4    # Hz  (Schwarzschild, l=m=2, for M = 1 M_sun)
T_QNM_SCHW = 5.54e-4    # s   (decay time, for M = 1 M_sun)
Q_QNM_SCHW = 2.0        # dimensionless quality factor (Schwarzschild)

# Kerr QNM fitting function coefficients (Berti, Cardoso & Will 2006 Table VIII)
# f_220(M, a*) ≈ (F1 + F2*(1-a*)^F3) / (2π * G*M/c^3)
QNM_F1 = 1.5251
QNM_F2 = -1.1568
QNM_F3 = 0.1292

# QNM quality factor: Q_220(a*) ≈ Q1 + Q2*(1-a*)^Q3
QNM_Q1 = 0.7000
QNM_Q2 = 1.4187
QNM_Q3 = -0.4990

# ── Radio / FRB ──────────────────────────────────────────────────────────────────
# DM dispersion constant: t_delay [ms] = K_DM * DM [pc/cm^3] / nu^2 [MHz]
K_DM = 4.148808e3        # MHz^2 pc^{-1} cm^3 ms

# Typical ISM scattering index
ISM_SCATT_INDEX = -4.0   # ν^{-4} frequency scaling

# ── X-ray ────────────────────────────────────────────────────────────────────────
KEV_J = 1.60218e-16      # J keV^{-1}
KEV_HZ = 2.41799e17      # Hz keV^{-1}

# ── Cosmology (Planck 2018, flat ΛCDM) ──────────────────────────────────────────
H0 = 67.4                # km s^{-1} Mpc^{-1}
OMEGA_M = 0.315
OMEGA_LAMBDA = 0.685
OMEGA_B = 0.0493

# DM–z relation coefficient (Macquart+ 2020)
DM_IGM_PER_Z = 855.0    # pc/cm^3 per unit redshift (average)


# ── Bounce burst waveform, relative to the ringdown ──────────────────────────
# Shared by simulators/grav_wave.py (injection), likelihoods/gw_likelihood.py
# (template) and models/bounce.py (prior derivation) so the three cannot drift
# apart.  The burst is modelled as a second damped sinusoid at
# BOUNCE_BURST_FREQ_FACTOR * f_ringdown with quality
# max(BOUNCE_BURST_Q_MIN, BOUNCE_BURST_Q_FACTOR * Q_ringdown).
BOUNCE_BURST_FREQ_FACTOR = 0.8
BOUNCE_BURST_Q_FACTOR = 0.5
BOUNCE_BURST_Q_MIN = 2.0
