"""Per-target known constants for the EHT image/VLBI channel.

The image channel analyses one named source at a time.  Its distance is not
something a VLBI image measures -- the image constrains the ring's *angular*
radius, which depends on mass and distance only through the ratio ``M / D_L``
-- so sampling ``D_L`` alongside ``M`` adds a 3.3 dex direction the data cannot
constrain and destroys the prior's representability (see
``GREternalWhiteHole.ring_radius_representable_range_muas`` and
docs/XRAY_IMAGE_PREFLIGHT_AUDIT.md X.6).  Distance is therefore a per-target
known constant, looked up from the target name the analysis declares.

How the target reaches the forward model mirrors the GW channel's handling of
per-event known quantities (``t_merger``, ``low_freq_cutoff`` in
``GWLikelihood._parse_data``): the observation's own metadata first, the
analysis context second.  The one deliberate difference is that there is no
third fallback -- a missing target raises, because silently adopting one
source's distance for another's data is a forward-model mismatch of exactly
the kind ``VisibilityLikelihood._require_uv_coverage`` exists to prevent.

Note on the name ``source``: ``EHTLoader`` already uses that key for the target
name ("M87" / "SgrA"), while the GW and radio paths use ``source`` for data
provenance ("GWOSC", "MOCK_EXPLICIT", ...).  This module introduces a separate
``target`` key rather than overloading either meaning; ``EHTLoader`` now emits
both, with ``source`` left untouched for backwards compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ImageTarget:
    """A VLBI target whose distance and mass scale are known a priori.

    Attributes
    ----------
    name : canonical target name.
    distance_mpc : luminosity distance [Mpc], held fixed during sampling.
    mass_msun : best-estimate mass [M_sun]; the prior is centred on this but
        is NOT this value -- see ``mass_prior_low_msun`` / ``mass_prior_high_msun``.
    mass_prior_low_msun, mass_prior_high_msun : the log-uniform mass prior this
        project samples for the target.  Chosen to bracket the *independent*
        published measurements, not to make the ring representable on any
        particular image grid; the grid is checked against the prior, never the
        other way round.
    distance_source, mass_source : literature citations for the two above.
    """

    name: str
    distance_mpc: float
    distance_source: str
    mass_msun: float
    mass_prior_low_msun: float
    mass_prior_high_msun: float
    mass_source: str


#: Targets this project's image channel knows how to analyse.
#:
#: M87* distance and mass: EHT Collaboration 2019, ApJL 875, L6 (Paper VI),
#: which adopts D = 16.8 (+0.8/-0.7) Mpc and reports
#: M = (6.5 +/- 0.2|stat +/- 0.7|sys) x 10^9 M_sun.  The prior is widened to
#: [3.0e9, 1.0e10] to cover the two pre-EHT dynamical measurements the paper
#: compares against, which differ by a factor ~1.8: stellar dynamics
#: 6.2e9 (Gebhardt et al. 2011, ApJ 729, 119) and gas dynamics 3.5e9
#: (Walsh et al. 2013, ApJ 770, 86).
#:
#: Sgr A* distance and mass: GRAVITY Collaboration 2019, A&A 625, L10, which
#: reports R0 = 8178 +/- 13|stat +/- 22|sys pc (= 0.008178 Mpc) and
#: M = (4.154 +/- 0.014|stat +/- 0.014|sys) x 10^6 M_sun.  The prior is widened
#: to [3.5e6, 5.0e6] to cover the independent Keck orbit fit,
#: M = 3.975e6 +/- 0.058e6 at R0 = 7959 pc (Do et al. 2019, Science 365, 664).
EHT_TARGETS: dict[str, ImageTarget] = {
    "M87*": ImageTarget(
        name="M87*",
        distance_mpc=16.8,
        distance_source="EHT Collaboration 2019, ApJL 875, L6",
        mass_msun=6.5e9,
        mass_prior_low_msun=3.0e9,
        mass_prior_high_msun=1.0e10,
        mass_source="EHT 2019 ApJL 875 L6; Gebhardt+2011 ApJ 729 119; Walsh+2013 ApJ 770 86",
    ),
    "SgrA*": ImageTarget(
        name="SgrA*",
        distance_mpc=0.008178,
        distance_source="GRAVITY Collaboration 2019, A&A 625, L10",
        mass_msun=4.154e6,
        mass_prior_low_msun=3.5e6,
        mass_prior_high_msun=5.0e6,
        mass_source="GRAVITY 2019 A&A 625 L10; Do+2019 Science 365 664",
    ),
}

#: Spellings accepted for each canonical name.  Deliberately an explicit table
#: rather than fuzzy matching: an unrecognised spelling must raise, not resolve
#: to whichever target happens to be closest.
_TARGET_ALIASES: dict[str, str] = {
    "m87": "M87*",
    "m87*": "M87*",
    "m 87": "M87*",
    "ngc4486": "M87*",
    "ngc 4486": "M87*",
    "sgra": "SgrA*",
    "sgra*": "SgrA*",
    "sgr a": "SgrA*",
    "sgr a*": "SgrA*",
    "sagittarius a*": "SgrA*",
}


def resolve_target_name(raw: Any) -> str:
    """Canonical target name for a user-supplied spelling.

    Raises
    ------
    ValueError
        If the name is not one of the known targets.  Fail-closed: there is no
        "default target", because the distance that would follow from guessing
        one is wrong by up to 3.3 dex.
    """
    key = str(raw).strip().lower()
    if key in _TARGET_ALIASES:
        return _TARGET_ALIASES[key]
    raise ValueError(
        f"Unknown image-channel target {raw!r}. Known targets: "
        f"{sorted(EHT_TARGETS)} (accepted spellings: {sorted(_TARGET_ALIASES)}). "
        "The target fixes the source distance, which the image channel treats "
        "as a known constant rather than a sampled parameter."
    )


def get_target(raw: Any) -> ImageTarget:
    """Look up an :class:`ImageTarget` by any accepted spelling."""
    if isinstance(raw, ImageTarget):
        return raw
    return EHT_TARGETS[resolve_target_name(raw)]


def require_target(*sources: Any) -> ImageTarget:
    """Resolve the analysis target from the first source that declares one.

    ``sources`` are searched in priority order -- pass observation metadata
    first and the analysis context second, matching
    ``GWLikelihood._parse_data``'s ``meta.get(..., context.get(...))`` ordering
    and ``VisibilityLikelihood._require_uv_coverage``.

    Raises
    ------
    KeyError
        If no source carries a ``target``.  There is no fallback by design.
    """
    seen: list[str] = []
    for source in sources:
        if isinstance(source, dict):
            seen.extend(source)
            value = source.get("target")
            if value is not None:
                return get_target(value)
    raise KeyError(
        "The image channel needs to know which target it is analysing: no "
        "'target' key in the observation metadata or the analysis context. "
        f"Set it to one of {sorted(EHT_TARGETS)}. Keys seen: {sorted(set(seen))}. "
        "The target supplies the source distance, which this channel holds "
        "fixed instead of sampling it."
    )


def target_distance_mpc(*sources: Any) -> float:
    """Distance [Mpc] of the target declared by the first source that has one."""
    return require_target(*sources).distance_mpc
