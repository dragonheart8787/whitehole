"""Unified data loading with provenance (fail-closed mock fallback)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .provenance import DataLoadError, DataProvenance


def load_observation_data(
    source: str,
    channel: str,
    *,
    event: str | None = None,
    inject_model: str,
    seed: int,
    context: dict[str, Any],
    allow_mock_fallback: bool = False,
    cache_dir: str | None = None,
) -> tuple[Any, DataProvenance]:
    """Load or synthesize observation data with full provenance.

    Parameters
    ----------
    source : str
        Requested source: mock, gwosc, chime, heasarc, eht.
    inject_model : str
        Model used to generate mock data (defaults should equal fit model).
    allow_mock_fallback : bool
        If False (default), any failure to load real data raises DataLoadError.

    Returns
    -------
    data : observation object (dict, SimData, DataFrame, ...)
    provenance : DataProvenance
    """
    if source == "mock":
        data = _load_mock(channel, inject_model, context, seed)
        prov = DataProvenance(
            requested_source="mock",
            actual_source="MOCK_EXPLICIT",
            fallback_used=False,
            inject_model=inject_model,
            event=event,
            channel=channel,
        )
        return data, prov

    if source == "gwosc":
        if not event:
            raise DataLoadError(
                "GWOSC requires --event (e.g. GW150914). "
                f"Validated events: see GWOSCLoader.list_validated_events()",
                requested_source="gwosc",
                reason="missing --event",
            )
        from .gwosc import GWOSCLoader

        loader = GWOSCLoader(
            cache_dir=cache_dir,
            allow_mock_fallback=allow_mock_fallback,
        )
        try:
            bundle = loader.load_event(event)
            det = list(bundle.keys())[0]
            record = bundle[det]
            record = _attach_gw_psd(record)
            actual = record.get("source", "GWOSC")
            prov = DataProvenance(
                requested_source="gwosc",
                actual_source=actual,
                fallback_used=actual == "MOCK_FALLBACK",
                fallback_reason=record.get("fallback_reason"),
                inject_model=None,
                event=event,
                channel=channel,
                extra={"detector": det},
            )
            return record, prov
        except DataLoadError:
            raise
        except Exception as exc:
            if not allow_mock_fallback:
                raise DataLoadError(
                    f"GWOSC load failed for {event!r}: {exc}",
                    requested_source="gwosc",
                    reason=str(exc),
                ) from exc
            data = _load_mock(channel, inject_model, context, seed)
            prov = DataProvenance(
                requested_source="gwosc",
                actual_source="MOCK_FALLBACK",
                fallback_used=True,
                fallback_reason=str(exc),
                inject_model=inject_model,
                event=event,
                channel=channel,
            )
            return data, prov

    if source == "chime":
        from .chime_frb import CHIMEFRBLoader

        loader = CHIMEFRBLoader(cache_dir=cache_dir)
        try:
            df = loader.load_catalog()
            prov = DataProvenance(
                requested_source="chime",
                actual_source="CHIME_CATALOG",
                inject_model=None,
                channel=channel,
            )
            return df, prov
        except Exception as exc:
            if not allow_mock_fallback:
                raise DataLoadError(
                    f"CHIME catalog load failed: {exc}",
                    requested_source="chime",
                    reason=str(exc),
                ) from exc
            data = _load_mock(channel, inject_model, context, seed)
            prov = DataProvenance(
                requested_source="chime",
                actual_source="MOCK_FALLBACK",
                fallback_used=True,
                fallback_reason=str(exc),
                inject_model=inject_model,
                channel=channel,
            )
            return data, prov

    if source in ("heasarc", "eht"):
        if not allow_mock_fallback:
            raise DataLoadError(
                f"Data source {source!r} is not fully implemented for production loads. "
                "Use --data mock for development, or pass --allow-mock-fallback "
                "only if you accept mock substitution.",
                requested_source=source,
                reason="source not implemented",
            )
        data = _load_mock(channel, inject_model, context, seed)
        prov = DataProvenance(
            requested_source=source,
            actual_source="MOCK_FALLBACK",
            fallback_used=True,
            fallback_reason=f"{source} pipeline not implemented; explicit fallback",
            inject_model=inject_model,
            channel=channel,
        )
        return data, prov

    raise DataLoadError(
        f"Unknown data source {source!r}",
        requested_source=source,
        reason="unknown source",
    )


def _attach_gw_psd(record: dict[str, Any]) -> dict[str, Any]:
    """Estimate one-sided PSD on the strain FFT grid for GW likelihoods."""
    from ..utils.math_utils import estimate_psd

    strain = np.asarray(record["strain"], dtype=np.float64)
    sr = float(record["sample_rate"])
    n = len(strain)
    dt = 1.0 / sr
    freqs = np.fft.rfftfreq(n, d=dt)
    freqs_w, psd_w = estimate_psd(strain, sr)
    psd = np.interp(freqs, freqs_w, psd_w, left=float(psd_w[0]), right=float(psd_w[-1]))
    psd = np.where(psd > 0, psd, 1.0e-30)
    out = dict(record)
    out["psd"] = psd
    out.setdefault("t_merger", 0.5 * n / sr)
    return out


def _load_mock(channel: str, inject_model: str, context: dict, seed: int) -> Any:
    from ..models import get_model
    from ..simulators import get_simulator

    model = get_model(inject_model)
    sim = get_simulator(channel)
    params = model.sample_prior(np.random.default_rng(seed))
    ctx = dict(context)
    ctx["rng_seed"] = seed
    return sim.simulate(params, ctx, rng=np.random.default_rng(seed))
