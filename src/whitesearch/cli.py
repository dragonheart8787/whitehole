"""WhiteSearch command-line interface (package entry point).

Installed as: whitesearch = whitesearch.cli:main

Core workflow is model comparison (Bayes factors), not single-model ln Z alone.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import click
import numpy as np
import yaml

from whitesearch.dataio.loader import load_observation_data
from whitesearch.dataio.provenance import DataLoadError, DataProvenance
from whitesearch.inference import BilbyRunner, compute_bayes_factor
from whitesearch.inference.bilby_runner import BILBY_AVAILABLE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("whitesearch.cli")

MODEL_CHOICES = [
    "bounce", "pbh_tunneling", "gr_eternal",
    "null", "magnetar", "grb_frb", "bh_ringdown", "bh_accretion",
]
CHANNEL_CHOICES = ["gw", "radio", "xray", "image"]
DATA_CHOICES = ["mock", "gwosc", "chime", "heasarc", "eht"]


def _allow_mock_opt(f):
    return click.option(
        "--allow-mock-fallback/--no-allow-mock-fallback",
        default=False,
        show_default=True,
        help="If real data load fails, substitute mock data (otherwise exit with error).",
    )(f)


@click.group()
@click.version_option(version="0.1.1", prog_name="whitesearch")
def main():
    """WhiteSearch — white hole candidate signal search and evidence ranking.

    This tool ranks candidate models; it does not prove white holes exist.
    """


@main.command()
@click.option("--model", required=True, type=click.Choice(MODEL_CHOICES, case_sensitive=False))
@click.option("--channel", default="gw", type=click.Choice(CHANNEL_CHOICES))
@click.option("--data", default="mock", type=click.Choice(DATA_CHOICES))
@click.option("--event", default=None, help="Event name (required for gwosc)")
@click.option("--inject-model", default=None, help="Model for mock synthesis (default: same as --model)")
@click.option("--config", default=None, help="Run config YAML")
@click.option("--nlive", default=500, show_default=True)
@click.option("--outdir", default="artifacts/bilby", show_default=True)
@click.option("--seed", default=42, show_default=True)
@click.option("--resume/--no-resume", default=True)
@_allow_mock_opt
def fit(model, channel, data, event, inject_model, config, nlive, outdir, seed, resume, allow_mock_fallback):
    """Fit one model and report ln Z with full data/sampler provenance."""
    inject_model = inject_model or model
    cfg = _load_run_config(config, channel)
    context = cfg.get("instrument", _default_context(channel))
    context["rng_seed"] = seed

    try:
        obs_data, prov = load_observation_data(
            data, channel, event=event, inject_model=inject_model,
            seed=seed, context=context, allow_mock_fallback=allow_mock_fallback,
        )
    except DataLoadError as exc:
        click.echo(f"ERROR: {exc}", err=True)
        sys.exit(1)

    result = _run_single_fit(
        model, channel, obs_data, context, prov, inject_model,
        nlive=nlive, outdir=outdir, seed=seed, resume=resume,
        event=event, data=data,
    )
    _print_warnings(prov, result)
    _print_fit_summary(model, inject_model, prov, result, channel, data, event)


@main.command("compare")
@click.option("--model", required=True, type=click.Choice(MODEL_CHOICES, case_sensitive=False),
              help="White-hole candidate model")
@click.option("--null", "null_model", default="null", show_default=True,
              type=click.Choice(MODEL_CHOICES, case_sensitive=False))
@click.option("--alt", "alt_model", default=None,
              type=click.Choice(MODEL_CHOICES, case_sensitive=False),
              help="Best astrophysical alternative (default: channel-specific)")
@click.option("--channel", default="gw", type=click.Choice(CHANNEL_CHOICES))
@click.option("--data", default="mock", type=click.Choice(DATA_CHOICES))
@click.option("--event", default=None)
@click.option("--inject-model", default=None)
@click.option("--config", default=None)
@click.option("--nlive", default=300, show_default=True)
@click.option("--outdir", default="artifacts/compare", show_default=True)
@click.option("--seed", default=42, show_default=True)
@_allow_mock_opt
def compare(model, null_model, alt_model, channel, data, event, inject_model, config,
            nlive, outdir, seed, allow_mock_fallback):
    """Compare candidate vs null and alternative via Bayes factors (core workflow)."""
    inject_model = inject_model or model
    alt_model = alt_model or _default_alt_model(channel)
    cfg = _load_run_config(config, channel)
    context = cfg.get("instrument", _default_context(channel))
    context["rng_seed"] = seed

    try:
        obs_data, prov = load_observation_data(
            data, channel, event=event, inject_model=inject_model,
            seed=seed, context=context, allow_mock_fallback=allow_mock_fallback,
        )
    except DataLoadError as exc:
        click.echo(f"ERROR: {exc}", err=True)
        sys.exit(1)

    _print_warnings(prov, None)

    results: dict[str, Any] = {}
    for m in [model, null_model, alt_model]:
        click.echo(f"\n--- Fitting model: {m} ---")
        results[m] = _run_single_fit(
            m, channel, obs_data, context, prov, inject_model,
            nlive=nlive, outdir=outdir, seed=seed + hash(m) % 1000,
            resume=False, event=event, data=data, label_suffix=m,
        )

    bf = compute_bayes_factor(
        results,
        wh_model=model,
        null_model=null_model,
        alt_model=alt_model,
    )

    click.echo("\n" + "=" * 60)
    click.echo("MODEL COMPARISON (candidate ranking — not detection)")
    click.echo("=" * 60)
    click.echo(f"Candidate model:     {model}")
    click.echo(f"Null reference:      {null_model}")
    click.echo(f"Alternative model:   {alt_model}")
    click.echo(f"Data:                {prov.summary_line()}")
    click.echo(f"Inject model (mock): {inject_model}")
    click.echo("-" * 60)
    click.echo(f"ln BF vs null:  {bf['ln_BF_vs_null']:.3f} ± {bf['ln_BF_vs_null_err']:.3f}")
    click.echo(f"  interpretation: {bf['interpretation_vs_null']}")
    click.echo(f"  internal gate:  {'PASS' if bf.get('gate_internal_passed') else 'FAIL'}")
    if "ln_BF_vs_alt" in bf:
        click.echo(f"ln BF vs alt:   {bf['ln_BF_vs_alt']:.3f} ± {bf['ln_BF_vs_alt_err']:.3f}")
        click.echo(f"  interpretation: {bf['interpretation_vs_alt']}")
        click.echo(f"  publication gate (alt): {'PASS' if bf.get('gate_alt_publication_passed') else 'FAIL'}")
    click.echo("=" * 60 + "\n")

    _save_compare_artifact(outdir, bf, prov, results, model, null_model, alt_model, inject_model)


@main.command()
@click.option("--models", required=True,
              help="Comma-separated models to rank, e.g. bounce,bh_ringdown,magnetar,null")
@click.option("--reference", default="null", show_default=True,
              type=click.Choice(MODEL_CHOICES, case_sensitive=False))
@click.option("--channel", default="gw", type=click.Choice(CHANNEL_CHOICES))
@click.option("--data", default="mock", type=click.Choice(DATA_CHOICES))
@click.option("--event", default=None)
@click.option("--inject-model", default=None)
@click.option("--nlive", default=200, show_default=True)
@click.option("--outdir", default="artifacts/rank", show_default=True)
@click.option("--seed", default=42, show_default=True)
@_allow_mock_opt
def rank(models, reference, channel, data, event, inject_model, nlive, outdir, seed, allow_mock_fallback):
    """Rank multiple models by ln Z and ln BF vs a reference."""
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    inject_model = inject_model or model_list[0]
    cfg = _load_run_config(None, channel)
    context = cfg.get("instrument", _default_context(channel))
    context["rng_seed"] = seed

    try:
        obs_data, prov = load_observation_data(
            data, channel, event=event, inject_model=inject_model,
            seed=seed, context=context, allow_mock_fallback=allow_mock_fallback,
        )
    except DataLoadError as exc:
        click.echo(f"ERROR: {exc}", err=True)
        sys.exit(1)

    results = {}
    for m in model_list:
        click.echo(f"Fitting {m} …")
        try:
            _get_likelihood(channel, m)
        except ValueError as exc:
            click.echo(f"ERROR: {exc}", err=True)
            sys.exit(1)
        results[m] = _run_single_fit(
            m, channel, obs_data, context, prov, inject_model,
            nlive=nlive, outdir=outdir, seed=seed + hash(m) % 1000,
            resume=False, event=event, data=data, label_suffix=m,
        )

    runner = BilbyRunner(nlive=nlive, outdir=outdir, seed=seed)
    table = runner.compare_models(results, reference=reference)
    click.echo("\n" + table.to_string(index=False))
    click.echo(f"\nReference model: {reference}")
    click.echo(f"Data provenance: {prov.summary_line()}")


@main.command()
@click.option("--model", required=True, type=click.Choice(
    ["bounce", "pbh_tunneling", "gr_eternal"], case_sensitive=False
))
@click.option("--channel", default="gw", type=click.Choice(CHANNEL_CHOICES))
@click.option("--n-injections", default=100, show_default=True)
@click.option("--nlive", default=200, show_default=True)
@click.option("--outdir", default="artifacts/injections", show_default=True)
@click.option("--seed", default=42, show_default=True)
def inject(model, channel, n_injections, nlive, outdir, seed):
    """Run injection / recovery and print coverage statistics."""
    from whitesearch.models import get_model
    from whitesearch.simulators import get_simulator
    from whitesearch.inference import BilbyRunner
    from whitesearch.validation import InjectionRecovery

    model_obj = get_model(model)
    sim = get_simulator(channel)
    ll = _get_likelihood(channel, model)
    context = _default_context(channel)

    runner = BilbyRunner(nlive=nlive, outdir=f"{outdir}/bilby", seed=seed)
    ir = InjectionRecovery(simulator=sim, runner=runner, n_injections=n_injections, rng_seed=seed)

    click.echo(f"Running {n_injections} injections for model={model}, channel={channel} …")
    result = ir.run_injections(model_obj, ll, context, save_dir=outdir)
    click.echo("\nCoverage Summary:")
    click.echo(result.summary().to_string(index=False))


@main.command()
@click.option("--model", required=True)
@click.option("--channel", default="radio", type=click.Choice(CHANNEL_CHOICES))
@click.option("--param", default=None)
@click.option("--n-injections", default=200, show_default=True)
@click.option("--outdir", default="artifacts/sensitivity", show_default=True)
@click.option("--seed", default=42, show_default=True)
def sensitivity(model, channel, param, n_injections, outdir, seed):
    """Generate sensitivity curves for a model parameter."""
    from whitesearch.models import get_model
    from whitesearch.simulators import get_simulator
    from whitesearch.inference import BilbyRunner
    from whitesearch.validation import InjectionRecovery
    from whitesearch.validation.sensitivity import SensitivityAnalyzer

    model_obj = get_model(model)
    if param is None:
        param = model_obj.parameter_names[0]
    sim = get_simulator(channel)
    ll = _get_likelihood(channel, model)
    context = _default_context(channel)

    runner = BilbyRunner(nlive=100, outdir=f"{outdir}/bilby", seed=seed)
    ir = InjectionRecovery(simulator=sim, runner=runner, n_injections=n_injections, rng_seed=seed)
    ir_result = ir.run_injections(model_obj, ll, context)

    sens = SensitivityAnalyzer().compute_sensitivity(
        ir_result.theta_true, ir_result.evidences, param
    )
    click.echo(f"\nSensitivity curve for {param}:")
    for b, r in zip(sens["param_bins"], sens["recovery_fraction"]):
        click.echo(f"  {b:.3g}: {r:.1%}")


@main.command()
@click.option("--run-dir", default="artifacts/bilby", show_default=True)
@click.option("--output", default="artifacts/report.md", show_default=True)
def report(run_dir, output):
    """Generate Markdown report from saved run artifacts."""
    run_path = Path(run_dir)
    if not run_path.exists():
        click.echo(f"Run directory {run_dir} not found.", err=True)
        sys.exit(1)

    meta_files = sorted(run_path.glob("*_metadata.json"))
    compare_file = run_path / "compare_summary.json"
    is_dev = False
    lines: list[str] = []

    if meta_files:
        for mf in meta_files:
            meta = json.loads(mf.read_text())
            if meta.get("sampler") == "toy_importance_sampling" or meta.get(
                "is_approximate_evidence"
            ):
                is_dev = True
                break

    title = (
        "# WhiteSearch Development / Smoke-Test Report\n"
        if is_dev
        else "# WhiteSearch Technical Report\n"
    )
    lines.append(title)
    lines.append("\n> Candidate ranking engine — Bayes factors do not constitute detection.\n\n")
    lines.append(f"**Run directory:** `{run_dir}`\n\n")

    # Provenance / fallback section
    lines.append("## Data provenance\n\n")
    for mf in meta_files:
        meta = json.loads(mf.read_text())
        prov = meta.get("provenance", {})
        if prov.get("fallback_used"):
            lines.append(
                f"- **WARNING** `{mf.stem}`: requested `{prov.get('requested_source')}` "
                f"but used `{prov.get('actual_source')}` — {prov.get('fallback_reason')}\n"
            )
        else:
            lines.append(f"- `{mf.stem}`: {prov.get('actual_source', 'unknown')}\n")
    lines.append("\n")

    if compare_file.exists():
        comp = json.loads(compare_file.read_text())
        lines.append("## Model comparison\n\n")
        lines.append(f"| Metric | Value |\n|--------|-------|\n")
        for k, v in comp.items():
            if isinstance(v, float):
                lines.append(f"| {k} | {v:.4f} |\n")
            else:
                lines.append(f"| {k} | {v} |\n")
        lines.append("\n")

    lines.append("## Evidence summary\n\n")
    lines.append("| Label | ln Z | ln Z_err | Sampler | Approximate? |\n")
    lines.append("|-------|------|----------|---------|-------------|\n")
    for mf in meta_files:
        meta = json.loads(mf.read_text())
        lines.append(
            f"| {meta.get('label', mf.stem)} | {meta.get('log_evidence', 'n/a'):.3f} | "
            f"{meta.get('log_evidence_err', 'n/a'):.3f} | {meta.get('sampler', '?')} | "
            f"{'YES' if meta.get('is_approximate_evidence') else 'no'} |\n"
        )

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("".join(lines), encoding="utf-8")
    click.echo(f"Report written to {output}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_single_fit(
    model: str,
    channel: str,
    obs_data: Any,
    context: dict,
    prov: DataProvenance,
    inject_model: str,
    *,
    nlive: int,
    outdir: str,
    seed: int,
    resume: bool,
    event: str | None,
    data: str,
    label_suffix: str | None = None,
):
    from whitesearch.models import get_model

    model_obj = get_model(model)
    ll = _get_likelihood(channel, model)
    label = label_suffix or f"{model}_{channel}_{event or data}"
    runner = BilbyRunner(sampler="dynesty", nlive=nlive, outdir=outdir, resume=resume, seed=seed)
    result = runner.run(ll, obs_data, context, model_obj, label=label)

    meta = {
        "label": label,
        "fit_model": model,
        "injected_model": inject_model,
        "channel": channel,
        "log_evidence": result.log_evidence,
        "log_evidence_err": result.log_evidence_err,
        "sampler": result.metadata.get("sampler"),
        "is_approximate_evidence": result.metadata.get("is_approximate_evidence", False),
        "provenance": prov.to_dict(),
    }
    _save_run_metadata(outdir, label, meta)
    return result


def _save_run_metadata(outdir: str, label: str, meta: dict) -> None:
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)
    safe = label.replace("/", "_")
    (path / f"{safe}_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def _save_compare_artifact(
    outdir: str, bf: dict, prov: DataProvenance,
    results: dict, model: str, null_m: str, alt_m: str, inject_model: str,
) -> None:
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        **bf,
        "provenance": prov.to_dict(),
        "fit_model": model,
        "null_model": null_m,
        "alt_model": alt_m,
        "inject_model": inject_model,
        "models": {k: v.log_evidence for k, v in results.items()},
    }
    (path / "compare_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _print_warnings(prov: DataProvenance, result: Any | None) -> None:
    if prov.fallback_used:
        click.echo("\n*** WARNING: MOCK DATA FALLBACK ***", err=True)
        click.echo(f"  requested_source: {prov.requested_source}", err=True)
        click.echo(f"  actual_source:      {prov.actual_source}", err=True)
        click.echo(f"  fallback_reason:    {prov.fallback_reason}", err=True)
        click.echo("  Results must NOT be interpreted as real-data science.\n", err=True)

    if result is not None and result.metadata.get("is_approximate_evidence"):
        click.echo("\n*** WARNING: APPROXIMATE EVIDENCE (toy sampler) ***", err=True)
        if not BILBY_AVAILABLE:
            click.echo(
                "  bilby/dynesty not installed; evidence is NOT publication-grade.",
                err=True,
            )
        click.echo("  Use only for development / smoke tests.\n", err=True)


def _print_fit_summary(
    model: str,
    inject_model: str,
    prov: DataProvenance,
    result: Any,
    channel: str,
    data: str,
    event: str | None,
) -> None:
    approx = result.metadata.get("is_approximate_evidence", False)
    click.echo("\n" + "=" * 60)
    click.echo("FIT SUMMARY (candidate model fit — not a detection claim)")
    click.echo("=" * 60)
    click.echo(f"Fit model:              {model}")
    click.echo(f"Injected / obs. source: {prov.requested_source}")
    click.echo(f"Actual source used:     {prov.actual_source}")
    if prov.fallback_used:
        click.echo(f"Fallback reason:        {prov.fallback_reason}")
    click.echo(f"Inject model (mock):    {inject_model}")
    click.echo(f"Channel:                {channel}")
    if event:
        click.echo(f"Event:                  {event}")
    click.echo(f"Sampler:                {result.metadata.get('sampler')}")
    click.echo(f"Approximate evidence:   {'YES' if approx else 'NO'}")
    click.echo(f"ln Z:                   {result.log_evidence:.3f} ± {result.log_evidence_err:.3f}")
    click.echo(f"N posterior:            {len(result.posterior)}")
    click.echo("=" * 60 + "\n")


def _default_alt_model(channel: str) -> str:
    return {"gw": "bh_ringdown", "radio": "magnetar", "xray": "pbh_tunneling", "image": "bh_accretion"}.get(
        channel, "null"
    )


def _load_run_config(config_path: str | None, channel: str) -> dict:
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    defaults = {
        "gw": "configs/runs/gw_run.yaml",
        "radio": "configs/runs/radio_run.yaml",
        "image": "configs/instruments/eht.yaml",
    }
    default_path = Path(defaults.get(channel, "configs/runs/gw_run.yaml"))
    if default_path.exists():
        with open(default_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _default_context(channel: str) -> dict:
    contexts = {
        "gw": {"sample_rate": 4096, "duration": 4.0, "t_merger": 1.0, "low_freq_cutoff": 20.0, "rng_seed": 42},
        "radio": {
            "freq_low_mhz": 400.0, "freq_high_mhz": 800.0, "n_freq_chans": 64, "n_time_bins": 512,
            "t_start_s": 0.1, "t_end_s": 0.3, "tsys_jy": 1000.0, "t_samp_ms": 0.1, "rng_seed": 42,
        },
        "xray": {"area_cm2": 1000.0, "bg_rate_cps": 0.5, "duration_s": 100.0, "dt_s": 1.0, "rng_seed": 42},
        "image": {"fov_muas": 200.0, "n_pixels": 64, "freq_ghz": 230.0, "thermal_noise_jy": 0.05, "rng_seed": 42},
    }
    return contexts.get(channel, {})


def _get_likelihood(channel: str, model: str):
    from whitesearch.likelihoods import (
        GWLikelihood, RadioBurstLikelihood, XRayBurstLikelihood, VisibilityLikelihood,
    )
    from whitesearch.models import get_model

    model_channel = get_model(model).channel
    compatible = {
        "gw": {"gw", "generic"},
        "radio": {"radio", "generic"},
        "xray": {"xray", "radio", "generic"},
        "image": {"image", "generic"},
    }
    if model_channel not in compatible.get(channel, set()):
        raise ValueError(
            f"Model '{model}' (native channel={model_channel}) "
            f"cannot be fit on data channel '{channel}'"
        )
    return {
        "gw": GWLikelihood(model),
        "radio": RadioBurstLikelihood(model),
        "xray": XRayBurstLikelihood(model),
        "image": VisibilityLikelihood(),
    }[channel]


if __name__ == "__main__":
    main()
