"""WhiteSearch command-line interface.

Usage
-----
  whitesearch fit     --model bounce --channel gw --data gwosc --event GW150914
  whitesearch inject  --model bounce --channel gw --n-injections 100
  whitesearch sensitivity --model pbh_tunneling --channel radio
  whitesearch compare --model bounce --null bh_ringdown --data gwosc
  whitesearch report  --run-dir artifacts/bilby
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("whitesearch.cli")


# ── Main group ────────────────────────────────────────────────────────────────

@click.group()
@click.version_option(version="0.1.0", prog_name="whitesearch")
def main():
    """WhiteSearch — White hole candidate signal search engine."""


# ── fit ───────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--model", required=True, type=click.Choice(
    ["bounce", "pbh_tunneling", "gr_eternal", "null", "magnetar", "bh_ringdown", "bh_accretion"],
    case_sensitive=False,
), help="White hole or alternative model")
@click.option("--channel", default="gw", type=click.Choice(["gw", "radio", "xray", "image"]))
@click.option("--data", default="mock", type=click.Choice(["mock", "gwosc", "chime", "heasarc", "eht"]))
@click.option("--event", default=None, help="Event name (e.g. GW150914)")
@click.option("--config", default=None, help="Run config YAML file")
@click.option("--nlive", default=500, show_default=True)
@click.option("--outdir", default="artifacts/bilby", show_default=True)
@click.option("--seed", default=42, show_default=True)
@click.option("--resume/--no-resume", default=True)
def fit(model, channel, data, event, config, nlive, outdir, seed, resume):
    """Run Bayesian inference on real or mock data and compute ln Z."""
    from whitesearch.models import get_model
    from whitesearch.simulators import get_simulator
    from whitesearch.likelihoods import GWLikelihood, RadioBurstLikelihood, XRayBurstLikelihood, VisibilityLikelihood
    from whitesearch.inference import BilbyRunner

    cfg = _load_run_config(config, channel)
    context = cfg.get("instrument", _default_context(channel))
    context["rng_seed"] = seed

    model_obj = get_model(model)
    label = f"{model}_{channel}_{event or 'mock'}"

    # Load or simulate data
    obs_data = _load_data(data, channel, event, context, seed)

    # Likelihood
    ll = _get_likelihood(channel, model)

    # Run inference
    runner = BilbyRunner(sampler="dynesty", nlive=nlive, outdir=outdir, resume=resume, seed=seed)
    result = runner.run(ll, obs_data, context, model_obj, label=label)

    click.echo(f"\n{'=' * 60}")
    click.echo(f"Model:       {model}")
    click.echo(f"Channel:     {channel}")
    click.echo(f"ln Z:        {result.log_evidence:.3f} ± {result.log_evidence_err:.3f}")
    click.echo(f"N posterior: {len(result.posterior)}")
    click.echo(f"{'=' * 60}\n")


# ── inject ────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--model", required=True, type=click.Choice(
    ["bounce", "pbh_tunneling", "gr_eternal"], case_sensitive=False
))
@click.option("--channel", default="gw", type=click.Choice(["gw", "radio", "xray", "image"]))
@click.option("--n-injections", default=100, show_default=True)
@click.option("--nlive", default=200, show_default=True)
@click.option("--outdir", default="artifacts/injections", show_default=True)
@click.option("--seed", default=42, show_default=True)
def inject(model, channel, n_injections, nlive, outdir, seed):
    """Run injection / recovery campaign and print coverage statistics."""
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

    summary = result.summary()
    click.echo("\nCoverage Summary:")
    click.echo(summary.to_string(index=False))


# ── sensitivity ───────────────────────────────────────────────────────────────

@main.command()
@click.option("--model", required=True)
@click.option("--channel", default="radio")
@click.option("--param", default=None, help="Parameter to scan (default: first param)")
@click.option("--n-injections", default=200, show_default=True)
@click.option("--outdir", default="artifacts/sensitivity", show_default=True)
@click.option("--seed", default=42, show_default=True)
def sensitivity(model, channel, param, n_injections, outdir, seed):
    """Generate sensitivity curves for a given model and channel."""
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

    click.echo(f"Running sensitivity analysis: model={model}, channel={channel}, param={param}")
    ir_result = ir.run_injections(model_obj, ll, context)

    analyzer = SensitivityAnalyzer()
    sensitivity_data = analyzer.compute_sensitivity(ir_result.theta_true, ir_result.evidences, param)

    click.echo(f"\nSensitivity curve for {param}:")
    for b, r in zip(sensitivity_data["param_bins"], sensitivity_data["recovery_fraction"]):
        click.echo(f"  {b:.3f}: {r:.1%}")


# ── report ────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--run-dir", default="artifacts/bilby", show_default=True)
@click.option("--output", default="artifacts/report.md", show_default=True)
def report(run_dir, output):
    """Generate a Markdown technical report from inference artifacts."""
    run_path = Path(run_dir)
    if not run_path.exists():
        click.echo(f"Run directory {run_dir} not found.")
        sys.exit(1)

    result_files = list(run_path.glob("*.json"))
    click.echo(f"Found {len(result_files)} result files in {run_dir}")

    lines = [
        "# WhiteSearch Technical Report\n",
        f"**Run directory:** {run_dir}\n",
        f"**Result files found:** {len(result_files)}\n",
        "\n## Summary\n",
        "| Model | ln Z | ln Z_err |\n",
        "|-------|------|----------|\n",
    ]

    for f in sorted(result_files):
        import json
        try:
            with open(f) as fp:
                data = json.load(fp)
            label = data.get("label", f.stem)
            ln_z = data.get("log_evidence", float("nan"))
            ln_z_err = data.get("log_evidence_err", float("nan"))
            lines.append(f"| {label} | {ln_z:.3f} | {ln_z_err:.3f} |\n")
        except Exception:
            lines.append(f"| {f.stem} | (error) | (error) |\n")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as fp:
        fp.writelines(lines)
    click.echo(f"Report written to {output}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_run_config(config_path: str | None, channel: str) -> dict:
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    defaults = {
        "gw": f"configs/runs/gw_run.yaml",
        "radio": f"configs/runs/radio_run.yaml",
        "image": f"configs/instruments/eht.yaml",
    }
    default_path = Path(defaults.get(channel, "configs/runs/gw_run.yaml"))
    if default_path.exists():
        with open(default_path) as f:
            return yaml.safe_load(f)
    return {}


def _default_context(channel: str) -> dict:
    contexts = {
        "gw": {
            "sample_rate": 4096, "duration": 4.0, "t_merger": 1.0,
            "low_freq_cutoff": 20.0, "rng_seed": 42,
        },
        "radio": {
            "freq_low_mhz": 400.0, "freq_high_mhz": 800.0,
            "n_freq_chans": 64, "n_time_bins": 512,
            "t_start_s": 0.1, "t_end_s": 0.3,
            "tsys_jy": 1000.0, "t_samp_ms": 0.1, "rng_seed": 42,
        },
        "xray": {
            "area_cm2": 1000.0, "bg_rate_cps": 0.5,
            "duration_s": 100.0, "dt_s": 1.0, "rng_seed": 42,
        },
        "image": {
            "fov_muas": 200.0, "n_pixels": 64,
            "freq_ghz": 230.0, "thermal_noise_jy": 0.05, "rng_seed": 42,
        },
    }
    return contexts.get(channel, {})


def _load_data(source: str, channel: str, event: str | None, context: dict, seed: int):
    if source == "mock":
        from whitesearch.simulators import get_simulator
        from whitesearch.models import get_model

        default_model_for_channel = {"gw": "bounce", "radio": "pbh_tunneling", "image": "gr_eternal", "xray": "pbh_tunneling"}
        model_name = default_model_for_channel.get(channel, "bounce")
        model = get_model(model_name)
        sim = get_simulator(channel)
        params = model.sample_prior(np.random.default_rng(seed))
        return sim.simulate(params, context, rng=np.random.default_rng(seed))

    elif source == "gwosc" and event:
        from whitesearch.dataio import GWOSCLoader
        loader = GWOSCLoader()
        data = loader.load_event(event)
        det = list(data.keys())[0]
        return data[det]  # return first detector

    elif source == "chime":
        from whitesearch.dataio import CHIMEFRBLoader
        loader = CHIMEFRBLoader()
        return loader.load_catalog()

    else:
        click.echo(f"Data source {source!r} not yet implemented; using mock data.")
        return _load_data("mock", channel, event, context, seed)


def _get_likelihood(channel: str, model: str):
    from whitesearch.likelihoods import (
        GWLikelihood, RadioBurstLikelihood, XRayBurstLikelihood, VisibilityLikelihood
    )
    return {
        "gw": GWLikelihood(model),
        "radio": RadioBurstLikelihood(model),
        "xray": XRayBurstLikelihood(model),
        "image": VisibilityLikelihood(),
    }[channel]


if __name__ == "__main__":
    main()
