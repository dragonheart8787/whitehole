"""image-channel gr_eternal SBC / coverage campaign, one injection per file.

Resumable by construction: each injection writes artifacts/image_sbc/<target>/
inj_XXXX.json and an existing file is skipped.  This container suspends between
turns, so the campaign is driven as a sequence of short foreground shards
rather than one long background job.

Per-run wall-clock cap is enforced by BilbyRunner's _BudgetGuard (inside the
likelihood), because dynesty's own maxcall is overwritten by bilby.
"""
from __future__ import annotations

import argparse, json, os, time
from pathlib import Path

import numpy as np

os.environ.setdefault("WHITESEARCH_FORCE_TOY", "0")

from whitesearch.cli import _default_context
from whitesearch.inference import BilbyRunner
from whitesearch.inference.bilby_runner import SamplingBudgetExceeded
from whitesearch.likelihoods import VisibilityLikelihood
from whitesearch.models import model_for_context
from whitesearch.simulators import get_simulator
from whitesearch.simulators.image_shadow import UnrepresentableRingError
from whitesearch.utils.math_utils import compute_credible_interval, compute_sbc_rank

L = 100                      # rank denominator: thinned posterior draws
CI_LEVELS = (0.68, 0.90)


def _override_brightness_prior(model, lo: float, hi: float):
    """Return `model` with its amplitude parameter re-bounded to [lo, hi].

    EXPERIMENTAL OVERRIDE, used only to measure what a different amplitude
    prior would cost.  It patches the ParameterSpec that both ``sample_prior``
    and ``to_bilby_priors`` read, so the injection and the fit see the same
    prior.  It does NOT change the shipped model.
    """
    from dataclasses import replace

    original = model.parameters

    def patched():
        out = []
        for spec in original():
            if spec.name == "log10_total_flux_jy":
                spec = replace(spec, prior_kwargs={"low": lo, "high": hi})
            out.append(spec)
        return out

    model.parameters = patched
    return model


def run_one(idx: int, target: str, nlive: int, timeout_s: float, outdir: Path,
            checkpoint_dt: float = 45.0,
            brightness_prior: tuple[float, float] | None = None) -> dict:
    path = outdir / f"inj_{idx:04d}.json"
    if path.exists():
        return json.loads(path.read_text())

    seed = 700_000 + idx
    ctx = {**_default_context("image"), "target": target, "rng_seed": seed}
    model = model_for_context("gr_eternal", ctx)
    if brightness_prior is not None:
        model = _override_brightness_prior(model, *brightness_prior)
    # amplitude-only, declared explicitly: the closure-phase triplets in
    # _default_eht_uv() do not close (audit X.13.4 / decision I-5), so this
    # campaign must not claim to use closure phases.
    like = VisibilityLikelihood("gr_eternal", use_closure_phases=False)
    sim = get_simulator("image")

    rng = np.random.default_rng(seed)
    theta = model.sample_prior(rng)
    try:
        data = sim.simulate(theta, ctx, rng=np.random.default_rng(seed + 1))
    except UnrepresentableRingError as exc:
        # The grid cannot represent this draw's ring (near-edge-on + thin).
        # Recorded as its own status rather than counted as a failure, so the
        # campaign's denominator stays honest.
        rec = {"idx": idx, "target": target, "seed": seed, "nlive": nlive,
               "status": "unrepresentable", "reason": str(exc), "wall_s": 0.0,
               "theta_true": {k: float(v) for k, v in theta.items()}}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec, indent=1))
        return rec

    runner = BilbyRunner(
        sampler="dynesty",
        nlive=nlive,
        outdir=str(outdir / "bilby"),
        resume=True,
        seed=seed,
        run_timeout_s=timeout_s,
        # bilby's default check_point_delta_t is 600 s, longer than a shard, so
        # nothing was ever written and every shard restarted from scratch.
        check_point_delta_t=checkpoint_dt,
        check_point_plot=False,
    )
    names = BilbyRunner.effective_parameter_names(model, like)

    sig = np.abs(data.metadata["vis_signal"])
    rec: dict = {
        "idx": idx, "target": target, "seed": seed, "nlive": nlive,
        "theta_true": {k: float(v) for k, v in theta.items()},
        "sampled_parameters": names,
        "r_ring_muas": float(data.metadata["r_ring_muas"]),
        "total_flux_jy": float(
            data.metadata["image"].sum() * (2 * ctx["fov_muas"] / ctx["n_pixels"]) ** 2
        ),
        "network_snr": float(np.sqrt(((sig / ctx["thermal_noise_jy"]) ** 2).sum())),
        "use_closure_phases": False,
        "brightness_prior_override": list(brightness_prior) if brightness_prior else None,
    }

    t0 = time.monotonic()
    try:
        label = f"img_{target.replace(chr(42), 'star')}_{idx:04d}"
        res = runner.run(like, data, ctx, model, label=label)
        post = res.posterior
        rec["status"] = "ok"
        rec["n_posterior"] = int(len(post))
        rec["log_evidence"] = float(res.log_evidence)
        rec["sampler_kwargs"] = {
            k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
            for k, v in res.metadata.get("sampler_kwargs", {}).items()
        }
        thin = post.iloc[
            np.random.default_rng(seed).choice(
                len(post), size=min(L, len(post)), replace=False
            )
        ]
        rec["L_effective"] = int(len(thin))
        rec["ranks"] = {}
        rec["ci"] = {}
        for p in names:
            s = thin[p].to_numpy()
            rec["ranks"][p] = compute_sbc_rank(float(theta[p]), s)
            rec["ci"][p] = {
                f"{int(100*lv)}": list(compute_credible_interval(s, lv))
                for lv in CI_LEVELS
            }
    except SamplingBudgetExceeded as exc:
        rec["status"] = "timeout"
        rec["reason"] = str(exc)
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        rec["status"] = "error"
        rec["reason"] = f"{type(exc).__name__}: {exc}"
    rec["wall_s"] = round(time.monotonic() - t0, 2)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="M87*")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--nlive", type=int, default=250)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--budget", type=float, default=520.0,
                    help="stop launching new injections after this many seconds")
    ap.add_argument("--checkpoint-dt", dest="checkpoint_dt", type=float, default=45.0)
    ap.add_argument("--brightness-prior", dest="brightness_prior", default=None,
                    help="EXPERIMENTAL 'lo,hi' override of log10_total_flux_jy bounds")
    ap.add_argument("--outroot", default="artifacts/image_sbc")
    a = ap.parse_args()

    bp = None
    if a.brightness_prior:
        bp = tuple(float(x) for x in a.brightness_prior.split(","))
    outdir = Path(a.outroot) / a.target.replace("*", "star")
    outdir.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()
    done = 0
    for i in range(a.start, a.start + a.n):
        if time.monotonic() - t_start > a.budget:
            print(f"SHARD-BUDGET-STOP after {done} injections", flush=True)
            break
        r = run_one(i, a.target, a.nlive, a.timeout, outdir, a.checkpoint_dt, bp)
        done += 1
        print(
            f"[{i:04d}] {r['status']:8s} {r['wall_s']:7.1f}s "
            f"snr={r.get('network_snr', float('nan')):9.1f} "
            f"npost={r.get('n_posterior', 0):5d} r={r.get('r_ring_muas', 0):6.2f}",
            flush=True,
        )
    print(f"SHARD-DONE launched={done} elapsed={time.monotonic()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
