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
            brightness_prior: tuple[float, float] | None = None,
            cap_s: float = 1500.0, nact: int | None = None,
            dlogz: float | None = None) -> dict:
    """Advance one injection by at most `timeout_s` of sampling.

    Injections are resumable: a shard that runs out of time leaves the record
    with ``final = False`` and a running ``cum_wall_s``, and the next shard
    picks it up from bilby's checkpoint.  ``cap_s`` is the TOTAL wall-clock an
    injection may consume across all shards; past it the record is closed as
    ``timeout_capped`` so one expensive draw cannot eat the whole campaign.

    ``nact`` overrides the random-walk chain length (bilby's
    ``AcceptanceTrackingRWalk`` accepts an average of ``2 * nact`` steps per
    proposal).  ``dlogz`` overrides dynesty's stopping threshold on the
    remaining evidence.  Leaving either ``None`` keeps ``BilbyRunner``'s
    defaults (2 and 0.1), which is what every archived campaign used.
    """
    path = outdir / f"inj_{idx:04d}.json"
    prior_rec: dict = {}
    cum = 0.0
    if path.exists():
        prior_rec = json.loads(path.read_text())
        if prior_rec.get("final", False):
            return prior_rec
        cum = float(prior_rec.get("cum_wall_s", 0.0))

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
               "cum_wall_s": 0.0, "final": True,
               "theta_true": {k: float(v) for k, v in theta.items()}}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec, indent=1))
        return rec

    extra_kwargs: dict = {}
    if nact is not None:
        extra_kwargs["nact"] = int(nact)
    if dlogz is not None:
        extra_kwargs["dlogz"] = float(dlogz)
    runner = BilbyRunner(
        sampler="dynesty",
        nlive=nlive,
        outdir=str(outdir / "bilby"),
        resume=True,
        seed=seed,
        run_timeout_s=min(timeout_s, max(cap_s - cum, 1.0)),
        # bilby's default check_point_delta_t is 600 s, longer than a shard, so
        # nothing was ever written and every shard restarted from scratch.
        check_point_delta_t=checkpoint_dt,
        check_point_plot=False,
        **extra_kwargs,
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
        "cap_s": cap_s,
        "nact_requested": None if nact is None else int(nact),
        "dlogz_requested": None if dlogz is None else float(dlogz),
    }

    t0 = time.monotonic()
    try:
        label = f"img_{target.replace(chr(42), 'star')}_{idx:04d}"
        res = runner.run(like, data, ctx, model, label=label)
        post = res.posterior
        rec["status"] = "ok"
        rec["n_posterior"] = int(len(post))
        rec["log_evidence"] = float(res.log_evidence)
        # Cumulative dynesty ncall: the behavioural evidence that a chain-length
        # change took effect, rather than the kwargs dict echoing the request.
        rec["ncall"] = int(res.metadata.get("num_likelihood_evaluations", 0))
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
        rec["final"] = True
    except SamplingBudgetExceeded as exc:
        rec["status"] = "timeout"
        rec["reason"] = str(exc)
        rec["final"] = False
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        rec["status"] = "error"
        rec["reason"] = f"{type(exc).__name__}: {exc}"
        rec["final"] = True
    rec["wall_s"] = round(time.monotonic() - t0, 2)
    rec["cum_wall_s"] = round(cum + rec["wall_s"], 2)
    rec["n_shards"] = int(prior_rec.get("n_shards", 0)) + 1
    if not rec["final"] and rec["cum_wall_s"] >= cap_s:
        # Spent its whole allowance without converging; close it so the
        # campaign moves on and the denominator stays honest.
        rec["status"] = "timeout_capped"
        rec["final"] = True

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
    ap.add_argument("--cap", type=float, default=1500.0,
                    help="total wall-clock an injection may use across shards")
    ap.add_argument("--brightness-prior", dest="brightness_prior", default=None,
                    help="EXPERIMENTAL 'lo,hi' override of log10_total_flux_jy bounds")
    ap.add_argument("--nact", type=int, default=None,
                    help="dynesty rwalk chain length (2*nact accepted steps); "
                         "default None keeps BilbyRunner's nact=2")
    ap.add_argument("--dlogz", type=float, default=None,
                    help="dynesty stopping threshold on remaining evidence; "
                         "default None keeps BilbyRunner's dlogz=0.1")
    ap.add_argument("--indices", default=None,
                    help="comma-separated injection indices to run, instead of "
                         "the --start/--n range")
    ap.add_argument("--outroot", default="artifacts/image_sbc")
    a = ap.parse_args()

    bp = None
    if a.brightness_prior:
        bp = tuple(float(x) for x in a.brightness_prior.split(","))
    outdir = Path(a.outroot) / a.target.replace("*", "star")
    outdir.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()
    done = 0
    if a.indices:
        todo = [int(x) for x in a.indices.split(",") if x.strip()]
    else:
        todo = list(range(a.start, a.start + a.n))
    for i in todo:
        if time.monotonic() - t_start > a.budget:
            print(f"SHARD-BUDGET-STOP after {done} injections", flush=True)
            break
        r = run_one(i, a.target, a.nlive, a.timeout, outdir, a.checkpoint_dt,
                    bp, a.cap, a.nact, a.dlogz)
        done += 1
        print(
            f"[{i:04d}] {r['status']:15s} cum={r.get('cum_wall_s', 0):7.1f}s "
            f"final={str(r.get('final', False)):5s} "
            f"snr={r.get('network_snr', float('nan')):9.1f} "
            f"npost={r.get('n_posterior', 0):5d}",
            flush=True,
        )
    print(f"SHARD-DONE launched={done} elapsed={time.monotonic()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
