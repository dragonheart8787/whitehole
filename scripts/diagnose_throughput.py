"""Diagnose the pilot/retest likelihood-throughput discrepancy (X.36.5).

X.36 measured 100-128 calls/s during the pilot and 149-214 calls/s during the
high-SNR retest, on the SAME injections with the SAME settings.  That was
reported as an unexplained measurement-condition difference.  Deciding whether
to spend ~60 h on the full campaign needs to know whether it recurs.

This measures the likelihood call rate directly -- no sampler, no bilby -- so
the only thing being timed is the forward model.  Sampler overhead is a small
constant on top and cannot manufacture a 1.49x swing.

CPU steal time is sampled alongside: this is a Firecracker microVM, so a busy
host shows up as steal, which is a cause NOTHING in this repository can fix.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# BLAS thread guard -- MUST run before numpy is imported.
#
# X.37 traced the 1.49x throughput discrepancy between the bh_accretion pilot
# and the high-SNR retest to CPU oversubscription inside the VM: numpy's BLAS
# is unbounded and takes all 4 cores (measured 3.92), while the sampler's
# likelihood is single-threaded (measured 1.00).  One numpy-heavy script run
# alongside a campaign is therefore enough to starve it and corrupt the timing
# record.  OpenBLAS/MKL read these at load time, so setting them after
# `import numpy` would be too late.
#
# setdefault, not assignment: an explicit outer setting still wins.
# ---------------------------------------------------------------------------
import os as _os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse, json, os, time
from pathlib import Path

import numpy as np

os.environ.setdefault("WHITESEARCH_FORCE_TOY", "0")

from whitesearch.cli import _default_context
from whitesearch.likelihoods import VisibilityLikelihood
from whitesearch.models import model_for_context
from whitesearch.simulators import get_simulator


def cpu_times() -> tuple[float, float]:
    """(total jiffies, steal jiffies) from /proc/stat."""
    parts = Path("/proc/stat").read_text().split("\n")[0].split()[1:]
    v = [float(x) for x in parts]
    return sum(v), (v[7] if len(v) > 7 else 0.0)


def verify_blas_threads() -> dict:
    """Behavioural check that the thread guard took effect (lesson 1).

    Setting an environment variable is not evidence that BLAS honoured it --
    the same trap as `walks` being echoed back while doing nothing (K.2.1).
    The evidence is CPU time over wall time on a matmul big enough to thread:
    unbounded BLAS on this 4-core box measured 3.92 cores in X.37, so a guarded
    process must come back at ~1.0.
    """
    import numpy as _np

    a = _np.random.rand(1500, 1500)
    a @ a                                     # warm up the BLAS thread pool
    t0, c0 = time.monotonic(), time.process_time()
    for _ in range(6):
        a @ a
    wall, cpu = time.monotonic() - t0, time.process_time() - c0
    return {
        "env": {v: os.environ.get(v) for v in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        "matmul_wall_s": wall, "matmul_cpu_s": cpu,
        "cores_used": cpu / wall,
    }


def bench(idx: int, seed_base: int, seconds: float, model_name: str) -> dict:
    seed = seed_base + idx
    ctx = {**_default_context("image"), "target": "M87*", "rng_seed": seed}
    model = model_for_context(model_name, ctx)
    like = VisibilityLikelihood(model_name, use_closure_phases=False)
    sim = get_simulator("image")
    rng = np.random.default_rng(seed)
    theta = model.sample_prior(rng)
    data = sim.simulate(theta, ctx, rng=np.random.default_rng(seed + 1))

    # Call at points drawn from the prior, not repeatedly at the truth: the
    # cost depends on the parameters, and a single point could sit in an
    # unrepresentatively cheap or expensive corner.
    draws = [model.sample_prior(np.random.default_rng(seed + 1000 + k))
             for k in range(64)]

    like.loglike(theta, data, ctx)          # warm up: JIT/caches/first alloc
    t0, (c0, s0) = time.monotonic(), cpu_times()
    n, marks = 0, []
    while time.monotonic() - t0 < seconds:
        for d in draws:
            try:
                like.loglike(d, data, ctx)
            except Exception:
                pass
            n += 1
        marks.append((time.monotonic() - t0, n))
    el, (c1, s1) = time.monotonic() - t0, cpu_times()

    # Per-second rate over the second half only, to expose a rate that drifts.
    half = [m for m in marks if m[0] >= el / 2]
    drift = None
    if len(half) >= 2:
        drift = (half[-1][1] - half[0][1]) / (half[-1][0] - half[0][0])
    return {
        "idx": idx, "calls": n, "wall_s": el, "calls_per_s": n / el,
        "calls_per_s_second_half": drift,
        "steal_frac": (s1 - s0) / max(c1 - c0, 1e-9),
        "snr": float(np.sqrt(((np.abs(data.metadata["vis_signal"])
                               / ctx["thermal_noise_jy"]) ** 2).sum())),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", default="73,43,25")
    ap.add_argument("--seed-base", dest="seed_base", type=int, default=710_000)
    ap.add_argument("--model", default="bh_accretion")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--tag", default="clean")
    ap.add_argument("--out", default=None)
    ap.add_argument("--verify-blas", dest="verify_blas", action="store_true",
                    help="check the thread guard behaviourally and exit")
    a = ap.parse_args()

    if a.verify_blas:
        v = verify_blas_threads()
        print("env:", v["env"])
        print(f"matmul 1500x1500 x6: wall {v['matmul_wall_s']:.2f}s  "
              f"cpu {v['matmul_cpu_s']:.2f}s  -> {v['cores_used']:.2f} cores used")
        print(f"X.37 measured 3.92 cores unguarded on this 4-core box.")
        ok = v["cores_used"] < 1.25
        print("GUARD", "EFFECTIVE" if ok else "NOT EFFECTIVE")
        raise SystemExit(0 if ok else 1)

    print(f"[{a.tag}] loadavg={Path('/proc/loadavg').read_text().split()[0]} "
          f"nproc={os.cpu_count()}")
    rows = []
    for i in [int(x) for x in a.indices.split(",")]:
        r = bench(i, a.seed_base, a.seconds, a.model)
        r["tag"] = a.tag
        rows.append(r)
        print(f"[{a.tag}] idx {r['idx']:>3} SNR {r['snr']:>9.1f}  "
              f"{r['calls_per_s']:>7.1f} calls/s  "
              f"(2nd half {r['calls_per_s_second_half'] or float('nan'):>7.1f})  "
              f"steal {100*r['steal_frac']:.2f}%  n={r['calls']}")
    v = [r["calls_per_s"] for r in rows]
    print(f"[{a.tag}] median {np.median(v):.1f} calls/s  "
          f"range {min(v):.1f}-{max(v):.1f}")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
