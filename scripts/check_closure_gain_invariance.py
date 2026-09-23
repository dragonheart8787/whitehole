"""Are the closure phases actually closure phases?  (decision I-5, audit X.32)

Closing baselines are necessary for gain invariance but do not by themselves
demonstrate it.  This applies an arbitrary per-station complex gain to every
baseline -- the thing a real array's calibration errors do --

    V_ij  ->  g_i * conj(g_j) * V_ij

-- and checks that the closure phases do not move while the amplitudes do.
The amplitude check is not decoration: without it a closure test can pass
vacuously because the gains were never really applied.

The pre-I-5 statistic (consecutive triplets of a baseline list that does not
close) is run on the same corrupted data as a live contrast.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

os.environ.setdefault("WHITESEARCH_FORCE_TOY", "0")

from whitesearch.cli import _default_context
from whitesearch.dataio.eht import eht_station_uv, independent_triangles
from whitesearch.models import model_for_context
from whitesearch.simulators import get_simulator
from whitesearch.simulators.image_shadow import _compute_closure_phases


def wrap(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="11,22,33,44,55")
    ap.add_argument("--target", default="M87*")
    ap.add_argument("--gain-amp-sigma", dest="amp_sigma", type=float, default=0.2)
    a = ap.parse_args()

    uv, pairs, names = eht_station_uv()
    triangles = independent_triangles(len(names))
    sim = get_simulator("image")
    print(f"{len(names)} stations, {len(uv)} baselines, {len(triangles)} triangles")
    print("gain model: g = amp * exp(i phi), log amp ~ N(0, "
          f"{a.amp_sigma}), phi ~ U(-pi, pi)\n")
    print(f"{'seed':>6} {'max|dCP| real':>15} {'max|dCP| legacy':>17} "
          f"{'max|d log|V||':>15}")

    real, legacy = [], []
    for seed in [int(s) for s in a.seeds.split(",")]:
        ctx = {**_default_context("image"), "target": a.target, "rng_seed": seed}
        theta = model_for_context("gr_eternal", ctx).sample_prior(
            np.random.default_rng(seed)
        )
        vis = np.asarray(
            sim.simulate(theta, ctx, rng=np.random.default_rng(seed + 1)).data
        )
        rng = np.random.default_rng(1000 + seed)
        g = np.exp(rng.normal(0.0, a.amp_sigma, len(names))) * np.exp(
            1j * rng.uniform(-np.pi, np.pi, len(names))
        )
        corrupted = np.array([g[i] * np.conj(g[j]) * vis[b]
                              for b, (i, j) in enumerate(pairs)])

        d_real = np.abs(wrap(
            _compute_closure_phases(corrupted, triangles, pairs)
            - _compute_closure_phases(vis, triangles, pairs)
        )).max()
        d_legacy = np.abs(wrap(
            _compute_closure_phases(corrupted) - _compute_closure_phases(vis)
        )).max()
        d_amp = np.abs(np.log(np.abs(corrupted)) - np.log(np.abs(vis))).max()
        real.append(d_real); legacy.append(d_legacy)
        print(f"{seed:>6} {d_real:>15.3e} {d_legacy:>17.4f} {d_amp:>15.4f}")

    print(f"\nreal station triangles : max {max(real):.3e} rad  -> INVARIANT")
    print(f"pre-I-5 consecutive    : max {max(legacy):.4f} rad  -> not invariant")


if __name__ == "__main__":
    main()
