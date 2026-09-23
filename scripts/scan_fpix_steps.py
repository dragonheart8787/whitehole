"""DIAGNOSTIC: is f_pix(theta) a stepped function near the truth?  (candidate I-6k)

`build_ring_image()` renders onto a finite pixel grid, so the worry is that as a
parameter varies continuously the rendered image can only change in discrete
jumps (a ring edge crossing a pixel boundary), and that those jumps distort the
shape of the lnL surface near the truth.

The test is a GRID-REFINEMENT test, not a single scan.  Looking at one scan and
calling it "rough" cannot distinguish a genuine discontinuity from a smooth but
steep curve sampled too coarsely -- and the image channel's lnL really is very
steep in some directions.  Halving the step separates them:

  * smooth, differentiable f  ->  max|dy| ~ |f'| h, so it HALVES with h,
                                  and max|dy|/h converges to |f'|max
  * genuine discontinuity     ->  max|dy| stays at the jump size, and
                                  max|dy|/h doubles every refinement

Two further traps this script is written to avoid, both hit on the first pass:

  1. The scan window must be CLIPPED TO THE PRIOR SUPPORT.  An unclipped
     "truth +/- 1.5 * w90" ran ring_width_frac negative and inclination past
     edge-on, where build_ring_image degenerates and lnL swings by hundreds of
     nats.  The sampler never evaluates there, so those swings say nothing.
  2. For a parameter the data barely constrains (here `i`), w90 is most of the
     prior, so "a few posterior widths" is the whole range and h is coarse.
     The refinement test is run on a bounded window so h actually gets small.

Runs no sampler and changes no production code.
"""
from __future__ import annotations

import argparse, json, os
from pathlib import Path

import numpy as np

os.environ.setdefault("WHITESEARCH_FORCE_TOY", "0")

from whitesearch.cli import _default_context
from whitesearch.likelihoods import VisibilityLikelihood
from whitesearch.models import model_for_context
from whitesearch.simulators import get_simulator
from whitesearch.simulators.image_shadow import (
    UnrepresentableRingError,
    build_ring_image,
)

PARAMS = ["M", "a_star", "i", "position_angle", "ring_width_frac",
          "log10_total_flux_jy"]
# A 1-parameter 1-sigma shift costs dlnL = 0.5.  A jump far below that cannot
# reshape a marginal posterior; one at or above it can.
LNL_SIGMA_SCALE = 0.5
GRIDS = (201, 401, 801, 1601)


def _finite(y: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(y) & (np.abs(y) < 1e29), y, np.nan)


def _max_step(y: np.ndarray) -> float:
    d = np.abs(np.diff(y))
    d = d[np.isfinite(d)]
    return float(d.max()) if d.size else float("nan")


def scan_param(like, theta, data, ctx, p, lo, hi, n) -> dict:
    xs = np.linspace(lo, hi, n)
    y = _finite(np.array([
        like.loglike({**theta, p: float(x)}, data, ctx) for x in xs
    ]))
    return {"h": float(xs[1] - xs[0]), "n_finite": int(np.isfinite(y).sum()),
            "max_step": _max_step(y), "range": float(np.nanmax(y) - np.nanmin(y))}


def image_response(theta, ctx, p, lo, hi, n, img_true) -> dict:
    xs = np.linspace(lo, hi, n)
    flux, dn = [], []
    for x in xs:
        try:
            img = build_ring_image({**theta, p: float(x)}, ctx).image
            flux.append(float(img.sum()))
            dn.append(float(np.linalg.norm(img - img_true)))
        except UnrepresentableRingError:
            flux.append(np.nan); dn.append(np.nan)
    return {"flux_max_step": _max_step(np.array(flux)),
            "dnorm_max_step": _max_step(np.array(dn)),
            "h": float(xs[1] - xs[0])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", default="29,76,62,96,80")
    ap.add_argument("--target", default="M87*")
    ap.add_argument("--archive", default="artifacts/image_sbc_fixA/M87star")
    ap.add_argument("--half-widths", dest="half_widths", type=float, default=0.5,
                    help="window is truth +/- this many 90%% interval widths, "
                         "clipped to the prior support")
    ap.add_argument("--image-points", dest="image_points", type=int, default=401)
    ap.add_argument("--out", default="docs/calibration/image_gr_eternal_fpix_scan.csv")
    a = ap.parse_args()

    rows = []
    for idx in [int(v) for v in a.indices.split(",")]:
        seed = 700_000 + idx
        ctx = {**_default_context("image"), "target": a.target, "rng_seed": seed}
        model = model_for_context("gr_eternal", ctx)
        theta = model.sample_prior(np.random.default_rng(seed))
        data = get_simulator("image").simulate(
            theta, ctx, rng=np.random.default_rng(seed + 1)
        )
        like = VisibilityLikelihood("gr_eternal", use_closure_phases=False)
        priors = model.to_bilby_priors()
        rec = json.loads((Path(a.archive) / f"inj_{idx:04d}.json").read_text())
        img_true = build_ring_image(theta, ctx).image
        sigma = float(data.metadata["thermal_noise_jy"])

        for p in PARAMS:
            lo90, hi90 = rec["ci"][p]["90"]
            w90 = float(hi90 - lo90)
            t = float(theta[p])
            pr = priors[p]
            lo = max(t - a.half_widths * w90, float(pr.minimum))
            hi = min(t + a.half_widths * w90, float(pr.maximum))

            grids = [scan_param(like, theta, data, ctx, p, lo, hi, n) for n in GRIDS]
            steps = np.array([g["max_step"] for g in grids])
            # refinement ratios: 0.5 => smooth (O(h)); 1.0 => genuine jump
            ratios = steps[1:] / steps[:-1]
            img = image_response(theta, ctx, p, lo, hi, a.image_points, img_true)

            rows.append({
                "idx": idx, "network_snr": rec["network_snr"], "param": p,
                "w90": w90, "window_lo": lo, "window_hi": hi,
                "clipped": int(t - a.half_widths * w90 < float(pr.minimum)
                               or t + a.half_widths * w90 > float(pr.maximum)),
                "lnl_range": grids[-1]["range"],
                "h_coarse": grids[0]["h"], "h_fine": grids[-1]["h"],
                "lnl_max_step_coarse": steps[0], "lnl_max_step_fine": steps[-1],
                "refine_ratio_mean": float(np.mean(ratios)),
                "refine_ratio_last": float(ratios[-1]),
                "lnl_max_step_fine_over_sigma_scale": steps[-1] / LNL_SIGMA_SCALE,
                "flux_max_step_jy": img["flux_max_step"],
                "flux_max_step_over_thermal": img["flux_max_step"] / sigma,
                "dnorm_max_step": img["dnorm_max_step"],
                "n_finite_fine": grids[-1]["n_finite"],
            })
        print(f"[{idx:04d}] snr={rec['network_snr']:8.1f} done", flush=True)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with out.open("w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(
                f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c])
                for c in cols) + "\n")
    print(f"wrote {out}\n")

    print(f"{'idx':>5} {'param':>20} {'lnL range':>10} {'step@coarse':>12} "
          f"{'step@fine':>11} {'refine':>7} {'fine/0.5nat':>12} {'flux step/sig':>14}")
    for r in rows:
        print(f"{r['idx']:>5} {r['param']:>20} {r['lnl_range']:>10.4g} "
              f"{r['lnl_max_step_coarse']:>12.4g} {r['lnl_max_step_fine']:>11.4g} "
              f"{r['refine_ratio_mean']:>7.3f} "
              f"{r['lnl_max_step_fine_over_sigma_scale']:>12.4g} "
              f"{r['flux_max_step_over_thermal']:>14.3g}")
    rr = np.array([r["refine_ratio_mean"] for r in rows])
    ff = np.array([r["lnl_max_step_fine_over_sigma_scale"] for r in rows])
    print(f"\nrefinement ratio over all {len(rows)} (injection, parameter) scans: "
          f"min {rr.min():.3f}  max {rr.max():.3f}  (0.5 = smooth, 1.0 = jump)")
    print(f"largest lnL step at the finest grid, in units of the 0.5-nat scale: "
          f"{ff.max():.4g}")


if __name__ == "__main__":
    main()
