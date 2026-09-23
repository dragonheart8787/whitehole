"""Does position_angle's circular topology bias its rank/coverage?  (audit X.30)

`position_angle` is declared as a plain Uniform(0, pi) like every other linear
parameter, but the gr_eternal image is EXACTLY pi-periodic in it (an
axisymmetric ring rotated by pi maps (xr,yr)->(-xr,-yr), leaving r_ellipse
unchanged), so the prior spans exactly one period and its two endpoints are the
same model.  Neither compute_sbc_rank nor compute_credible_interval does any
wraparound.  This measures how often that matters and whether it can bias
calibration.

The answer to the second question is no, and the reason is worth stating
because it is not obvious: SBC rank uniformity follows from EXCHANGEABILITY of
the truth with the posterior draws, which holds for any fixed total order and
does not require that order to respect the topology.  Equal-tailed interval
coverage is essentially a function of that rank, so it inherits the guarantee.
Wrapping makes intervals WIDER than necessary, never miscalibrated -- and the
simulation here confirms it at wrapping fractions up to 99.7%.

One trap this script is written to avoid: a continuous KS test on a DISCRETE
rank (101 values) at large N rejects uniformity for a perfectly uniform linear
control too.  The chi-square test over the 101 rank values is the right test.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chisquare, fisher_exact

from whitesearch.utils.math_utils import compute_credible_interval, compute_sbc_rank

PERIOD = np.pi
L = 100


def thinned_position_angle(rec: dict, archive: Path) -> np.ndarray:
    """Reproduce exactly the L=100 subset the campaign ranked against."""
    label = f"img_{rec['target'].replace('*', 'star')}_{rec['idx']:04d}"
    d = json.loads((archive / "bilby" / f"{label}_result.json").read_text())
    post = pd.DataFrame(d["posterior"]["content"])
    post = post.drop(columns=[c for c in ("log_likelihood", "log_prior")
                              if c in post.columns])
    idx = np.random.default_rng(rec["seed"]).choice(
        len(post), size=min(L, len(post)), replace=False)
    return post.iloc[idx]["position_angle"].to_numpy()


def minimal_arc(samples: np.ndarray, frac: float = 0.90) -> float:
    """Smallest arc on the period-PERIOD circle containing `frac` of samples."""
    x = np.sort(np.mod(samples, PERIOD))
    n = len(x)
    k = int(np.ceil(frac * n))
    xx = np.concatenate([x, x + PERIOD])
    return float(np.min(xx[k - 1:k - 1 + n] - xx[:n]))


def simulate_circular(sd: float, n_trials: int, rng) -> dict:
    """A perfectly calibrated circular posterior, scored the project's way."""
    ranks = np.empty(n_trials, int)
    c68 = c90 = wrapped = 0
    for t in range(n_trials):
        draws = np.mod(rng.uniform(0, PERIOD) + rng.normal(0, sd, L + 1), PERIOD)
        truth, post = draws[0], draws[1:]
        ranks[t] = compute_sbc_rank(truth, post)
        lo, hi = compute_credible_interval(post, 0.90); c90 += lo <= truth <= hi
        lo6, hi6 = compute_credible_interval(post, 0.68); c68 += lo6 <= truth <= hi6
        wrapped += int((post < 0.1 * PERIOD).any() and (post > 0.9 * PERIOD).any())
    return {"sd": sd, "wrapped_fraction": wrapped / n_trials,
            "cov68": c68 / n_trials, "cov90": c90 / n_trials,
            "rank_mean": float(ranks.mean()),
            "rank_chi2_p": float(chisquare(np.bincount(ranks, minlength=L + 1)).pvalue)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="artifacts/image_sbc_fixA/M87star")
    ap.add_argument("--sim-trials", dest="sim_trials", type=int, default=40000)
    ap.add_argument("--out",
                    default="docs/calibration/image_gr_eternal_position_angle_wrap.csv")
    a = ap.parse_args()
    archive = Path(a.archive)

    ok = [r for r in (json.loads(p.read_text())
                      for p in sorted(archive.glob("inj_*.json")))
          if r.get("status") == "ok"]
    rows = []
    for r in ok:
        s = thinned_position_angle(r, archive)
        truth = r["theta_true"]["position_angle"]
        lo, hi = r["ci"]["position_angle"]["90"]
        arc = minimal_arc(s)
        rows.append({
            "idx": r["idx"], "truth": truth,
            "edge_distance": min(truth, PERIOD - truth),
            "linear_w90": hi - lo, "circular_w90": arc,
            "width_ratio": (hi - lo) / arc if arc > 0 else np.nan,
            "resultant_R": float(np.abs(np.mean(np.exp(2j * s)))),
            "n_in_bottom_decile": int((s < 0.1 * PERIOD).sum()),
            "n_in_top_decile": int((s > 0.9 * PERIOD).sum()),
            "rank": r["ranks"]["position_angle"],
            "hit90": int(lo <= truth <= hi),
        })
    df = pd.DataFrame(rows)
    split = (df.n_in_bottom_decile > 0) & (df.n_in_top_decile > 0)
    conc = split & (df.resultant_R > 0.5)
    print(f"n = {len(df)}")
    print(f"  samples in both deciles: {int(split.sum())} ({100*split.mean():.1f}%)")
    print(f"  of those, circularly concentrated (R>0.5): {int(conc.sum())}")
    print(f"  linear/circular width ratio: median {df.width_ratio.median():.4f}, "
          f"max {df.width_ratio.max():.4f}, #>1.2 {int((df.width_ratio>1.2).sum())}")
    for label, mask in (("wrapped+concentrated", conc), ("ratio>1.05", df.width_ratio > 1.05)):
        tab = [[int((mask & (df.hit90 == 1)).sum()), int((mask & (df.hit90 == 0)).sum())],
               [int((~mask & (df.hit90 == 1)).sum()), int((~mask & (df.hit90 == 0)).sum())]]
        print(f"  {label}: {tab[0]} vs {tab[1]}  Fisher p = {fisher_exact(tab)[1]:.4f}")
    print(f"  position_angle coverage 90% = {df.hit90.mean():.4f}")

    print(f"\nsimulation: perfectly calibrated CIRCULAR posterior, scored linearly "
          f"({a.sim_trials} trials each)")
    rng = np.random.default_rng(4242)
    sims = [simulate_circular(sd, a.sim_trials, rng) for sd in (0.05, 0.15, 0.40, 0.80)]
    print(f"{'sd':>6} {'wrapped':>9} {'cov68':>8} {'cov90':>8} {'rank mean':>10} {'chi2 p':>8}")
    for s in sims:
        print(f"{s['sd']:>6.2f} {s['wrapped_fraction']:>9.3f} {s['cov68']:>8.4f} "
              f"{s['cov90']:>8.4f} {s['rank_mean']:>10.2f} {s['rank_chi2_p']:>8.4f}")
    print("reference (X.25): cov68 0.6671, cov90 0.8825, rank mean 50.0")

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    with out.open("a") as fh:
        fh.write("\n# simulation: perfectly calibrated circular posterior, scored linearly\n")
        fh.write("# sd,wrapped_fraction,cov68,cov90,rank_mean,rank_chi2_p\n")
        for s in sims:
            fh.write(f"# {s['sd']},{s['wrapped_fraction']},{s['cov68']},"
                     f"{s['cov90']},{s['rank_mean']},{s['rank_chi2_p']}\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
