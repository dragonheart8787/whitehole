"""X.20-style rank / coverage report for an image-channel SBC campaign.

Model-agnostic: the parameter vector is read from the records themselves, so
this runs on gr_eternal and bh_accretion alike.

Three things this reports that the first gr_eternal round did not, each one a
lesson from that investigation's postmortem:

* **Lesson 3 (discrete ranks vs continuous tests).**  A rank is an integer in
  {0..L}; the continuity-corrected one-sample KS against Uniform(0,1) is only
  approximately calibrated for it.  At N=200000 that approximation manufactured
  p ~ 1e-5 out of a perfectly uniform control (X.30).  So every KS p here is
  accompanied by (a) an exact chi-square on the L+1 rank cells and (b) a
  CONTROL: the same statistic on ranks drawn from the exact discrete null at
  this N, reported as the fraction of null trials at least as extreme.  If the
  KS test is miscalibrated at this N, the control says so.
* **Lesson 5 (test the gap before explaining it).**  Coverage is reported with
  an injection-level bootstrap and a formal p-value in the same pass, not
  after a round of candidate hunting.
* **Lesson 2 (decompose before believing an aggregate).**  Coverage is also
  split by SNR tercile, so a gap concentrated in one regime cannot hide inside
  a flat-looking average.

The reference coverage is the Monte-Carlo finite-L expectation of THIS
credible-interval estimator (0.8825 / 0.6671 at L=100, X.25), not the
level*L/(L+1) figure the first round used, which is 0.0085 too high.
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

import argparse, json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import chisquare, kstest, norm

REFERENCE = {"90": 0.8825, "68": 0.6671}
N_NULL = 20_000


def covered(rec: dict, p: str, level: str) -> int:
    lo, hi = rec["ci"][p][level]
    return int(lo <= rec["theta_true"][p] <= hi)


def required_widening(observed: float, reference: float) -> float:
    if observed >= reference:
        return 0.0
    z = norm.ppf(0.5 + reference / 2)
    return brentq(lambda k: 2 * norm.cdf(z / k) - 1 - observed, 1.0, 10.0) - 1.0


def rank_tests(ranks: np.ndarray, L: int, rng) -> dict:
    """KS + chi-square on ranks, each with an exact-discrete-null control."""
    n = len(ranks)
    u = (ranks + 0.5) / (L + 1.0)
    ks_p = float(kstest(u, "uniform").pvalue)

    # Exact discrete null: ranks ~ Uniform{0..L}.  Same n, same statistic.
    null = rng.integers(0, L + 1, size=(N_NULL, n))
    null_ks = np.array([kstest((r + 0.5) / (L + 1.0), "uniform").pvalue
                        for r in null[:2000]])
    # chi-square on coarse bins: L+1=101 cells at n=100 is far too sparse for
    # the asymptotic chi-square, so bin into 10 equal-width rank bins.
    nb = 10
    edges = np.linspace(-0.5, L + 0.5, nb + 1)
    obs, _ = np.histogram(ranks, bins=edges)
    chi2_p = float(chisquare(obs, f_exp=np.full(nb, n / nb)).pvalue)
    null_chi2 = np.array([
        chisquare(np.histogram(r, bins=edges)[0], f_exp=np.full(nb, n / nb)).pvalue
        for r in null[:2000]])
    return {
        "ks_p": ks_p,
        "ks_null_frac_below": float((null_ks <= ks_p).mean()),
        "ks_null_median": float(np.median(null_ks)),
        "chi2_p": chi2_p,
        "chi2_null_frac_below": float((null_chi2 <= chi2_p).mean()),
        "extreme_frac": float(np.mean((ranks == 0) | (ranks == L))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--label", default="campaign")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    recs = [json.loads(p.read_text())
            for p in sorted(Path(a.archive).glob("inj_*.json"))]
    ok = [r for r in recs if r.get("status") == "ok"]
    n = len(ok)
    if not ok:
        raise SystemExit(f"no converged injections in {a.archive}")
    params = list(ok[0]["ranks"])
    L = int(ok[0].get("L_effective", 100))
    model = ok[0].get("model", "gr_eternal")
    rng = np.random.default_rng(20260921)

    other = [r for r in recs if r.get("status") != "ok"]
    print(f"=== {a.label}: model={model}  archive={a.archive} ===")
    print(f"launched {len(recs)}, converged {n}, other "
          f"{sorted({r['status'] for r in other}) if other else 'none'}")
    if other:
        print("  non-converged SNR: "
              + str([round(r.get("network_snr", float('nan')), 1) for r in other]))
    wall = np.array([r["cum_wall_s"] for r in ok])
    print(f"compute {sum(r['cum_wall_s'] for r in recs)/3600:.2f} h; "
          f"converged wall 5/50/95% = {np.round(np.percentile(wall,[5,50,95]),0)}")
    print(f"L = {L}, reference coverage {REFERENCE['90']} / {REFERENCE['68']} "
          f"(MC finite-L, X.25), Bonferroni threshold p > {0.05/len(params):.4f}\n")

    snr = np.array([r["network_snr"] for r in ok])
    rows = []
    print(f"{'param':>20} {'rank_mean':>10} {'rank_sd':>8} {'extreme':>8} "
          f"{'KS p':>9} {'KSnull':>7} {'chi2 p':>9} {'cov90':>7} {'cov68':>7}")
    for p in params:
        rk = np.array([r["ranks"][p] for r in ok], float)
        t = rank_tests(rk, L, rng)
        c9 = np.mean([covered(r, p, "90") for r in ok])
        c6 = np.mean([covered(r, p, "68") for r in ok])
        print(f"{p:>20} {rk.mean():>10.1f} {rk.std(ddof=1):>8.1f} "
              f"{t['extreme_frac']:>8.3f} {t['ks_p']:>9.4f} "
              f"{t['ks_null_frac_below']:>7.3f} {t['chi2_p']:>9.4f} "
              f"{c9:>7.3f} {c6:>7.3f}")
        rows.append({"param": p, "n": n, "rank_mean": rk.mean(),
                     "rank_sd": rk.std(ddof=1), "cov90": c9, "cov68": c6, **t})
    print(f"\n(theory: rank mean {L/2:.1f}, sd {np.sqrt(((L+1)**2-1)/12):.2f}, "
          f"extreme frac {2/(L+1):.4f})")
    print("KSnull = fraction of exact-discrete-null trials with a KS p at least "
          "as small.\n  If KS were calibrated at this N it tracks the KS p "
          "itself; a large gap means\n  the continuous test is misreading "
          "discrete ranks (lesson 3).\n")

    # ---- coverage, injection-level bootstrap (lesson 5) ----
    for lv, ref in REFERENCE.items():
        mat = np.array([[covered(r, p, lv) for p in params] for r in ok])
        tot = mat.sum(axis=1)
        obs = tot.mean() / len(params)
        boot = tot[rng.integers(0, n, (200_000, n))].mean(axis=1) / len(params)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        pv = float(((boot - obs + ref) <= obs).mean())
        print(f"nominal {lv}%: coverage {obs:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  "
              f"ref {ref:.4f}  bootstrap one-sided p = {pv:.5f}")
        print(f"            required widening {100*required_widening(obs,ref):+.1f}%  "
              f"CI [{100*required_widening(hi,ref):+.1f}%, "
              f"{100*required_widening(lo,ref):+.1f}%]")
        # lesson 2: decompose by SNR tercile before believing the aggregate
        qs = np.percentile(snr, [100/3, 200/3])
        for name, sel in (("low  ", snr <= qs[0]),
                          ("mid  ", (snr > qs[0]) & (snr <= qs[1])),
                          ("high ", snr > qs[1])):
            if sel.sum():
                print(f"            SNR {name} (n={sel.sum():>2}, "
                      f"{snr[sel].min():.2g}-{snr[sel].max():.3g}): "
                      f"coverage {mat[sel].mean():.4f}")
        rows.append({"param": f"COMBINED_{lv}", "n": n * len(params),
                     "rank_mean": np.nan, "rank_sd": np.nan,
                     "cov90": obs if lv == "90" else np.nan,
                     "cov68": obs if lv == "68" else np.nan,
                     "ks_p": pv, "ks_null_frac_below": np.nan,
                     "ks_null_median": np.nan, "chi2_p": np.nan,
                     "chi2_null_frac_below": np.nan, "extreme_frac": np.nan})
        print()

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with out.open("w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[c]:.6g}" if isinstance(r[c], float)
                              else str(r[c]) for c in cols) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
