"""Is the residual coverage gap statistically significant?  (audit X.28)

Eight candidate mechanisms were tested and excluded before anyone formally
asked whether the gap they were being tested against is distinguishable from
noise.  This does that test, from the archived campaign only -- no sampling.

The one design decision that matters: the six parameters of a single injection
are NOT independent (same data, same posterior), so pooling 6 x 95 indicators
into one binomial test understates the variance.  The independent unit is the
INJECTION, so the combined test counts covered parameters per injection and
bootstraps over injections.  The naive pooled binomial and Fisher's method are
reported alongside, marked as the anti-conservative comparisons they are.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binomtest, combine_pvalues, norm

PARAMS = ["M", "a_star", "i", "position_angle", "ring_width_frac",
          "log10_total_flux_jy"]
# Finite-L reference for this credible-interval estimator, measured in X.25 at
# L=100 over 200000 trials -- NOT level*L/(L+1), which is 0.0085 too high.
REFERENCE = {"90": 0.8825, "68": 0.6671}
N_BOOT = 200_000


def covered(rec: dict, p: str, level: str) -> int:
    lo, hi = rec["ci"][p][level]
    return int(lo <= rec["theta_true"][p] <= hi)


def required_widening(observed: float, reference: float) -> float:
    """Factor by which intervals must widen to bring coverage up to reference."""
    if observed >= reference:
        return 0.0
    z = norm.ppf(0.5 + reference / 2)
    return brentq(lambda k: 2 * norm.cdf(z / k) - 1 - observed, 1.0, 10.0) - 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="artifacts/image_sbc_fixA/M87star")
    ap.add_argument("--out",
                    default="docs/calibration/image_gr_eternal_coverage_significance.csv")
    a = ap.parse_args()

    recs = [json.loads(p.read_text())
            for p in sorted(Path(a.archive).glob("inj_*.json"))]
    ok = [r for r in recs if r.get("status") == "ok"]
    n = len(ok)
    print(f"{len(recs)} launched, {n} converged\n")

    rng = np.random.default_rng(20260918)
    rows = []
    for level, ref in REFERENCE.items():
        print(f"===== nominal {level}%  (reference {ref:.4f}) =====")
        print(f"{'param':>20} {'hits':>8} {'coverage':>9} "
              f"{'p(1-sided)':>11} {'Bonf x6':>9}")
        per_p, mat = [], np.zeros((n, len(PARAMS)), int)
        for j, p in enumerate(PARAMS):
            mat[:, j] = [covered(r, p, level) for r in ok]
            k = int(mat[:, j].sum())
            pv = binomtest(k, n, ref, alternative="less").pvalue
            per_p.append(pv)
            rows.append({"level": level, "scope": p, "hits": k, "n": n,
                         "coverage": k / n, "reference": ref, "p_value": pv,
                         "p_bonferroni": min(1.0, 6 * pv), "method": "binomial"})
            print(f"{p:>20} {k:>5}/{n:<3} {k/n:>9.3f} {pv:>11.4f} "
                  f"{min(1.0, 6*pv):>9.4f}")

        tot = mat.sum(axis=1)
        obs_mean, expected = tot.mean(), len(PARAMS) * ref
        boot = tot[rng.integers(0, n, (N_BOOT, n))].mean(axis=1)
        p_cluster = float(((boot - obs_mean + expected) <= obs_mean).mean())
        se = tot.std(ddof=1) / np.sqrt(n)
        cov_boot = boot / len(PARAMS)
        lo, hi = np.percentile(cov_boot, [2.5, 97.5])
        obs_cov = obs_mean / len(PARAMS)

        print(f"\n  combined, injection-level ({n} independent units):")
        print(f"    mean covered parameters/injection {obs_mean:.4f} "
              f"vs expected {expected:.4f}  (diff {obs_mean-expected:+.4f})")
        print(f"    cluster-robust SE {se:.4f}  ->  z = {(obs_mean-expected)/se:+.3f}")
        print(f"    bootstrap one-sided p = {p_cluster:.5f}")
        print(f"    coverage {obs_cov:.4f}, 95% CI [{lo:.4f}, {hi:.4f}]")
        print(f"    required widening {100*required_widening(obs_cov, ref):.1f}%  "
              f"CI [{100*required_widening(hi, ref):.1f}%, "
              f"{100*required_widening(lo, ref):.1f}%]")
        naive = binomtest(int(mat.sum()), mat.size, ref, alternative="less").pvalue
        fisher = combine_pvalues(per_p, method="fisher").pvalue
        print(f"    [naive pooled binomial p = {naive:.3g}; Fisher p = {fisher:.5f}"
              f"  -- both assume independence, anti-conservative]\n")

        rows.append({
            "level": level, "scope": "COMBINED", "hits": int(tot.sum()),
            "n": n * len(PARAMS), "coverage": obs_cov, "reference": ref,
            "p_value": p_cluster, "p_bonferroni": p_cluster,
            "method": "injection-level bootstrap",
        })
        rows.append({"level": level, "scope": "COMBINED_naive_pooled",
                     "hits": int(tot.sum()), "n": n * len(PARAMS),
                     "coverage": obs_cov, "reference": ref, "p_value": naive,
                     "p_bonferroni": naive, "method": "pooled binomial"})
        rows.append({"level": level, "scope": "COMBINED_fisher", "hits": -1,
                     "n": len(PARAMS), "coverage": obs_cov, "reference": ref,
                     "p_value": fisher, "p_bonferroni": fisher,
                     "method": "Fisher on per-parameter p"})

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with out.open("w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[c]:.6g}" if isinstance(r[c], float)
                              else str(r[c]) for c in cols) + "\n")
    print(f"wrote {out}")

    capped = [r for r in recs if r.get("status") != "ok"]
    if capped:
        print(f"\nconvergence-selection caveat: {len(capped)} injections excluded "
              f"({sorted({r['status'] for r in capped})}), SNR "
              f"{[round(r.get('network_snr', float('nan')), 1) for r in capped]}; "
              f"converged upper quartile is "
              f"{np.percentile([r['network_snr'] for r in ok], 75):.1f}")


if __name__ == "__main__":
    main()
