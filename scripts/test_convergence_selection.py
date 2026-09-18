"""Does convergence selection bias the residual coverage gap?  (audit X.29)

The X.20 campaign launched 100 injections and analysed the 95 that converged.
The 5 that hit the wall-clock cap all sit above the converged set's upper SNR
quartile, so the analysed set is not a uniform draw from the prior.  This asks
whether that matters, from the archived data only -- no new sampling.

Two things this script is careful about:

  * The selection variable is COST, not SNR.  Injections were dropped for
    exceeding a 1500 s cap; SNR only correlates with that.  Regressing coverage
    on SNR alone would test the proxy rather than the mechanism, so both are
    fitted.
  * The six parameters of one injection share a posterior, so all inference
    bootstraps over INJECTIONS, never over the 6*N indicators.

The decisive number is not either regression but the arithmetic ceiling: with
5 of 100 injections missing, the pooled coverage cannot move by more than
5/100 * (1 - observed), whatever those 5 would have shown.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.stats import norm

PARAMS = ["M", "a_star", "i", "position_angle", "ring_width_frac",
          "log10_total_flux_jy"]
REFERENCE = {"90": 0.8825, "68": 0.6671}   # X.25 finite-L values
N_BOOT = 4000


def covered(rec: dict, p: str, level: str) -> int:
    lo, hi = rec["ci"][p][level]
    return int(lo <= rec["theta_true"][p] <= hi)


def logistic_slope(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    def nll(b):
        z = b[0] + b[1] * x
        return -np.sum(y * z - np.logaddexp(0.0, z))
    return minimize(nll, [0.0, 0.0], method="BFGS").x


def required_widening(observed: float, reference: float) -> float:
    if observed >= reference:
        return 0.0
    z = norm.ppf(0.5 + reference / 2)
    return brentq(lambda k: 2 * norm.cdf(z / k) - 1 - observed, 1.0, 10.0) - 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="artifacts/image_sbc_fixA/M87star")
    ap.add_argument("--out",
                    default="docs/calibration/image_gr_eternal_selection_effect.csv")
    a = ap.parse_args()

    recs = [json.loads(p.read_text())
            for p in sorted(Path(a.archive).glob("inj_*.json"))]
    ok = [r for r in recs if r.get("status") == "ok"]
    capped = [r for r in recs if r.get("status") != "ok"]
    n = len(ok)
    snr = np.array([r["network_snr"] for r in ok])
    wall = np.array([r["cum_wall_s"] for r in ok])
    print(f"converged {n}, excluded {len(capped)} "
          f"(SNR {np.round([r['network_snr'] for r in capped], 1).tolist()})")
    print(f"converged cost {wall.min():.0f}-{wall.max():.0f} s; "
          f"the excluded all exceeded the 1500 s cap\n")

    rng = np.random.default_rng(20260919)
    rows = []
    for level, ref in REFERENCE.items():
        mat = np.array([[covered(r, p, level) for p in PARAMS] for r in ok])
        obs = mat.mean()
        print(f"===== nominal {level}%  (reference {ref:.4f}, observed {obs:.4f}) =====")

        for name, cov_x in (("log10(SNR)", np.log10(snr)),
                            ("log10(cost)", np.log10(wall))):
            b = logistic_slope(np.repeat(cov_x, len(PARAMS)), mat.ravel())
            bs = np.array([
                logistic_slope(np.repeat(cov_x[i], len(PARAMS)), mat[i].ravel())[1]
                for i in (rng.integers(0, n, n) for _ in range(N_BOOT))])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            p2 = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
            print(f"  logistic slope vs {name:>12}: {b[1]:+.4f}  "
                  f"95% CI [{lo:+.4f}, {hi:+.4f}]  p = {p2:.4f}")
            rows.append({"level": level, "quantity": f"logit_slope_{name}",
                         "value": b[1], "ci_low": lo, "ci_high": hi,
                         "p_value": p2})
            q = np.quantile(cov_x, [1 / 3, 2 / 3])
            g = np.digitize(cov_x, q)
            terts = [mat[g == k].mean() for k in (0, 1, 2)]
            print(f"      tertiles: {terts[0]:.4f} / {terts[1]:.4f} / {terts[2]:.4f}")
            for k, t in enumerate(terts):
                rows.append({"level": level,
                             "quantity": f"tertile{k}_{name}", "value": t,
                             "ci_low": "", "ci_high": "", "p_value": ""})

        near = wall >= 1000
        print(f"  converged injections nearest the cap (>=1000 s, n={near.sum()}): "
              f"coverage {mat[near].mean():.4f}  vs rest {mat[~near].mean():.4f}")

        ceiling = 5 / 100 * (1 - obs)
        print(f"\n  what if the 5 excluded were included?")
        g = np.digitize(np.log10(wall), np.quantile(np.log10(wall), [1/3, 2/3]))
        for label, pred in (("high-cost tertile rate", mat[g == 2].mean()),
                            ("near-cap rate", mat[near].mean()),
                            ("BOUND: all 5 fully covered", 1.0),
                            ("BOUND: all 5 fully missed", 0.0)):
            pooled = (n * obs + 5 * pred) / (n + 5)
            print(f"    {label:>28}: pooled {pooled:.4f} "
                  f"(shift {pooled-obs:+.4f})  needs "
                  f"+{100*required_widening(pooled, ref):.1f}%")
            rows.append({"level": level, "quantity": f"pooled_{label}",
                         "value": pooled, "ci_low": "", "ci_high": "",
                         "p_value": ""})
        print(f"    arithmetic ceiling on any upward shift: "
              f"5/100 * (1 - {obs:.4f}) = {ceiling:.4f} "
              f"= {100*ceiling/(ref-obs):.0f}% of the {ref-obs:.4f} deficit\n")
        rows.append({"level": level, "quantity": "selection_ceiling_fraction",
                     "value": ceiling / (ref - obs), "ci_low": "",
                     "ci_high": "", "p_value": ""})

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
