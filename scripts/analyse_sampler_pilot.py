"""Paired sampler-setting comparison for the image-channel gr_eternal pilots.

Reads the archived X.20 campaign (nlive=250, nact=2) and a pilot re-run that
changes exactly one sampler knob, for the same injection indices.  Truth and
data are reproduced from the same seed, so the only difference between the two
arms is that knob; the comparison is therefore paired injection by injection.

The read-out that carries the conclusion is the PAIRED 90% interval width
ratio, not the coverage point estimate: at pilot sample sizes the coverage SE
is ~0.03, while the width ratio resolves a few percent.
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

PARAMS = ["M", "a_star", "i", "position_angle", "ring_width_frac",
          "log10_total_flux_jy"]
L = 100


def load(root: Path, idx: int) -> dict | None:
    p = root / f"inj_{idx:04d}.json"
    return json.loads(p.read_text()) if p.exists() else None


def bilby_result(root: Path, rec: dict) -> dict | None:
    label = f"img_{rec['target'].replace('*', 'star')}_{rec['idx']:04d}"
    p = root / "bilby" / f"{label}_result.json"
    return json.loads(p.read_text()) if p.exists() else None


def diagnostics(root: Path, rec: dict) -> dict:
    """Behavioural quantities: ncall, iterations, ln Z error, information gain.

    These are what a sampler-setting change has to move.  A kwargs dict is not
    evidence -- `walks` was echoed back while doing nothing (K.2.1) and `nact`
    does not appear in bilby's own kwargs even when fully in effect (K.2.3).
    """
    d = bilby_result(root, rec)
    if d is None:
        return {"ncall": rec.get("ncall"), "niter": None, "zerr": None, "H": None}
    ns = d.get("nested_samples", {}).get("content", {})
    niter = len(next(iter(ns.values()))) if ns else None
    return {
        "ncall": int(d.get("num_likelihood_evaluations", 0)) or rec.get("ncall"),
        "niter": niter,
        "zerr": d.get("log_evidence_err"),
        "H": d.get("information_gain"),
    }


def covered(rec: dict, p: str, level: str) -> bool:
    lo, hi = rec["ci"][p][level]
    return bool(lo <= rec["theta_true"][p] <= hi)


def width(rec: dict, p: str, level: str) -> float:
    lo, hi = rec["ci"][p][level]
    return float(hi - lo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="artifacts/image_sbc_fixA/M87star")
    ap.add_argument("--test", required=True)
    ap.add_argument("--label", default="test")
    ap.add_argument("--indices", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    lbl = a.label
    base_root, test_root = Path(a.base), Path(a.test)
    idxs = [int(x) for x in a.indices.split(",")]

    rows, paired = [], []
    for i in idxs:
        b, t = load(base_root, i), load(test_root, i)
        if b is None:
            continue
        bd = diagnostics(base_root, b)
        td = diagnostics(test_root, t) if t else {}
        rows.append((i, b, t, bd, td))
        if b.get("status") == "ok" and t and t.get("status") == "ok":
            paired.append((b, t, bd, td))

    print(f"selected {len(rows)}  paired-converged {len(paired)}")
    hdr = (f"{'idx':>4} {'snr':>8} {'base_s':>8} {lbl+'_s':>10} {'w_ratio':>7} "
           f"{'base_ncall':>11} {lbl+'_ncall':>13} {'c_ratio':>7} "
           f"{'niter_r':>7} {'zerr_r':>7} {'H_ratio':>7}")
    print(hdr)
    for i, b, t, bd, td in rows:
        if t is None or t.get("status") != "ok":
            print(f"{i:>4} {b['network_snr']:>8.1f} {b['cum_wall_s']:>8.1f} "
                  f"{(t or {}).get('status', 'not_run'):>10} " + "-" * 50)
            continue
        print(f"{i:>4} {b['network_snr']:>8.1f} {b['cum_wall_s']:>8.1f} "
              f"{t['cum_wall_s']:>10.1f} {t['cum_wall_s']/b['cum_wall_s']:>7.2f} "
              f"{bd['ncall']:>11} {td['ncall']:>13} "
              f"{td['ncall']/bd['ncall']:>7.2f} "
              f"{td['niter']/bd['niter']:>7.2f} {td['zerr']/bd['zerr']:>7.3f} "
              f"{td['H']/bd['H']:>7.3f}")

    # ---- cost and behavioural verification ----
    def rat(key, field=None):
        if field:
            return np.array([t[field] / b[field] for b, t, _, _ in paired])
        return np.array([td[key] / bd[key] for _, _, bd, td in paired])

    wr = rat(None, "cum_wall_s")
    cr, nr, zr, hr = rat("ncall"), rat("niter"), rat("zerr"), rat("H")
    print(f"\ncost (paired, n={len(paired)}): "
          f"ncall ratio median {np.median(cr):.2f} (mean {cr.mean():.2f}); "
          f"wall ratio median {np.median(wr):.2f} (mean {wr.mean():.2f})")
    print(f"base ncall median {np.median([bd['ncall'] for _,_,bd,_ in paired]):.3g}, "
          f"{lbl} {np.median([td['ncall'] for _,_,_,td in paired]):.3g}")
    print(f"base wall median {np.median([b['cum_wall_s'] for b,_,_,_ in paired]):.1f} s, "
          f"{lbl} {np.median([t['cum_wall_s'] for _,t,_,_ in paired]):.1f} s")
    print(f"total wall: base {sum(b['cum_wall_s'] for b,_,_,_ in paired)/3600:.2f} h, "
          f"{lbl} {sum(t['cum_wall_s'] for _,t,_,_ in paired)/3600:.2f} h")
    print(f"\nbehavioural: iterations ratio median {np.median(nr):.2f}  "
          f"ln Z error ratio median {np.median(zr):.3f}  "
          f"information gain ratio median {np.median(hr):.3f}")

    # ---- coverage / width ----
    print(f"\n{'param':>20} {'cov90_b':>8} {'cov90_t':>8} {'cov68_b':>8} "
          f"{'cov68_t':>8} {'w90_ratio_med':>14}")
    summary = {}
    for p in PARAMS:
        c9b = np.mean([covered(b, p, "90") for b, _, _, _ in paired])
        c9t = np.mean([covered(t, p, "90") for _, t, _, _ in paired])
        c6b = np.mean([covered(b, p, "68") for b, _, _, _ in paired])
        c6t = np.mean([covered(t, p, "68") for _, t, _, _ in paired])
        wrat = np.array([width(t, p, "90") / width(b, p, "90")
                         for b, t, _, _ in paired])
        summary[p] = (c9b, c9t, c6b, c6t)
        print(f"{p:>20} {c9b:>8.3f} {c9t:>8.3f} {c6b:>8.3f} {c6t:>8.3f} "
              f"{np.median(wrat):>14.4f}")

    m = [np.mean([summary[p][j] for p in PARAMS]) for j in range(4)]
    allw = np.array([width(t, p, "90") / width(b, p, "90")
                     for b, t, _, _ in paired for p in PARAMS])
    lo = allw.mean() - 1.96 * allw.std(ddof=1) / np.sqrt(len(allw))
    hi = allw.mean() + 1.96 * allw.std(ddof=1) / np.sqrt(len(allw))
    print(f"\nmean 90% coverage: base {m[0]:.3f} -> {lbl} {m[1]:.3f} "
          f"(theory {0.90 * L / (L + 1):.3f})")
    print(f"mean 68% coverage: base {m[2]:.3f} -> {lbl} {m[3]:.3f} "
          f"(theory {0.68 * L / (L + 1):.3f})")
    print(f"90% width ratio over {len(allw)} (injection, parameter) pairs: "
          f"median {np.median(allw):.4f}, mean {allw.mean():.4f}, "
          f"95% CI {lo:.4f}..{hi:.4f}")

    from scipy.stats import wilcoxon, binomtest
    for p in PARAMS:
        d = np.array([width(t, p, "90") - width(b, p, "90")
                      for b, t, _, _ in paired])
        if np.any(d != 0):
            print(f"  wilcoxon w90 {p:>20}: p = {wilcoxon(d).pvalue:.4f}")
    print(f"  wilcoxon w90 {'ALL':>20}: p = {wilcoxon(np.log(allw)).pvalue:.4f}")

    for lv in ("90", "68"):
        up = sum(1 for b, t, _, _ in paired for p in PARAMS
                 if not covered(b, p, lv) and covered(t, p, lv))
        dn = sum(1 for b, t, _, _ in paired for p in PARAMS
                 if covered(b, p, lv) and not covered(t, p, lv))
        n = up + dn
        pv = binomtest(up, n, 0.5).pvalue if n else 1.0
        print(f"  McNemar {lv}%: discordant {n}/{len(paired)*len(PARAMS)} "
              f"(miss->hit {up}, hit->miss {dn}), p = {pv:.3f}")

    # ---- ranks ----
    print(f"\n{'param':>20} {'rank_mean_b':>12} {'rank_mean_t':>12} "
          f"{'rank_sd_b':>10} {'rank_sd_t':>10}")
    for p in PARAMS:
        rb = np.array([b["ranks"][p] for b, _, _, _ in paired], float)
        rt = np.array([t["ranks"][p] for _, t, _, _ in paired], float)
        print(f"{p:>20} {rb.mean():>12.1f} {rt.mean():>12.1f} "
              f"{rb.std(ddof=1):>10.1f} {rt.std(ddof=1):>10.1f}")
    print(f"(theory: rank mean {L/2:.1f}, sd {np.sqrt(((L+1)**2-1)/12):.2f})")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["idx", "network_snr", "base_status", "base_wall_s", "base_ncall",
            "base_niter", "base_logZ_err", "base_information_gain",
            f"{lbl}_status", f"{lbl}_wall_s", f"{lbl}_ncall", f"{lbl}_niter",
            f"{lbl}_logZ_err", f"{lbl}_information_gain"]
    for p in PARAMS:
        cols += [f"base_in90_{p}", f"{lbl}_in90_{p}", f"base_w90_{p}",
                 f"{lbl}_w90_{p}", f"base_rank_{p}", f"{lbl}_rank_{p}"]
    with out.open("w") as fh:
        fh.write(",".join(cols) + "\n")
        for i, b, t, bd, td in rows:
            okb = b.get("status") == "ok"
            okt = bool(t) and t.get("status") == "ok"
            v = [str(i), f"{b['network_snr']:.6g}", b["status"],
                 f"{b['cum_wall_s']:.6g}", str(bd["ncall"]), str(bd["niter"]),
                 f"{bd['zerr']:.6g}", f"{bd['H']:.6g}",
                 (t or {}).get("status", "not_run"),
                 f"{t['cum_wall_s']:.6g}" if t else "",
                 str(td.get("ncall", "")), str(td.get("niter", "")),
                 f"{td['zerr']:.6g}" if td.get("zerr") else "",
                 f"{td['H']:.6g}" if td.get("H") else ""]
            for p in PARAMS:
                v += [str(int(covered(b, p, "90"))) if okb else "",
                      str(int(covered(t, p, "90"))) if okt else "",
                      f"{width(b, p, '90'):.6g}" if okb else "",
                      f"{width(t, p, '90'):.6g}" if okt else "",
                      str(b["ranks"][p]) if okb else "",
                      str(t["ranks"][p]) if okt else ""]
            fh.write(",".join(v) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
