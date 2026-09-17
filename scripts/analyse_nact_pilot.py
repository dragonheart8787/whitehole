"""Paired nact=2 vs nact=8 comparison for the image-channel gr_eternal pilot.

Reads the archived X.20 campaign (nact=2) and the pilot re-run (nact=8) for the
same injection indices.  Truth and data are reproduced from the same seed, so
the only difference between the two arms is the random-walk chain length; the
comparison is therefore paired injection by injection.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np

PARAMS = ["M", "a_star", "i", "position_angle", "ring_width_frac",
          "log10_total_flux_jy"]
L = 100


def load(root: Path, idx: int) -> dict | None:
    p = root / f"inj_{idx:04d}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def ncall_of(rec: dict, root: Path) -> int | None:
    """ncall, from the record if present else from bilby's own result file."""
    if rec.get("ncall"):
        return int(rec["ncall"])
    label = f"img_{rec['target'].replace('*', 'star')}_{rec['idx']:04d}"
    p = root / "bilby" / f"{label}_result.json"
    if not p.exists():
        return None
    return int(json.loads(p.read_text()).get("num_likelihood_evaluations", 0))


def covered(rec: dict, p: str, level: str) -> bool:
    lo, hi = rec["ci"][p][level]
    t = rec["theta_true"][p]
    return bool(lo <= t <= hi)


def width(rec: dict, p: str, level: str) -> float:
    lo, hi = rec["ci"][p][level]
    return float(hi - lo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="artifacts/image_sbc_fixA/M87star")
    ap.add_argument("--test", default="artifacts/image_sbc_nact8/M87star")
    ap.add_argument("--indices", required=True)
    ap.add_argument("--out", default="docs/calibration/image_gr_eternal_nact_pilot.csv")
    a = ap.parse_args()

    base_root, test_root = Path(a.base), Path(a.test)
    idxs = [int(x) for x in a.indices.split(",")]

    rows, paired = [], []
    for i in idxs:
        b, t = load(base_root, i), load(test_root, i)
        if b is None:
            continue
        row = {
            "idx": i, "network_snr": b["network_snr"],
            "base_status": b["status"], "base_wall_s": b["cum_wall_s"],
            "base_ncall": ncall_of(b, base_root),
            "nact8_status": (t or {}).get("status", "not_run"),
            "nact8_wall_s": (t or {}).get("cum_wall_s"),
            "nact8_ncall": ncall_of(t, test_root) if t else None,
        }
        rows.append(row)
        if b.get("status") == "ok" and t and t.get("status") == "ok":
            paired.append((b, t))

    print(f"selected {len(rows)}  paired-converged {len(paired)}")
    print(f"{'idx':>4} {'snr':>8} {'base_s':>8} {'n8_s':>8} {'w_ratio':>7} "
          f"{'base_ncall':>11} {'n8_ncall':>11} {'c_ratio':>7}")
    for r in rows:
        if r["nact8_wall_s"] is None:
            print(f"{r['idx']:>4} {r['network_snr']:>8.1f} {r['base_wall_s']:>8.1f} "
                  f"{'-':>8} {'-':>7} {r['base_ncall']:>11} {'-':>11} {'-':>7}")
            continue
        wr = r["nact8_wall_s"] / r["base_wall_s"]
        cr = (r["nact8_ncall"] / r["base_ncall"]) if r["nact8_ncall"] else float("nan")
        print(f"{r['idx']:>4} {r['network_snr']:>8.1f} {r['base_wall_s']:>8.1f} "
              f"{r['nact8_wall_s']:>8.1f} {wr:>7.2f} {r['base_ncall']:>11} "
              f"{str(r['nact8_ncall']):>11} {cr:>7.2f}")

    # ---- cost ----
    wr = np.array([t["cum_wall_s"] / b["cum_wall_s"] for b, t in paired])
    cr = np.array([ncall_of(t, test_root) / ncall_of(b, base_root) for b, t in paired])
    print(f"\ncost (paired, n={len(paired)}): "
          f"ncall ratio median {np.median(cr):.2f} (mean {cr.mean():.2f}); "
          f"wall ratio median {np.median(wr):.2f} (mean {wr.mean():.2f})")
    print(f"total wall: base {sum(b['cum_wall_s'] for b, _ in paired)/3600:.2f} h, "
          f"nact8 {sum(t['cum_wall_s'] for _, t in paired)/3600:.2f} h")

    # ---- coverage / width / rank ----
    print(f"\n{'param':>20} {'cov90_b':>8} {'cov90_8':>8} {'cov68_b':>8} "
          f"{'cov68_8':>8} {'w90_ratio_med':>14}")
    summary = {}
    for p in PARAMS:
        c9b = np.mean([covered(b, p, "90") for b, _ in paired])
        c98 = np.mean([covered(t, p, "90") for _, t in paired])
        c6b = np.mean([covered(b, p, "68") for b, _ in paired])
        c68 = np.mean([covered(t, p, "68") for _, t in paired])
        wrat = np.array([width(t, p, "90") / width(b, p, "90") for b, t in paired])
        summary[p] = (c9b, c98, c6b, c68, float(np.median(wrat)))
        print(f"{p:>20} {c9b:>8.3f} {c98:>8.3f} {c6b:>8.3f} {c68:>8.3f} "
              f"{np.median(wrat):>14.4f}")

    m9b = np.mean([summary[p][0] for p in PARAMS])
    m98 = np.mean([summary[p][1] for p in PARAMS])
    m6b = np.mean([summary[p][2] for p in PARAMS])
    m68 = np.mean([summary[p][3] for p in PARAMS])
    allw = np.array([width(t, p, "90") / width(b, p, "90")
                     for b, t in paired for p in PARAMS])
    print(f"\nmean 90% coverage: base {m9b:.3f} -> nact8 {m98:.3f} "
          f"(theory {0.90 * L / (L + 1):.3f})")
    print(f"mean 68% coverage: base {m6b:.3f} -> nact8 {m68:.3f} "
          f"(theory {0.68 * L / (L + 1):.3f})")
    print(f"90% width ratio over all {len(allw)} (injection, parameter) pairs: "
          f"median {np.median(allw):.4f}, mean {allw.mean():.4f}")

    # Wilcoxon signed-rank on the per-pair log width ratio (is the widening
    # systematic?), matching the G-1 read-out.
    try:
        from scipy.stats import wilcoxon
        for p in PARAMS:
            d = np.array([width(t, p, "90") - width(b, p, "90") for b, t in paired])
            if np.any(d != 0):
                print(f"  wilcoxon w90 {p:>20}: p = {wilcoxon(d).pvalue:.4f}")
        print(f"  wilcoxon w90 {'ALL':>20}: "
              f"p = {wilcoxon(np.log(allw)).pvalue:.2e}")
    except ImportError:
        print("  (scipy unavailable; Wilcoxon skipped)")

    # ---- ranks ----
    print(f"\n{'param':>20} {'rank_mean_b':>12} {'rank_mean_8':>12} "
          f"{'rank_sd_b':>10} {'rank_sd_8':>10}")
    for p in PARAMS:
        rb = np.array([b["ranks"][p] for b, _ in paired], float)
        rt = np.array([t["ranks"][p] for _, t in paired], float)
        print(f"{p:>20} {rb.mean():>12.1f} {rt.mean():>12.1f} "
              f"{rb.std(ddof=1):>10.1f} {rt.std(ddof=1):>10.1f}")
    print(f"(theory: rank mean {L / 2:.1f}, sd "
          f"{np.sqrt(((L + 1) ** 2 - 1) / 12):.2f})")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["idx", "network_snr", "base_status", "base_wall_s", "base_ncall",
            "nact8_status", "nact8_wall_s", "nact8_ncall"]
    pcols = []
    for p in PARAMS:
        pcols += [f"base_in90_{p}", f"nact8_in90_{p}", f"base_w90_{p}",
                  f"nact8_w90_{p}", f"base_rank_{p}", f"nact8_rank_{p}"]
    with out.open("w") as fh:
        fh.write(",".join(cols + pcols) + "\n")
        for r in rows:
            b, t = load(base_root, r["idx"]), load(test_root, r["idx"])
            vals = [str(r[c]) for c in cols]
            for p in PARAMS:
                okb = b.get("status") == "ok"
                okt = bool(t) and t.get("status") == "ok"
                vals += [
                    str(int(covered(b, p, "90"))) if okb else "",
                    str(int(covered(t, p, "90"))) if okt else "",
                    f"{width(b, p, '90'):.6g}" if okb else "",
                    f"{width(t, p, '90'):.6g}" if okt else "",
                    str(b["ranks"][p]) if okb else "",
                    str(t["ranks"][p]) if okt else "",
                ]
            fh.write(",".join(vals) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
