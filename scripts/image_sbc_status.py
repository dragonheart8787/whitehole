"""Compact status of the image SBC campaign directory."""

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
import json, sys
from pathlib import Path
import numpy as np

d = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/image_sbc_n20/M87star")
recs = sorted((json.loads(p.read_text()) for p in d.glob("inj_*.json")), key=lambda r: r["idx"])
ok = [r for r in recs if r["status"] == "ok"]
capped = [r for r in recs if r["status"] == "timeout_capped"]
other = [r for r in recs if r["status"] not in ("ok", "timeout_capped", "timeout")]
inflight = [r for r in recs if not r.get("final", False)]
tot = sum(r["cum_wall_s"] for r in recs)
print(f"started={len(recs):3d}  ok={len(ok):3d}  capped={len(capped):2d}  "
      f"other={len(other):2d}  in_flight={len(inflight):2d}  "
      f"compute={tot/3600:5.2f}h")
if ok:
    t = np.array([r["cum_wall_s"] for r in ok])
    print(f"  converged wall: 5/50/95% = {np.round(np.percentile(t,[5,50,95]),0)}  mean {t.mean():.0f}s")
if inflight:
    print("  in flight: " + ", ".join(
        f"{r['idx']}(snr={r.get('network_snr',0):.0f},{r['cum_wall_s']:.0f}s)" for r in inflight))
if other:
    print("  other: " + ", ".join(f"{r['idx']}:{r['status']}" for r in other))
