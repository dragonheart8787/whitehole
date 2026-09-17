"""Compact status of the image SBC campaign directory."""
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
