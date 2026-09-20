#!/bin/bash
# run_torus.sh — re-execute the Császár-torus results of optimized_algorithms.md Part III.
#
#   bash run_torus.sh            # explicit 11-domain certificate (SC_strict(T²) <= 10) + independent audit  (CPU, ~1 min)
#   bash run_torus.sh ilp        # also recompute the minimum rectangle cover rho(torus) = 11 (scipy milp, ~10 s)
#   bash run_torus.sh anneal     # also the GPU chain-first annealer runs of §15 (k=4, corrected energy, 12 min each)
#   bash run_torus.sh hybrid     # also one hybrid run (6 rectangles + 3 annealed chains, 6 min, GPU)
#
# Outputs: results/torus_starcover_opt/{cover.txt,planners.txt,summary.json} (audited),
#          results/torus_k4_c40_b5/, results/torus_hybrid_m6_k3/ for the optional GPU runs.
set -e
cd "$(dirname "$0")"
PY=${PY:-$HOME/Documents/Abstract_Motion_GPU/venv/bin/python}
AUDIT=$HOME/Documents/Abstract_Motion_GPU/independent_check.py
[ -x "$PY" ] || { echo "python with numba not found at $PY (set PY=...)"; exit 1; }
MODE=${1:-cert}

echo "== explicit product certificate: 11 rectangles star(v) x star(w) (rho(torus) = 11) =="
"$PY" star_cover.py torus results/torus_starcover_opt 0,0 0,4 1,1 2,4 2,6 3,3 4,0 4,2 5,5 6,2 6,6
echo "== independent audit (Abstract_Motion_GPU/independent_check.py) =="
python3 "$AUDIT" results/torus_starcover_opt | tail -3

if [ "$MODE" = "ilp" ] || [ "$MODE" = "all" ]; then
  echo "== exact minimum rectangle cover (integer program) =="
  "$PY" - <<'PYEOF'
import numpy as np, gpu_sc
from scipy.optimize import milp, LinearConstraint, Bounds
P = gpu_sc.build("torus"); tris = [set(f) for f in P.kfacets]; nt = len(tris); nv = P.nv
rects = [(v, w) for v in range(nv) for w in range(nv)]
M = np.zeros((nt*nt, len(rects)), np.int8)
for r, (v, w) in enumerate(rects):
    for s in range(nt):
        if v in tris[s]:
            for t in range(nt):
                if w in tris[t]: M[s*nt+t, r] = 1
res = milp(np.ones(len(rects)), constraints=LinearConstraint(M, lb=np.ones(nt*nt), ub=np.full(nt*nt, np.inf)),
           integrality=np.ones(len(rects)), bounds=Bounds(0, 1))
print("rho(torus) =", round(res.fun), "rectangles:", [rects[r] for r in np.flatnonzero(res.x > 0.5)])
PYEOF
fi

if [ "$MODE" = "anneal" ] || [ "$MODE" = "all" ]; then
  nvidia-smi -L || { echo "no GPU"; exit 1; }
  echo "== chain-first annealer, corrected energy (expect a plateau near 730/1176; §15) =="
  "$PY" -u gpu_sc.py --complex torus --k 4 --cmax 40 --S 4096 --iters 20000 --epochs 400 --beta 5.0 --gamma 0.5 --ptrack 0.3 \
      --seed 42 --time-limit 720 --outdir results/torus_k4_c40_b5 2>&1 | grep --line-buffered -v Warning
fi

if [ "$MODE" = "hybrid" ] || [ "$MODE" = "all" ]; then
  nvidia-smi -L || { echo "no GPU"; exit 1; }
  echo "== hybrid: 6 largest rectangles fixed + 3 annealed chains on the 288 remaining facets (§16.6) =="
  "$PY" -u hybrid_cover.py torus results/torus_starcover_opt/summary.json 6 3 results/torus_hybrid_m6_k3 --time-limit 360 --seed 12 \
      2>&1 | grep --line-buffered -v Warning
fi
echo "done."
