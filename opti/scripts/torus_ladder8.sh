#!/bin/bash
# torus_ladder8.sh — hybrid certificates: keep the m largest product rectangles
# of results/torus_starcover_opt, anneal k chains on the remainder (6 min each),
# then a k=3 non-product probe.  Waits for pid $1 (the k=11 validation run).
cd "$(dirname "$0")"
PY=${PY:-$HOME/Documents/Abstract_Motion_GPU/venv/bin/python}
WAIT=${1:-0}
while [ "$WAIT" -gt 0 ] && kill -0 "$WAIT" 2>/dev/null; do sleep 10; done
hyb() { m=$1; k=$2; tl=$3; seed=$4; shift 4
  echo "=== $(date -u +%FT%TZ) hybrid m=$m k=$k time-limit=$tl seed=$seed $* ==="
  "$PY" -u hybrid_cover.py torus results/torus_starcover_opt/summary.json $m $k results/torus_hybrid_m${m}_k${k} \
      --time-limit $tl --seed $seed "$@" 2>&1 | grep --line-buffered -v Warning
}
hyb 7 2 360 11
hyb 6 3 360 12
hyb 5 3 360 13
hyb 5 4 360 14
hyb 4 4 360 15
hyb 3 5 360 16
echo "=== $(date -u +%FT%TZ) torus k=3 cmax=20 gamma=0.1 probe ==="
"$PY" -u gpu_sc.py --complex torus --k 3 --cmax 20 --S 4096 --iters 20000 --epochs 400 --beta 5.0 --gamma 0.1 --ptrack 0.3 \
    --seed 91 --time-limit 1200 --outdir results/torus_k3_c20_g01 2>&1 | grep --line-buffered -v Warning
echo "=== LADDER8 DONE $(date -u +%FT%TZ) ==="
