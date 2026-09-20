#!/bin/bash
# push_s4.sh — S⁴ = ∂Δ⁵ with the chain-first annealer (corrected energy):
# k=3 should find the 3-rectangle product cover; k=2 is predicted to plateau
# at 34 x 70 = 2380 / 2520 (two rectangles miss two facet pairs).  Waits for pid $1.
cd "$(dirname "$0")"
PY=${PY:-$HOME/Documents/Abstract_Motion_GPU/venv/bin/python}
WAIT=${1:-0}
while [ "$WAIT" -gt 0 ] && kill -0 "$WAIT" 2>/dev/null; do sleep 10; done
run() { k=$1; tl=$2; seed=$3; shift 3
  echo "=== $(date -u +%FT%TZ) s4 k=$k cmax=40 beta=5 gamma=0.5 ptrack=0.3 time-limit=$tl seed=$seed ==="
  "$PY" -u gpu_sc.py --complex s4 --k $k --cmax 40 --S 4096 --iters 20000 --epochs 400 --beta 5.0 --gamma 0.5 --ptrack 0.3 \
      --seed $seed --time-limit $tl --outdir results/s4_k${k}_v2 "$@" 2>&1 | grep --line-buffered -v Warning
}
run 3 600 61
run 2 600 62
echo "=== PUSH_S4 DONE $(date -u +%FT%TZ) ==="
