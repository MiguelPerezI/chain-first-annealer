#!/bin/bash
# torus_ladder5.sh — after the discovery that gamma-0.5 domains are products
# A x A of 6-triangle sub-complexes (216 facets, product covers need >= 7):
#  1. k=1, gamma 0: fair single-chain ceiling probe (can a non-product domain beat 216?)
#  2. k=1, gamma 0.5: product baseline (expect 216)
#  3. k=7 and 4. k=6 certificate attempts with the corrected energy
#  5. gamma 0.2 / no-track ablation at k=4
cd "$(dirname "$0")"
PY=${PY:-$HOME/Documents/Abstract_Motion_GPU/venv/bin/python}
WAIT=${1:-0}
while [ "$WAIT" -gt 0 ] && kill -0 "$WAIT" 2>/dev/null; do sleep 10; done
run() { k=$1; cm=$2; tl=$3; seed=$4; tag=$5; shift 5
  echo "=== $(date -u +%FT%TZ) torus k=$k cmax=$cm beta=5 time-limit=$tl seed=$seed $* ==="
  "$PY" -u gpu_sc.py --complex torus --k $k --cmax $cm --S 4096 --iters 20000 --epochs 400 --beta 5.0 \
      --seed $seed --time-limit $tl --outdir results/torus_${tag} "$@" 2>&1 | grep --line-buffered -v Warning
}
run 1 40 600 81 k1_c40_g0 --gamma 0.0 --ptrack 0.3
run 1 40 480 82 k1_c40_b5 --gamma 0.5 --ptrack 0.3
run 7 40 900 83 k7_c40_b5 --gamma 0.5 --ptrack 0.3
run 6 40 900 84 k6_c40_b5 --gamma 0.5 --ptrack 0.3
run 4 20 720 62 k4_c20_g02 --gamma 0.2 --ptrack 0.05
echo "=== LADDER5 DONE $(date -u +%FT%TZ) ==="
