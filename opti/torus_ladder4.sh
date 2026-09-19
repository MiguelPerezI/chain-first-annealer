#!/bin/bash
# torus_ladder4.sh — corrected energy (beta 5, gamma 0.5, ptrack 0.3):
#  1. k=8 certificate attempt (any full cover gives the first torus bound),
#  2. k=1 single-chain ceiling probe,
#  3. continuation of the best k=4 run (still climbing at its 12-min limit),
#  4. k=6 certificate attempt,
#  5. gamma 0.2 / no-track ablation at k=4.
cd "$(dirname "$0")"
PY=${PY:-$HOME/Documents/Abstract_Motion_GPU/venv/bin/python}
WAIT=${1:-0}
while [ "$WAIT" -gt 0 ] && kill -0 "$WAIT" 2>/dev/null; do sleep 10; done
run() { k=$1; cm=$2; tl=$3; seed=$4; tag=$5; shift 5
  echo "=== $(date -u +%FT%TZ) torus k=$k cmax=$cm beta=5 time-limit=$tl seed=$seed $* ==="
  "$PY" -u gpu_sc.py --complex torus --k $k --cmax $cm --S 4096 --iters 20000 --epochs 400 --beta 5.0 \
      --seed $seed --time-limit $tl --outdir results/torus_${tag} "$@" 2>&1 | grep --line-buffered -v Warning
}
run 8 40 600 71 k8_c40_b5 --gamma 0.5 --ptrack 0.3
run 1 40 480 72 k1_c40_b5 --gamma 0.5 --ptrack 0.3
run 4 20 1500 61 k4_c20_b5_cont --gamma 0.5 --ptrack 0.3 --init results/torus_k4_c20_b5/summary.json
run 6 40 900 73 k6_c40_b5 --gamma 0.5 --ptrack 0.3
run 4 20 720 62 k4_c20_g02 --gamma 0.2 --ptrack 0.05
echo "=== LADDER4 DONE $(date -u +%FT%TZ) ==="
