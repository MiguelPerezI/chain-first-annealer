#!/bin/bash
# torus_ladder3.sh — torus ladder with the settings that solve the circle:
# steep penalty slope (beta 5), 30 % track moves, optional per-chain
# cleanliness (GAMMA), short chains.  Waits for pid $1 to exit first.
cd "$(dirname "$0")"
PY=${PY:-$HOME/Documents/Abstract_Motion_GPU/venv/bin/python}
GAMMA=${GAMMA:-0.0}
WAIT=${1:-0}
while [ "$WAIT" -gt 0 ] && kill -0 "$WAIT" 2>/dev/null; do sleep 10; done
run() { k=$1; cm=$2; tl=$3; seed=$4; tag=$5; shift 5
  echo "=== $(date -u +%FT%TZ) torus k=$k cmax=$cm beta=5 ptrack=0.3 gamma=$GAMMA time-limit=$tl seed=$seed $* ==="
  "$PY" -u gpu_sc.py --complex torus --k $k --cmax $cm --S 4096 --iters 20000 --epochs 400 \
      --beta 5.0 --ptrack 0.3 --gamma $GAMMA \
      --seed $seed --time-limit $tl --outdir results/torus_${tag} "$@" 2>&1 | grep --line-buffered -v Warning
}
run 4 20 720 41 k4_c20_b5
run 4 40 720 42 k4_c40_b5
run 3 20 720 43 k3_c20_b5
echo "=== LADDER3 DONE $(date -u +%FT%TZ) ==="
