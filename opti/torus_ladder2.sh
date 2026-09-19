#!/bin/bash
# torus_ladder2.sh — second ladder: steep penalty slope (beta 5) and short
# chains, the settings that fixed the circle.  Waits for the running baseline
# job (pid $1) to exit before using the GPU.
cd "$(dirname "$0")"
PY=${PY:-$HOME/Documents/Abstract_Motion_GPU/venv/bin/python}
WAIT=${1:-0}
while [ "$WAIT" -gt 0 ] && kill -0 "$WAIT" 2>/dev/null; do sleep 10; done
run() { k=$1; cm=$2; tl=$3; seed=$4; tag=$5; shift 5
  echo "=== $(date -u +%FT%TZ) torus k=$k cmax=$cm beta=5 time-limit=$tl seed=$seed $* ==="
  "$PY" -u gpu_sc.py --complex torus --k $k --cmax $cm --S 4096 --iters 20000 --epochs 400 --beta 5.0 \
      --seed $seed --time-limit $tl --outdir results/torus_${tag} "$@" 2>&1 | grep --line-buffered -v Warning
}
run 4 10 720 41 k4_c10_b5
run 4 20 720 42 k4_c20_b5
run 3 10 720 43 k3_c10_b5
echo "=== LADDER2 DONE $(date -u +%FT%TZ) ==="
