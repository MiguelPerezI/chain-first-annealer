#!/bin/bash
# torus_ladder.sh — first chain-first annealer ladder on the Császár torus:
# k=6 (10 min), k=4 (15 min), k=3 (15 min), sequential (one GPU).
cd "$(dirname "$0")"
PY=${PY:-$HOME/Documents/Abstract_Motion_GPU/venv/bin/python}
run() { k=$1; tl=$2; seed=$3; shift 3
  echo "=== $(date -u +%FT%TZ) torus k=$k time-limit=$tl seed=$seed $* ==="
  "$PY" -u gpu_sc.py --complex torus --k $k --cmax 40 --S 4096 --iters 20000 --epochs 400 \
      --seed $seed --time-limit $tl --outdir results/torus_k${k}_L1 "$@" 2>&1 | grep -v Warning
}
run 6 600 31
run 4 900 32
run 3 900 33
echo "=== LADDER DONE $(date -u +%FT%TZ) ==="
