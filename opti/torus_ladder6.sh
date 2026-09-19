#!/bin/bash
# torus_ladder6.sh — non-product probes: k=3 with weak / no cleanliness term
# (gamma 0.5 drives the population onto star(v) x star(v) products, which
# cannot cover with fewer than 9 domains).  Waits for pid $1.
cd "$(dirname "$0")"
PY=${PY:-$HOME/Documents/Abstract_Motion_GPU/venv/bin/python}
WAIT=${1:-0}
while [ "$WAIT" -gt 0 ] && kill -0 "$WAIT" 2>/dev/null; do sleep 10; done
run() { k=$1; cm=$2; tl=$3; seed=$4; tag=$5; shift 5
  echo "=== $(date -u +%FT%TZ) torus k=$k cmax=$cm beta=5 time-limit=$tl seed=$seed $* ==="
  "$PY" -u gpu_sc.py --complex torus --k $k --cmax $cm --S 4096 --iters 20000 --epochs 400 --beta 5.0 \
      --seed $seed --time-limit $tl --outdir results/torus_${tag} "$@" 2>&1 | grep --line-buffered -v Warning
}
run 3 20 1200 91 k3_c20_g01 --gamma 0.1 --ptrack 0.3
run 3 20 1200 92 k3_c20_g0 --gamma 0.0 --ptrack 0.3
echo "=== LADDER6 DONE $(date -u +%FT%TZ) ==="
