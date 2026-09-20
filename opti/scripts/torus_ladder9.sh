#!/bin/bash
# torus_ladder9.sh — follow-ups to the hybrid ladder: (6,4) would give 10 domains
# (SC <= 9), and a longer (6,3) run.  Waits for pid $1.
cd "$(dirname "$0")"
PY=${PY:-$HOME/Documents/Abstract_Motion_GPU/venv/bin/python}
WAIT=${1:-0}
while [ "$WAIT" -gt 0 ] && kill -0 "$WAIT" 2>/dev/null; do sleep 10; done
hyb() { m=$1; k=$2; tl=$3; seed=$4; tag=$5; shift 5
  echo "=== $(date -u +%FT%TZ) hybrid m=$m k=$k time-limit=$tl seed=$seed $* ==="
  "$PY" -u hybrid_cover.py torus results/torus_starcover_opt/summary.json $m $k results/torus_hybrid_$tag \
      --time-limit $tl --seed $seed "$@" 2>&1 | grep --line-buffered -v Warning
}
hyb 6 4 600 21 m6_k4
hyb 6 3 900 22 m6_k3_long --cmax 40
echo "=== LADDER9 DONE $(date -u +%FT%TZ) ==="
