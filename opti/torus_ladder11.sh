#!/bin/bash
# torus_ladder11.sh — two fresh-seed (6,4) hybrids (the H1 wall of the first
# (6,4) run is basin-specific).  Waits for pid $1.
cd "$(dirname "$0")"
PY=${PY:-$HOME/Documents/Abstract_Motion_GPU/venv/bin/python}
WAIT=${1:-0}
while [ "$WAIT" -gt 0 ] && kill -0 "$WAIT" 2>/dev/null; do sleep 10; done
for seed in 41 42; do
  echo "=== $(date -u +%FT%TZ) hybrid m=6 k=4 seed=$seed 600 s ==="
  "$PY" -u hybrid_cover.py torus results/torus_starcover_opt/summary.json 6 4 results/torus_hybrid_m6_k4_s$seed \
      --time-limit 600 --seed $seed 2>&1 | grep --line-buffered -v Warning
done
echo "=== LADDER11 DONE $(date -u +%FT%TZ) ==="
