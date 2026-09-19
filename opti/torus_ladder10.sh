#!/bin/bash
# torus_ladder10.sh — continue the (6,4) hybrid (283/288 after 10 min) from its best state, 20 min.
cd "$(dirname "$0")"
PY=${PY:-$HOME/Documents/Abstract_Motion_GPU/venv/bin/python}
WAIT=${1:-0}
while [ "$WAIT" -gt 0 ] && kill -0 "$WAIT" 2>/dev/null; do sleep 10; done
echo "=== $(date -u +%FT%TZ) hybrid m=6 k=4 continuation (init from results/torus_hybrid_m6_k4) 1200 s ==="
"$PY" -u hybrid_cover.py torus results/torus_starcover_opt/summary.json 6 4 results/torus_hybrid_m6_k4_cont \
    --time-limit 1200 --seed 31 --init results/torus_hybrid_m6_k4/summary.json 2>&1 | grep --line-buffered -v Warning
echo "=== LADDER10 DONE $(date -u +%FT%TZ) ==="
