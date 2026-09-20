#!/bin/bash
# push_s3_v2.sh — retry the S³ two-chain search with the corrected energy
# (beta 5, gamma 0.5; see optimized_algorithms.md §14), plus regression runs
# (S² k=3, S³ k=3) under the new settings.  Waits for pid $1 first.
cd "$(dirname "$0")"
PY=${PY:-$HOME/Documents/Abstract_Motion_GPU/venv/bin/python}
WAIT=${1:-0}
while [ "$WAIT" -gt 0 ] && kill -0 "$WAIT" 2>/dev/null; do sleep 10; done
run() { cx=$1; k=$2; cm=$3; tl=$4; seed=$5; tag=$6; shift 6
  echo "=== $(date -u +%FT%TZ) $cx k=$k cmax=$cm time-limit=$tl seed=$seed $* ==="
  "$PY" -u gpu_sc.py --complex $cx --k $k --cmax $cm --S 4096 --iters 20000 --epochs 400 \
      --beta 5.0 --gamma 0.5 --seed $seed --time-limit $tl --outdir results/$tag "$@" 2>&1 | grep --line-buffered -v Warning
}
run s2 3 20 120 51 s2_k3_v2 --ptrack 0.3
run s3 3 40 300 52 s3_k3_v2 --ptrack 0.3
run s3 2 40 300 53 s3_k2_v2_track --ptrack 0.3
run s3 2 40 300 54 s3_k2_v2_point --ptrack 0.05
echo "=== PUSH_S3_V2 DONE $(date -u +%FT%TZ) ==="
