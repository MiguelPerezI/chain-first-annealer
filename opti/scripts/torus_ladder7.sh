#!/bin/bash
# torus_ladder7.sh — after rho(torus) = 11 (product covers certify exactly
# SC_strict <= 10): k=11 as an end-to-end validation of the search on the
# torus, then k=10 and k=9 with a weak cleanliness term (non-product domains
# needed to beat the product optimum), a k=3 non-product probe, an ablation.
cd "$(dirname "$0")"
PY=${PY:-$HOME/Documents/Abstract_Motion_GPU/venv/bin/python}
WAIT=${1:-0}
while [ "$WAIT" -gt 0 ] && kill -0 "$WAIT" 2>/dev/null; do sleep 10; done
run() { k=$1; cm=$2; tl=$3; seed=$4; tag=$5; shift 5
  echo "=== $(date -u +%FT%TZ) torus k=$k cmax=$cm beta=5 time-limit=$tl seed=$seed $* ==="
  "$PY" -u gpu_sc.py --complex torus --k $k --cmax $cm --S 4096 --iters 20000 --epochs 400 --beta 5.0 \
      --seed $seed --time-limit $tl --outdir results/torus_${tag} "$@" 2>&1 | grep --line-buffered -v Warning
}
run 11 20 600 101 k11_c20_b5 --gamma 0.5 --ptrack 0.3
run 10 40 1200 102 k10_c40_g01 --gamma 0.1 --ptrack 0.3
run 9 40 1200 103 k9_c40_g01 --gamma 0.1 --ptrack 0.3
run 3 20 1200 91 k3_c20_g01 --gamma 0.1 --ptrack 0.3
run 4 20 720 62 k4_c20_g02 --gamma 0.2 --ptrack 0.05
echo "=== LADDER7 DONE $(date -u +%FT%TZ) ==="
