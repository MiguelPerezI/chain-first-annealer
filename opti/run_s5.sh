#!/bin/bash
# run_s5.sh — S⁵ = ∂Δ⁶ with the chain-first GPU annealer (corrected energy).
#
# Theory target: rho(∂Δ⁶) = 3 (integer program, same as S³/S⁴), so the best
# product cover gives SC_strict(∂Δ⁶) <= 2 by 3 domains — the article's
# Theorem 1.  This script searches for such a 3-domain system directly
# (no product ansatz in the energy: the domains are found, not imposed).
#
# The search is heavy: ∂Δ⁶x∂Δ⁶ has 12348 facets and starmax = 9072
# (vs 2520 / 1750 for S⁴), so the kernel runs at ~0.019 Mmoves/s with
# cmax 16 instead of the ~0.10 Mmoves/s of the S⁴ run.  One 82M-move epoch
# of S⁴ would take ~71 min here, hence the smaller --iters (finer epochs,
# more annealing restarts and tournaments for the same move budget).
#
#   ./run_s5.sh main     # k=3, cmax 16 (default): the search for 3 domains
#   ./run_s5.sh cont     # continuation: warm start from main's summary.json, cmax 40
#   ./run_s5.sh all      # main, then cont if main did not reach full coverage
#   TL=21600 ./run_s5.sh main     # override the time limit (seconds)
#
# Outputs: results/s5_k3_v2/{cover.txt,planners.txt,summary.json}, log in logs/.
set -u
cd "$(dirname "$0")"
PY=${PY:-$HOME/Documents/Abstract_Motion_GPU/venv/bin/python}
AUDIT="$(pwd)/independent_check.py"
TL=${TL:-10800}
MODE=${1:-main}
OUT=results/s5_k3_v2

[ -x "$PY" ] || { echo "python with numba not found at $PY (set PY=...)"; exit 1; }
nvidia-smi -L >/dev/null 2>&1 || { echo "no NVIDIA GPU visible"; exit 1; }

audit_and_md() {
  [ -f "$1/cover.txt" ] || return 1
  echo "--- code-independent audit of $1 ---"
  python3 "$AUDIT" "$1" | tail -6
  "$PY" make_certificate_md.py s5 "$1" "$1/certificate.md" \
      "Chain-first GPU annealer (gpu_sc.py, corrected energy beta 5 / gamma 0.5 / track moves); see optimized_algorithms.md and the S5 run log." \
      2>&1 | grep -v Warning
}

case "$MODE" in
  main|all)
    echo "=== $(date -u +%FT%TZ) s5 k=3 cmax=16 beta=5 gamma=0.5 ptrack=0.3 time-limit=$TL ==="
    "$PY" -u gpu_sc.py --complex s5 --k 3 --cmax 16 --S 4096 --iters 5000 --epochs 400 \
        --beta 5.0 --gamma 0.5 --ptrack 0.3 --T0 0.6 --T1 0.03 \
        --seed 71 --time-limit "$TL" --outdir "$OUT" 2>&1 | grep --line-buffered -v Warning
    audit_and_md "$OUT" || echo "main: no full cover yet (partial summary.json written)"
    ;;
esac

case "$MODE" in
  cont|all)
    if [ -f "$OUT/cover.txt" ] && [ "$MODE" = all ]; then
      echo "=== main already produced a full cover — skipping the continuation ==="
    else
      echo "=== $(date -u +%FT%TZ) s5 continuation: seeded from $OUT/summary.json, cmax=40 ==="
      "$PY" -u gpu_sc.py --complex s5 --k 3 --cmax 40 --S 4096 --iters 5000 --epochs 400 \
          --beta 5.0 --gamma 0.5 --ptrack 0.3 --T0 0.6 --T1 0.03 --seed 72 \
          --time-limit "$TL" --init "$OUT/summary.json" --outdir results/s5_k3_cont 2>&1 | grep --line-buffered -v Warning
      audit_and_md results/s5_k3_cont || echo "cont: still no full cover"
    fi
    ;;
esac
echo "=== RUN_S5 DONE $(date -u +%FT%TZ) ==="
