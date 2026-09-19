#!/bin/bash
# run_bary.sh — the barycenter triangle (Δ² = {0,1,2} subdivided by its
# barycenter 3 into the three triangles {0,1,3},{0,2,3},{1,2,3}) with the
# chain-first GPU annealer.
#
# Theory: vertex 3 lies in every facet, so K is a cone on ∂Δ² and ‖K‖ is
# contractible — exactly the case in which, as the paper notes right before
# defining SC_strict, a motion planner exists on the whole of K x K.  The
# chain π₁ ∼ c₃ ∼ π₂ (c₃ the constant map at the barycenter) is valid on all
# 54 facets, so ONE domain and 3 maps suffice: the target is SC_strict = 0.
#
#   ./run_bary.sh k1      # k=1: the single-domain search (target 54/54)
#   ./run_bary.sh k1c3    # k=1, cmax 2: chains forced to be exactly π₁,m,π₂
#   ./run_bary.sh k3      # k=3: the generic setting, for comparison (≤ 2)
#   ./run_bary.sh all
#   TL=600 ./run_bary.sh k1     # override the time limit (seconds)
#
# Outputs: results/bary_k1/, results/bary_k1_c3/, results/bary_k3/; logs in logs/.
set -u
cd "$(dirname "$0")"
PY=${PY:-$HOME/Documents/Abstract_Motion_GPU/venv/bin/python}
AUDIT="$(pwd)/independent_check.py"
TL=${TL:-300}
MODE=${1:-k1}

[ -x "$PY" ] || { echo "python with numba not found at $PY (set PY=...)"; exit 1; }
nvidia-smi -L >/dev/null 2>&1 || { echo "no NVIDIA GPU visible"; exit 1; }

audit_and_md() {
  [ -f "$1/cover.txt" ] || return 1
  echo "--- code-independent audit of $1 ---"
  python3 "$AUDIT" "$1" | tail -6
  "$PY" make_certificate_md.py bary "$1" "$1/certificate.md" \
      "Chain-first GPU annealer (gpu_sc.py, beta 5 / gamma 0.5 / track moves); see optimized_algorithms.md and the barycenter run log." \
      2>&1 | grep -v Warning
  "$PY" make_definition_md.py bary "$1" "barycenter triangle, chain-first annealer" 2>&1 | grep -v Warning
}

run_one() {  # $1 = k, $2 = outdir, $3 = cmax, $4 = seed
  echo "=== $(date -u +%FT%TZ) bary k=$1 cmax=$3 beta=5 gamma=0.5 ptrack=0.3 time-limit=$TL seed=$4 ==="
  "$PY" -u gpu_sc.py --complex bary --k "$1" --cmax "$3" --S 4096 --iters 5000 --epochs 60 \
      --beta 5.0 --gamma 0.5 --ptrack 0.3 --T0 0.6 --T1 0.03 \
      --seed "$4" --time-limit "$TL" --outdir "$2" 2>&1 | grep --line-buffered -v Warning
  if audit_and_md "$2"; then :; else echo "k=$1: no full cover (partial summary.json written)"; fi
}

case "$MODE" in
  k1|all)   run_one 1 results/bary_k1    16 71 ;;
esac
case "$MODE" in
  k1c3|all) run_one 1 results/bary_k1_c3  2 71 ;;
esac
case "$MODE" in
  k3|all)   run_one 3 results/bary_k3    16 71 ;;
esac
echo "=== RUN_BARY DONE $(date -u +%FT%TZ) ==="
