#!/bin/bash
# run_wedge.sh — the paper's §5.2 complex (K_4 minus the edge {2,3}, whose
# realization is homotopic to a wedge of two circles) with the chain-first GPU
# annealer.
#
# Theory target: TC(S^1 v S^1) = 2, so the paper reports 3 optimal domains
# J0,J1,J2 and SC_strict = 2.  The product ansatz (star rectangles) only
# reaches rho = 4 -> SC_strict <= 3 here, so this is the first complex where
# the free-form search must beat every product cover.
#
#   ./run_wedge.sh k3     # k=3: the search for the paper's 3 domains
#   ./run_wedge.sh k2     # k=2: negative control (theory says no 2-domain cover)
#   ./run_wedge.sh all    # both
#   TL=600 ./run_wedge.sh k3      # override the time limit (seconds)
#
# Outputs: results/wedge_k3_v2/, results/wedge_k2/; logs in logs/.
set -u
cd "$(dirname "$0")"
PY=${PY:-$HOME/Documents/Abstract_Motion_GPU/venv/bin/python}
AUDIT="$(pwd)/independent_check.py"
TL=${TL:-300}
MODE=${1:-k3}

[ -x "$PY" ] || { echo "python with numba not found at $PY (set PY=...)"; exit 1; }
nvidia-smi -L >/dev/null 2>&1 || { echo "no NVIDIA GPU visible"; exit 1; }

audit_and_md() {
  [ -f "$1/cover.txt" ] || return 1
  echo "--- code-independent audit of $1 ---"
  python3 "$AUDIT" "$1" | tail -6
  "$PY" make_certificate_md.py wedge "$1" "$1/certificate.md" \
      "Chain-first GPU annealer (gpu_sc.py, beta 5 / gamma 0.5 / track moves); see optimized_algorithms.md and the wedge run log." \
      2>&1 | grep -v Warning
  "$PY" make_definition_md.py wedge "$1" "wedge of two circles, chain-first annealer" 2>&1 | grep -v Warning
}

run_one() {  # $1 = k, $2 = outdir, $3 = seed
  echo "=== $(date -u +%FT%TZ) wedge k=$1 beta=5 gamma=0.5 ptrack=0.3 time-limit=$TL seed=$3 ==="
  "$PY" -u gpu_sc.py --complex wedge --k "$1" --cmax 16 --S 4096 --iters 5000 --epochs 60 \
      --beta 5.0 --gamma 0.5 --ptrack 0.3 --T0 0.6 --T1 0.03 \
      --seed "$3" --time-limit "$TL" --outdir "$2" 2>&1 | grep --line-buffered -v Warning
  if audit_and_md "$2"; then :; else echo "k=$1: no full cover (partial summary.json written)"; fi
}

case "$MODE" in
  k3|all) run_one 3 results/wedge_k3_v2 71 ;;
esac
case "$MODE" in
  k2|all) run_one 2 results/wedge_k2 71 ;;
esac
echo "=== RUN_WEDGE DONE $(date -u +%FT%TZ) ==="
