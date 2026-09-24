#!/usr/bin/env bash
# anneal_then_sat.sh — Run the annealer, then offer CP-SAT hybrid on stall.
#
# Usage:
#   bash scripts/anneal_then_sat.sh rp2 5 8 results/rp2_k6/summary.json 2
#                                   ^   ^  ^  ^                          ^
#                                   |   k  L  warm-start from            free domains
#
# Flow:
#   1. Annealer runs (GPU) until time-limit or full cover
#   2. If stall detected (< 100% coverage), asks user to launch CP-SAT
#   3. CP-SAT (CPU) runs in hybrid mode: freezes (k - free) domains,
#      re-solves the freed ones to cover the residual
#
set -euo pipefail

COMPLEX="${1:?usage: $0 COMPLEX K L WARM_JSON N_FREE [TIME_LIMIT] [SEED]}"
K="${2:?}"
L="${3:?}"
WARM="${4:?path to warm-start summary.json}"
N_FREE="${5:?number of domains to free for CP-SAT}"
TIME_LIMIT="${6:-1800}"
SEED="${7:-42}"

PY="${PY:-python3}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SAT_SCRIPT="$BASE_DIR/paper-overleaf/paper/scs_sat.py"

OUTDIR="$BASE_DIR/results/${COMPLEX}_k${K}_hybrid"
LOGDIR="$BASE_DIR/logs"
mkdir -p "$OUTDIR" "$LOGDIR"

echo "=============================================="
echo " PHASE 1: GPU ANNEALER"
echo " complex=$COMPLEX  k=$K  seed=$SEED"
echo " warm-start: $WARM"
echo " time-limit: ${TIME_LIMIT}s"
echo "=============================================="

$PY -u "$BASE_DIR/src/gpu_sc.py" \
    --complex "$COMPLEX" --k "$K" --cmax 40 --S 4096 --iters 5000 \
    --epochs 200 --beta 5.0 --gamma 0.5 --mu 0.5 --ptrack 0.3 \
    --T0 0.6 --T1 0.03 --seed "$SEED" --time-limit "$TIME_LIMIT" \
    --guide 3.0 --guide-max 11 \
    --init "$WARM" --outdir "$OUTDIR" \
    2>&1 | tee "$LOGDIR/${COMPLEX}_k${K}_hybrid.log"

echo ""
echo "=============================================="
echo " ANNEALER DONE — checking coverage..."
echo "=============================================="

# Check if cover is complete
SUMMARY="$OUTDIR/summary.json"
if [ ! -f "$SUMMARY" ]; then
    echo "No summary.json found — annealer may not have saved state."
    echo "Try running with a longer --time-limit."
    exit 1
fi

COVERED=$($PY -c "
import json
d = json.load(open('$SUMMARY'))
covered = set()
for dom in d['domains']:
    covered.update(dom)
nf = d['complex']['num_facets']
print(f'{len(covered)}/{nf}')
if len(covered) >= nf:
    exit(0)
else:
    exit(1)
" 2>/dev/null) && COMPLETE=1 || COMPLETE=0

echo "Coverage: $COVERED"

if [ "$COMPLETE" -eq 1 ]; then
    echo ""
    echo "FULL COVER — no CP-SAT needed!"
    echo "SC_strict($COMPLEX) <= $((K - 1))"
    exit 0
fi

echo ""
echo "=============================================="
echo " STALL DETECTED at $COVERED"
echo ""
echo " PHASE 2: CP-SAT hybrid"
echo "   freeze $((K - N_FREE)) best domains from annealer"
echo "   free $N_FREE domains for CP-SAT to re-solve"
echo "   chain length L=$L"
echo "=============================================="
echo ""
read -p "Launch CP-SAT now? [y/N] " answer

case "$answer" in
    [yY]|[yY][eE][sS])
        echo ""
        echo "Starting CP-SAT (CPU, this may take a while)..."
        echo ""

        SAT_OUT="$BASE_DIR/results/${COMPLEX}_k${K}_sat"

        $PY -u "$SAT_SCRIPT" \
            --complex "$COMPLEX" --k "$K" --L "$L" \
            --timeout 7200 --witness \
            --warm "$SUMMARY" --free "$N_FREE" \
            --outdir "$SAT_OUT" \
            2>&1 | tee "$LOGDIR/${COMPLEX}_k${K}_sat.log"

        echo ""
        echo "CP-SAT done. Check $SAT_OUT for witness."
        ;;
    *)
        echo ""
        echo "Skipped. To run CP-SAT later:"
        echo ""
        echo "  $PY $SAT_SCRIPT \\"
        echo "    --complex $COMPLEX --k $K --L $L --timeout 7200 \\"
        echo "    --witness --warm $SUMMARY --free $N_FREE \\"
        echo "    --outdir $BASE_DIR/results/${COMPLEX}_k${K}_sat"
        ;;
esac
