#!/bin/bash
# run_s3_k2_push.sh — the guided two-chain campaign on ∂Δ⁴ = S³ (500 facets).
#
# The question (guided_sphere_k2_plan.md): is the k=2 plateau at exactly
# 460/500 — which is the best cover by ρ−1 = 2 rectangles — a *structural*
# attractor (Conjecture C1/C2), or were the twelve recorded two-chain searches
# simply starved of budget (every one of them ran 300 s, i.e. 2–3 epochs)?
# A two-domain cover would give SC_strict(∂Δ⁴) = 1, sharp against TC(S³) = 1.
#
# Stages (each is one gpu_sc.py invocation, 3600 s = ~69 epochs at ptrack 0.05):
#
#   p1cor   pass 1, corrected energy (β 5, γ 0.5): the energy every successful
#           certificate in this repo used, and what optimized_algorithms.md
#           §14.3 asks for.  seed 5.
#   p1flat  pass 1, plan-literal energy (β 0.05, γ 0 — the gpu_sc.py defaults
#           and what s3_k2_A–D ran).  Same seed 5, so the two runs share an
#           initial population and differ in the energy alone: a controlled
#           comparison of "is the plateau an artefact of the flat slope?".
#   p1      both of the above, back to back (~2 h).
#
#   esc     one escalation rung, configured by environment so the invocation
#           reads as what it is (see the ladder in s3_k2_push_log.md):
#             INIT=<summary.json to seed from>  OUT=<outdir>   (both required)
#             BETA=5.0 GAMMA=0.5 GUIDE=3 GMAX=11 CMAX=40 PTRACK=0.05 SEED=6
#           Warm-start: half the population is seeded with the reduced chains
#           of INIT's best state, half stays random.
#           `--guide-max` matters: a persistently uncovered facet saturates the
#           weight cap at 1 + guide/0.3, so with the old fixed cap of 4 an
#           increment of 1.0 already saturates — escalating `--guide` alone was
#           a no-op until gpu_sc.py exposed `--guide-max` (2026-09-19).
#
#   audit   independent_check.py on every k=2 directory that has a cover.txt
#           (it needs cover.txt; it crashes on the partial directories).
#
# Usage:
#   ./run_s3_k2_push.sh p1                    # the two controlled pass-1 runs
#   INIT=results/s3_k2_longcor_5/summary.json OUT=results/s3_k2_esc_6 \
#       GUIDE=3 GMAX=11 SEED=6 ./run_s3_k2_push.sh esc
#
# Outputs: results/<outdir>/{cover.txt,planners.txt,summary.json} on success,
# results/<outdir>/summary.json (partial: best_chains of the best state) on a
# miss — the partial summary is what the next rung seeds from.  Logs: logs/.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/Documents/Abstract_Motion_GPU/venv/bin/python}
AUDIT=${AUDIT:-independent_check.py}
TL=${TL:-3600}   # SEED is deliberately *not* defaulted here: p1 defaults to 5, esc to 6
mkdir -p results logs

[ -x "$PY" ] || { echo "python with numba not found at $PY (set PY=...)"; exit 1; }
nvidia-smi -L >/dev/null 2>&1 || { echo "no NVIDIA GPU visible"; exit 1; }

# one run: stage outdir beta gamma guide guide-max cmax seed [init] [ptrack]
run_one() {
  local stage=$1 out=$2 beta=$3 gamma=$4 guide=$5 gmax=$6 cmax=$7 seed=$8 init=${9:-} ptrack=${10:-0.05}
  local log="logs/${out##*/}.log" INIT=""
  [ -n "$init" ] && INIT="--init $init"
  echo "[$(date -u +%H:%M:%SZ)] $stage -> $out  (β $beta γ $gamma guide $guide/max $gmax cmax $cmax seed $seed ptrack $ptrack ${init:+init ${init##*/}})"
  $PY -u gpu_sc.py --complex s3 --k 2 --cmax "$cmax" --S 4096 --iters 20000 \
      --epochs 200 --time-limit "$TL" --beta "$beta" --gamma "$gamma" \
      --ptrack "$ptrack" --guide "$guide" --guide-max "$gmax" \
      --seed "$seed" $INIT --outdir "$out" 2>&1 \
      | grep --line-buffered -v -i "NumbaPerformanceWarning\|warn(" | tee "$log" \
      | grep --line-buffered -E "^  epoch|best coverage|VERIFIED|NOT a covering|full coverage|time limit|Traceback|Error"
  if [ -f "$out/cover.txt" ]; then
    echo "[$(date -u +%H:%M:%SZ)] *** COVER FOUND: $out — auditing"
    python3 "$AUDIT" "$out" | tail -4 || true
  else
    echo "[$(date -u +%H:%M:%SZ)] $out: partial (see $log)"
  fi
}

case "${1:-}" in
  p1cor) run_one p1cor "results/s3_k2_longcor_${SEED:-5}"  5.0 0.5 1.0 4 40 "${SEED:-5}" ;;
  p1flat) run_one p1flat "results/s3_k2_longflat_${SEED:-5}" 0.05 0.0 1.0 4 40 "${SEED:-5}" ;;
  p1)    "$0" p1cor; "$0" p1flat ;;
  esc)
    : "${INIT:?set INIT=<summary.json to seed from>}" "${OUT:?set OUT=<outdir>}"
    run_one esc "$OUT" "${BETA:-5.0}" "${GAMMA:-0.5}" "${GUIDE:-3.0}" "${GMAX:-11}" \
             "${CMAX:-40}" "${SEED:-6}" "$INIT" "${PTRACK:-0.05}" ;;
  audit)
    for d in results/s3_k2*; do
      [ -f "$d/cover.txt" ] || continue
      echo "=== $d"; python3 "$AUDIT" "$d" | tail -3
    done ;;
  *) sed -n '2,40p' "$0"; exit 1 ;;
esac
