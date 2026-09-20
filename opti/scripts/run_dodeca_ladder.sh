#!/bin/bash
# run_dodeca_ladder.sh — chained short warm-start escalations on the
# dodecahedron boundary (∂Δ⁸-style ladder, but on the 36-triangle S² mesh).
#
# Why short rungs: the within-rung yield of a warm-started escalation fits
#     gain(t) ≈ c · t^0.8        (measured on escalation 3: +33 at 7 min,
#                                 +58 at 15 min, +170 at 60 min)
# and every rung also pays a fixed ~40 s of JIT compile + host setup.  Total
# coverage per hour is therefore (3600/(t+40)) · c·t^0.8, which peaks at
# t = 4·40 s ≈ 2.7 min at ≈ 274 facets/hour, against 182 for hour-long rungs.
# 2.5 min is the author's chosen setting, essentially at that peak.  See
# dodeca_anneal_log.md for the curve and its caveats — the fit is three points
# from one rung and its constant drifts as coverage rises.
#
# Each rung seeds half its population from the previous rung's best state
# (gpu_sc --init), so the ladder carries the search forward rather than
# restarting it.  A rung that ever covers all 7,776 facets writes cover.txt
# and the script stops to audit it.
#
# Resumable: re-running continues from the highest-numbered rung already on
# disk, so the ladder can be stopped and restarted without losing the chain.
#
# Usage:  ./run_dodeca_ladder.sh                 # 20 rungs from the last state
#         RUNGS=40 ./run_dodeca_ladder.sh        # longer
#         TL=300 RUNGS=12 ./run_dodeca_ladder.sh # 5-minute rungs instead
#         INIT=results/dodeca_k3/summary.json OUTDIR=results/my_ladder \
#             ./run_dodeca_ladder.sh
set -uo pipefail          # deliberately NOT -e: a rung that finds nothing must
cd "$(dirname "$0")"      # not kill the ladder
PY=${PY:-$HOME/Documents/Abstract_Motion_GPU/venv/bin/python}
TL=${TL:-150}                                   # seconds per rung
# EPOCHS, if set, bounds each rung by epoch count instead of wall-clock: the
# run does exactly that many epochs and stops.  Use it when the budget should
# be a fixed amount of *search* rather than a fixed amount of time (epochs
# vary in duration, so --time-limit gives a variable number of them).
EPOCHS=${EPOCHS:-}
RUNGS=${RUNGS:-20}
SEED0=${SEED0:-90}
OUTDIR=${OUTDIR:-results/dodeca_ladder}
INIT=${INIT:-}
mkdir -p "$OUTDIR" logs

# --- resume: find the highest rung already on disk, continue from it --------
start=1
if [ -z "$INIT" ]; then
  last=$(ls -d "$OUTDIR"/rung* 2>/dev/null | sort | tail -1 || true)
  if [ -n "$last" ] && [ -f "$last/summary.json" ]; then
    INIT="$last/summary.json"
    start=$(( 10#$(basename "$last" | sed 's/rung//') + 1 ))
    echo "resuming: initialised from $INIT, starting at rung $start"
  else
    INIT=results/dodeca_k3_escal5/summary.json
    echo "fresh ladder: initialised from $INIT"
  fi
fi
[ -f "$INIT" ] || { echo "no init summary at $INIT — nothing to seed from"; exit 1; }
start_cov=$(python3 -c "import json,sys;print(json.load(open('$INIT'))['coverage'])")
echo "seeding state covers $start_cov of 7776 facets"

end=$(( start + RUNGS - 1 ))
for i in $(seq "$start" "$end"); do
  out="$OUTDIR/rung$(printf '%03d' "$i")"
  seed=$(( SEED0 + i ))
  cov_before=$(python3 -c "import json;print(json.load(open('$INIT'))['coverage'])" 2>/dev/null || echo 0)
  echo "=== rung $i  $(date -u +%H:%M:%SZ)  seed $seed  from $cov_before facets  -> $out ==="
  if [ -n "$EPOCHS" ]; then budget=(--epochs "$EPOCHS")
  else budget=(--epochs 200 --time-limit "$TL"); fi
  $PY -u gpu_sc.py --complex dodeca --k 3 --cmax 40 --S 4096 --iters 5000 \
      "${budget[@]}" --beta 5.0 --gamma 0.5 --ptrack 0.3 \
      --guide 3.0 --guide-max 11 --seed "$seed" \
      --init "$INIT" --outdir "$out" 2>&1 \
    | grep --line-buffered -v -i "warn(" | tee "logs/$(basename "$out").log" \
    | grep --line-buffered -E "best coverage|VERIFIED|NOT a covering|full coverage|Traceback|Error" \
    || true
  if [ ! -f "$out/summary.json" ]; then
    echo "!!! rung $i produced no summary.json — stopping so the state is not lost"
    exit 1
  fi
  cov_after=$(python3 -c "import json;print(json.load(open('$out/summary.json'))['coverage'])")
  echo "--- rung $i done: $cov_before -> $cov_after  (+$(( cov_after - cov_before )))"
  if [ -f "$out/cover.txt" ]; then
    echo "*** FULL COVER in $out — auditing"
    python3 independent_check.py "$out" | tail -5
    exit 0
  fi
  INIT="$out/summary.json"
done
cov_end=$(python3 -c "import json;print(json.load(open('$INIT'))['coverage'])")
echo "=== ladder finished at rung $end: $cov_end of 7776 facets ($(python3 -c "print(f'{$cov_end/7776*100:.1f}')")%)"
