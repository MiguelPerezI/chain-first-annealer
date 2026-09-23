#!/bin/bash
# run_klein_overnight.sh — unattended overnight escalation on the minimal
# Klein bottle (1536 facets, k=5 target for the sharp SC_strict = 4 = TC(K^2)).
#
# Alternates two rung types, each 10 min (TL configurable):
#   even: RE-MELT   — warm-start from the current best state, hot T0=1.2,
#                     strong guide (5, cap 30), mu redundancy penalty: shake
#                     the basin apart and re-anneal it slowly.
#   odd:  COLD      — fresh seed, T0=0.9, 30-min schedule: a brand-new basin.
# After each rung the new summary is compared with the best on disk; if it is
# better its directory becomes the new best pointer (results/klein_best).
# A rung that ever covers all 1536 facets writes cover.txt and the script
# audits it and STOPS (exit 0).  Otherwise it keeps going until ROUNDS.
#
# Resumable: re-running continues from the best state found so far.
#   ./run_klein_overnight.sh            # 30 rounds x ~10-30 min
#   TL=1800 ./run_klein_overnight.sh    # 30-min rungs
set -u
cd "$(dirname "$0")"
PY=${PY:-$HOME/Documents/Abstract_Motion_GPU/venv/bin/python}
TL=${TL:-600}
ROUNDS=${ROUNDS:-30}
BEST=${BEST:-results/klein_night_melt1}  # campaign best at launch time (1174/1536)
mkdir -p logs results

for i in $(seq 1 "$ROUNDS"); do
  seed=$(( 400 + i * 7 ))
  if [ $(( i % 2 )) -eq 1 ]; then
    # re-melt rung: hot restart from the best known state
    OUT=results/klein_night_r$i
    T0=1.2; EPOCHS=30
    INIT="--init $BEST/summary.json"
    MODE="remelt from $BEST"
  else
    # cold rung: fresh basin, complete fast schedule (18 epochs ~ 9.5 min < TL)
    OUT=results/klein_night_r$i
    T0=0.9; EPOCHS=18
    INIT=""
    MODE="cold new basin"
  fi
  echo "=== $(date -u +%FT%TZ) round $i ($MODE) TL=$TL T0=$T0 seed=$seed -> $OUT ==="
  "$PY" -u gpu_sc.py --complex klein --k 5 --cmax 40 --S 4096 --iters 5000 \
      --epochs "$EPOCHS" --beta 5.0 --gamma 0.5 --mu 0.5 --ptrack 0.3 \
      --T0 "$T0" --T1 0.02 --guide 5.0 --guide-max 30 \
      --seed "$seed" --time-limit "$TL" $INIT --outdir "$OUT" 2>&1 \
      | grep --line-buffered -v -i "warn(" | tee "logs/$(basename "$OUT").log" \
      | grep --line-buffered -E "best coverage|VERIFIED|NOT a covering|full coverage|Traceback|Error" \
      || true
  [ -f "$OUT/summary.json" ] || { echo "!!! round $i produced no summary.json — stopping"; exit 1; }
  if [ -f "$OUT/cover.txt" ]; then
    echo "*** FULL COVER in $OUT — auditing"
    python3 independent_check.py "$OUT" | tail -8
    exit 0
  fi
  newcov=$("$PY" -c "import json;print(json.load(open('$OUT/summary.json'))['coverage'])")
  bestcov=$("$PY" -c "import json;print(json.load(open('$BEST/summary.json'))['coverage'])")
  if [ "$newcov" -gt "$bestcov" ]; then
    echo "--- new best: $OUT ($newcov > $bestcov)"
    BEST="$OUT"
  else
    echo "--- no improvement ($newcov vs best $bestcov)"
  fi
done
echo "=== night finished at $(date -u +%FT%TZ); best = $BEST ==="