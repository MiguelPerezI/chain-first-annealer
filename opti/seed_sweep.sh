#!/bin/bash
# seed_sweep.sh — one rung of the ladder, tried under as many seeds as you can
# afford: every try is the SAME warm restart (same IN, same params, fresh
# seed), and the most successful try's summary.json is copied to
# $OUT/summary.json, ready to seed the next rung.
#
#   IN=results/klein_rung_next OUT=results/klein_rung2 ./seed_sweep.sh
#   IN=results/klein_rung_next OUT=results/klein_rung2 \
#       TRIES=12 PARALLEL=3 BUDGET=7200 ./seed_sweep.sh
#
# Env knobs (defaults = the current rung command):
#   IN        required — run to seed every try from (must have summary.json)
#   OUT       required — rung name you choose; tries land in $OUT/tryN_sSEED/,
#             the winner is copied to $OUT/summary.json.  Must differ from IN.
#   TRIES     how many seeds to try                        (default 5)
#   SEED0     first seed; try t uses SEED0+t-1             (default 780)
#   PARALLEL  tries running simultaneously                 (default 1)
#   BUDGET    stop launching new tries after N seconds     (default 0 = off)
#   LAUNCH_DELAY  seconds between launches when PARALLEL>1 — staggers the
#             first-kernel-launch race: N processes hitting cuLaunchKernel in
#             the same seconds can each die with CUDA_ERROR_OUT_OF_MEMORY even
#             though the card has room for all of them at steady state
#             (default 45; 0 disables)
#   FREE_MB   wait until this much VRAM is free before each launch (default 1200)
#   EPOCHS TL ITERS S K CMAX T0 T1 BETA GAMMA PTRACK GUIDE GUIDEMAX COMPLEX
#             per-try annealer settings (defaults: 4, 600, 5000, 4096, 5, 40,
#             0.6, 0.03, 5.0, 0.5, 0.3, 3.0, 11, klein).  TL is multiplied by
#             PARALLEL so GPU time-sharing never truncates a schedule.
#
# Resume/retry: a try dir that already has summary.json (or cover.txt) is
# skipped, so re-running the same command continues where it stopped.  Raise
# TRIES to add more seeds on top; change SEED0 to add a fresh batch that
# competes with the old ones — the best try wins regardless of batch.
#
# Verbose: every try streams its full output tagged [tryN sSEED], and each
# epoch line is annotated with the epoch counter and the facet gains:
#   [try2 s781] epoch 2: best 1178.0/1536 ...  [ep 2/4 | vs IN +2 | this epoch +1]
# After every wave each try's final gain is reported, plus the sweep-wide best.
#
# A try that covers every facet is promoted at once, audited with
# independent_check.py, and stops the sweep.
#
# Cost to plan around: each try pays ~30 s of numba JIT before its first epoch
# (gpu_sc.py does not cache compiled kernels), then ~32 s per epoch on an
# otherwise-idle T4.  Default sweep = 5 x (30 s + 4x32 s) ~= 13 min.
# Without MPS, PARALLEL overlaps the JIT but barely raises tries/hour; with
# 'sudo nvidia-cuda-mps-control -d' running, tries genuinely overlap.

set -u
cd "$(dirname "$0")"
PY=${PY:-$HOME/Documents/Abstract_Motion_GPU/venv/bin/python}

IN=${IN:?usage: IN=results/klein_rung_next OUT=results/klein_rung2 ./seed_sweep.sh}
OUT=${OUT:?usage: IN=results/klein_rung_next OUT=results/klein_rung2 ./seed_sweep.sh}
TRIES=${TRIES:-5};     SEED0=${SEED0:-780};     PARALLEL=${PARALLEL:-1}
EPOCHS=${EPOCHS:-4};  ITERS=${ITERS:-5000};    S=${S:-4096}
K=${K:-5};            CMAX=${CMAX:-40};        TL=${TL:-600}
T0=${T0:-0.6};        T1=${T1:-0.03};          BUDGET=${BUDGET:-0}
BETA=${BETA:-5.0};    GAMMA=${GAMMA:-0.5};     PTRACK=${PTRACK:-0.3}
GUIDE=${GUIDE:-3.0};  GUIDEMAX=${GUIDEMAX:-11}; COMPLEX=${COMPLEX:-klein}

[ "$TRIES" -ge 0 ] && [ "$PARALLEL" -ge 1 ] || { echo "!! TRIES>=0 and PARALLEL>=1 required"; exit 1; }
LAUNCH_DELAY=${LAUNCH_DELAY:-45}
FREE_MB=${FREE_MB:-1200}
[ "$PARALLEL" -gt 1 ] || LAUNCH_DELAY=0
[ -f "$IN/summary.json" ] || { echo "!! $IN/summary.json not found — nothing to seed from"; exit 1; }
mkdir -p "$OUT" logs
[ "$(cd "$IN" && pwd)" != "$(cd "$OUT" && pwd)" ] || {
  echo "!! IN and OUT must be different directories — a losing sweep must not clobber the state it seeds from"; exit 1; }

TLTRY=$((TL * PARALLEL))
OUTBASE=$(basename "$OUT")

cov() { "$PY" -c "import json,sys
try: print(json.load(open(sys.argv[1]))['coverage'])
except Exception: print(-1)" "$1"; }
sgn() { if [ "$1" -ge 0 ]; then echo "+$1"; else echo "$1"; fi; }

NF=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['complex']['num_facets'])" "$IN/summary.json")
INCOV=$(cov "$IN/summary.json")

echo "=== $COMPLEX rung sweep ==="
echo "  IN   $IN  (coverage $INCOV/$NF)"
echo "  OUT  $OUT  — tries=$TRIES seeds $SEED0..$((SEED0 + TRIES - 1)) parallel=$PARALLEL budget=${BUDGET}s"
echo "  try  epochs=$EPOCHS iters=$ITERS k=$K cmax=$CMAX S=$S T=$T0->$T1 tl=${TLTRY}s guide=$GUIDE max=$GUIDEMAX"
echo "  launch stagger=${LAUNCH_DELAY}s  vram gate=${FREE_MB} MiB free per try"
if [ "$PARALLEL" -gt 1 ] && ! pgrep -f nvidia-cuda-mps >/dev/null; then
  echo "  note: MPS daemon not running — $PARALLEL processes time-slice the GPU (~${PARALLEL}x slower each);"
  echo "        'sudo nvidia-cuda-mps-control -d' lets them overlap for real AND shares one CUDA"
  echo "        context, which also removes the first-launch memory race."
fi

promote() {
  local best="" bc=-1 f c
  for f in "$OUT"/try*/summary.json; do
    [ -e "$f" ] || continue
    c=$(cov "$f")
    [ "$c" -ge 0 ] && [ "$c" -gt "$bc" ] && { bc=$c; best=$f; }
  done
  if [ -z "$best" ]; then
    echo "!! no completed try to keep — not writing $OUT/summary.json"
    return 1
  fi
  cp "$best" "$OUT/summary.json"
  echo "  kept $best  (coverage $bc/$NF)  ->  $OUT/summary.json"
}

bestline() {
  local best="" bc=-1 f c
  for f in "$OUT"/try*/summary.json; do
    [ -e "$f" ] || continue
    c=$(cov "$f")
    [ "$c" -ge 0 ] && [ "$c" -gt "$bc" ] && { bc=$c; best=$f; }
  done
  [ -n "$best" ] && echo "== sweep best so far: $(basename "$(dirname "$best")") at $bc/$NF ($(sgn $((bc - INCOV))) vs IN)"
}

on_intr() {
  trap - INT TERM
  echo; echo "== interrupted — keeping the best try completed so far"
  wait 2>/dev/null
  promote
  exit 130
}
trap on_intr INT TERM

start=$(date +%s)
t=0
while [ "$t" -lt "$TRIES" ]; do
  if [ "$BUDGET" -gt 0 ] && [ $(( $(date +%s) - start )) -ge "$BUDGET" ]; then
    echo "== budget spent after $t tries — wrapping up"; break
  fi
  wave=()
  while [ "$t" -lt "$TRIES" ] && [ "${#wave[@]}" -lt "$PARALLEL" ]; do
    t=$((t + 1)); seed=$((SEED0 + t - 1)); d="$OUT/try${t}_s${seed}"
    if [ -f "$d/summary.json" ] || [ -f "$d/cover.txt" ]; then
      echo "== try$t (seed $seed) already in $d — skipping"; continue
    fi
    # stagger: let the previous try get through its JIT + first kernel launch
    # (where it grabs context memory + local-memory pools) before the next one
    if [ "${#wave[@]}" -gt 0 ] && [ "$LAUNCH_DELAY" -gt 0 ]; then sleep "$LAUNCH_DELAY"; fi
    # vram gate: steady footprint is ~1 GiB per try; wait if the card is fuller
    for _ in $(seq 1 60); do
      fm=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
      [ -n "$fm" ] || fm=999999
      [ "$fm" -ge "$FREE_MB" ] && break
      echo "== only ${fm} MiB free — waiting 10 s before try$t (seed $seed)"
      sleep 10
    done
    if [ "$fm" -lt "$FREE_MB" ]; then
      echo "== warning: still only ${fm} MiB free — launching try$t anyway"
    fi
    wave+=("$d")
    echo "== $(date -u +%H:%M:%S)  try$t seed=$seed -> $d"
    ( "$PY" -u gpu_sc.py --complex "$COMPLEX" --k "$K" --cmax "$CMAX" --S "$S" --iters "$ITERS" \
        --epochs "$EPOCHS" --beta "$BETA" --gamma "$GAMMA" --ptrack "$PTRACK" \
        --T0 "$T0" --T1 "$T1" --seed "$seed" --time-limit "$TLTRY" \
        --guide "$GUIDE" --guide-max "$GUIDEMAX" \
        --init "$IN/summary.json" --outdir "$d" 2>&1 \
      | grep --line-buffered -v "NumbaPerformanceWarning\|warn(" \
      | awk -v tag="[try${t} s${seed}]" -v tot="$EPOCHS" -v incov="$INCOV" '
          BEGIN { prev = incov }
          $1 == "epoch" {
            e = $2; sub(/:$/, "", e); c = $4; sub(/\..*/, "", c)
            printf "%s %s  [ep %s/%s | vs IN %+d | this epoch %+d]\n", tag, $0, e, tot, c - incov, c - prev
            fflush(); prev = c; next
          }
          { print tag" "$0; fflush() }
        ' \
      | tee -a "logs/${OUTBASE}_try${t}_s${seed}.log" ) &
  done
  [ "${#wave[@]}" -eq 0 ] && continue
  wait
  for d in "${wave[@]}"; do
    if [ -f "$d/cover.txt" ]; then
      echo; echo "*** FULL COVER in $d — promoting and auditing ***"
      cp "$d"/cover.txt "$d"/planners.txt "$d"/summary.json "$OUT/"
      "$PY" independent_check.py "$OUT" | tail -10
      echo "*** covering system in $OUT — sweep stops here ***"
      exit 0
    fi
    c=$(cov "$d/summary.json")
    if [ "$c" -lt 0 ]; then
      echo "== $(basename "$d") FAILED — no readable summary (see logs/${OUTBASE}_$(basename "$d").log)"
    else
      echo "== $(basename "$d") done: $c/$NF  ($(sgn $((c - INCOV))) vs IN $INCOV)"
    fi
  done
  bestline
done

echo
echo "=== tries, best first ==="
"$PY" - "$OUT" "$INCOV" <<'EOF'
import glob, json, sys
rows = []
for f in sorted(glob.glob(sys.argv[1] + "/try*/summary.json")):
    try:
        s = json.load(open(f))
        rows.append((s.get("coverage", -1), s.get("params", {}).get("seed", "?"), f))
    except Exception:
        rows.append((-1, "?", f + "  (unreadable)"))
incov = int(sys.argv[2])
for c, seed, f in sorted(rows, reverse=True):
    print(f"  {c:>5}  seed {seed:>4}  {c - incov:+d} vs IN  {f}")
if not rows:
    print("  (no completed tries)")
EOF

promote || exit 1
bc=$(cov "$OUT/summary.json")
echo
if [ "$bc" -gt "$INCOV" ]; then
  echo "== rung improved $INCOV -> $bc (+$((bc - INCOV))) — next: IN=$OUT OUT=<new name> $0"
else
  echo "== best try $bc did not beat IN ($INCOV) — next: keep IN=$IN, or IN=$OUT for the fresh state, OUT=<new name> $0"
fi