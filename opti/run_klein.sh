#!/bin/bash
# run_klein.sh — the vertex-minimal Klein bottle (8 vertices 0..7, 16 triangles,
# Cervone's "242" triangulation from his Figure 5.10, relabelled A..H -> 0..7)
# with the chain-first GPU annealer: 16 x 16 triangle pairs x 6 staircase paths
# = 1536 maximal facets of K x K.
#
# Theory: every edge lies in exactly two triangles, every vertex link is a
# single cycle (f-vector (8, 24, 16), chi = 0), and NO coherent orientation of
# the triangles exists, so ||K|| is a closed non-orientable surface with
# chi = 0: the Klein bottle K^2 (the orientable chi = 0 alternative would be
# the torus).  8 vertices is provably the minimum for K^2 (Cervone: V >= 7 by
# V^2 - 7V + 6 chi >= 0, and V = 7 would make the 1-skeleton K_7, against
# Franklin's 6-colour theorem for K^2) — this is the fewest-facets Klein
# bottle there is.
#
# Hence SC_strict(K^2) >= TC(K^2) = 4 (Cohen–Vandembroucq 2017, also Davis and
# Iwase–Sakai–Tsutaya): at least FIVE domains in any system of piecewise
# linear motion planners, at any subdivision level.  A 5-domain level-0 cover
# is therefore sharp, SC_strict = 4 = TC, and the k=4 control below must never
# cover.
#
#   ./run_klein.sh k5      # k=5 cold, beta 5 / gamma 0.5 / track moves: TARGET
#   ./run_klein.sh k5w     # k=5 continuation of a previous run's best state
#   ./run_klein.sh k5g     # k=5 continuation + the coverage guide
#   ./run_klein.sh k5c     # k=5 cold + the coverage guide
#   ./run_klein.sh k4      # k=4 control — TC(K^2) = 4 forbids a 4-domain cover
#   ./run_klein.sh k6      # k=6 comparison (covers trivially, gives SC <= 5)
#   TL=900 ./run_klein.sh k5   # override the time limit (seconds)
#
# Outputs: results/klein_k5/, klein_k4/, klein_k6/, ...; logs in logs/.
set -u
cd "$(dirname "$0")"
PY=${PY:-$HOME/Documents/Abstract_Motion_GPU/venv/bin/python}
AUDIT="$(pwd)/independent_check.py"
TL=${TL:-1200}
MODE=${1:-k5}
GUIDE=${GUIDE:-1.0}

[ -x "$PY" ] || { echo "python with numba not found at $PY (set PY=...)"; exit 1; }
nvidia-smi -L >/dev/null 2>&1 || { echo "no NVIDIA GPU visible"; exit 1; }

audit_and_md() {
  [ -f "$1/cover.txt" ] || return 1
  echo "--- code-independent audit of $1 ---"
  python3 "$AUDIT" "$1" | tail -8
}

run_one() {  # $1 = k, $2 = outdir, $3 = cmax, $4 = seed, $5 = init summary, $6 = guide
  local INIT="" GD=""
  [ -n "${5:-}" ] && INIT="--init $5"
  [ -n "${6:-}" ] && [ "$6" != "0" ] && GD="--guide $6"
  echo "=== $(date -u +%FT%TZ) klein k=$1 cmax=$3 beta=5 gamma=0.5 ptrack=0.3 time-limit=$TL seed=$4 $INIT $GD ==="
  "$PY" -u gpu_sc.py --complex klein --k "$1" --cmax "$3" --S 4096 --iters 5000 --epochs 60 \
      --beta 5.0 --gamma 0.5 --ptrack 0.3 --T0 0.6 --T1 0.03 \
      --seed "$4" --time-limit "$TL" $INIT $GD --outdir "$2" 2>&1 | grep --line-buffered -v Warning
  if audit_and_md "$2"; then :; else echo "k=$1: no full cover (partial summary.json written)"; fi
}

case "$MODE" in
  k5) run_one 5 results/klein_k5 24 201 ;;
esac
case "$MODE" in
  k5w) run_one 5 results/klein_k5_v2 24 202 "${INIT:-results/klein_k5/summary.json}" 0 ;;
esac
case "$MODE" in
  k5g) run_one 5 results/klein_k5_guided 24 203 "${INIT:-results/klein_k5_v2/summary.json}" "$GUIDE" ;;
esac
case "$MODE" in
  k5c) run_one 5 results/klein_k5_gcold 24 204 "" "$GUIDE" ;;
esac
case "$MODE" in
  k4) run_one 4 results/klein_k4 24 205 ;;
esac
case "$MODE" in
  k6) run_one 6 results/klein_k6 24 206 ;;
esac
echo "=== RUN_KLEIN DONE $(date -u +%FT%TZ) ==="
