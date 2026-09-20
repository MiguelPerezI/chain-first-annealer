#!/bin/bash
# run_cube.sh — the cube boundary (8 vertices 0..7, the six square faces cut
# along their drawn diagonals into 12 triangles) with the chain-first GPU
# annealer.
#
# Theory: every edge lies in exactly two triangles and every vertex link is a
# single cycle (f-vector (8, 18, 12), chi = 2), so ||K|| is a closed surface
# with chi = 2: the sphere S² — the same space as ∂Δ³, in a different
# (8-vertex, non-minimal) triangulation.  Hence SC_strict >= TC(S²) = 2, i.e.
# at least 3 domains, and the sharp answer is 3 domains.
#
# Product domains cannot reach it here (unlike ∂Δ³, where three stars suffice):
# the stars of K cover 6, 4, 5, 3, 4, 4, 5, 5 triangles, so the rectangles
# star(v) x star(w) cover at most 36 of the 144 facet pairs of K x K each
# (36+30+30+30 = 126 < 144 already forces rho >= 5), and the exact
# branch-and-bound of rect_cover.py proves rho = 8: no 7 rectangles cover
# K x K.  A product cover therefore certifies only SC <= 7 (the explicit
# 8-domain one is results/cube_starcover), so the sharp 2 can only come from
# the search — which it does: the guided k=3 run lands on 3 domains
# (482+236+146, chains 5/5/4 links), SC_strict(cube) = 2, sharp.
#
#   ./run_cube.sh k3      # k=3 cold (no guide): 662/864 in 300 s — short
#   ./run_cube.sh k3w     # k=3 continuation of a previous run's best state:
#                         #   820/864 after 814 s — 44 facets short
#   ./run_cube.sh k3g     # k=3 continuation + the coverage guide: THE RESULT
#                         #   864/864 in 304 s => sharp SC_strict(cube) = 2
#   ./run_cube.sh k3c     # k=3 cold + the coverage guide: 858/864 in 815 s,
#                         #   the whole population pinned at 856-858 — 6 short
#                         #   (the guide needs a good seed, not just a budget)
#   ./run_cube.sh k2      # k=2 control — TC(S²) = 2 forbids a 2-domain cover
#   ./run_cube.sh k4      # k=4 comparison (covers, gives SC <= 3)
#   ./run_cube.sh all     # k3, k2, k4
#   TL=900 ./run_cube.sh k3c    # override the time limit (seconds)
#
# Mode details:
#   k3w/k3g seed from a previous annealer run (INIT=<summary.json>, default
#   results/cube_k3_v2/summary.json) — a star_cover.py certificate records
#   domains but no map tables, so it cannot seed the population.
#   The chain that produced the sharp result, in order:
#     TL=300 ./run_cube.sh k3                                  # -> cube_k3   (662/864)
#     INIT=results/cube_k3/summary.json TL=900 ./run_cube.sh k3w   # -> cube_k3_v2 (820/864)
#     ./run_cube.sh k3g                                        # -> cube_k3_guided (864/864)
#
# Outputs: results/cube_k3/, cube_k3_v2/, cube_k3_guided/, cube_k2/, cube_k4/;
# logs in logs/.
set -u
cd "$(dirname "$0")"
PY=${PY:-$HOME/Documents/Abstract_Motion_GPU/venv/bin/python}
AUDIT="$(pwd)/independent_check.py"
TL=${TL:-300}
MODE=${1:-k3}
GUIDE=${GUIDE:-1.0}

[ -x "$PY" ] || { echo "python with numba not found at $PY (set PY=...)"; exit 1; }
nvidia-smi -L >/dev/null 2>&1 || { echo "no NVIDIA GPU visible"; exit 1; }

audit_and_md() {
  [ -f "$1/cover.txt" ] || return 1
  echo "--- code-independent audit of $1 ---"
  python3 "$AUDIT" "$1" | tail -6
  "$PY" make_certificate_md.py cube "$1" "$1/certificate.md" \
      "Chain-first GPU annealer (gpu_sc.py, beta 5 / gamma 0.5 / track moves); see optimized_algorithms.md §26 and the cube run log." \
      2>&1 | grep -v Warning
  "$PY" make_definition_md.py cube "$1" "cube boundary, chain-first annealer" 2>&1 | grep -v Warning
}

run_one() {  # $1 = k, $2 = outdir, $3 = cmax, $4 = seed, $5 = init summary, $6 = guide
  local INIT="" GD=""
  [ -n "${5:-}" ] && INIT="--init $5"
  [ -n "${6:-}" ] && [ "$6" != "0" ] && GD="--guide $6"
  echo "=== $(date -u +%FT%TZ) cube k=$1 cmax=$3 beta=5 gamma=0.5 ptrack=0.3 time-limit=$TL seed=$4 $INIT $GD ==="
  "$PY" -u gpu_sc.py --complex cube --k "$1" --cmax "$3" --S 4096 --iters 5000 --epochs 60 \
      --beta 5.0 --gamma 0.5 --ptrack 0.3 --T0 0.6 --T1 0.03 \
      --seed "$4" --time-limit "$TL" $INIT $GD --outdir "$2" 2>&1 | grep --line-buffered -v Warning
  if audit_and_md "$2"; then :; else echo "k=$1: no full cover (partial summary.json written)"; fi
}

case "$MODE" in
  k3|all)  run_one 3 results/cube_k3 16 81 ;;
esac
case "$MODE" in
  k2|all)  run_one 2 results/cube_k2 16 81 ;;
esac
case "$MODE" in
  k4|all)  run_one 4 results/cube_k4 16 81 "${INIT:-}" ;;   # INIT=… to warm-start
esac
# k3w/k3g: continue from a previous annealer summary (the k largest domains of
# that summary are handed to half the population as seeds).  k3g adds --guide:
# the adaptive weight increment on uncovered facets that closed the last 44.
case "$MODE" in
  k3w) run_one 3 results/cube_k3_v2 16 82 "${INIT:-results/cube_k3/summary.json}" 0 ;;
esac
case "$MODE" in
  k3g) run_one 3 results/cube_k3_guided 16 84 "${INIT:-results/cube_k3_v2/summary.json}" "$GUIDE" ;;
esac
case "$MODE" in
  k3c) run_one 3 results/cube_k3_gcold 16 86 "" "$GUIDE" ;;
esac
echo "=== RUN_CUBE DONE $(date -u +%FT%TZ) ==="
