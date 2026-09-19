#!/bin/bash
# reproduce_all.sh — replay ONLY the successful, independently audited results
# of optimized_algorithms.md (Parts II–III) on a fresh machine.
#
#   bash reproduce_all.sh          # CPU-only certificates (about 2 minutes):
#                                  #   S¹, S², S³, S⁴ product covers (3 domains each), torus 11-domain cover,
#                                  #   cube boundary: exact rho = 8 and the 8-domain product cover
#   bash reproduce_all.sh gpu      # additionally the GPU searches that succeeded:
#                                  #   S² k=3 (≈ 25 s), S³ k=3 (≈ 3 min), S⁴ k=3 (≈ 13 min) with the corrected energy
#                                  #   the wedge of two circles k=3 (≈ 41 s) and a re-verification of
#                                  #   the paper's own §5.2 certificate (wedge_paper_check.py)
#                                  #   the barycenter triangle k=1 (≈ 15 s: the sharp SC = 0, one domain)
#                                  #   the cube boundary guided k=3 (≈ 5 min: the sharp SC = 2; replaying the
#                                  #   whole chain is ./run_cube.sh k3 + k3w + k3g ≈ 24 min) and its k=2 control
#
# Every certificate is verified with the plain Definition 2.3 check and then
# audited by the code-independent independent_check.py (shipped in this repo).
# Requirements: a python with numpy, numba, scipy (set PY=...);
# a CUDA GPU only for the 'gpu' mode.
set -e
cd "$(dirname "$0")"
PY=${PY:-$HOME/Documents/Abstract_Motion_GPU/venv/bin/python}
AUDIT="$(dirname "$0")/independent_check.py"
[ -x "$PY" ] || { echo "python with numba not found at $PY (set PY=...)"; exit 1; }
audit() { python3 "$AUDIT" "$1" | tail -1; }

echo "=== 1. spheres: explicit product certificates star(v) x star(v), v = 0, 1, 2  (Theorem, §19) ==="
for c in circle s2 s3 s4 s5; do
  "$PY" star_cover.py $c results/${c}_starcover 0,0 1,1 2,2 2>&1 | grep -v Warning
  audit results/${c}_starcover
done

echo "=== 2. Császár torus: optimal product cover, rho = 11  =>  SC_strict(T²) <= 10 ==="
"$PY" star_cover.py torus results/torus_starcover_opt 0,0 0,4 1,1 2,4 2,6 3,3 4,0 4,2 5,5 6,2 6,6 2>&1 | grep -v Warning
audit results/torus_starcover_opt

echo "=== 2b. cube boundary (8 vertices, 12 triangles): optimal product cover, rho = 8 => SC <= 7 ==="
"$PY" rect_cover.py cube 2>&1 | tail -1        # exact branch-and-bound, ~2 s
"$PY" star_cover.py cube results/cube_starcover 0,0 0,6 7,2 2,7 6,6 2,5 6,0 5,2 2>&1 | grep -v Warning
audit results/cube_starcover
"$PY" make_certificate_md.py cube results/cube_starcover results/cube_starcover/certificate.md "Optimal product cover: 8 rectangles star(v) x star(w), the exact rectangle-cover number rho = 8 (rect_cover.py); see optimized_algorithms.md §26." 2>&1 | grep -v Warning
"$PY" make_definition_md.py cube results/cube_starcover "cube boundary, optimal product cover (8 rectangles, rho = 8)" 2>&1 | grep -v Warning

echo "=== 3. human-readable certificates (all domains + planner maps) ==="
for c in circle s2 s3 s4 s5; do "$PY" make_certificate_md.py $c results/${c}_starcover results/${c}_starcover/certificate.md "Explicit product construction (optimized_algorithms.md §16, §19): D(v) = star(v) x star(v), v = 0, 1, 2, chains pi1 ~ c_v ~ pi2; built by star_cover.py, verified (Definition 2.3) and audited by independent_check.py."; done
  "$PY" make_certificate_md.py torus results/torus_starcover_opt results/torus_starcover_opt/certificate.md "Optimal product cover (11 rectangles, integer program); see T2_certificate.md for the full account." 2>&1 | grep -v Warning

if [ "${1:-}" = "gpu" ]; then
  nvidia-smi -L || { echo "no GPU"; exit 1; }
  echo "=== 4. GPU: chain-first annealer with the corrected energy (beta 5, gamma 0.5, track moves) ==="
  "$PY" -u gpu_sc.py --complex s2 --k 3 --cmax 20 --S 4096 --iters 20000 --epochs 400 --beta 5.0 --gamma 0.5 --ptrack 0.3 \
      --seed 51 --time-limit 120 --outdir results/s2_k3_v2 2>&1 | grep --line-buffered -v Warning
  audit results/s2_k3_v2
  "$PY" -u gpu_sc.py --complex s3 --k 3 --cmax 40 --S 4096 --iters 20000 --epochs 400 --beta 5.0 --gamma 0.5 --ptrack 0.3 \
      --seed 52 --time-limit 300 --outdir results/s3_k3_v2 2>&1 | grep --line-buffered -v Warning
  audit results/s3_k3_v2
  "$PY" -u gpu_sc.py --complex s4 --k 3 --cmax 40 --S 4096 --iters 20000 --epochs 400 --beta 5.0 --gamma 0.5 --ptrack 0.3 \
      --seed 61 --time-limit 900 --outdir results/s4_k3_v2 2>&1 | grep --line-buffered -v Warning     # ≈ 13 min (one epoch)
  audit results/s4_k3_v2
  "$PY" make_certificate_md.py s2 results/s2_k3_v2 results/s2_k3_v2/certificate.md "Chain-first GPU annealer (gpu_sc.py) with the corrected energy, seed 51; see optimized_algorithms.md §17 and S2_certificate.md." 2>&1 | grep -v Warning
  echo "=== 5. GPU: the wedge of two circles (paper §5.2): sharp SC = 2 in 41 s ==="
  "$PY" -u gpu_sc.py --complex wedge --k 3 --cmax 16 --S 4096 --iters 5000 --epochs 60 --beta 5.0 --gamma 0.5 --ptrack 0.3 \
      --seed 71 --time-limit 180 --outdir results/wedge_k3_v2 2>&1 | grep --line-buffered -v Warning
  audit results/wedge_k3_v2
  "$PY" make_certificate_md.py wedge results/wedge_k3_v2 results/wedge_k3_v2/certificate.md "Chain-first GPU annealer (gpu_sc.py, beta 5 / gamma 0.5 / track moves); see optimized_algorithms.md §23." 2>&1 | grep -v Warning
  "$PY" make_definition_md.py wedge results/wedge_k3_v2 "wedge of two circles, chain-first annealer" 2>&1 | grep -v Warning
  echo "=== 6. the paper's own §5.2 certificate, re-verified (CPU, ~1 s) ==="
  "$PY" wedge_paper_check.py results/wedge_k3_v2 2>&1 | grep -v Warning
  echo "=== 7. GPU: the barycenter triangle — the sharp SC = 0 from a single domain (3 s + 11 s) ==="
  "$PY" -u gpu_sc.py --complex bary --k 1 --cmax 2 --S 4096 --iters 5000 --epochs 60 --beta 5.0 --gamma 0.5 --ptrack 0.3 \
      --seed 71 --time-limit 120 --outdir results/bary_k1_c3 2>&1 | grep --line-buffered -v Warning
  audit results/bary_k1_c3
  "$PY" make_certificate_md.py bary results/bary_k1_c3 results/bary_k1_c3/certificate.md "Chain-first GPU annealer (gpu_sc.py, beta 5 / gamma 0.5 / track moves, cmax 2); see optimized_algorithms.md §24." 2>&1 | grep -v Warning
  "$PY" make_definition_md.py bary results/bary_k1_c3 "barycenter triangle, chain-first annealer" 2>&1 | grep -v Warning
  "$PY" -u gpu_sc.py --complex bary --k 1 --cmax 16 --S 4096 --iters 5000 --epochs 60 --beta 5.0 --gamma 0.5 --ptrack 0.3 \
      --seed 71 --time-limit 120 --outdir results/bary_k1 2>&1 | grep --line-buffered -v Warning
  audit results/bary_k1
  "$PY" make_certificate_md.py bary results/bary_k1 results/bary_k1/certificate.md "Chain-first GPU annealer (gpu_sc.py, beta 5 / gamma 0.5 / track moves, cmax 16); see optimized_algorithms.md §24." 2>&1 | grep -v Warning
  "$PY" make_definition_md.py bary results/bary_k1 "barycenter triangle, chain-first annealer" 2>&1 | grep -v Warning
  echo "=== 8. GPU: the cube boundary — the search reaches the sphere's sharp 2 where products stall at 7 (304 s) ==="
  # The guided k=3 run, warm-started from the shipped best state of the k=3
  # continuation (results/cube_k3_v2/summary.json: 820/864).  The full chain
  # that produced the sharp result is  ./run_cube.sh k3  (300 s, 662/864)  ->
  # INIT=results/cube_k3/summary.json TL=900 ./run_cube.sh k3w  (814 s, 820/864)  ->
  # ./run_cube.sh k3g  (304 s, 864/864);  replaying the last step alone
  # certifies SC_strict(cube) <= 2 = TC(S²), sharp.  The k=2 control
  # (./run_cube.sh k2) must never cover: TC = 2 forbids it.
  "$PY" -u gpu_sc.py --complex cube --k 3 --cmax 16 --S 4096 --iters 5000 --epochs 60 --beta 5.0 --gamma 0.5 --ptrack 0.3 \
      --T0 0.6 --T1 0.03 --guide 1.0 --seed 84 --time-limit 900 --init results/cube_k3_v2/summary.json \
      --outdir results/cube_k3_guided 2>&1 | grep --line-buffered -v Warning
  audit results/cube_k3_guided
  "$PY" make_certificate_md.py cube results/cube_k3_guided results/cube_k3_guided/certificate.md "Chain-first GPU annealer (gpu_sc.py, beta 5 / gamma 0.5 / track moves, --guide 1.0 continuation); sharp SC_strict = 2 at subdivision level 0. See optimized_algorithms.md §26." 2>&1 | grep -v Warning
  "$PY" make_definition_md.py cube results/cube_k3_guided "cube boundary, chain-first annealer (sharp SC = 2)" 2>&1 | grep -v Warning
  echo "=== 9. GPU: the k=2 control on the cube — the theory forbids a 2-domain cover (must NOT cover) ==="
  "$PY" -u gpu_sc.py --complex cube --k 2 --cmax 16 --S 4096 --iters 5000 --epochs 60 --beta 5.0 --gamma 0.5 --ptrack 0.3 \
      --T0 0.6 --T1 0.03 --seed 81 --time-limit 300 --outdir results/cube_k2 2>&1 | grep --line-buffered -v Warning
  audit results/cube_k2 || echo "k=2: no full cover, as TC(S²) = 2 requires"
fi
echo "ALL DONE"
