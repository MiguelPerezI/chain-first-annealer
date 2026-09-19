#!/bin/bash
# run_starcovers.sh — explicit product (star) covers of ∂Δ^{n+1} for n = 5..9
# (S^5 .. S^9): build with star_cover.py, verify with the plain Definition 2.3
# check, audit with the code-independent independent_check.py, write
# definition.md and certificate.md.
#
# Above S^8 the full facet-by-facet certificate is not representable here:
# cover.txt is ≈0.8 GB at S^9 (5,883,020 facets) and ≈4.2 / 21 / 106 GB at
# S^10 / S^11 / S^12, ngrid = (n+2)^2 > 127 overflows the int8 grid ids from
# n = 10 on, and independent_check.py holds ≈2.8 KB per facet in RAM (≈16.5 GB
# at S^9).  Those are handled by stream_starcover.py instead — written and
# cross-checked, but NOT run: the set stops at S^8 (optimized_algorithms.md §21).
#
#   ./run_starcovers.sh 6 7 8 9      # (default: 6 7 8 9)
set -u
cd "$(dirname "$0")"
PY=${PY:-$HOME/Documents/Abstract_Motion_GPU/venv/bin/python}
AUDIT="$(pwd)/independent_check.py"
NS=${*:-"6 7 8 9"}

for N in $NS; do
  OUT=results/s${N}_starcover
  echo "=== $(date -u +%FT%TZ)  S^$N = ∂Δ$((N+1))  -> $OUT ==="
  /usr/bin/time -v "$PY" star_cover.py s$N "$OUT" 0,0 1,1 2,2 2>&1 \
      | grep -v Warning | grep -E "SC_strict|Maximum resident|Elapsed \(wall|^\s*∂Δ" || true
  echo "--- audit ---"
  /usr/bin/time -v python3 "$AUDIT" "$OUT" 2>&1 | grep -E "VERIFIED|\[2\]|Maximum resident|Elapsed \(wall" || true
  "$PY" make_definition_md.py s$N "$OUT" "Explicit product certificate (star_cover.py)" 2>&1 | grep -v Warning
  "$PY" make_certificate_md.py s$N "$OUT" "$OUT/certificate.md" \
      "Explicit product construction (Corollary 3): D(v) = star(v) x star(v), v = 0, 1, 2, chains pi1 ~ c_v ~ pi2; built by star_cover.py, verified (Definition 2.3) and audited by independent_check.py." 2>&1 | grep -v Warning
  ls -la "$OUT" | awk '{print "   ", $5, $9}'
done
echo "=== RUN_STARCOVERS DONE $(date -u +%FT%TZ) ==="
