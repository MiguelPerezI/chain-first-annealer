#!/bin/bash
# run_starcovers_big.sh — the far end of the star-cover table.  NOT RUN: the
# set was stopped at S^8 (the largest full certificate the shipped auditor can
# check on this machine); started, S^9 would build a 0.8 GB cover.txt that the
# auditor cannot load here (≈16.5 GB needed, 15 GB available).  Kept for a
# machine with more RAM — every step below is executable as-is.
#
#   S^9  = ∂Δ^10 : full facet-by-facet certificate (5,883,020 facets, ≈0.8 GB
#                  cover.txt — the last n that fits: ngrid = 121 <= 127).
#   S^10, S^11, S^12 = ∂Δ^11..13 : streamed certificate.  ngrid = 144/169/196
#                  overflows the int8 grid ids and cover.txt would be ≈4.2/21/106
#                  GB, so stream_starcover.py verifies the construction facet by
#                  facet without materializing it and writes the compact
#                  certificate (planners.txt, definition.md, summary.json,
#                  certificate.md) instead of a cover.txt.
#
#   ./run_starcovers_big.sh 9 10 11 12
set -u
cd "$(dirname "$0")"
PY=${PY:-$HOME/Documents/Abstract_Motion_GPU/venv/bin/python}
AUDIT="$(pwd)/independent_check.py"
NS=${*:-"9 10 11 12"}

for N in $NS; do
  OUT=results/s${N}_starcover
  if [ "$N" -le 9 ]; then
    echo "=== $(date -u +%FT%TZ)  S^$N (full certificate)  -> $OUT ==="
    # --no-reduce: on the sphere the constructed chain pi1 ~ c_v ~ pi2 is
    # already minimal (reduce_chain removes nothing, verified byte-identical
    # for n = 5), and reduction materializes per-vertex star lists that cost
    # ~10 GB of RAM at n = 9 for no change in the result.
    /usr/bin/time -v "$PY" star_cover.py s$N "$OUT" 0,0 1,1 2,2 --no-reduce 2>&1 \
        | grep -v Warning | grep -E "SC_strict|Maximum resident|Elapsed \(wall" || true
    if [ -f "$OUT/cover.txt" ]; then
      echo "--- audit (code-independent) ---"
      # ulimit -v: the auditor holds the whole facet table as Python objects
      # (~2.8 KB/facet); at n = 9 that is past the machine's RAM, so cap it
      # and let it raise MemoryError cleanly instead of OOM-killing anything.
      /usr/bin/time -v bash -c "ulimit -v ${AUDIT_ULIMIT_KB:-12500000}; exec python3 '$AUDIT' '$OUT'" 2>&1 \
          | grep -E "VERIFIED|\[2\]|MemoryError|Maximum resident|Elapsed \(wall" || true
    fi
    "$PY" make_definition_md.py s$N "$OUT" "Explicit product certificate (star_cover.py)" 2>&1 | grep -v Warning
  else
    echo "=== $(date -u +%FT%TZ)  S^$N (streamed certificate)  -> $OUT ==="
    "$PY" -u stream_starcover.py $N "$OUT" 2>&1 | grep -v Warning | tee logs/s${N}_stream.log
    "$PY" make_definition_md.py s$N "$OUT" "Explicit product certificate, verified by streaming brute force (stream_starcover.py)" 2>&1 | grep -v Warning
  fi
  ls -la "$OUT" | awk 'NR>3 {printf "    %10d  %s\n", $5, $9}'
done
echo "=== RUN_STARCOVERS_BIG DONE $(date -u +%FT%TZ) ==="
