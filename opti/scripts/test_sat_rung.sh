#!/usr/bin/env bash
# Smoke test for sat_rung.sh on the Moebius band (fast).
set -euo pipefail
OPTI=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d)
bash "$OPTI/scripts/sat_rung.sh" mobius 5 4 120 4 "$tmp/hyb" "$tmp/log.md" \
     "$OPTI/results/mobius_k5/summary.json" 1
bash "$OPTI/scripts/sat_rung.sh" mobius 1 3 120 4 "$tmp/infeas" "$tmp/log.md"
cat "$tmp/log.md"
grep -qE '\| mobius \| 5 \| 4 \| hybrid \| 1 \| (OPTIMAL|FEASIBLE) \|.*\| audited \|' "$tmp/log.md"
grep -qE '\| mobius \| 1 \| 3 \| cold \| - \| INFEASIBLE \|.*\| - \|' "$tmp/log.md"
echo "SAT_RUNG OK"
