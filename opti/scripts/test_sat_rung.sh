#!/usr/bin/env bash
# Smoke test for sat_rung.sh on the Moebius band (fast).
set -euo pipefail
OPTI=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$OPTI/.venv-sat/bin/python}
tmp=$(mktemp -d)
# hybrid needs warm chains collapsed to their active links (scs_sat rejects
# frozen chains longer than --L)
"$PY" "$OPTI/scripts/sat_validation/reduce_warm.py" mobius \
      "$OPTI/results/mobius_k5/summary.json" "$tmp/warm_red.json" > "$tmp/reduce.log"
L=$(sed -nE 's/.*--L ([0-9]+).*/\1/p' "$tmp/reduce.log" | tail -1)
[ "$L" -ge 4 ] || L=4
bash "$OPTI/scripts/sat_rung.sh" mobius 5 "$L" 120 4 "$tmp/hyb" "$tmp/log.md" \
     "$tmp/warm_red.json" 1
bash "$OPTI/scripts/sat_rung.sh" mobius 1 3 120 4 "$tmp/infeas" "$tmp/log.md"
if bash "$OPTI/scripts/sat_rung.sh" mobius 5 4 120 4 "$tmp/nofree" "$tmp/log.md" \
     "$OPTI/results/mobius_k5/summary.json" 0 2>/dev/null; then
  echo "FAIL: WARM with FREE=0 was accepted"; exit 1
fi
# images without GNU time (e.g. the Kubeflow notebook image) fall back to rss_run.py
TIME_BIN=/nonexistent bash "$OPTI/scripts/sat_rung.sh" mobius 1 3 120 4 "$tmp/notime" "$tmp/notime.md"
grep -qE '\| mobius \| 1 \| 3 \| cold \| - \| INFEASIBLE \|' "$tmp/notime.md"
grep -q "measured by rss_run.py" "$tmp/notime/time.txt" || { echo "FAIL: TIME_BIN ignored"; exit 1; }
grep -qE '\| 0\.[0-9][1-9] \||\| [1-9][0-9]*\.[0-9]+ \|' "$tmp/notime.md" || { echo "FAIL: no RSS without GNU time"; cat "$tmp/notime.md"; exit 1; }
cat "$tmp/log.md"
grep -qE "\| mobius \| 5 \| $L \| hybrid \| 1 \| (OPTIMAL|FEASIBLE) \|.*\| audited \|" "$tmp/log.md"
grep -qE '\| mobius \| 1 \| 3 \| cold \| - \| INFEASIBLE \|.*\| - \|' "$tmp/log.md"
echo "SAT_RUNG OK"
