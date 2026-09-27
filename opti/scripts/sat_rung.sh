#!/usr/bin/env bash
# One CP-SAT rung: solve, measure, convert, audit, log.
# Uso: sat_rung.sh COMPLEX K L TIMEOUT WORKERS OUTDIR LOG_MD [WARM FREE]
set -euo pipefail
[ $# -ge 7 ] || { echo "usage: $0 COMPLEX K L TIMEOUT WORKERS OUTDIR LOG_MD [WARM FREE]" >&2; exit 64; }
cx=$1 k=$2 L=$3 to=$4 wk=$5 out=$6 log=$7 warm=${8:-} free=${9:-0}
if [ -n "$warm" ] && [ "$free" -lt 1 ]; then
  echo "WARM needs FREE >= 1 (scs_sat runs a cold solve with --free 0)" >&2; exit 64
fi
OPTI=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$OPTI/.venv-sat/bin/python}
mkdir -p "$out"
args=(--complex "$cx" --k "$k" --L "$L" --timeout "$to" --workers "$wk" --outdir "$out" --witness)
mode=cold; freecol=-
if [ -n "$warm" ]; then args+=(--warm "$warm" --free "$free"); mode=hybrid; freecol=$free; fi

/usr/bin/time -v -o "$out/time.txt" "$PY" -u "$OPTI/paper-overleaf/paper/scs_sat.py" "${args[@]}" \
    > "$out/solver.log" 2>&1 || true

line=$(grep -oE '(OPTIMAL|FEASIBLE|INFEASIBLE|UNKNOWN|MODEL_INVALID) in [0-9.]+s \(vars [0-9]+, cons [0-9]+\)' "$out/solver.log" | tail -1 || true)
status=$(echo "$line" | awk '{print $1}'); status=${status:-CRASH}
wall=$(echo "$line" | sed -nE 's/.* in ([0-9.]+)s.*/\1/p')
nvars=$(echo "$line" | sed -nE 's/.*vars ([0-9]+).*/\1/p')
ncons=$(echo "$line" | sed -nE 's/.*cons ([0-9]+).*/\1/p')
rss_kb=$(sed -nE 's/.*Maximum resident set size \(kbytes\): ([0-9]+).*/\1/p' "$out/time.txt")
rss_gb=$(LC_ALL=C awk -v kb="${rss_kb:-0}" 'BEGIN{printf "%.2f", kb/1048576}')

cert=-
if [ "$status" = OPTIMAL ] || [ "$status" = FEASIBLE ]; then
  conv=("$OPTI/scripts/sat_witness_to_certificate.py" "$out/scs_witness_k${k}_L${L}.json" "$out/certificate"
        --complex "$cx" --k "$k" --L "$L" --timeout "$to" --status "$status" --wall-s "${wall:-0}")
  [ -n "$warm" ] && conv+=(--warm "$warm" --free "$free")
  set +e; "$PY" "${conv[@]}" > "$out/convert.log" 2>&1; rc=$?; set -e
  case $rc in 0) cert=audited ;; 2) cert=not-a-cover ;; *) cert=audit-failed ;; esac
fi

if [ ! -f "$log" ]; then
  printf '| date | complex | k | L | mode | free | status | wall_s | vars | cons | rss_gb | certificate | outdir |\n|---|---|---:|---:|---|---:|---|---:|---:|---:|---:|---|---|\n' > "$log"
fi
printf '| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n' \
  "$(date -u +%Y-%m-%dT%H:%MZ)" "$cx" "$k" "$L" "$mode" "$freecol" "$status" \
  "${wall:--}" "${nvars:--}" "${ncons:--}" "$rss_gb" "$cert" "$out" >> "$log"
echo "$cx k=$k L=$L $mode: $status, certificate=$cert"
