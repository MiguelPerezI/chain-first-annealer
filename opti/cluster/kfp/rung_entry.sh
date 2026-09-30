#!/usr/bin/env bash
# KFP container-component entrypoint for one CP-SAT rung.
# Args: COMPLEX K L TIMEOUT WORKERS OUT_ROOT STATUS_OUT CERT_OUT ARTIFACT_OUT [WARM FREE]
#   STATUS_OUT / CERT_OUT: KFP output-parameter files (status, certificate label)
#   ARTIFACT_OUT: KFP output-artifact path (tar.gz of the rung directory)
#   WARM FREE: optional hybrid case (e.g. orbit case A1); WARM relative to opti/
#              or absolute. An empty WARM means a cold rung; WARM needs FREE >= 1.
# Env: PY (default $HOME/venv-sat/bin/python)
set -euo pipefail
[ $# -eq 9 ] || [ $# -eq 11 ] || { echo "usage: $0 COMPLEX K L TIMEOUT WORKERS OUT_ROOT STATUS_OUT CERT_OUT ARTIFACT_OUT [WARM FREE]" >&2; exit 64; }
cx=$1 k=$2 L=$3 to=$4 wk=$5 root=$6 status_out=$7 cert_out=$8 art_out=$9
warm=${10:-} free=${11:-0}
if [ -n "$warm" ] && [ "$free" -lt 1 ]; then
  echo "WARM needs FREE >= 1 (a warm file with FREE=0 would run a cold rung)" >&2; exit 64
fi
OPTI=$(cd "$(dirname "$0")/../.." && pwd)
export PY=${PY:-$HOME/venv-sat/bin/python}
hyb=()
if [ -n "$warm" ]; then
  case $warm in /*) ;; *) warm="$OPTI/$warm" ;; esac
  hyb=("$warm" "$free")
fi

tag=""; [ ${#hyb[@]} -gt 0 ] && tag="_$(basename "${hyb[0]}" .json)_free${free}"
out="$root/${cx}_k${k}_L${L}${tag}"
mkdir -p "$out" "$(dirname "$status_out")" "$(dirname "$cert_out")" "$(dirname "$art_out")"
git -C "$OPTI/.." rev-parse HEAD > "$out/revision.txt" 2> /dev/null || echo unknown > "$out/revision.txt"
nproc > "$out/nproc.txt"

bash "$OPTI/scripts/sat_rung.sh" "$cx" "$k" "$L" "$to" "$wk" "$out/run" "$out/row.md" "${hyb[@]}"

# row: | date | complex | k | L | mode | free | status | wall_s | vars | cons | rss_gb | certificate | outdir |
row=$(tail -1 "$out/row.md")
echo "$row"
echo "$row" | awk -F'|' '{gsub(/ /, "", $8); printf "%s", $8}' > "$status_out"
echo "$row" | awk -F'|' '{gsub(/ /, "", $13); printf "%s", $13}' > "$cert_out"
tar -C "$out" -czf "$art_out" .
