#!/usr/bin/env bash
# KFP container-component entrypoint for one CP-SAT rung.
# Args: COMPLEX K L TIMEOUT WORKERS OUT_ROOT STATUS_OUT CERT_OUT ARTIFACT_OUT
#   STATUS_OUT / CERT_OUT: KFP output-parameter files (status, certificate label)
#   ARTIFACT_OUT: KFP output-artifact path (tar.gz of the rung directory)
# Env: PY (default $HOME/venv-sat/bin/python)
set -euo pipefail
[ $# -eq 9 ] || { echo "usage: $0 COMPLEX K L TIMEOUT WORKERS OUT_ROOT STATUS_OUT CERT_OUT ARTIFACT_OUT" >&2; exit 64; }
cx=$1 k=$2 L=$3 to=$4 wk=$5 root=$6 status_out=$7 cert_out=$8 art_out=$9
OPTI=$(cd "$(dirname "$0")/../.." && pwd)
export PY=${PY:-$HOME/venv-sat/bin/python}

out="$root/${cx}_k${k}_L${L}"
mkdir -p "$out" "$(dirname "$status_out")" "$(dirname "$cert_out")" "$(dirname "$art_out")"
git -C "$OPTI/.." rev-parse HEAD > "$out/revision.txt" 2> /dev/null || echo unknown > "$out/revision.txt"
nproc > "$out/nproc.txt"

bash "$OPTI/scripts/sat_rung.sh" "$cx" "$k" "$L" "$to" "$wk" "$out/run" "$out/row.md"

# row: | date | complex | k | L | mode | free | status | wall_s | vars | cons | rss_gb | certificate | outdir |
row=$(tail -1 "$out/row.md")
echo "$row"
echo "$row" | awk -F'|' '{gsub(/ /, "", $8); printf "%s", $8}' > "$status_out"
echo "$row" | awk -F'|' '{gsub(/ /, "", $13); printf "%s", $13}' > "$cert_out"
tar -C "$out" -czf "$art_out" .
