#!/usr/bin/env bash
# Pod entrypoint: one CP-SAT rung per Indexed-Job index.
# Env:
#   COMPLEX, K, L_LIST ("4 5 6 8"), TIMEOUT, WORKERS   rung parameters
#   JOB_COMPLETION_INDEX                                set by Kubernetes (Indexed Job)
#   OUT_ROOT                                            results directory on the PVC
#   REPO_REF, GIT_TOKEN                                 commit to run and read-only token
#   SRC_DIR, PY (optional)                              use a local checkout / python (local test)
set -euo pipefail

for tool in git /usr/bin/time; do
  if ! command -v "$tool" > /dev/null; then
    apt-get update -qq && apt-get install -y -qq git time > /dev/null
    break
  fi
done

read -r -a Ls <<< "$L_LIST"
L=${Ls[$JOB_COMPLETION_INDEX]}

if [ -z "${SRC_DIR:-}" ]; then
  SRC_DIR=/tmp/src
  git clone --quiet "https://x-access-token:${GIT_TOKEN}@github.com/MiguelPerezI/chain-first-annealer.git" "$SRC_DIR"
  git -C "$SRC_DIR" checkout --quiet "$REPO_REF"
fi
if [ -z "${PY:-}" ]; then
  python3 -m venv /tmp/venv
  /tmp/venv/bin/pip install --quiet -r "$SRC_DIR/opti/requirements-sat.txt"
  PY=/tmp/venv/bin/python
fi
export PY

out="$OUT_ROOT/${COMPLEX}_k${K}_L${L}"
mkdir -p "$out"
git -C "$SRC_DIR" rev-parse HEAD > "$out/revision.txt"
nproc > "$out/nproc.txt"
cd "$SRC_DIR/opti"
bash scripts/sat_rung.sh "$COMPLEX" "$K" "$L" "$TIMEOUT" "$WORKERS" "$out/run" "$out/row.md"
cat "$out/row.md"
