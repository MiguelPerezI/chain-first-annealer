#!/usr/bin/env bash
# One-time setup in the lab notebook terminal: repo and CP-SAT venv on the
# workspace PVC (mounted at /home/jovyan in every pipeline pod).
# Uso: bash setup_workspace.sh   (run from anywhere inside the notebook)
set -euo pipefail
REPO=$HOME/chain-first-annealer
VENV=$HOME/venv-sat
if [ ! -d "$REPO/.git" ]; then
  git clone https://github.com/MiguelPerezI/chain-first-annealer.git "$REPO"   # private: uses your git credentials
else
  git -C "$REPO" pull --ff-only
fi
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet -r "$REPO/opti/requirements-sat.txt"
"$VENV/bin/python" -c "import ortools; print('ortools', ortools.__version__)"
PY=$VENV/bin/python bash "$REPO/opti/scripts/test_sat_rung.sh" | tail -1
pip install --quiet --user "kfp==2.17.0" kfp-kubernetes 2> /dev/null || true
echo "workspace ready: repo $REPO at $(git -C "$REPO" rev-parse --short HEAD), venv $VENV"
