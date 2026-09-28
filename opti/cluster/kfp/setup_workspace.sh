#!/usr/bin/env bash
# One-time setup in the lab notebook terminal: repo and CP-SAT venv on the
# workspace PVC (mounted at /home/jovyan in every pipeline pod).
# Uso: bash setup_workspace.sh   (run from anywhere inside the notebook)
set -euo pipefail
REPO=$HOME/chain-first-annealer
VENV=$HOME/venv-sat
t0=$SECONDS
step() { echo; echo "==> [$((SECONDS - t0))s] $*"; }

step "1/5 repo en $REPO"
if [ ! -d "$REPO/.git" ]; then
  git clone https://github.com/MiguelPerezI/chain-first-annealer.git "$REPO"   # private: uses your git credentials
else
  git -C "$REPO" pull --ff-only
fi
echo "    commit $(git -C "$REPO" rev-parse --short HEAD)"

step "2/5 venv en $VENV con $(python3 --version) ($(command -v python3))"
if [ -x "$VENV/bin/python" ]; then
  echo "    ya existe, se reutiliza"
else
  python3 -m venv "$VENV"
  echo "    creado"
fi

step "3/5 instalando opti/requirements-sat.txt (ortools, numpy, numba, scipy; puede tardar unos minutos)"
"$VENV/bin/pip" install --disable-pip-version-check --progress-bar on \
  -r "$REPO/opti/requirements-sat.txt"
"$VENV/bin/python" -c "import ortools, numba, numpy; print('    ortools', ortools.__version__, '| numba', numba.__version__, '| numpy', numpy.__version__)"

step "4/5 prueba corta (Möbius, ~1-2 min): debe terminar en SAT_RUNG OK"
PY=$VENV/bin/python bash "$REPO/opti/scripts/test_sat_rung.sh" | tail -1

step "5/5 SDK de KFP para compilar el pipeline"
pip install --quiet --disable-pip-version-check --user "kfp==2.17.0" kfp-kubernetes 2> /dev/null \
  && python3 -c "import kfp, kfp.kubernetes; print('    kfp', kfp.__version__)" \
  || echo "    (no se pudo instalar kfp-kubernetes; el notebook ya trae kfp)"

step "listo: repo $REPO ($(git -C "$REPO" rev-parse --short HEAD)), venv $VENV"
