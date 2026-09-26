#!/usr/bin/env bash
# serve.sh -- start the K x K webapp on this machine.
#
#   bash webapp/serve.sh                 # http://127.0.0.1:8770
#   bash webapp/serve.sh --port 9000
#
# Runs the server out of the venv that has numba + matplotlib (the system
# python3 has neither, and kxk_state imports gpu_sc which imports numba).
set -u
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-/home/mike666/whisper_env/bin/python}
[ -x "$PY" ] || { echo "no python with numba/matplotlib at $PY (set PY=...)"; exit 1; }
exec "$PY" -u webapp/server.py "$@"
