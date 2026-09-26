#!/usr/bin/env bash
# setup_remote.sh -- provision the GPU server for the annealer webapp
# Automates: pip install, numpy-2 compat patch, CUDA verification, smoke test.
# Usage:  bash webapp/setup_remote.sh [user@host]  (default: mike666@136.108.172.74)
set -euo pipefail

HOST="${1:-mike666@136.108.172.74}"
PIP="~/miniforge3/bin/pip"
PY="~/miniforge3/bin/python3"

echo "=== Step 1: Install numba + dependencies ==="
ssh "$HOST" "$PIP install numpy 'numba>=0.67' scipy 'numba-cuda[cu12]>=0.30.4'"

echo ""
echo "=== Step 2: Patch numba-cuda for numpy 2.x (np.row_stack removed) ==="
ssh "$HOST" "$PY -c \"
import importlib, pathlib, re
mod = importlib.import_module('numba_cuda.numba.cuda.np.arrayobj')
p = pathlib.Path(mod.__file__)
src = p.read_text()
needle = 'overload(np.row_stack)(impl_np_vstack)'
if needle in src and 'hasattr(np' not in src:
    patched = src.replace(
        needle,
        '# patched: np.row_stack removed in numpy 2.0\\n'
        '    if hasattr(np, \\\"row_stack\\\"):\\n'
        '        overload(np.row_stack)(impl_np_vstack)')
    p.write_text(patched)
    print('  patched', p)
else:
    print('  already patched or not needed')
\""

echo ""
echo "=== Step 3: Verify numba + CUDA ==="
ssh "$HOST" "$PY -c \"import numba; print('numba', numba.__version__); from numba import cuda; print('GPU:', cuda.gpus[0].name)\""

echo ""
echo "=== Step 4: Smoke test (2-epoch circle anneal) ==="
ssh "$HOST" "cd ~/chain-first-annealer/opti && rm -rf /tmp/smoke_test && mkdir -p /tmp/smoke_test && $PY -u src/annealer_stretch.py --complex circle --k 3 --S 256 --iters 500 --epochs 2 --cmax 8 --seed 42 --outdir /tmp/smoke_test --jsonl /tmp/smoke_test/epochs.jsonl --checkpoint-every 1 --graceful-stop --time-limit 60"

echo ""
echo "=== Step 5: Verify output files ==="
ssh "$HOST" "ls /tmp/smoke_test/ && echo '---' && cat /tmp/smoke_test/epochs.jsonl && echo '---' && head -c 200 /tmp/smoke_test/summary.json"

echo ""
echo "=== GPU server ready ==="
