#!/bin/bash
# run_s3.sh — re-execute the S^3 = ∂Δ⁴ experiment end to end (GPU required).
#
#   ./run_s3.sh            # 3-chain certificate (SC_strict ≤ 2), ~2-5 min, then independent audit
#   ./run_s3.sh push       # + the two-chain push (5 min per configuration, seeded from the certificate)
#   SEED=7 ./run_s3.sh     # different random seed (the certificate found may differ; the bound is the same)
#
# Outputs: results/s3_k3_<timestamp>/{cover.txt,planners.txt,summary.json}, logs/ (verbose epochs).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/Documents/Abstract_Motion_GPU/venv/bin/python}
AUDIT=${AUDIT:-$HOME/Documents/Abstract_Motion_GPU/independent_check.py}
SEED=${SEED:-3}
TS=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p results logs

[ -x "$PY" ] || { echo "python with numba not found at $PY (set PY=...)"; exit 1; }
nvidia-smi -L >/dev/null 2>&1 || { echo "no NVIDIA GPU visible"; exit 1; }
$PY -c "from numba import cuda; assert cuda.is_available()" || { echo "numba cannot see CUDA"; exit 1; }

OUT=results/s3_k3_$TS
echo "[1/3] three-chain search on ∂Δ⁴×∂Δ⁴ (500 facets) -> $OUT"
$PY -u gpu_sc.py --complex s3 --k 3 --cmax 40 --S 4096 --iters 20000 --epochs 60 --seed "$SEED" \
    --time-limit 1800 --outdir "$OUT" 2>&1 | grep -v -i warning | tee "logs/s3_k3_$TS.log"

if [ -f "$OUT/cover.txt" ]; then
  echo "[2/3] code-independent audit"
  python3 "$AUDIT" "$OUT" | tail -5
  echo "certificate: $OUT  (see S3_certificate.md for the documented instance)"
else
  echo "[2/3] no full covering within the budget — rerun with another SEED or a larger --time-limit"; exit 2
fi

if [ "${1:-}" = "push" ]; then
  echo "[3/3] two-chain push (open question: can 2 domains cover at level 0?)"
  for cfg in "--cmax 40 --ptrack 0.05 --guide 1.0" "--cmax 80 --ptrack 0.05 --guide 1.0 --T1 0.1"; do
    O=results/s3_k2_$TS_$RANDOM
    $PY -u gpu_sc.py --complex s3 --k 2 --S 4096 --iters 20000 --epochs 200 --time-limit 300 --seed "$SEED" \
        --init "$OUT/summary.json" --outdir "$O" $cfg 2>&1 | grep -v -i warning | grep -E "epoch|best coverage|VERIF|NOT"
    [ -f "$O/cover.txt" ] && { echo "TWO-DOMAIN COVER FOUND: $O"; python3 "$AUDIT" "$O" | tail -3; }
  done
fi
