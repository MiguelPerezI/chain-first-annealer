#!/bin/bash
# Sequential S^3 two-chain experiments (5 min each) — logs in logs/, results in results/
cd /home/ubuntu/Documents/opti
PY=/home/ubuntu/Documents/Abstract_Motion_GPU/venv/bin/python
run() { name=$1; shift; echo "### $name: $*" ; $PY -u gpu_sc.py --complex s3 --k 2 --S 4096 --iters 20000 --epochs 200 --time-limit 300 --outdir results/$name "$@" 2>&1 | grep -v -i warning > logs/$name.log; grep -E "best coverage|VERIF|NOT" logs/$name.log; }
run s3_k2_A --cmax 40 --ptrack 0.3 --guide 0   --seed 21
run s3_k2_B --cmax 40 --ptrack 0.3 --guide 1.0 --seed 22
run s3_k2_C --cmax 80 --ptrack 0.3 --guide 1.0 --T1 0.1 --seed 23
run s3_k2_D --cmax 40 --ptrack 0.3 --guide 1.0 --seed 24 --init results/s3_k3_run1/summary.json
echo "### all done"
