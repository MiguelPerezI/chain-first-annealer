#!/usr/bin/env python3
"""Turn a scs_sat.py witness into an audited certificate directory.

Hybrid witnesses carry frozen chains cut to L + 1 maps; the full chains are
restored from the warm summary, in the order scs_sat.load_warm froze them.
Exit codes: 0 audited, 2 not a certificate, 3 independent audit failed.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

OPTI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OPTI / "src"))
sys.path.insert(0, str(OPTI / "paper-overleaf" / "paper"))
import gpu_sc  # noqa: E402
import scs_sat  # noqa: E402


def full_chains(witness, warm, k, n_free, dims):
    chains = [list(c) for c in witness["chains"]]
    if warm:
        raw_chains, raw_domains, _, order = scs_sat.load_warm(warm, *dims)
        n_frozen = min(k - n_free, len(raw_domains))
        chains[:n_frozen] = [raw_chains[i] for i in order[:n_frozen]]
    return [np.array(c, np.int8) for c in chains]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("witness")
    ap.add_argument("outdir")
    ap.add_argument("--complex", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--warm")
    ap.add_argument("--free", type=int, default=0)
    ap.add_argument("--timeout", type=int)
    ap.add_argument("--status")
    ap.add_argument("--wall-s", type=float)
    a = ap.parse_args(argv)

    NV, NG, NF, FAC, TRIS, _, _, P = scs_sat.build_complex(a.complex)
    witness = json.loads(Path(a.witness).read_text())
    chains = full_chains(witness, a.warm, a.k, a.free, (NV, NG, NF, FAC, TRIS))
    parts = gpu_sc.partition_from_chains(P, chains)
    if parts is None:
        print("witness chains do not cover F_L", file=sys.stderr)
        return 2
    import ortools
    meta = {"params": {
        "complex": a.complex, "method": "CP-SAT witness (scs_sat.py)",
        "ortools": ortools.__version__, "k": a.k, "L": a.L,
        "mode": "hybrid" if a.warm else "cold", "init": a.warm,
        "free": a.free, "timeout": a.timeout, "status": a.status,
        "wall_s": a.wall_s, "witness": str(a.witness)}}
    try:
        gpu_sc.write_certificate(P, parts, a.outdir, meta)
    except ValueError as e:
        print(f"certificate rejected: {e}", file=sys.stderr)
        return 2
    audit = subprocess.run([sys.executable, str(OPTI / "src" / "independent_check.py"),
                            a.outdir])
    return 0 if audit.returncode == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
