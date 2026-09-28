#!/usr/bin/env python3
"""Family A cases for RP2 k=5.

A1: domain 0 = star(0) x star(0), given as the frozen chain pi1 ~ c_0 ~ pi2
    (a verified planner, so hybrid SAT is unconditional). No symmetry is
    used: it is one concrete case.
A2: if scripts/orbits/a2_orbits.json exists ({name: [facet ids]}), A1 minus
    each listed set of private cells, fixed by facet set (--fix-domain0).
B:  pending Mike's decision on the transport between triangulations
    (spec, Decisions 2); see test_rp2_aut.py for why facets do not carry A5.

Writes one warm file per case and cases.json under --out.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

OPTI = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OPTI / "src"))
import gpu_sc  # noqa: E402


def star_chain(P, v):
    pi1 = [g // P.nv for g in range(P.ngrid)]
    pi2 = [g % P.nv for g in range(P.ngrid)]
    return [pi1, [v] * P.ngrid, pi2]


def write_case(out, cid, family, P, chain, dom, extra):
    warm = Path(out, f"{cid}.json")
    warm.write_text(json.dumps({"best_chains": [chain], "status": "partial",
                                "chain_domain_sizes": [len(dom)], "coverage": len(dom)}))
    return {"id": cid, "family": family, "domain0_facets": sorted(int(f) for f in dom),
            "warm_file": str(warm), "residual": P.nf - len(dom), **extra}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OPTI / "results" / "rp2_cases"))
    ap.add_argument("--family", default="A", choices=["A", "B"])
    a = ap.parse_args()
    if a.family == "B":
        raise SystemExit("Family B split pending Mike's decision (spec, Decisions 2)")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    P = gpu_sc.build("rp2")
    chain = star_chain(P, 0)
    dom = [int(f) for f in P.domain_of(np.array(chain, np.int8))]
    cases = [write_case(a.out, "A1_star0", "A1", P, chain, dom, {})]
    orbits = OPTI / "scripts" / "orbits" / "a2_orbits.json"
    if orbits.exists():
        for name, cells in json.loads(orbits.read_text()).items():
            sub = [f for f in dom if f not in set(cells)]
            cases.append(write_case(a.out, f"A2_{name}", "A2", P, chain, sub,
                                    {"fix_by_facets": True}))
    Path(a.out, "cases.json").write_text(json.dumps(cases, indent=1))
    print(f"{len(cases)} cases -> {a.out}/cases.json")


if __name__ == "__main__":
    main()
