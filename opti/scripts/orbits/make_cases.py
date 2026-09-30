#!/usr/bin/env python3
"""Family A cases for RP2 k=5.

A1: domain 0 = star(0) x star(0), given as the frozen chain pi1 ~ c_0 ~ pi2
    (a verified planner, so hybrid SAT is unconditional). No symmetry is
    used: it is one concrete case.
A2: if scripts/orbits/a2_orbits.json exists ({name: [facet ids]}), A1 minus
    each listed set of private cells, pinned by facet set: a cold case run
    with --fix-domain0 FILE. It gets no warm file, because a frozen chain
    always covers its whole domain (load_warm recomputes the full 150).
B:  pending Mike's decision on the transport between triangulations
    (spec, Decisions 2); see test_rp2_aut.py for why facets do not carry A5.

Writes cases.json under --out, plus one warm file (A1) or one facet-list
file (A2) per case.
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


def rel(path):
    try:
        return str(path.resolve().relative_to(OPTI))   # portable to the cluster checkout
    except ValueError:
        return str(path.resolve())


def case(cid, family, P, dom, **files):
    return {"id": cid, "family": family, "domain0_facets": sorted(int(f) for f in dom),
            **files, "residual": P.nf - len(dom)}


def write_warm_case(out, cid, family, P, chain, dom):
    """Domain 0 is a frozen chain: hybrid run, --warm FILE --free k-1."""
    warm = Path(out, f"{cid}.json")
    warm.write_text(json.dumps({"best_chains": [chain], "status": "partial",
                                "chain_domain_sizes": [len(dom)], "coverage": len(dom)}))
    return case(cid, family, P, dom, warm_file=rel(warm))


def write_pin_case(out, cid, family, P, dom):
    """Domain 0 contains a facet set: cold run, --fix-domain0 FILE."""
    pin = Path(out, f"{cid}_fix_domain0.json")
    pin.write_text(json.dumps(sorted(int(f) for f in dom)))
    return case(cid, family, P, dom, fix_domain0_file=rel(pin))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OPTI / "results" / "rp2_cases"))
    ap.add_argument("--family", default="A", choices=["A", "B"])
    ap.add_argument("--a2-orbits", default=str(OPTI / "scripts" / "orbits" / "a2_orbits.json"),
                    help="JSON {name: [facet ids]} of private cells (absent: A1 only)")
    a = ap.parse_args()
    if a.family == "B":
        raise SystemExit("Family B split pending Mike's decision (spec, Decisions 2)")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    P = gpu_sc.build("rp2")
    chain = star_chain(P, 0)
    dom = [int(f) for f in P.domain_of(np.array(chain, np.int8))]
    cases = [write_warm_case(a.out, "A1_star0", "A1", P, chain, dom)]
    orbits = Path(a.a2_orbits)
    if orbits.exists():
        for name, cells in json.loads(orbits.read_text()).items():
            sub = [f for f in dom if f not in set(cells)]
            cases.append(write_pin_case(a.out, f"A2_{name}", "A2", P, sub))
    Path(a.out, "cases.json").write_text(json.dumps(cases, indent=1))
    print(f"{len(cases)} cases -> {a.out}/cases.json")


if __name__ == "__main__":
    main()
