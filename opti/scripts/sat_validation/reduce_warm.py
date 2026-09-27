"""Collapse padded annealer chains to their active links for scs_sat.py --warm.

anneal_cover(P, k, cmax=40) stores every chain as 41 maps (40 links) where
only 3-8 links change anything; the rest repeat a map (padding).  Collapsing
runs of equal consecutive maps:
  * keeps pi1 at the head and pi2 at the tail (endpoints unchanged);
  * deletes only (phi,phi) links, so the covered domain can only grow;
  * leaves every other link literally identical.

scs_sat.py solve_hybrid raises ValueError when a warm chain is longer than
--L, so hybrid runs need these reduced chains (then --L only has to match
the active length, not cmax=40).

Usage:
  python reduce_warm.py <complex> <summary.json> <out.json>

Prints per-chain active lengths and verifies domain(reduced) superset of
domain(original) before writing out.json (same keys, recomputed 'domains').
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_sc


def domains_of(P, chains):
    tri_set = [set(tuple(sorted(t))) for t in P.kfacets]
    FAC = [sorted(int(v) for v in P.facet_verts[i][:P.facet_nv[i]])
           for i in range(P.nf)]
    out = []
    for ch in chains:
        arr = np.asarray(ch)
        dom = []
        for f in range(P.nf):
            verts = FAC[f]
            ok = True
            for i in range(len(arr) - 1):
                ua = {int(arr[i][v]) for v in verts}
                ub = {int(arr[i + 1][v]) for v in verts}
                if not any((ua | ub) <= ts for ts in tri_set):
                    ok = False
                    break
            if ok:
                dom.append(f)
        out.append(dom)
    return out


def collapse(chain):
    keep = [chain[0]]
    for m in chain[1:]:
        if m != keep[-1]:
            keep.append(m)
    return keep


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    cname, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    P = gpu_sc.build(cname)
    data = json.load(open(src))
    chains = [list(map(lambda r: list(map(int, r)), c)) for c in data["best_chains"]]
    doms_old = domains_of(P, chains)
    red = [collapse(c) for c in chains]
    doms_new = domains_of(P, red)
    PI1 = [v // P.nv for v in range(P.ngrid)]
    PI2 = [v % P.nv for v in range(P.ngrid)]
    max_links = 0
    for i, (co, cn, do, dn) in enumerate(zip(chains, red, doms_old, doms_new)):
        assert list(cn[0]) == PI1 and list(cn[-1]) == PI2, f"chain {i}: endpoints broken"
        assert set(do) <= set(dn), f"chain {i}: domain shrank {len(do)}->{len(dn)}"
        links = len(cn) - 1
        max_links = max(max_links, links)
        print(f"  chain {i}: {len(co)-1} links -> {links} active, "
              f"domain {len(do)} -> {len(dn)}")
    data["best_chains"] = red
    data["chain_domain_sizes"] = [len(d) for d in doms_new]
    data["domains"] = doms_new
    data["reduced_from"] = src
    json.dump(data, open(dst, "w"))
    union = len(set().union(*map(set, doms_new)))
    print(f"  union: {union}/{P.nf}")
    print(f"  use with scs_sat.py --warm {dst} --L {max_links} "
          f"(max active links = {max_links})")


if __name__ == "__main__":
    main()
