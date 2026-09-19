#!/usr/bin/env python3
"""star_cover.py — explicit ("product") motion-planner systems for K x K.

For vertices v, w of K the sub-complex  D(v,w) = star(v) x star(w)  of the
staircase product (all staircase facets over pairs of facets (s, t) with
v in s, w in t) is a contiguity domain:

    pi1  ~  c_v  ~  c_w  ~  pi2          (c_x = constant map to x)

since on a facet sigma over (s, t): pi1(sigma) u c_v(sigma) = s (a simplex),
c_v(sigma) u c_w(sigma) = {v, w} (an edge if v ~ w in K; for non-adjacent
v, w insert the constants along a shortest path), c_w(sigma) u pi2(sigma) = t.
Hence  SC_strict(K) <= rho(K) - 1  where rho(K) is the minimum number of
rectangles star(v) x star(w) covering all pairs of facets of K.  For the
Csaszar torus rho = 7 (exact search), for dDelta^4 rho = 3 (matching the
3-domain certificate of Part II).

Usage:  star_cover.py <complex> <outdir> v0,w0 v1,w1 ...
        (rectangles given as vertex pairs; if omitted, a greedy cover is used)
"""
import sys, json, itertools
from pathlib import Path
import numpy as np
import gpu_sc


def rectangle_facets(P, v, w):
    tris = [set(f) for f in P.kfacets]; nt = len(tris)
    npath = P.nf // (nt * nt)
    out = []
    for s in range(nt):
        if v in tris[s]:
            for t in range(nt):
                if w in tris[t]:
                    out.extend(range((s * nt + t) * npath, (s * nt + t + 1) * npath))
    return out


def rectangle_chain(P, v, w):
    const = lambda x: np.full(P.ngrid, x, np.int8)
    # shortest path v -> w in the 1-skeleton (dk is the BFS distance table)
    dk = P.dk.reshape(P.nv, P.nv)
    path = [v]
    while path[-1] != w:
        cur = path[-1]
        nxt = min((u for u in range(P.nv) if dk[cur, u] == 1 or u == cur), key=lambda u: dk[u, w])
        path.append(nxt)
    chain = [P.pi1.copy()] + [const(x) for x in path] + [P.pi2.copy()]
    return chain


def greedy_cover(P):
    tris = [set(f) for f in P.kfacets]; nt = len(tris)
    pairs = set(itertools.product(range(nt), range(nt)))
    rects = []
    while pairs:
        best = max(itertools.product(range(P.nv), range(P.nv)),
                   key=lambda vw: sum(1 for (s, t) in pairs if vw[0] in tris[s] and vw[1] in tris[t]))
        rects.append(best)
        pairs = {(s, t) for (s, t) in pairs if not (best[0] in tris[s] and best[1] in tris[t])}
    return rects


def build(P, rects, reduce=True):
    covered = np.zeros(P.nf, bool); parts = []
    for v, w in rects:
        mine = [f for f in rectangle_facets(P, v, w) if not covered[f]]
        covered[mine] = True
        if mine:
            chain = rectangle_chain(P, v, w)
            if reduce:
                chain = gpu_sc.reduce_chain(P, chain, mine)
            parts.append((mine, chain))
    assert covered.all(), "rectangles do not cover K x K"
    parts.sort(key=lambda t: -len(t[0]))
    return parts


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--no-reduce"]
    name, outdir = args[0], args[1]
    P = gpu_sc.build(name)
    rects = [tuple(int(x) for x in a.split(",")) for a in args[2:]] or greedy_cover(P)
    parts = build(P, rects, reduce="--no-reduce" not in sys.argv)
    gpu_sc.verify_system(P, parts)
    summ = gpu_sc.write_certificate(P, parts, outdir, dict(method="star(v) x star(w) product domains (star_cover.py)",
                                                          rectangles=[list(r) for r in rects]))
    print(f"{P.name}: {len(parts)} domains {summ['domain_sizes']}, chain lengths {summ['chain_lengths']}, "
          f"SC_strict <= {len(parts) - 1}; verified, written to {outdir}")
