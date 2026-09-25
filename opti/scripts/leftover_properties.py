#!/usr/bin/env python3
"""leftover_properties.py — profile the uncovered facets of a stalled annealer state.

For a stalled summary.json (status incomplete, best_chains saved), decode every
uncovered product facet into its cell (s,t) and staircase path, and report:
  * dead-path counts per cell (how many of the 6 paths of each cell are dead)
  * |s cap t| (near-diagonal vs far pair) and dual-graph distance d(s,t)
  * rectangle membership: how many maximal D(v,w) = star(v)xstar(w) contain it,
    uncovered vs covered
  * plateau-domain rectangle identity: is each D_j (near) a rectangle?
  * defect depth: min_j bad_j[f] — how many non-contiguous links the best
    chain still has on each uncovered facet
  * cells fully dead: are they exactly the cells of one missing rectangle
    from the optimal rho-cover?

Usage: python3 leftover_properties.py [complex] [summary.json]
"""
import json, sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import gpu_sc
from rect2_optimum import rect_masks

name = sys.argv[1] if len(sys.argv) > 1 else "rp2"
summary = sys.argv[2] if len(sys.argv) > 2 else "results/rp2_k5_hybrid/summary.json"

P = gpu_sc.build(name)
s = json.loads(Path(summary).read_text())
chains = [np.array(c, np.int8) for c in s["best_chains"]]
doms = [sorted(int(f) for f in P.domain_of(c)) for c in chains]
nt = len(P.kfacets)
npath = P.nf // (nt * nt)

covered = np.zeros(P.nf, bool)
for d in doms:
    covered[d] = True
L = [f for f in range(P.nf) if not covered[f]]
print(f"== {name}: {summary}")
print(f"   |F|={P.nf} k={len(doms)} covered={int(covered.sum())} |L|={len(L)} "
      f"sizes={[len(d) for d in doms]}")

def decode(f):
    cell = f // npath
    st, t = divmod(cell, nt)
    return cell, st, t, f % npath

# ---- dual graph of K: triangles adjacent iff they share an edge ----
tris = [set(map(int, t)) for t in P.kfacets]
adj = [[] for _ in range(nt)]
for a in range(nt):
    for b in range(nt):
        if a != b and len(tris[a] & tris[b]) == 2:
            adj[a].append(b)
def dual_dist(a, b):
    if a == b: return 0
    seen = {a}; frontier = [a]; d = 0
    while frontier:
        d += 1
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v == b: return d
                if v not in seen:
                    seen.add(v); nxt.append(v)
        frontier = nxt
    return -1

# ---- per-facet rectangle membership ----
rmasks = rect_masks(P)
rect_of = defaultdict(list)   # facet -> [(v,w)]
for (v, w), mask in rmasks.items():
    fs = [i for i in range(P.nf) if (mask >> i) & 1] if P.nf <= 64 else None
    if fs is None:
        # bigint path: python ints handle arbitrary size via bit ops on range
        for i in range(P.nf):
            if (mask >> i) & 1:
                rect_of[i].append((v, w))
    else:
        for i in fs:
            rect_of[i].append((v, w))

# ---- cells: dead path counts ----
dead_paths = Counter()
for f in L:
    cell, st, t, p = decode(f)
    dead_paths[(st, t)] += 1
full_cells = sorted(c for c, n in dead_paths.items() if n == npath)
partial_cells = sorted(c for c, n in dead_paths.items() if n < npath)
print(f"\n-- cells with dead facets: {len(dead_paths)} "
      f"(fully dead: {len(full_cells)}, partial: {len(partial_cells)})")
for (st, t) in sorted(dead_paths):
    inter = len(tris[st] & tris[t])
    dd = dual_dist(st, t)
    print(f"   cell ({st:2d},{t:2d}): {dead_paths[(st,t)]}/{npath} dead   "
          f"|s∩t|={inter}   dual_dist={dd}   tris s={sorted(tris[st])} t={sorted(tris[t])}")

# ---- |s∩t| and dual dist distributions: L vs covered ----
inter_L = Counter(len(tris[decode(f)[1]] & tris[decode(f)[2]]) for f in L)
inter_C = Counter(len(tris[decode(f)[1]] & tris[decode(f)[2]]) for f in range(P.nf) if covered[f])
print(f"\n-- |s∩t| distribution   L: {dict(sorted(inter_L.items()))}   "
      f"covered: {dict(sorted(inter_C.items()))}")
ddist_L = Counter(dual_dist(decode(f)[1], decode(f)[2]) for f in L)
ddist_C = Counter(dual_dist(decode(f)[1], decode(f)[2]) for f in range(P.nf) if covered[f])
print(f"-- dual distance  L: {dict(sorted(ddist_L.items()))}   "
      f"covered: {dict(sorted(ddist_C.items()))}")

# ---- rectangle membership counts ----
memL = [len(rect_of[f]) for f in L]
memC = [len(rect_of[f]) for f in range(P.nf) if covered[f]]
print(f"-- #rectangles containing facet: L mean {np.mean(memL):.2f} (min {min(memL)}, max {max(memL)}), "
      f"covered mean {np.mean(memC):.2f}")

# ---- plateau domains vs rectangles ----
print("\n-- plateau domain rectangle identity")
dom_sets = [set(d) for d in doms]
for j, ds in enumerate(dom_sets):
    best = None
    for (v, w), mask in rmasks.items():
        rs = {i for i in range(P.nf) if (mask >> i) & 1}
        miss = len(rs - ds); extra = len(ds - rs)
        if best is None or (miss, extra) < best[:2]:
            best = (miss, extra, (v, w))
    print(f"   D_{j} ({len(ds)}): closest rect {best[2]}  missing {best[0]}, extra {best[1]}")

# ---- defect depth on uncovered facets ----
bads = [P.chain_bad_counts(c) for c in chains]
depth = [min(int(b[f]) for b in bads) for f in L]
dc = Counter(depth)
print(f"\n-- defect depth min_j bad_j[f] on L: {dict(sorted(dc.items()))}")

# ---- is the fully-dead cell set a missing rectangle's cells? ----
if full_cells:
    fc = set(full_cells)
    print(f"\n-- fully-dead cells ({len(fc)}) vs the cells of each rectangle:")
    for (v, w), mask in sorted(rmasks.items()):
        rs = {((i // npath) // nt, (i // npath) % nt) for i in range(P.nf) if (mask >> i) & 1}
        inter = len(rs & fc)
        if inter >= len(fc) * 0.5:
            print(f"   rect ({v},{w}): contains {inter}/{len(fc)} fully-dead cells")