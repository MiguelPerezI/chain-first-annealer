#!/usr/bin/env python3
"""rect2_optimum.py — what is the 460 in the two-chain plateau?

The k=2 searches of the S³ = ∂Δ⁴ campaign all pin at exactly 460 of the 500
product facets.  guided_sphere_k2_plan.md explains that as "the best cover by
ρ − 1 = 2 rectangles".  This script checks that explanation combinatorially,
with no search and no annealing:

  1. exhaustive optimum of the rectangle family star(v) × star(w)
     (25 rectangles for ∂Δ⁴, so C(25,2) = 300 pairs, C(25,3) = 2300 triples),
     reported as the best cover by 1, 2 and 3 rectangles;

  2. for every recorded two-chain run, the domains the annealer actually
     reached — are they (sub)rectangles, and is their union exactly the
     2-rectangle optimum?

Rectangle = the product star(v) × star(w) of the stars of two vertices v, w of
K, i.e. all product facets whose cell (s, t) has v ∈ s and w ∈ t.  This is the
product-domain family whose minimum cover number is ρ(K) (rect_cover.py).

Usage:  python3 rect2_optimum.py            # s3 = ∂Δ⁴ and its recorded runs
        python3 rect2_optimum.py s4         # any complex name gpu_sc knows
"""
import itertools
import json
import sys
from pathlib import Path

import gpu_sc

NF = 500  # facets of ∂Δ⁴ × ∂Δ⁴, asserted below


def rect_masks(P):
    """Bitmask over product facets for every rectangle star(v) × star(w)."""
    tris = [set(f) for f in P.kfacets]
    nt = len(tris)
    # facets are enumerated cell by cell: block (s, t) holds the staircase
    # facets of that cell, all cells the same size (all facets of ∂Δ^4 are
    # 3-simplices, so every cell has the same staircase count)
    assert P.nf % (nt * nt) == 0, "ragged cell blocks"
    per_cell = P.nf // (nt * nt)
    cell_of = [i // per_cell for i in range(P.nf)]
    out = {}
    for v in range(P.nv):
        for w in range(P.nv):
            m = 0
            for f, cell in enumerate(cell_of):
                s, t = divmod(cell, nt)
                if v in tris[s] and w in tris[t]:
                    m |= 1 << f
            out[(v, w)] = m
    return out


def popcount(x):
    return bin(x).count("1")


def best_union(masks, k):
    """Largest union of k distinct rectangles: (size, [(v, w), ...])."""
    keys = list(masks)
    best = (-1, None)
    for combo in itertools.combinations(keys, k):
        u = 0
        for c in combo:
            u |= masks[c]
        if popcount(u) > best[0]:
            best = (popcount(u), list(combo))
    return best


def best_union_cells(P, k):
    """Largest union of k rectangles, at CELL level — exact, because
    membership in star(v) x star(w) is a property of the cell, so every
    rectangle is a union of whole cells.  This is the only tractable form
    once the facet count runs to six figures (∂Δ⁸: 277,992 facets), and it
    agrees with the facet-level count on the small complexes.

    Returns (facets covered, [(v, w), ...], cells covered, cells per cell)."""
    tris = [set(f) for f in P.kfacets]
    nt = len(tris)
    assert P.nf % (nt * nt) == 0, "ragged cells"
    per = P.nf // (nt * nt)
    masks = {}
    for v in range(P.nv):
        for w in range(P.nv):
            m = 0
            for cell in range(nt * nt):
                s, t = divmod(cell, nt)
                if v in tris[s] and w in tris[t]:
                    m |= 1 << cell
            masks[(v, w)] = m
    best = (-1, None, -1)
    for combo in itertools.combinations(masks, k):
        u = 0
        for c in combo:
            u |= masks[c]
        n = u.bit_count()
        if n > best[0]:
            best = (n, list(combo), per)
    cells, combo, per = best
    return cells * per, combo, cells, per


def domains_of(P, path):
    """Domains of a recorded run's best state, or None if the summary is not an
    annealer state of *this* complex (a ∂Δ⁵ chain indexed into ∂Δ⁴'s tables
    would index out of bounds, so the grid size is checked first)."""
    s = json.loads(Path(path).read_text())
    if "best_chains" not in s:
        return None
    nc = s.get("complex", {})
    if nc.get("nv") != P.nv or nc.get("num_facets") != P.nf:
        return None
    return [P.domain_of(c) for c in s["best_chains"]]


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "s3"
    P = gpu_sc.build(name)
    print(f"{P.name}: {P.nf} product facets, {P.nv} vertices, "
          f"{P.nv ** 2} rectangles of the form star(v) x star(w)\n")
    masks = rect_masks(P)
    if name == "s3":
        assert P.nf == NF, f"∂Δ⁴ × ∂Δ⁴ should have {NF} facets, has {P.nf}"

    # 1 — exhaustive optimum of the rectangle family
    print("exhaustive rectangle-family optimum (no search involved):")
    big = P.nf > 100_000
    for k in (1, 2, 3):
        if big:      # facet-level masks would be 34 KB ints x 85k unions
            size, combo, cells, per = best_union_cells(P, k)
            print(f"  best cover by {k} rectangle(s): {size}/{P.nf}"
                  f"  ({cells} of {P.nf // per} cells x {per} facets/cell)"
                  f"   {combo if k > 1 else ''}")
        else:
            size, combo = best_union(masks, k)
            print(f"  best cover by {k} rectangle(s): {size}/{P.nf}"
                  f"   {combo if k > 1 else ''}")
    print()

    # 2 — what the recorded two-chain runs actually reached
    runs = [p.parent.name for p in Path("results").glob(f"{name}_k2*/summary.json")]
    runs.sort()
    if not runs:
        return
    best2 = best_union(masks, 2)[0]
    print(f"recorded k=2 runs — the plateau value is the 2-rectangle optimum "
          f"{best2}/{P.nf}:\n")
    print(f"  {'run':22s} {'cov':>4s} {'sizes':>12s}  domains, as products of stars")
    for r in runs:
        doms = domains_of(P, f"results/{r}/summary.json")
        if doms is None:
            print(f"  {r:22s} skipped (not a state of this complex)")
            continue
        union = set()
        for d in doms:
            union |= set(d.tolist())
        flags = []
        for d in doms:
            m = 0
            for f in d.tolist():
                m |= 1 << f
            contained = [vw for vw, rm in masks.items() if (m & ~rm) == 0]
            equal = [vw for vw, rm in masks.items() if m == rm]
            flags.append(f"= star{equal[0][0]} x star{equal[0][1]}" if equal else
                         (f"smaller than {len(contained)} rect(s), max "
                          f"{max((popcount(masks[c]) for c in contained), default=0)}"
                          if contained else "NOT a rectangle"))
        print(f"  {r:22s} {len(union):4d} "
              f"{str([len(d) for d in doms]):>12s}  {' '.join(flags)}")


if __name__ == "__main__":
    main()
