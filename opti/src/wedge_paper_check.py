#!/usr/bin/env python3
"""wedge_paper_check.py — cross-check §5.2 of arXiv:2008.13290 (the wedge of
two circles) against this repository's conventions, and re-verify the paper's
own certificate with our Definition 2.3 checker.

The paper lists its three optimal domains J0, J1, J2 as triangles of K x K
(writing the product vertex (i, j) as the label 4i + j) and their three
contiguity chains in Tables 3-5 (one row per map; the header row lists the
grid vertices of the domain, each cell is the image of that grid vertex — a
vertex of K, since the planner maps run K x K -> K).  This script reads all of
that straight out of the paper's own HTML (article.html, the tracked copy of
the arXiv page — no transcription and no reliance on the figure), and then

  1. decodes each triangle: label L -> (L // 4, L % 4), so the paper's label
     IS our grid-vertex id, id(a, b) = 4a + b; the two edges of K the triangle
     spans in the first/second factor; and which of the two monotone staircase
     paths of that edge pair it is — giving the facet id our code assigns;
  2. checks those ids are exactly 0..49, once each: the paper's facets are
     exactly the staircase triangulation of K = K_4 minus the edge {2,3}, and
     J0, J1, J2 partition it (20 + 18 + 12 = 50 facets);
  3. runs the H^1 necessary condition — an obstructed set cannot be a
     contiguity domain, so this catches a decoding mistake;
  4. re-verifies the paper's three chains (Tables 3-5) on the paper's domains
     with the plain Definition 2.3 check: first/last map on the domain are
     pi1/pi2, every map simplicial, consecutive maps 1-contiguous;
  5. compares the paper's partition and chain lengths with ours in
     results/wedge_k3_v2.

This file is standard library only — no numpy, no numba, no code from the
search — so that `python3 wedge_paper_check.py` works on a bare interpreter,
like independent_check.py.  When gpu_sc.py and h1_obstruction.py ARE
importable (they need numpy/numba), the script additionally cross-checks its
own facet enumeration, its simplex table and its H^1 numbers against theirs.

usage: python3 wedge_paper_check.py [our results dir]
"""

import re
import sys
import json
import html
from itertools import combinations
from pathlib import Path

EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)]   # facets of K, build("wedge") order
NAMES = ("J0", "J1", "J2")


# --------------------------------------------------------------------------
#  K x K, rebuilt here from scratch (no search code)
# --------------------------------------------------------------------------

class Wedge:
    """K = K_4 minus the edge {2,3} (5 edges, one 4-vertex component) and its
    product with itself, in exactly the enumeration of gpu_sc.Product: grid
    vertex (a, b) has id a*nv + b, facets ordered by (facet of K, facet of K,
    path by increasing bitmask).

    Planner maps run K x K -> K (images are vertices of K, not grid vertices),
    so 'is_simplex' is indexed by bitmasks over the nv = 4 vertices of K —
    again as in gpu_sc.Product."""

    def __init__(self, kfacets=EDGES):
        self.name = "wedge of two circles"
        self.kfacets = [tuple(sorted(f)) for f in kfacets]
        nv = self.nv = 1 + max(v for f in self.kfacets for v in f)
        assert nv <= 11, "grid ids as ints assume nv <= 11"
        # a subset of V_K is a simplex iff it is contained in a facet of K
        # (the empty set included, as in gpu_sc's table)
        self.is_simplex = set()
        for f in self.kfacets:
            m = sum(1 << v for v in f)
            for s in range(1 << nv):
                if s & ~m == 0:
                    self.is_simplex.add(s)
        facets = []
        for s0 in self.kfacets:
            for s1 in self.kfacets:
                p, q = len(s0) - 1, len(s1) - 1
                for mask in range(1 << (p + q)):
                    bits = [(mask >> b) & 1 for b in range(p + q)]
                    if sum(bits) != q:
                        continue
                    pi = pj = 0
                    tet = [s0[0] * nv + s1[0]]
                    for step in bits:
                        if step == 0:
                            pi += 1
                        else:
                            pj += 1
                        tet.append(s0[pi] * nv + s1[pj])
                    facets.append(tuple(tet))
        self.facets = facets
        self.nf = len(facets)


# --------------------------------------------------------------------------
#  GF(2) linear algebra and the H^1 obstruction test (stdlib only)
# --------------------------------------------------------------------------

def _rref(rows, ncols):
    """Row-reduce a GF(2) matrix given as a list of row bitmasks over the
    ncols columns -> (reduced rows, pivot columns)."""
    full = (1 << ncols) - 1
    rows = [r & full for r in rows if r & full]
    piv, r = [], 0
    for c in range(ncols):
        p = next((i for i in range(r, len(rows)) if (rows[i] >> c) & 1), None)
        if p is None:
            continue
        rows[r], rows[p] = rows[p], rows[r]
        for i in range(len(rows)):
            if i != r and (rows[i] >> c) & 1:
                rows[i] ^= rows[r]
        piv.append(c)
        r += 1
        if r == len(rows):
            break
    return rows, piv


def gf2_rank(rows, ncols):
    return len(_rref(rows, ncols)[1])


def gf2_nullspace(rows, ncols):
    """Basis of {x : M x = 0} over GF(2), each vector a bitmask over columns."""
    red, piv = _rref(rows, ncols)
    basis = []
    for f in range(ncols):
        if f in piv:
            continue
        x = 1 << f
        for i, pc in enumerate(piv):
            if (red[i] >> f) & 1:
                x |= 1 << pc
        basis.append(x)
    return basis


def h1_obstruction_local(P, J):
    """(dim H1(J;Z2), rank of (pi1 - pi2)_* : H1(J) -> H1(K;Z2)) — the same
    computation as h1_obstruction.h1_map_obstruction, but over GF(2) with
    bitmask rows and no numpy."""
    facets = [P.facets[f] for f in J]
    V0 = sorted({v for f in facets for v in f})
    V1 = sorted({tuple(sorted(p)) for f in facets for p in combinations(f, 2)})
    V2 = sorted({tuple(sorted(t)) for f in facets for t in combinations(f, 3)})
    # d1: vertices -> edges, d2: edges -> triangles (rows = source, bits = target)
    vi = {v: i for i, v in enumerate(V0)}
    d1 = [0] * len(V0)
    for j, (a, b) in enumerate(V1):
        d1[vi[a]] |= 1 << j
        d1[vi[b]] |= 1 << j
    ei = {e: i for i, e in enumerate(V1)}
    d2 = [0] * len(V1)
    for j, t in enumerate(V2):
        for a, b in combinations(t, 2):
            d2[ei[(a, b)]] |= 1 << j
    Z = gf2_nullspace(d1, len(V1))                 # cycles, as bitmasks over V1
    h1J = len(Z) - gf2_rank(d2, len(V2))
    # K: edges and triangles
    KE = sorted({tuple(sorted(p)) for f in P.kfacets for p in combinations(f, 2)})
    KT = sorted({tuple(sorted(f)) for f in P.kfacets if len(f) >= 3})
    ke = {e: i for i, e in enumerate(KE)}
    dK2 = [0] * len(KE)
    for j, t in enumerate(KT):
        for a, b in combinations(t, 2):
            dK2[ke[(a, b)]] |= 1 << j
    # (pi1 - pi2)_* on the cycles of J, as rows over the K-edges: push each
    # cycle forward by both projections and add (GF(2)) the two 1-chains
    nv = P.nv
    D = [0] * len(KE)
    for ci, z in enumerate(Z):
        push = [[0] * len(KE), [0] * len(KE)]
        for j in range(len(V1)):
            if (z >> j) & 1:
                a, b = V1[j]
                for t, (pa, pb) in enumerate(((a // nv, b // nv), (a % nv, b % nv))):
                    if pa != pb:
                        push[t][ke[tuple(sorted((pa, pb)))]] ^= 1
        for i in range(len(KE)):
            if push[0][i] ^ push[1][i]:
                D[i] |= 1 << ci
    # rank modulo the boundaries of K
    comb = [dK2[i] | (D[i] << len(KT)) for i in range(len(KE))]
    rk = gf2_rank(comb, len(KT) + len(Z)) - gf2_rank(dK2, len(KT))
    return h1J, rk


# --------------------------------------------------------------------------
#  reading the paper's data out of article.html
# --------------------------------------------------------------------------

def _wedge_region(path):
    t = Path(path).read_text(encoding="utf-8", errors="replace")
    i = t.find("A wedge of two circles")
    j = t.find("A wedge of two circles", i + 10)      # body, not the TOC entry
    assert j > 0, "§5.2 heading not found twice in article.html"
    k = t.find("References", j)                       # §5.2 is the last section
    return t[j:k if k > j else len(t)]


def parse_list(a):
    """A LaTeX facet list of the paper's form -> tuple of label triples.

    '\\{5,9,10\\},\\{0,3,15\\}.'  ->  ((5,9,10), (0,3,15));  None for any
    alttext that is not a list of such groups."""
    a = html.unescape(a).strip().rstrip(".")
    if not (a.startswith("\\{") and a.endswith("\\}")):
        return None
    out = []
    for g in a.split("\\},"):
        g = g.strip()
        if g.startswith("\\{"):
            g = g[2:]
        if g.endswith("\\}"):
            g = g[:-2]
        if not g or not all(c.isdigit() or c == "," for c in g):
            return None
        out.append(tuple(int(x) for x in g.split(",")))
    return tuple(out)


def paper_lists(region):
    """The three facet lists of §5.2, in document order."""
    # each list is split over several alttext chunks (one per rendered line);
    # a chunk ending in "\\}" or "\\}." ends the list.
    out, buf = [], []
    for a in re.findall(r'alttext="([^"]*)"', region):
        a = html.unescape(a).strip()
        if not a.startswith("\\{"):
            continue
        buf.append(a.rstrip(".").rstrip(","))
        if a.endswith("\\}") or a.endswith("\\}."):
            out.append(parse_list(",".join(buf)))
            buf = []
    return [L for L in out if L]


def read_tables(region):
    """The chain tables of §5.2 -> {name: (maps, header)} where maps is a list
    of {grid vertex id: image vertex of K} (one per map, in chain order) and
    header the list of grid vertices the maps are defined on."""
    tables = {}
    for m in re.finditer(r"<table\b", region):
        blk, depth = region[m.start():m.start() + 200000], 0
        for mm in re.finditer(r"<table\b|</table>", blk):
            depth += 1 if mm.group(0).startswith("<table") else -1
            if depth == 0:
                blk = blk[:mm.end()]
                break
        rows = [[html.unescape(x) for x in re.findall(r'alttext="([^"]*)"', r)]
                for r in re.findall(r"<tr\b.*?</tr>", blk, re.S)]
        if len(rows) < 3 or not rows[0] or not all(x.isdigit() for x in rows[0]):
            continue
        if not rows[1] or not rows[1][0].startswith("\\varphi"):
            continue
        header = [int(x) for x in rows[0]]
        maps, idx = [], []
        for pr in rows[1:]:
            assert pr[0].startswith("\\varphi"), pr[0]
            k = re.search(r"\\varphi_\{(\d+)\}", pr[0])
            idx.append(int(k.group(1)) if k else None)
            vals = [int(x) for x in pr[1:]]
            assert len(vals) == len(header), "row is not aligned with the header"
            maps.append(dict(zip(header, vals)))
        assert idx == list(range(len(idx))), f"chain is not phi_0..phi_c: {idx}"
        name = re.search(r"J_\{(\d+)\}", rows[1][0])
        assert name, rows[1][0]
        tables[f"J{name.group(1)}"] = (maps, header)
    return tables


def decode(tri):
    """Paper's label triple -> facet id of K x K in our enumeration (facet
    ids run over (edge of K in factor 1, edge of K in factor 2, path))."""
    a = sorted({L // 4 for L in tri})
    b = sorted({L % 4 for L in tri})
    assert len(a) == 2 and len(b) == 2, f"{tri} is not a triangle over two K-edges"
    e0, e1 = tuple(a), tuple(b)
    assert e0 in EDGES and e1 in EDGES, f"{tri}: {e0} or {e1} is not an edge of K"
    T = {(L // 4, L % 4) for L in tri}
    if (e0[0], e1[1]) in T:
        path = 0        # staircase path with mask 1: factor 2 advances first
    elif (e0[1], e1[0]) in T:
        path = 1        # mask 2: factor 1 advances first
    else:
        raise AssertionError(f"{tri} is not a staircase triangle of {e0} x {e1}")
    return (EDGES.index(e0) * len(EDGES) + EDGES.index(e1)) * 2 + path


# --------------------------------------------------------------------------
#  re-verifying the paper's certificate with our own Definition 2.3 check
# --------------------------------------------------------------------------

def check_paper_chain(P, name, facets, maps, header):
    """Definition 2.3 on the paper's own domain and chain.  The maps run
    K x K -> K, so their images are vertices of K (0..nv-1) and 'simplicial'
    / '1-contiguous' are tested with the simplex table of K."""
    verts = sorted({v for f in facets for v in P.facets[f]})
    assert sorted(header) == verts, \
        f"{name}: maps are given on {len(header)} vertices, the domain has {len(verts)}"
    for mi, mp in enumerate(maps):
        for v in verts:
            assert 0 <= mp[v] < P.nv, \
                f"{name}: map {mi} sends grid vertex {v} to {mp[v]}, not a vertex of K"
    def mask(vals):
        m = 0
        for v in vals:
            m |= 1 << int(v)
        return m
    for mi, mp in enumerate(maps):
        for f in facets:
            assert mask(mp[v] for v in P.facets[f]) in P.is_simplex, \
                f"{name}: map {mi} is not simplicial on facet {f}"
    for mi, (a, b) in enumerate(zip(maps, maps[1:])):
        for f in facets:
            assert mask([a[v] for v in P.facets[f]] +
                        [b[v] for v in P.facets[f]]) in P.is_simplex, \
                f"{name}: maps {mi},{mi+1} are not 1-contiguous on facet {f}"
    for v in verts:
        assert maps[0][v] == v // P.nv, f"{name}: first map is not pi1 at {v}"
        assert maps[-1][v] == v % P.nv, f"{name}: last map is not pi2 at {v}"
    return len(maps) - 1, len(maps)


def cross_check(P, paper):
    """When the search code is importable, check this file's enumeration, its
    simplex table and its H^1 numbers against gpu_sc / h1_obstruction."""
    try:
        import gpu_sc
        from h1_obstruction import h1_map_obstruction
    except Exception as exc:                       # numpy/numba not installed
        print(f"[x] gpu_sc/h1_obstruction not importable here "
              f"({exc.__class__.__name__}) — ran the self-contained checks only")
        return
    Q = gpu_sc.build("wedge")
    assert [tuple(f) for f in Q.facets] == [tuple(f) for f in P.facets], \
        "facet enumeration differs from gpu_sc.Product"
    assert {s for s in range(1 << P.nv) if Q.is_simplex[s]} == P.is_simplex, \
        "is_simplex table differs from gpu_sc.Product"
    for name, J in paper.items():
        assert h1_map_obstruction(Q, J) == h1_obstruction_local(P, J), \
            f"{name}: the H^1 numbers differ from h1_obstruction.py"
    print(f"[x] cross-checked against gpu_sc.Product (facets, is_simplex) and "
          f"h1_obstruction.py (H^1 on all 3 domains): identical")


def main():
    ourdir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/wedge_k3_v2")
    P = Wedge()
    nv = P.nv
    print(f"our K: {P.name}, {len(P.kfacets)} facets, {P.nf} facets of K x K, grid {nv} x {nv}")

    region = _wedge_region("article.html")
    lists = paper_lists(region)
    assert len(lists) == 3, f"expected the 3 lists of §5.2, extracted {len(lists)}"
    assert [len(L) for L in lists] == [20, 18, 12], [len(L) for L in lists]
    paper = {name: [decode(t) for t in L] for name, L in zip(NAMES, lists)}
    print(f"[0] read the paper's three lists from article.html: "
          f"{[len(paper[n]) for n in NAMES]} facets")

    # (a) each decoded id is a facet of K x K with the vertex set the paper's
    #     labels describe (label L = 4i + j = our grid id i*nv + j)
    for name, L in zip(NAMES, lists):
        for fid, tri in zip(paper[name], L):
            assert set(P.facets[fid]) == {(x // 4) * nv + (x % 4) for x in tri}, (name, tri, fid)
    print("[1] every decoded triangle is that facet of K x K, with the paper's vertices")

    # (b) the three domains partition K x K
    flat = [f for ids in paper.values() for f in ids]
    assert len(flat) == len(set(flat)), "the paper's domains overlap"
    assert sorted(flat) == list(range(P.nf)), "the paper's domains do not cover K x K"
    print(f"[2] the paper's J0, J1, J2 partition K x K: "
          f"{[len(paper[n]) for n in NAMES]} = {P.nf} facets")

    # (c) H^1 necessary condition: (pi1 - pi2)_* must vanish on H_1 of a domain
    for name in NAMES:
        h1, rk = h1_obstruction_local(P, paper[name])
        print(f"[3] {name}: dim H1(J;Z2) = {h1}, rank (pi1-pi2)_* = {rk}  -> "
              f"{'OBSTRUCTED (decode error)' if rk else 'passes the H1 test'}")

    # (d) the paper's chains, re-verified here
    tables = read_tables(region)
    assert set(tables) == set(NAMES), sorted(tables)
    paper_links = {}
    for name in NAMES:
        maps, header = tables[name]
        links, nmaps = check_paper_chain(P, name, paper[name], maps, header)
        paper_links[name] = links
        print(f"[4] {name}: the paper's Table for {name} — {nmaps} maps ({links} links), "
              f"pi1 -> pi2, simplicial + 1-contiguous on all {len(paper[name])} facets: OK")

    cross_check(P, paper)

    # (e) compare with our annealed partition
    if (ourdir / "summary.json").exists():
        s = json.loads((ourdir / "summary.json").read_text())
        ours = [sorted(int(f) for f in d) for d in s["domains"]]
        print(f"[5] {ourdir}: domains {s['domain_sizes']} = {sum(s['domain_sizes'])} facets, "
              f"chain links {s['chain_lengths']} / maps "
              f"{[c + 1 for c in s['chain_lengths']]}  ->  SC_strict <= {s['sc_strict_bound']}")
        print(f"    the paper's chain links {[paper_links[n] for n in NAMES]} / maps "
              f"{[paper_links[n] + 1 for n in NAMES]}")
        print("    overlap matrix (rows = paper J0..J2, cols = our domains):")
        for name in NAMES:
            ps = set(paper[name])
            print(f"      {name} ({len(ps):2d}): " + " ".join(f"{len(ps & set(o)):3d}" for o in ours))
        same = sorted(sorted(d) for d in ours) == sorted(sorted(paper[n]) for n in NAMES)
        print(f"    identical to the paper's partition? {same}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
