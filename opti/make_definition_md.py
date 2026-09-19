#!/usr/bin/env python3
"""make_definition_md.py — generate the per-result `definition.md` (exact
complex + product indexing conventions used by the code) for a sphere
∂Δ^{n+1} = S^n, for any n — including n >= 10, where `gpu_sc.Product` cannot
be instantiated (int8 grid ids) and only the streaming tools apply — or for
the wedge of two circles of §5.2 of arXiv:2008.13290, for the barycenter
triangle (Δ² subdivided by its barycenter into three triangles), or for the
cube boundary (its six square faces cut along their drawn diagonals).

usage:  make_definition_md.py <sN|wedge|bary|cube> <results dir> "<result label>"

The emitted file is self-contained: it lists the facets of K, the grid
indexing, the staircase facet order and the file formats, i.e. everything a
reader needs to replicate or hand-check cover.txt / planners.txt.
"""
import sys
from math import comb
from pathlib import Path

SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def dsup(k):
    """k as a superscript exponent, for sphere/simplex names only ("∂Δ⁶", "S⁵")."""
    return str(k).translate(SUP)


def sphere_spec(N):
    """K = ∂Δ^{N+1} = S^N and everything the report says about it."""
    import itertools
    nv = N + 2                            # vertices of K = ∂Δ^{N+1}
    facets = list(itertools.combinations(range(nv), nv - 1))
    # f-vector of ∂Δ^{N+1}: C(nv, k+1) faces of dimension k, k = 0..nv-2
    fvec = [comb(nv, k + 1) for k in range(nv - 1)]
    chi = sum((-1) ** k * fvec[k] for k in range(nv - 1))
    return dict(
        nv=nv, facets=facets, npath=comb(2 * N, N), maxfacet=2 * N + 1,
        title=f"∂Δ{dsup(nv - 1)} (S{dsup(N)})",
        facets_intro=(f"Facets (maximal simplices), in the order used by the code — defined as\n"
                      f"all {nv - 1}-tuples of {{0..{nv - 1}}}: "
                      f"itertools.combinations(range({nv}), {nv - 1}):"),
        topology=[
            f"K = ∂Δ{dsup(nv - 1)} is the boundary of the {nv - 1}-simplex: these {nv} facets are",
            f"exactly the facets of Δ{dsup(nv - 1)} (each omitting one vertex); K has f-vector",
            f"({' '.join(str(x) for x in fvec)}) and χ = {chi}, and is a closed",
            f"({nv - 3})-connected {N}-manifold, hence homeomorphic to S{dsup(N)}.",
        ],
        nf_note=f"= (n+1)^2 * C(2n-2, n-1) with n = {nv - 1}",
    )


def wedge_spec():
    """K = K₄ minus the edge {2,3}: the wedge of two circles of §5.2 of
    arXiv:2008.13290 (4 vertices, 5 edges, χ = −1)."""
    return dict(
        nv=4, facets=[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)],
        npath=2, maxfacet=3,
        title="K₄ − the edge {2,3}  (‖K‖ ≃ S¹ ∨ S¹)",
        facets_intro=("Facets (maximal simplices), in the order used by the code — the five edges\n"
                      "of K₄ minus the edge {2,3}, listed explicitly rather than generated:"),
        topology=[
            "K is the 1-dimensional complex with vertices {0,1,2,3} and these 5 edges:",
            "the complete graph K₄ with the edge {2,3} deleted; f-vector (4, 5), χ = 4 − 5 = −1.",
            "The triangles {0,1,2} and {0,1,3} share the edge {0,1}; that edge is contractible",
            "and so is each triangle, so collapsing it in both gives",
            "‖K‖ ≃ (triangle/{0,1}) ∨ (triangle/{0,1}) ≃ S¹ ∨ S¹, the wedge of two circles",
            "of §5.2 of arXiv:2008.13290.  (The five edges above are exactly the ones that",
            "occur in the paper's three domain listings J₀, J₁, J₂ — the edge {2,3} never does.)",
        ],
        nf_note="= 5 x 5 edge pairs x 2 staircase paths per pair",
    )


def bary_spec():
    """K = Δ² = {0,1,2} subdivided by its barycenter 3 into three triangles
    (a cone on ∂Δ²; 4 vertices, all 6 edges, χ = 1)."""
    return dict(
        nv=4, facets=[(0, 1, 3), (0, 2, 3), (1, 2, 3)],
        npath=6, maxfacet=4,
        title="the barycenter triangle  (Δ² split by its barycenter 3; ‖K‖ ≅ Δ²)",
        facets_intro=("Facets (maximal simplices), in the order used by the code — the three\n"
                      "triangles of the triangle {0,1,2} after adding the barycenter 3:"),
        topology=[
            "K has vertices {0,1,2,3} where {0,1,2} is the outer triangle and 3 is the",
            "barycenter, joined to all three of its vertices.  2-faces: 3 triangles;",
            "1-faces: all 6 edges of K₄ (the 3 triangles have 9 edge slots, the 3 inner",
            "edges {0,3},{1,3},{2,3} shared by two triangles each, so 9 − 3 = 6 edges);",
            "f-vector (4, 6, 3), χ = 4 − 6 + 3 = 1: a disk, the triangulation of a triangle.",
            "Vertex 3 lies in EVERY facet, so K is the cone on ∂Δ²; in particular ‖K‖ is",
            "contractible, the case in which — as noted right before the definition of",
            "SC_strict in arXiv:2008.13290 — a motion planner exists on all of K x K.",
            "The constant map c₃ and the chain π₁ ∼ c₃ ∼ π₂ are therefore valid on every",
            "facet of the product, giving a one-domain system (SC_strict = 0), and the",
            "whole search problem is to discover c₃ (or an equivalent middle map).",
        ],
        nf_note="= 3 x 3 triangle pairs x 6 staircase paths per pair",
    )


def cube_spec():
    """K = the boundary of the cube: 8 vertices, the six square faces cut along
    their drawn diagonals into 12 triangles (a triangulated S², 8 vertices)."""
    return dict(
        nv=8,
        facets=[(0, 3, 7), (0, 4, 7), (1, 2, 6), (1, 5, 6), (0, 1, 5), (0, 4, 5),
                (2, 3, 7), (2, 6, 7), (4, 5, 6), (4, 6, 7), (0, 1, 2), (0, 2, 3)],
        npath=6, maxfacet=5,
        title="the cube boundary  (8 vertices, 12 triangles; ‖K‖ ≅ S²)",
        facets_intro=("Facets (maximal simplices), in the order used by the code — the six square\n"
                      "faces of the cube, each split along the drawn diagonal into two triangles:"),
        topology=[
            "K is the boundary surface of the cube, triangulated: vertices 0..7 (bottom face",
            "0,1,2,3 at even height, top face 4,5,6,7 directly above), the six square faces",
            "{0,3,7,4} (with diagonal 0-7), {1,2,6,5} (diagonal 1-6), {0,1,5,4} (diagonal 0-5),",
            "{2,3,7,6} (diagonal 2-7), {4,5,6,7} (diagonal 4-6) and {0,1,2,3} (diagonal 0-2),",
            "each cut along its diagonal into the two triangles listed above: 12 triangles.",
            "Counts: 8 vertices, 18 edges (12 x 3 / 2, every edge in exactly two triangles —",
            "there is no boundary), f-vector (8, 18, 12), χ = 8 − 18 + 12 = 2.  Every vertex",
            "link is a single cycle (link sizes 6, 4, 5, 3, 4, 4, 5, 5), so the complex is a",
            "closed surface, and χ = 2 forces it to be orientable and connected: ‖K‖ ≅ S² —",
            "the same space as ∂Δ³, in a different (8-vertex, non-minimal) triangulation.",
            "The target is therefore SC_strict ≥ TC(S²) = 2, i.e. at least 3 domains, sharp at 2.",
            "Product domains fare far worse here than on ∂Δ³ (where 3 stars suffice): the stars",
            "of K cover 6, 4, 5, 3, 4, 4, 5, 5 triangles, so a rectangle star(v) x star(w) covers",
            "at most 36 of the 144 facet pairs of K x K, and the counting bound 36+30+30+30 =",
            "126 < 144 already forces ρ ≥ 5; the exact branch-and-bound of rect_cover.py gives",
            "ρ = 8 (no 7 rectangles cover), so a product cover certifies only SC ≤ 7 — an",
            "explicit 8-domain one (216+180+108+108+102+78+48+24 facets) is the companion",
            "certificate results/cube_starcover.  Any 3-domain (sharp) certificate must come",
            "from the search, as for the wedge of two circles.",
        ],
        nf_note="= 12 x 12 triangle pairs x 6 staircase paths per pair",
    )


def main():
    name, resdir, label = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    if name == "wedge":
        spec = wedge_spec()
    elif name == "bary":
        spec = bary_spec()
    elif name == "cube":
        spec = cube_spec()
    else:
        spec = sphere_spec(int(name[1:]))
    nv, facets, npath = spec["nv"], spec["facets"], spec["npath"]
    nf = len(facets) ** 2 * npath        # staircase facets of K x K
    lines = []
    w = lines.append
    w("# Definition of the complex K and of the product K x K")
    w("")
    w(f"Result: {resdir.name}   ({label})")
    w(f"Complex: {spec['title']}")
    w("This is exactly the definition used by the code that produced")
    w("cover.txt / planners.txt in this directory (`build(\"%s\")` and the" % name)
    w("`Product` class in gpu_sc.py; star_cover.py imports the same definitions).")
    w("The auditor independent_check.py rebuilds K x K with these same")
    w("conventions, so this file is what you need to replicate the run or to")
    w("re-check the certificate by hand.")
    w("")
    w("## 1. The complex K")
    w("")
    w(f"Vertices: V = {{0, ..., {nv - 1}}}  (nv = {nv})")
    w("")
    w(spec["facets_intro"])
    w("")
    for i, f in enumerate(facets):
        w(f"  facet {i}: [{', '.join(str(v) for v in f)}]")
    w("")
    for line in spec["topology"]:
        w(line)
    w("")
    w("## 2. Vertex indexing of the product K x K")
    w("")
    w(f"The ordered (staircase) product lives on the {nv} x {nv} grid V x V.")
    w("Grid vertex (a, b) has id")
    w("")
    w(f"      id(a, b) = {nv}*a + b")
    w("")
    w(f"and the projections are pi1(id) = a, pi2(id) = b.  In cover.txt each")
    w(f"vertex of a facet of K x K is printed back as the pair ({nv}*a+b -> (a,b)).")
    w("")
    w("## 3. Facet enumeration of K x K (facet ids)")
    w("")
    w("Facet ids run over: first coordinate facet i of K (order of Section 1),")
    w("then second coordinate facet j of K, then all staircase paths over")
    w("(facet_i, facet_j) ordered by INCREASING BITMASK.  A path is a mask of")
    w("p + q bits, where p = dim(facet_i), q = dim(facet_j), with exactly q")
    w("bits equal to 1; bits are read from the lowest, and bit b is")
    w("")
    w("    0 = advance one vertex in the FIRST factor (facet_i),")
    w("    1 = advance one vertex in the SECOND factor (facet_j).")
    w("")
    w("The facet is the vertex list that starts at (facet_i[0], facet_j[0]) and")
    w(f"walks according to the mask, so it has p + q + 1 vertices ({spec['maxfacet']} here).")
    w("")
    w(f"Total facets of K x K: {nf} = {len(facets)} x {len(facets)} facet pair(s) x")
    w(f"{npath} staircase path(s) per pair {spec['nf_note']}.")
    w("")
    w("## 4. Format of cover.txt and planners.txt")
    w("")
    w("cover.txt: per domain, `facet <id>: (a,b) ...` = the vertex list of facet")
    w("<id> of K x K in the convention of Sections 2-3.")
    w("")
    w("planners.txt: per domain, the reduced contiguity chain map 0, map 1, ...")
    w("starting at pi1 and ending at pi2.  Each map is printed as a nv x nv")
    w("table: row a, column b is the value (a vertex of K) of the map at the")
    w(f"grid vertex id = {nv}*a + b.  `@` marks grid vertices NOT in the domain")
    w("(the map's values there are irrelevant).")
    w("")
    w("## 5. Rebuild and re-verify")
    w("")
    w("    python3 independent_check.py <this directory>   # exit 0 iff all checks pass")
    w("")
    resdir.mkdir(parents=True, exist_ok=True)
    (resdir / "definition.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {resdir}/definition.md  ({name}: {nv} vertices, {nf} facets of K x K)")


if __name__ == "__main__":
    main()
