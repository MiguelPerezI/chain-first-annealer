# Definition of the complex K and of the product K x K

Result: bary_k3   (barycenter triangle, chain-first annealer)
Complex: the barycenter triangle  (Δ² split by its barycenter 3; ‖K‖ ≅ Δ²)
This is exactly the definition used by the code that produced
cover.txt / planners.txt in this directory (`build("bary")` and the
`Product` class in gpu_sc.py; star_cover.py imports the same definitions).
The auditor independent_check.py rebuilds K x K with these same
conventions, so this file is what you need to replicate the run or to
re-check the certificate by hand.

## 1. The complex K

Vertices: V = {0, ..., 3}  (nv = 4)

Facets (maximal simplices), in the order used by the code — the three
triangles of the triangle {0,1,2} after adding the barycenter 3:

  facet 0: [0, 1, 3]
  facet 1: [0, 2, 3]
  facet 2: [1, 2, 3]

K has vertices {0,1,2,3} where {0,1,2} is the outer triangle and 3 is the
barycenter, joined to all three of its vertices.  2-faces: 3 triangles;
1-faces: all 6 edges of K₄ (the 3 triangles have 9 edge slots, the 3 inner
edges {0,3},{1,3},{2,3} shared by two triangles each, so 9 − 3 = 6 edges);
f-vector (4, 6, 3), χ = 4 − 6 + 3 = 1: a disk, the triangulation of a triangle.
Vertex 3 lies in EVERY facet, so K is the cone on ∂Δ²; in particular ‖K‖ is
contractible, the case in which — as noted right before the definition of
SC_strict in arXiv:2008.13290 — a motion planner exists on all of K x K.
The constant map c₃ and the chain π₁ ∼ c₃ ∼ π₂ are therefore valid on every
facet of the product, giving a one-domain system (SC_strict = 0), and the
whole search problem is to discover c₃ (or an equivalent middle map).

## 2. Vertex indexing of the product K x K

The ordered (staircase) product lives on the 4 x 4 grid V x V.
Grid vertex (a, b) has id

      id(a, b) = 4*a + b

and the projections are pi1(id) = a, pi2(id) = b.  In cover.txt each
vertex of a facet of K x K is printed back as the pair (4*a+b -> (a,b)).

## 3. Facet enumeration of K x K (facet ids)

Facet ids run over: first coordinate facet i of K (order of Section 1),
then second coordinate facet j of K, then all staircase paths over
(facet_i, facet_j) ordered by INCREASING BITMASK.  A path is a mask of
p + q bits, where p = dim(facet_i), q = dim(facet_j), with exactly q
bits equal to 1; bits are read from the lowest, and bit b is

    0 = advance one vertex in the FIRST factor (facet_i),
    1 = advance one vertex in the SECOND factor (facet_j).

The facet is the vertex list that starts at (facet_i[0], facet_j[0]) and
walks according to the mask, so it has p + q + 1 vertices (4 here).

Total facets of K x K: 54 = 3 x 3 facet pair(s) x
6 staircase path(s) per pair = 3 x 3 triangle pairs x 6 staircase paths per pair.

## 4. Format of cover.txt and planners.txt

cover.txt: per domain, `facet <id>: (a,b) ...` = the vertex list of facet
<id> of K x K in the convention of Sections 2-3.

planners.txt: per domain, the reduced contiguity chain map 0, map 1, ...
starting at pi1 and ending at pi2.  Each map is printed as a nv x nv
table: row a, column b is the value (a vertex of K) of the map at the
grid vertex id = 4*a + b.  `@` marks grid vertices NOT in the domain
(the map's values there are irrelevant).

## 5. Rebuild and re-verify

    python3 independent_check.py <this directory>   # exit 0 iff all checks pass

