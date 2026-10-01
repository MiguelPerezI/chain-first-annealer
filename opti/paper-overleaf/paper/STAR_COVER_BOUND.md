# The star-rectangle bound: why it always covers, why it is only an upper bound, and why it is sharp on the even spheres

Companion note to `main.tex` §5 (Theorem `thm:prod`, Corollaries `cor:rho`,
`cor:spheres`; section `sec:products`) and the results of §6 — written
2026-09-29.  Three questions
are answered here, with every number verified against the repository
(`opti/src/star_cover.py`, `opti/src/rect_cover.py`,
`opti/results/*_starcover*/summary.json`):

1. **Mechanism** — how does `star_cover.py` always produce a *complete,
   verified* system, for any connected K, in seconds and without search?
2. **Slack** — ρ(K) is computed exactly, so why is ρ(K)−1 only an *upper*
   bound on SCst(K), and where exactly does the inequality live?
3. **Sharpness** — why is the bound nevertheless *the* minimum on the even
   spheres, provably, while it is provably not on the circle, the wedge of
   two circles, the Möbius band, the cube boundary, the hemi-octahedron or
   the icosahedron?

Notation follows the paper: SCst(K) = SD(π₁,π₂), a cover of K×K by k
contiguity domains certifies SCst(K) ≤ k−1, TC is Farber's invariant in the
reduced convention, ρ(K) is the minimum number of rectangles
D(v,w) = star(v) × star(w) covering all facet pairs of K.

## 1. Mechanism: every rectangle is a certified contiguity domain

`star_cover.py` turns a vertex pair (v, w) into the chain

    π₁ ~ c_v ~ c_w ~ π₂        (c_x = constant map; if v ≁ w, the
                                constants of a BFS shortest path are
                                interpolated — `rectangle_chain`)

and claims D(v,w) ⊆ D(Φ) for that chain Φ.  The verification is pure
simplex membership — no search.  Take any staircase facet σ over a facet
pair (s, t) with v ∈ s, w ∈ t.  Then:

* π₁(σ) ∪ c_v(σ) = s ∪ {v} = **s** — a simplex, because v ∈ s is exactly
  the defining condition of the rectangle;
* c_v(σ) ∪ c_w(σ) = {v, w} — an edge if v ~ w; otherwise each pair of
  consecutive path constants gives {xᵢ, xᵢ₊₁}, an edge of the 1-skeleton
  by construction of the shortest path.  This is where connectedness of K
  is used (Theorem `thm:prod`): the BFS path must exist;
* c_w(σ) ∪ π₂(σ) = {w} ∪ t = **t** — a simplex because w ∈ t.

Every link is 1-contiguous on every facet of the rectangle, and
`gpu_sc.verify_system` re-audits the assembled system afterwards.  This is
why the construction is unconditional and instant: contiguity reduces to
"is the vertex in the facet."

## 2. Completeness: the rectangle family covers K×K for free

Completeness of the *family* (before any selection) is trivial: every
facet pair (s, t) has some v ∈ s and some w ∈ t (facets are nonempty), so
(s, t) ∈ D(v, w).  The full |V|² grid of rectangles covers all |F|² pairs
of K×K, which is why `build()` can `assert covered.all()` — a star run
cannot fail to produce a complete cover.  ρ(K) is then the *minimum
subcover*: a set-cover over |V|² candidate sets and |F|² elements, solved
exactly by the bitmask branch-and-bound of `rect_cover.py` (the paper also
reports HiGHS ILP optimality certificates for the cube and the
hemi-octahedron; no HiGHS run records live in this repository — the
docstring's "integer program of rect_cover.py / scipy's HiGHS" misattributes
the tool, rect_cover.py contains no ILP).

Two consequences:

* **ρ is exact.**  The set-cover step has no slack: no system of fewer
  than ρ(K) star rectangles covers K×K.
* **The bound is ρ(K) − 1**, since ρ rectangles give ρ domains and
  k domains certify SCst ≤ k−1 (Corollary `cor:rho`).

So the entire distance between ρ(K)−1 and SCst(K) lives in one place: the
restriction to the rectangle family.  The next two sections say why that
restriction can cost domains, and when it costs none.

## 3. Why the bound is not the minimum in general

SCst(K) minimises over **all** contiguity chains: k-tuples of arbitrary
vertex maps Φ¹,…,Φᵏ on the whole grid whose domains ∪ᵢ D(Φᵢ) cover F_{K×K}.
A general domain D(Φ) is whatever shape its chain carves out.  The star
construction restricts two things at once:

* **Domain shapes** — only axis-aligned products star(v) × star(w): |V|²
  candidate domains out of an astronomical space of shapes;
* **Chains** — only constant-type chains (π₁, c_{x₀}, …, c_{x_m}, π₂).
  Homotopically these factor through a point: they are the most degenerate
  chains possible.  A general chain's intermediate maps are nonconstant
  and may vary from facet to facet.

A minimum over a subfamily is ≥ the true minimum, and the gap opens along
two structurally distinct axes.

### 3.1 Winding (topology)

A rectangle anchored at constant maps cannot wrap around a cycle or a
handle; a constant chain is homotopically trivial.  Complexes with
nontrivial H₁ admit covers whose chains *wind*, using fewer domains than
any rectangle cover can express:

* **Circle** (∂Δ²): ρ = 3 (`results/circle_starcover`: 8+6+4 facets ⇒
  SCst ≤ 2), but the paper's two-domain winding cover certifies
  SCst(S¹) = 1 = TC.  The minimal example of the family overcounting.
* **Möbius band**: ρ = 4 (`results/mobius_starcover_opt`: 54+36+36+24 ⇒
  SCst ≤ 3), but the CP-SAT cover of 3 non-product domains
  (`results/mobius_k3_sat`, chains of 5–6 links) certifies SCst ≤ 2, and
  the k = 2 CP-SAT instance is UNSAT at chain length L = 6 — so SCst = 2
  *within the bounded planner class* (main.tex is careful to say "within
  chains of up to 8 links"; no unbounded-chain impossibility proof exists).
  "The product family can be a domain short" (`star_cover.py` docstring).
* **RP²** (hemi-icosahedron): ρ = 6 = |V| (exact set-cover computation), so
  *no* 5-rectangle system exists.  More sharply, the best 5-rectangle cover
  misses exactly 10 cells = 60 facets — exhaustive over the C(36,5) = 376,992
  rectangle choices, attained by any 5 of the 6 diagonal rectangles (each
  far cell |s ∩ t| = 1 is private to one diagonal) — and whether a general
  5-domain cover exists (⇒ SCst ≤ 4) is exactly the open decisive
  experiment; the bracket is 2 ≤ SCst⁰ ≤ 5 (lower: TC(RP²) = 2, reduced
  convention).

### 3.2 Sparsity (area)

A rectangle covers |star(v)|·|star(w)| of the |F|² facet pairs.  Low
valence means skinny stars, so ρ inflates structurally no matter how
cleverly the exact cover selects; general domains are not area-limited and can be
arbitrary blobs.  The size of ρ tracks how much of K a facet pair can
miss:

* **Simplex boundaries** — every facet misses one vertex, so a pair (s,t)
  shares all but ≤ 2 vertices; three fixed diagonals star(a)×star(a),
  star(b)×star(b), star(c)×star(c) hit every pair (one of the three
  survives in s ∩ t).  Hence ρ(∂Δ^{n+1}) = 3 for all n ≥ 1
  (Corollary `cor:spheres`; verified `results/s3` … `s8_starcover`, e.g.
  S⁸: 1,042,470 + 218,790 + 25,740 of 1,287,000 facets).
* **Hemi-polyhedra** — only the hemi-icosahedron has all facet pairs
  meeting, so the six diagonal rectangles cover (ρ ≤ 6) and the ρ = |V|
  proposition of the paper (Proposition `prop:hemi`, which concerns
  *exactly three* RP² triangulations) gives ρ(hemi-icosa) = 6,
  ρ(hemicube) = 7, ρ(hemi-octa) = 9.  The diagonal mechanism does *not*
  cover the other two: the stored hemicube and hemi-octa triangulations
  have 6 and 36 disjoint facet pairs, and their optimum covers use
  off-diagonal rectangles (6 of 7 for the hemicube; the |S|² = 9
  corner/midpoint grid for the hemi-octa, Lemma `lem:covers` second
  clause).  The fourth hemi-polyhedron is not a ρ = |V| case at all:
  ρ(hemi-dodeca) = 15 > 10 = |V|, with 52 disjoint facet pairs — see the
  sparse bullet.
* **Sparse complexes** — many pairs share no vertex, off-diagonal
  rectangles are needed and ρ can exceed |V|: cube 8 = |V|, Császár torus
  11 > 7, Klein bottle 12 > 8 (the stored Klein is the 8-vertex, 16-triangle
  Cervone triangulation), hemi-dodecahedron 15 > 10, icosahedron 24 > 12
  (opposite faces are disjoint; 100 disjoint facet pairs), dodecahedron
  36 > 20 (stored-triangulation stars of size 3–8, dual-graph diameter 7,
  triangle density 36/C(20,3) ≈ 3.2%).

On the sparse complexes the product bound is far from everything known:
the icosa star certificate (ρ = 24 ⇒ SCst ≤ 23) seeds the annealer via
`best_chains` → `--init`, and the star-seeded ladder closed every rung
k = 23 down to k = 5 (`results/icosa_k5_v2`: a verified 5-domain
certificate over the 2400-facet KxK, domains of 894+755+387+185+179
facets, reached at epoch 67 of the warm continuation from the first
k = 5 run's 2395/2400 stall) ⇒ **SCst(icosa) ≤ 4** — the annealer beat
its own seed's family by nineteen domains and closed to within two of
the topological floor: TC(S²) = 2 forces 2 ≤ SCst(icosa) ≤ 4.  On the
dodecahedron the star bound (≤ 35) remains the only certified bound;
the k = 3 campaign stalled at ~2602/7776 facets.

## 4. When the bound *is* the minimum: the even-sphere sandwich

For every simplex boundary the star bound is 2 (ρ = 3).  On the **even**
spheres the sandwich closes from below:

    TC(S^{2m}) = 2  ≤  SCst(∂Δ^{2m+1})  ≤  ρ − 1 = 2    ⟹    SCst = 2,

certified from both sides — this is Corollary `cor:spheres` ("sharp at
subdivision level 0").  The star cover is minimal there *not because the
product family is exact in general*, but because the topological floor —
Farber's TC(S^{2m}) = 2, forced by χ(S^{2m}) ≠ 0: no motion planner on an
even sphere has fewer than 3 sections — happens to sit exactly at the
crudest constructive family's ceiling.  On a simplex boundary the stars
are maximally fat (each vertex lies in all facets except the opposite
one), so three rectangles saturate a bound that nothing, product or not,
can beat.

The two neighbouring cases show this is a coincidence of the complex, not
a property of the method:

* **Odd spheres** (∂Δ⁴ = S³): TC = 1 < 2 = star bound, so the sandwich
  does not close.  Topologically a 2-domain planner exists (antipodal
  trick), but realising it simplicially requires a winding band around
  the antidiagonal of K×K, and the k = 2 searches pin at exactly
  460/500: the circle's winding mechanism needs a cyclic order in the
  facet structure, which the K₅ 1-skeleton of ∂Δ⁴ does not have.  So the
  star cover is still *conjecturally* minimal (the paper's Conjecture (C1)),
  but the support is only search exhaustion — the Conjecture itself warns
  that "a failed search is not one"; the SCst − TC gap
  is a discretisation cost, not a star-family cost.
* **The circle** (∂Δ² = S¹): TC = 1 *and* the facet structure is a
  3-cycle, so winding is expressible — the 2-domain cover exists and
  beats ρ − 1 = 2.  Here the family restriction provably costs a full
  domain.

## 5. Where the numbers stand

Star bounds vs. best certified SCst, all verified against the repository
(ρ values exact by set-cover branch-and-bound — re-verified independently
here for the circle, Möbius, RP², cube, hemicube and hemi-octahedron;
for the Császár torus the same computation proves ρ ≥ 10 but did not
finish the limit-10 infeasibility search, so ρ = 11 there rests on the
covering certificate plus the paper's solver; star domains from
`results/*_starcover*`; non-star bounds from the annealer/CP-SAT records):

| K | ρ | star bound ρ−1 | best known SCst | star sharp? |
|---|---|---|---|---|
| ∂Δ³, ∂Δ⁵, ∂Δ⁷, ∂Δ⁹ (S², S⁴, S⁶, S⁸) | 3 | 2 | 2 = TC | **yes — sandwich closes** |
| ∂Δ⁴ = S³ | 3 | 2 | 2 > TC = 1 (Conjecture C1; 460/500 plateau) | conjecturally yes |
| ∂Δ² = S¹ | 3 | 2 | 1 = TC (winding cover) | no — overcounts by a domain |
| wedge S¹∨S¹ | 4 | 3 | 2 = TC (annealer, 3 domains) | no — a domain short |
| cube boundary (S²) | 8 | 7 | 2 = TC (annealer, 3 domains) | no — gap of 5 |
| Möbius band | 4 | 3 | ≤ 2 (CP-SAT, 3 domains); = 2 within chains ≤ 8 links | no — a domain short |
| hemi-octahedron (RP² family) | 9 = \|V\| | 8 | ≤ 7 (k = 8 certificate) | no — beaten by 1 |
| hemicube (RP² family) | 7 = \|V\| | 6 | ≤ 6 (k = 7 certificate; k = 6 stalls at 836/864) | bound attained; sharpness open |
| hemi-icosahedron (RP²) | 6 = \|V\| | 5 | 2 ≤ SCst⁰ ≤ 5, k = 5 open | unknown (in-family 5 impossible) |
| hemi-dodecahedron | 15 > \|V\| = 10 | 14 | ≤ 14 (star only) | open |
| Császár torus | 11 | 10 | ≤ 10 (star only; TC(T²) = 2 below) | open |
| Klein bottle | 12 | 11 | ≤ 9 (verified k = 10; k = 9/8 stall at 1494/1470, k = 5 at 1186, of 1536) | open, gap likely large |
| icosahedron | 24 | 23 | ≤ 4 (star-seeded ladder; k = 5 closed) | no — beaten by 19 |
| dodecahedron | 36 | 35 | ≤ 35 (k = 3 stalled ~2602/7776) | no (expected), unproven |

Reading of the table: the star cover is the minimum exactly where a
certified lower bound rises to meet it (even spheres).  Where topology
demands less *and* winding is expressible (circle, wedge, Möbius), it
overcounts; where topology demands less but winding is not
expressible (odd spheres), it is *conjecturally* still minimal for a
combinatorial reason
the lower bound cannot see; where the complex is sparse, it is merely the
constructive starting point the annealer descends from.

## 6. Role in the pipeline

Even where not sharp, the star cover is load-bearing:

* **Guaranteed zone.**  ρ(K) marks the constructive boundary: k ≥ ρ
  domains exist by explicit certificate, so the search is only
  interesting below ρ.
* **Seeding.**  `summary.json` records the full-grid chains
  (`best_chains`), which `gpu_sc.py` / `annealer_stretch.py` read via
  `--init`, and `hybrid_cover.py` freezes the m largest star domains and
  anneals only the remainder.  Two of the table's sub-star records do
  descend from a star certificate this way — the icosahedron ladder
  (k = 23 seeded from `icosa_starcover_opt`, then warm-started down
  through k = 5) and the hemi-octahedron k = 8 certificate — but the
  cube, Möbius and Klein records are annealer/CP-SAT native (cold starts;
  the verified Klein ladder k = 15 → 10 originates in a cold run), so
  "star-seeded" is not a universal property of the improved bounds.
* **Verification.**  Every star system is re-audited internally by
  `gpu_sc.verify_system`.  The sphere runs (s2–s8) are additionally
  audited by this repo's code-independent `src/independent_check.py`
  (invoked from `scripts/run_starcovers.sh`), whose facet order matches
  the Abstract_Motion_GPU pipeline.  Since 2026-09-30 the checker also
  rebuilds the icosahedron (2400-facet KxK; it re-proves the identity
  from the facet list alone — the unique 5-regular triangulated sphere,
  antipodal quotient the hemi-icosahedron — and its planner-grid parser
  was fixed for the two-digit vertex labels that nv = 12 produces), and
  with it the entire icosahedron ladder passes the full independent
  audit: all twenty certificates, the `icosa_starcover_opt` seed (24
  domains) and every rung k = 23 … 5, including the SCst ≤ 4 record.
  The remaining small-complex `*_starcover_opt` certificates record
  `independent_audit: not_run` in their provenance, so for those the
  internal audit is currently the only one executed.

One-line summary: **complete by a trivial membership argument, minimal
within its family by exact set-cover computation, an upper bound because
its family (constants through points, products of stars) cannot bend or
wind, and provably equal to the true minimum only where the topological
lower bound rises to meet it — the even spheres (the odd spheres are
conjecturally sharp too, by the combinatorial mechanism of §4).**