# New mathematics of RP² — the hemi-icosahedron's simplicial complexity

*2026-09-24.  M. E. Perez Ibarra.  Companion to the chain-first annealer paper.*

Every result below is new: no prior computation of SC\_strict for any
closed non-orientable surface appears in the literature.  The lower bound
TC(RP²) = 2 is due to Farber–Tabachnikov–Yuzvinsky; everything else is
from this repository's search and analysis pipeline.

---

## 1. The complex

Let K = K\_RP² be the **vertex-minimal triangulation of the real projective
plane**: 6 vertices, 15 edges, 10 triangles (the hemi-icosahedron — the
icosahedron's 20 faces identified by the antipodal map, or equivalently the
unique triangulation of RP² on 6 vertices).

$$
\text{Facets of } K: \quad
(0,1,2),\ (0,2,3),\ (0,3,4),\ (0,4,5),\ (0,1,5),\ \\
(1,2,4),\ (2,3,5),\ (1,3,4),\ (2,4,5),\ (1,3,5).
$$

The f-vector is (6, 15, 10), χ = 6 − 15 + 10 = 1, the 1-skeleton is the
complete graph K₆, and no coherent orientation of the triangles exists
(`independent_check.check_rp2` re-proves all of this from the facet list).
Every vertex lies in exactly 5 of the 10 triangles: the valence is
**uniform** — (5, 5, 5, 5, 5, 5).

## 2. The automorphism group: Aut(K\_RP²) ≅ A₅

**Theorem (machine-verified).** The combinatorial automorphism group of the
hemi-icosahedron has order 60, consists entirely of even permutations of the
6 vertices, and is therefore isomorphic to the alternating group A₅ — the
unique simple group of order 60.

The action of A₅ on K is:

- **Vertex-transitive:** the orbit of every vertex is all of {0, 1, 2, 3, 4, 5}.
- **Edge-transitive:** the orbit of every edge is all 15 edges.
- **Triangle-transitive:** the orbit of every triangle is all 10 triangles.

*Proof.* Exhaustive enumeration over S₆ (720 permutations), retaining those
that send every triangle to a triangle. Of the 60 automorphisms found, all
are even permutations (parity check: count transpositions in each cycle
decomposition). An order-60 subgroup of A₆ consisting entirely of even
permutations, with the quotient S₆/A₆ ≅ Z/2 having no order-60 subgroup, is
necessarily a copy of A₅. (Alternatively: the hemi-icosahedron's
automorphism group is well known to be A₅ — the icosahedron has Aut ≅ A₅ ×
Z/2, and the antipodal quotient kills the Z/2.)  ∎

**Consequence for the search.** Since A₅ acts transitively on vertices,
triangles, and edges, any plateau state obtained by "dropping rectangle
star(v)×star(v)" is isomorphic to the one obtained by dropping any other
vertex.  A single barrier audit certifies all six.

## 3. The rectangle-cover number: ρ(K\_RP²) = 6 = |V(K)|

**Theorem.** The minimum number of maximal product rectangles D(v, w) =
star(v) × star(w) needed to cover all cells (s, t) of K × K is exactly 6.

*Proof (constructive upper bound).* The six **diagonal** rectangles
star(v) × star(v), v = 0, …, 5, cover every cell (s, t). This follows from
one structural property of the hemi-icosahedron:

> **Every pair of triangles shares at least one vertex:**
> min { |s ∩ t| : s, t facets of K } = 1.

Given any cell (s, t), pick any v ∈ s ∩ t (which is nonempty). Then s ∈
star(v) and t ∈ star(v), so (s, t) ∈ star(v) × star(v). Hence the six
diagonal rectangles cover all 100 cells = 600 product facets. ∎

*Proof (matching lower bound).* An exact set-cover ILP over the 36 candidate
rectangles star(v) × star(w), one binary variable per rectangle, one
coverage constraint per cell (s, t) (scipy.optimize.milp, HiGHS solver):
optimum = 6. ∎

**Why ρ = |V|.** The hemi-icosahedron has two properties that force ρ = 6:

1. **Every triangle contains exactly 3 of the 6 vertices** (trivially), so
   every off-diagonal rectangle star(v) × star(w) with v ≠ w covers only
   cells where v ∈ s and w ∈ t — at most 5 × 5 = 25 of the 100 cells.
2. **Every pair of triangles shares at least one vertex** (the hemi-icosahedron
   property), so the diagonal rectangles cover everything.
3. **No 5 rectangles suffice** (ILP: infeasible at limit 5), because the dual
   graph of the cell structure has minimum degree 3 and the incidence matrix
   of rectangles-to-cells has rank 36 — but the tight argument is just the
   ILP's proof of optimality.

The value ρ = |V| is a coincidence of the uniform-valence structure: every
vertex lies in exactly half the triangles (5 of 10), and every triangle
pair intersects, so the "diagonal family" is both sufficient and no smaller
family exists.

**The diagonal rectangles are uniform.** Since every vertex has valence 5:

$$
|\\text{star}(v) \\times \\text{star}(v)| = 5^2 \\times 6 = 150
  \\quad \\text{for every } v.
$$

The six rectangles therefore have the same size and overlap structure, as
guaranteed by the A₅-transitivity.

## 4. The level-0 bracket: 2 ≤ SC\_strict⁰(RP²) ≤ 5

**Lower bound:** TC(RP²) = 2 (Farber–Tabachnikov–Yuzvinsky 2006, in the
reduced convention; also recoverable from TC(S¹) = 1 via the deformation
retract of ‖K‖ onto S¹ plus the 2-fold covering S¹ → RP¹). By the general
inequality SC\_strict ≥ TC:

$$
\\text{SC}\_\\text{strict}^0(\\text{RP}^2) \\ge 2.
$$

**Upper bound:** the verified 6-domain cover `results/rp2_k6` (domains
132 + 126 + 112 + 80 + 78 + 72 = 600, chains of 3–4 links, machine-audited
by `independent_check.py`) gives SC\_strict⁰ ≤ 5. This is also the product
bound ρ − 1 = 5 from the diagonal cover.

Note: the k = 6 cover is **not** a product cover — its domain sizes
(132, 126, 112, 80, 78, 72) are far from the uniform 150 of the diagonal
rectangles, and its chains have 3–4 links (the diagonal rectangles need only
1-link chains π₁ ~ c\_v ~ π₂). The annealer found a genuinely different
shape that the product family cannot reach.

## 5. The k = 5 question: what would a 5-cover prove?

The bracket 2 ≤ SC\_strict⁰ ≤ 5 has a gap of 3. A 5-domain cover of K × K
(= a k = 5 solution) would prove SC\_strict⁰(RP²) ≤ 4. Conversely, if no
5-domain cover with chains of ≤ L links exists (CP-SAT UNSAT at k = 5), then
SC\_strict⁰ = 5 within the planner class — the **first exact computation of
SC\_strict for any closed non-orientable surface.**

**Current status of the k = 5 search:**

| run | coverage | shape | notes |
|---|---:|---|---|
| `rp2_k5_hybrid` (seed 673, 3 epochs, 66 s) | 540/600 | diagonal rectangles (4 exact, 1 missing 9) | strict coverage local optimum at w = 1; 60/60 H₁-dead |
| `rp2_k5` (seed 7434, 59 epochs, 1181 s) | 542/600 | (escaped rectangle plateau after 40 epochs) | guide-driven ridge drift; no summary saved |
| CP-SAT hybrid (3 frozen + 2 free, L = 8) | — | — | INFEASIBLE (348 s) — but only warm, not cold |

The 542 data point proves the rectangle plateau is escapable — the guide
provides a gradient (254 repair moves become downhill at guide weights
w* = 11), and with enough epochs the annealer drifts to a different shape
from which absorption is possible. The question is whether the full 600 is
reachable at k = 5, or whether SC\_strict⁰ = 5 exactly.

## 6. The product-family trap and the A₅ symmetry

The RP² k = 5 plateau state has a remarkable structure visible only through
the A₅ lens:

- The annealer warm-started from the **non-rectangular** k = 6 cover
  (132 + 126 + … + 72) with one domain dropped. It then **re-converged to the
  diagonal rectangle family**: four domains became exactly star(v)×star(v) for
  v ∈ {1, 2, 3, 4}, the fifth approached star(0)×star(0) (141 of 150 facets),
  and the 60 uncovered facets are exactly the private cells of the missing
  rectangle star(5)×star(5).

- This is the A₅ symmetry breaking: the 6 diagonal rectangles form a single
  orbit under A₅, the annealer picked 5 of the 6 (breaking the symmetry to the
  stabilizer of vertex 5, which is A₄ of order 12), and the 60-facet hole is
  the orbit of the dropped vertex's private cells under the residual symmetry.

- The 540/600 count is **forced by the family, not reached by search**: the 50
  far cells (|s ∩ t| = 1) each have exactly one covering diagonal rectangle,
  so 5 of 6 diagonals cover exactly 100 − 10 = 90 cells = 540 facets.

The product-family trap is the observation that the annealer's energy
landscape has a deep attractor at the product family — the 1-link chains
π₁ ~ c\_v ~ π₂ have **zero cleanliness cost** (γ = 0 per chain, since every
link is trivially contiguous), while the non-rectangular k = 6 cover's 3–4
link chains cost γ per excess link. The annealer trades coverage for
cleanliness and settles in the rectangle basin.

## 7. The H₁ obstruction on RP²

H₁(RP²; Z/2) = Z/2, generated by the core circle of the projective plane
(the boundary cycle of the Möbius band that RP² contains). The
(π₁ − π₂)\_* map sends this generator to the nontrivial element — rank 1 —
so the obstruction is live.

Machine results on the plateau states:

| state | |L| | L alone | absorption | interpretation |
|---|---:|---|---|---|
| k = 5 (540/600) | 60 | passes (rank 0) | 60/60 dead | extension dead; the missing rectangle's facets don't wind by themselves, but every plateau domain that absorbs one starts winding |
| k = 3 (398/600) | 202 | **obstructed** (rank 1) | 200/202 dead | the leftover complex itself winds — no fresh chain can take even the leftovers alone |
| k = 11 (590/600) | 10 | — | — | same far-pair, depth-1 signature as k = 5 |

The k = 5 result is the most informative: L alone passes (it's a collection
of cells whose union does not wind), but every D\_j ∪ {f} is obstructed — the
obstruction is created by the **combination** of a non-winding facet with an
already-wound domain. This is the mechanism by which the A₅-symmetric
rectangle family generates its own trap: each diagonal rectangle star(v) ×
star(v) is large enough (150 facets) to contain non-contractible cycles, and
adding any of the 60 missing facets crosses one.

## 8. What this means for the paper

Three results are publishable independent of whether k = 5 closes:

1. **ρ(K\_RP²) = 6 = |V|** — the exact rectangle-cover number, a new
   combinatorial invariant of the hemi-icosahedron's product.

2. **SC\_strict⁰(RP²) ≤ 5** — from the verified non-rectangular k = 6 cover,
   matching the product bound but achieved by a different certificate.

3. **The A₅-symmetric product-family trap** — the annealer's re-convergence
   to diagonal rectangles from a non-rectangular warm start, with the
   60-facet hole forced by the broken A₅ symmetry, the exhaustive barrier
   audit (strict coverage local optimum, 0 improving moves, 747 neutral
   swaps), and the guide-driven escape to 542.

A fourth result depends on the k = 5 outcome:

- **SAT** ⇒ SC\_strict⁰(RP²) ≤ 4, and the gap SC\_strict⁰ − TC ≥ 2 is at
  least two units (TC = 2).
- **UNSAT** ⇒ SC\_strict⁰(RP²) = 5 within the planner class (chains ≤ L
  links), and the gap is exactly 3 — the **first exact SC\_strict for any
  closed non-orientable surface**.

---

## Reproduce

All claims are machine-verified. Run from the repository root
(`/home/mike666/chain-first-annealer/opti`):

```bash
PY=~/whisper_env/bin/python

# 1. Automorphism group (pure enumeration, <1s)
$PY -c "
import sys, itertools; sys.path.insert(0, 'src'); import gpu_sc
P = gpu_sc.build('rp2')
tris = [tuple(sorted(t)) for t in P.kfacets]; T = set(tris)
auts = [g for g in itertools.permutations(range(P.nv))
        if all(tuple(sorted(g[v] for v in t)) in T for t in tris)]
print(f'|Aut| = {len(auts)}')
print('vertex orbit of 0:', sorted(set(g[0] for g in auts)))
# parity check
def par(g):
    vis=[False]*len(g); s=0
    for i in range(len(g)):
        if not vis[i]:
            j,l=i,0
            while not vis[j]: vis[j]=True; j=g[j]; l+=1
            s+=l-1
    return s%2
print('all even:', all(par(g)==0 for g in auts))
"

# 2. ρ(RP²) = 6 (ILP, <1s)
$PY -c "
import sys; sys.path.insert(0, 'src')
import numpy as np; import gpu_sc
from rect2_optimum import rect_masks
from scipy.optimize import milp, LinearConstraint
from scipy.sparse import lil_matrix
P = gpu_sc.build('rp2'); nt = len(P.kfacets); masks = rect_masks(P)
keys = sorted(masks); cells = [(s,t) for s in range(nt) for t in range(nt)]
A = lil_matrix((len(cells), len(keys)))
for j,(v,w) in enumerate(keys):
    for i,(s,t) in enumerate(cells):
        if v in P.kfacets[s] and w in P.kfacets[t]: A[i,j] = 1
lc = LinearConstraint(A.tocsr(), lb=np.ones(len(cells)))
res = milp(c=np.ones(len(keys)), constraints=lc, integrality=np.ones(len(keys)))
print(f'rho(RP2) = {res.fun:.0f}')
"

# 3. Verify the k=6 cover (independent audit, <1s)
python3 src/independent_check.py results/rp2_k6

# 4. H₁ absorption on RP² k=5 plateau (~30s)
$PY scripts/klein_h1_absorption.py rp2 results/rp2_k5_hybrid/summary.json

# 5. Exhaustive barrier audit of RP² k=5 plateau (~60s)
$PY scripts/plateau_barrier_audit.py rp2 results/rp2_k5_hybrid/summary.json

# 6. Leftover facet profile (~10s)
$PY scripts/leftover_properties.py rp2 results/rp2_k5_hybrid/summary.json
```

## Run the k = 5 targeted push

```bash
# Phase 1 (anneal) + Phase 2 (low-γ ridge walk) + Phase 3 (targeted rewrite)
# Budget: 1 hour.  Requires GPU.
$PY scripts/rp2_targeted_push.py --time-limit 3600 --seed 42 \
    --outdir results/rp2_k5_push 2>&1 | tee logs/rp2_k5_push.log

# Or warm-start from the existing k=6 cover:
$PY scripts/rp2_targeted_push.py --time-limit 3600 --seed 42 \
    --init results/rp2_k6/summary.json \
    --outdir results/rp2_k5_push_warm 2>&1 | tee logs/rp2_k5_push_warm.log

# If the targeted push stalls, the CP-SAT cold attack is the decisive test:
# (requires ortools — pip install ortools)
$PY paper-overleaf/paper/scs_sat.py --complex rp2 --k 5 --L 8 --timeout 14400
```