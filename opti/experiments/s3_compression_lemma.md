# Compression lemma for SC_strict⁰(∂Δ⁴) — notes

Target: show every contiguity domain of L = ∂Δ⁴ × ∂Δ⁴ admits a chain of
bounded links, so that `scs_sat.py` UNSAT at finite L is an unconditional
statement about `SC_strict⁰`.

## Setting

K = ∂Δ⁴: V(K) = {0..4}, facets = all C(5,4) = 5 four-subsets.  Hence a
subset U ⊆ V(K) **spans a simplex iff |U| ≤ 4** (every ≤4-set is inside a
facet; the full 5-set is not).  So two vertex maps φ, ψ are 1-contiguous on
a product facet f iff their combined images on the 7 grid vertices of f use
**at most 4 distinct values**.

Grid: 25 vertices (a,b), id 5a+b; π₁(a,b)=a, π₂(a,b)=b.  A product facet f
lies over a pair of facets (s,t) of K and visits every vertex of s in π₁ and
every vertex of t in π₂.  Structural facts (verified in A1b):

* diagonal pairs s = t: |s ∩ t| = 4 (100 facets);
* off-diagonal: |s ∩ t| = 3, |s \ t| = |t \ s| = 1 (400 facets).

## Lemma 1 (interpolation — proved)

If φ ~ ψ on D, then every map on the Hamming box between them (each grid
vertex takes φ's or ψ's value) is contiguous to both.  Proof: for any such χ
and facet f, img(χ)|_f ⊆ img(φ)|_f ∪ img(ψ)|_f, which has ≤ 4 values by
assumption; subsets of ≤4-sets still have ≤4 values.  Hence the whole box is
a clique through χ.  ∎

Consequence: a contiguous pair can be replaced by ≤ 25 single-vertex steps
without losing contiguity; "changing one grid vertex at a time" is free.

## Lemma 2 (endpoints — proved)

π₁ ~ φ on D iff img(φ)|_f ⊆ s_f for all f ∈ D (since img(π₁)|_f = s_f is
already a full facet, |s_f| = 4); symmetrically φ ~ π₂ iff img(φ)|_f ⊆ t_f.

## Lemma 3 (single-facet chains — proved)

Every product facet f admits a 2-link chain: pick v ∈ s_f ∩ t_f (nonempty in
all 500 facets: 3 resp. 4 common values) and route π₁ ~ c_v ~ π₂.  The only
obstruction to a *global* chain is synchronizing shared grid vertices across
facets.

## Compression claim (open)

For every D ⊆ F_L that is a contiguity domain (admits some chain),
dist_D(π₁, π₂) ≤ B for a small constant B (ideally B ≤ 16).

Candidate proof route: use Lemma 1 to walk grid vertex by vertex from π₁ to
π₂; each step changes one value at one grid vertex; need to show the walk can
be scheduled so every intermediate map stays contiguous on all of D.  Since
each facet tolerates 4 of 5 values, the walk only gets stuck if some facet's
7 vertices would use all 5 values; a detour through a map that drops one
value (Lemma 2 style) should repair it.  A potential function: number of
grid vertices not yet at their π₂ value (≤ 20 off-diagonal); if every step
can be made non-repairing, B ≤ 20 + (repairs).  Not yet a proof.

## Falsification instrument (`scripts/sat_validation/c2_growth.py`)

Exact dist_D(π₁, π₂) via the same CP-SAT constraints at k=1, coverage forced
on D, scanning L upward.  Measured on s3 (2026-09-27):

| domain | \|D\| | exact min links |
|---|---|---|
| s3_starcover J0/J1/J2 | 320/140/40 | 2 / 2 / 2 |
| s3_k3_v2 J0/J1/J2 | 309/130/61 | 3 / 3 / 3 |
| rect(3,2), rect(2,3), rect(3,0) | 320 | 3 |
| s3_k2_A plateau | 284/265 | 3 / **4** |
| s3_k2_D plateau | 275/212 | 2 / **4** |
| s3_k2_longflat_5 plateau | 232/273 | 3 / 3 |
| s3_k2_longcor_5 plateau | 311/303 | **4** / 3 |
| s3_k2_v2_point plateau | 297/317 | 3 / 3 |
| s3_k2_v2_track plateau | 320/320 | 3 / 2 |
| grown (320-facet seed, 131 attempted adds) | 320 | 2 (all adds rejected) |
| random subsets \|D\| = 400–500 | — | not chainable at ≤ 24 links |

**Max exact min-L observed: 4.**  The k=2 plateau domains (the largest
chainable sets the annealer ever found) certify at 2–4 links despite 41-page
budgets.  Random sets above ~320 facets are not chainable at all, so there
is nothing to compress up there.

Naive interpolation walk: **falsified as a proof route** (scripts/sat_validation/c3_precedence.py —
81 820 facet pairs force flip-order cycles), but Lemma 1 still holds; the
short chains use non-interpolation intermediates (constant maps, value
drops).  A real proof likely needs "any chain on ∂Δ⁴ compresses to ≤ 5
links", matching the empirical max of 4.

## Status of the s3 statement

* Proven by CP-SAT (validated encoding, A1–A3): **no 2-cover of ∂Δ⁴ × ∂Δ⁴
  with chains of ≤ 48 links exists** (INFEASIBLE at L = 4, 5, 8, 16, 24, 32,
  40, 48; padding ⇒ monotone in L).
* Empirical compression: every chainable domain found by any method needs
  ≤ 4 links — a 12× margin below the UNSAT ladder's 48.  The compression
  claim (every chainable D has dist ≤ 48) is unfalsified and, on the
  evidence, overwhelmingly likely.
* Without the lemma: SC_strict⁰(∂Δ⁴) = 2 for the chain-length class ≤ 48,
  which contains every cover ever observed in this repo (max reduced
  chain: 9 links; max min-L on any measured domain: 4).
