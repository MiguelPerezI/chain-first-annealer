# RP² / Klein / Möbius stall analysis — properties of the leftover facets

*2026-09-24. Follow-up to `klein_anneal_log.md` (§3–§5): the same H₁
absorption battery and a new facet-property profiler
(`scripts/leftover_properties.py`) applied to the stalled RP² states, the
never-before-tested Klein k = 9 and k = 5 states, and the full eight-run
Möbius k = 3 seed set.
New exact values: **ρ(RP²) = 6**, ρ(Möbius) = 4 (ILP, below), ρ(Klein) = 12
(known, §1 of the klein log). All statements at subdivision level 0 and all
obstruction results are **state-relative** (they rule out completing the
recorded plateau states; they bound nothing from below — same status as C1).*

The RP² ladder: k = 11 → 6 closed (SC_strict⁰ ≤ 5, `results/rp2_k6` verified,
domains 132+126+112+80+78+72, chains 3–4 links — **not** the rectangle
cover); k = 5 stalls 540/600, k = 4 at 469/600 (killed before writing a
summary — no state on disk, not analyzable), k = 3 at 398/600.

## 1. ρ(RP²) = 6 and ρ(Möbius) = 4

Exact set-cover ILP over the 36 / 25 candidate rectangles star(v) × star(w)
(scipy/HiGHS, one constraint per cell (s, t), same encoding as the klein §1
computation):

- **ρ(RP²) = 6**, witness (v, w) = (0,5), (1,0), (2,1), (3,3), (4,2), (5,4).
  Also the six *diagonal* rectangles star(v) × star(v), v = 0..5, cover: the
  hemi-icosahedron's triangle pairs all intersect (|s ∩ t| ∈ {1, 2, 3}), so
  cell (s, t) is taken by any v ∈ s ∩ t. Both products certify SC_strict⁰ ≤ 5.
- **ρ(Möbius) = 4**, witness (1,4), (2,2), (3,4), (4,2) — matching the
  paper's "product covers reach only ≤ 3" (mobius_sc_strict.tex).

So the annealer's k-ladder on RP² closed every rung **down to ρ** and stalls
at k = ρ − 1 = 5; on the Möbius band it closed k = 4 = ρ and stalls at
k = ρ − 1 = 3. The Klein bottle (ρ = 12) never gets near its rectangle
number: no cover at k ≤ 8, and the k = 10 cover that was found crashed at
verification.

## 2. Properties of the RP² leftovers

`scripts/leftover_properties.py` decodes every uncovered facet into its cell
(s, t) and staircase path and measures: dead paths per cell, |s ∩ t|, dual
distance d(s, t), defect depth min_j bad_j[f], plateau-domain rectangle
identity, and rectangle membership.

### k = 5 (`results/rp2_k5_hybrid`, 540/600, L = 60)

The plateau state is a **pure product-family state one rectangle short**:

- The five domains are the diagonal rectangles: D_0 = star(1)×star(1),
  D_1 = star(4)×star(4), D_3 = star(2)×star(2), D_4 = star(3)×star(3),
  D_2 ⊂ star(0)×star(0) (missing 9 facets); sizes 150+150+141+150+150.
- The 60 leftovers are **entire cells, all 6 paths dead**: exactly the 10
  cells whose only diagonal covering vertex is 5 — i.e. the private cells of
  the missing rectangle star(5)×star(5). All have s ∩ t = {5}.
- All 60 are **far pairs**: |s ∩ t| = 1, dual distance 2 = the diameter of
  the hemi-icosahedron's dual (the Petersen graph).
- All 60 sit at **defect depth exactly 1** (min_j bad_j[f] = 1).
- H₁ absorption (below): 60/60 dead — no chain of the five can absorb any.
- Yet **L alone passes** (dim H₁ = 0): the missing rectangle's facets are
  exactly what a fresh sixth domain takes — consistent with the k = 6 cover
  that exists. The state was warm-started from that non-rectangular k = 6
  cover (5 largest domains) and **annealed back into the rectangle family**,
  trading 60 facets for the 1-link chains π₁ ~ c_v ~ π₂ of the diagonal
  rectangles.
- CP-SAT hybrid on this state (3 frozen + 2 free, L = 8): INFEASIBLE in 348 s
  (`logs/rp2_k5_sat.log`) — the plateau cannot be completed to a 5-cover by
  adding two L ≤ 8 domains.

**Is there a fixed area of RP² that won't budge?** No — and the distinction
matters. The 60 leftovers are the cells (s, t) with s ∩ t = {5}: pairs of
triangles touching at vertex 5 alone, a well-defined region of K × K. But
(a) that region is perfectly coverable — star(5) × star(5) covers exactly
it, and the verified k = 6 cover does; (b) the location is an accident of
the run (any of the six diagonal rectangles could be the dropped one — the
invariant is the hole's *shape*, one missing rectangle's private cells, not
its position; same "the hole moves" phenomenon as the S³ k = 2 plateau,
where the union of covered sets over runs is all 500 facets); (c) the
540/600 count is forced by arithmetic, not reached by search: the 50 far
cells (|s ∩ t| = 1) have exactly one covering diagonal rectangle each, so
five diagonal rectangles cover exactly 100 − 10 = 90 cells = 540 facets.
What is genuinely stuck is the **family**: the second k = 5 run (seed 7434,
`logs/rp2_k5.log`, no summary written) pinned its *entire population* at
540 for 40 consecutive epochs and gained +2 facets over the next 19
(~1.2 × 10⁹ moves for +2) — the annealer does not fail against a hard
region, it fails to leave the product family, and inside that family the
hole is a forced, provably unfixable-by-extension consequence of dropping
one rectangle.

### k = 3 (`results/rp2_k3_cold`, 398/600, L = 202)

- The domains are still near-diagonal rectangles (D_0 = star(4)×star(4)
  exactly, D_1 = star(3)×star(3) minus 2, D_2 a half-reshaped star(5)×star(5)),
  sizes 150+148+146.
- The hole has spread to **45 cells (29 fully dead, 16 partial)**; it is no
  longer only far pairs: 154/202 at |s ∩ t| = 1, but 42 at = 2 and 6 diagonal
  (= 3) facets are dead too.
- Defect depth: 174 at 1, 28 at 2 — the first depth-2 leftovers on any
  surface tested.
- H₁ absorption: **200/202 dead** (only f = 41, 53 have one H₁-pass absorber,
  both into chain 2). And **L alone is OBSTRUCTED** (dim H₁ = 2,
  rank (π₁−π₂)_* = 1): the leftover complex itself winds — no fresh chain
  could take even just the leftovers.

### k = 11 (`results/rp2_k11`, 590/600, L = 10) — the top rung

Same signature at the top of the ladder: 10 leftovers, **all far pairs**
(|s ∩ t| = 1, dual distance 2), spread over 7 partially-dead cells, all at
defect depth exactly 1. The domains here are non-rectangular (sizes
73–133; closest rectangles miss 26–109 facets).

## 3. Klein: the untested rungs

The same battery on the two Klein states that had never been absorption-
tested (k = 8's 424/424 theorem is `klein_anneal_log.md` §3):

| state | covered | \|L\| | L alone | per-(f, j) absorption |
|---|---:|---:|---|---|
| `klein_k9` (k = 9) | 1519/1536 | 17 | passes | **15/17 dead**; f = 1152 → chain 7, f = 1211 → chain 8 |
| `klein_cold666_seed_999_2` (k = 5) | 1198/1536 | 338 | **OBSTRUCTED** (dim H₁ = 7, rank 2) | **330/338 dead** |

- k = 9: the 17 leftovers sit on 7 cells (2 fully dead: cell (12, 0) and
  (12, 2) — the private pair of one base triangle), all at defect depth 1,
  mixed |s ∩ t| ∈ {0, 1, 2}. Extension of this state can gain at most 2
  facets (→ 1521); the rest needs restructuring.
- k = 5: the leftover complex winds on its own (rank 2 — both the Z and the
  Z/2 torsion directions), so no fresh chain can take the 338 leftovers, and
  330/338 are individually unabsorbable besides.
- k = 8 (`klein_k8_r1`), profiled for comparison: all 53 leftovers at defect
  depth 1, on 14 cells, 37/53 at |s ∩ t| = 0 and 38/53 at dual distance 3
  (the dual diameter is 4); one plateau domain (D_1, 294 facets) is exactly
  star(0)×star(0) — the same product-family contamination as RP²'s k = 5.

## 4. Möbius: eight seeds, always the same two-facet hole — shape invariant, position not

All eight k = 3 runs (seven seeds + a restart) stall at **exactly 148/150 —
two dead facets, never one, never three**. Full sweep
(`scripts/leftover_properties.py mobius` on every summary):

| run | the two dead facets (cell · staircase path) | far pair(s) shared vertex |
|---|---|---|
| seed 999 / 666 / restart | (0,3)·p5 and (3,0)·p0 — facets 23 & 90 | 0 (π-mirror pair) |
| seed 72341 / 72341_1 | (0,2)·p5 and (2,0)·p0 | 2 (π-mirror pair) |
| seed 523 | (2,4)·p0 and (4,2)·p5 | 4 (π-mirror pair) |
| seed 723 | (2,0)·p0 and (3,0)·p0 | 2 and 0 (mixed) |
| seed 875 | (2,0)·p0 and (2,4)·p0 | 2 and 4 (mixed) |

The earlier two-seed observation ("exactly the same two facets as seed 666")
does **not** generalize: the seed-independent fact is the hole's *shape*, not
its position — the same "the hole moves" phenomenon as RP² (§2) and the
S³ k = 2 plateau. 6 of 8 runs leave the π-mirror pair (s, t) ↔ (t, s), with
the path index flipping too (p5 ↔ p0 — the mirror swaps the staircase's two
factors), which is itself evidence the hole is a family-level property
rather than a pinned region.

**What the two facets are (the invariant shape):**

1. **Far pairs at the dual diameter.** Every dead facet lies on a cell (s, t)
   with |s ∩ t| = 1 and dual distance 2 — the diameter of the 5-triangle dual
   cycle. The Möbius triangulation's five unordered far pairs are
   (sᵢ, sᵢ₊₂), which share exactly one boundary vertex c whose link path
   sᵢ–sᵢ₊₁–sᵢ₊₂ has the two triangles at its ends: the hole is always
   "the two ends of a boundary vertex's fan". Of the 20 far-pair
   extreme-path facets (10 cells × 2 extreme paths), 18 are always eaten and
   exactly 2 never are.
2. **The dead path is always an extreme staircase ordering.** 16/16 dead
   facets across the eight runs are path 0 or path 5 of the 6 — the two
   *pure* orderings (path 5 crosses all of factor s before any of t, path 0
   the reverse: the L-shaped corner tetrahedra of the 3×3 prism grid,
   confirmed by decoding the facet vertex lists). The four interleaved
   middle paths are never the ones left dead. A uniform-random dead path
   would give this pattern with probability (1/3)¹⁶ ≈ 2 × 10⁻⁸ —
   structural, not chance.
3. **Partial cells at defect depth exactly 1.** Each holed cell has 1/6
   paths dead (never fully dead), and every leftover is one non-contiguous
   link away from the nearest chain — a flat-score plateau in the
   `STALL_VERIFICATION.md` sense.

**Why the annealer cannot reach them.** Every topological door is open: H₁
absorption is **0/2 dead** (seed 999: facet 23 → chain 2, facet 90 → chain 0;
the matching differs per seed but always exists), L alone passes, and CP-SAT
closes the cover outright (`results/mobius_k3_sat`: 150/150, domains
65 + 43 + 42, reduced chains 5–6 links, SC_strict⁰(Möbius) ≤ 2). No facet of
the band is intrinsically uncoverable, and **link count is not the obstacle
either**: the plateau chains reduce to 5–8 links (`reduce_chain` on the
plateau domains), the same budget the SAT cover uses. The binding is the
**move set**:

- An interleaved middle path is reachable mid-chain — a chain crossing the
  cell picks up the middle interleavings on the way through. A *pure*
  ordering path sits at the end of the cell's staircase block: covering it
  requires a chain to commit its entire traversal of that cell to one
  direction, a coordinated re-route of several links at once.
- Every single-facet edit toward that re-route breaks contiguity somewhere
  else — defect depth 1 means the best chain is exactly one bad link short,
  and repairing that link creates the next bad one — so each step on the path
  to the corner path is uphill at w = 1. The annealer prices the steps,
  never the correlated sequence.
- At k = ρ − 1 = 3 the 18-of-20 far-corner frontier is shared out among the
  three chains; every redistribution that releases a corner path is a
  multi-chain edit. The population has converged to a shape family that
  contains no completion, and crossover recombines only within the family
  (klein log §7).

Same door, different key — now confirmed against eight runs rather than two:
each seed finds an equivalent H₁-matching (a different key) and never walks
through any of them.

## 5. The correlation: one signature, three depths

Across all 14 stalled states tested (RP² k = 11/5/3, Klein k = 9/8/5,
Möbius k = 3 × 8 runs), the leftovers share the same fingerprint:

1. **Far pairs.** Leftovers concentrate on cells with |s ∩ t| ≤ 1 at or near
   the dual-graph diameter (RP²: distance 2 of 2; Möbius: 2 of 2; Klein:
   3 of 4 at k = 8). Near-diagonal and adjacent cells are always eaten first
   by the star-shaped cores. The only departures are the deep stalls
   (RP² k = 3, Klein k = 9), where the hole has grown past the far frontier.
   On the Möbius band the concentration is exact and stronger: the leftovers
   are always *extreme* staircase paths of far cells (§4, point 2) — the
   fingerprint includes the path index, not just the cell.
2. **Defect depth exactly 1.** Every leftover on every surface is one bad
   link short (only RP² k = 3 has 28 facets at depth 2). The stalls are
   flat-score plateaus in the precise sense of `STALL_VERIFICATION.md`.
3. **Rectangle membership does not discriminate** — every facet of every
   surface lies in exactly 3·3 = 9 maximal rectangles (leftover and covered
   means identical). Unlike the dodecahedron (3.2 % density, `gpu_pending.md`),
   these stalls are *not* a product-sparsity effect.
4. **The surviving domains are all H₁-trivial; the winding lives in L.**
   Every plateau domain passes `h1_map_obstruction` on every surface —
   sanity — while L alone is already obstructed at Klein k = 8/k = 5 and
   RP² k = 3.
5. **Extension is monotonically H₁-dead** at rates that deepen with the
   surface's torsion and the depth of the k-shortage:

   | surface | state | k vs ρ | dead / \|L\| | L alone |
   |---|---|---|---|---|
   | Möbius | k = 3 | ρ − 1 | **0/2** (perfect matching) | passes |
   | RP² | k = 5 | ρ − 1 | **60/60** | passes |
   | RP² | k = 3 | ρ − 3 | 200/202 | **obstructed** |
   | Klein | k = 9 | ρ − 3 | 15/17 | passes |
   | Klein | k = 8 | ρ − 4 | **53/53** (424/424) | **obstructed** |
   | Klein | k = 5 | ρ − 7 | 330/338 | **obstructed** |

The three surfaces are the **same failure at three depths**:

- **Möbius — move-set binding.** The topological door is open (an H₁-valid
  completion exists as a perfect matching, and a 3-cover exists,
  `mobius_k3_sat`); the annealer's move set never produces the correlated
  edit. Crossover does not help (klein log §7): the population has converged
  to one shape family that does not contain the cover.
- **RP² — product-family trap.** At k = ρ − 1 the annealer falls back into
  the diagonal rectangle family — the warm start from a genuine
  non-rectangular k = 6 cover *lost shape* and re-converged to 1-link
  rectangle chains, paying exactly the missing rectangle's 60 private
  facets. The leftovers are provably unabsorbable (60/60) but H₁-clean
  alone: the completion is "add the sixth domain", which is a k = 6 cover,
  not a k = 5 one. Escaping needs a 5-cover of a genuinely new shape
  (CP-SAT hybrid: infeasible for this state at L = 8; existence of any
  5-cover open — it would give SC_strict⁰(RP²) ≤ 4 against the bracket
  3 ≤ SC_strict⁰ ≤ 5 from TC(RP²) = 2 and ρ = 6).
- **Klein — topology binding.** Below k ≈ 9 every extension door closes
  (424/424 at k = 8, 330/338 at k = 5) and the leftover complex itself winds
  with both the Z and the Z/2 torsion. This is the only surface of the
  three where the plateau states are provably dead *and* the leftovers are
  jointly unchainable — the deepest layer of the winding family failure.

Practical consequence, same as klein log §6: before spending GPU on any
k → k − 1 rung below ρ, run the absorption battery
(`scripts/klein_h1_absorption.py <complex> <summary.json>`, ~0.03 s/test) on
the incumbent. A state whose leftovers are H₁-dead for all chains needs
restructuring-aware search (domain-cutting moves) or external construction
(CP-SAT), never more epochs.

## Reproduce

```bash
PY=whisper_env  # any python with numpy+numba
$PY scripts/leftover_properties.py rp2   results/rp2_k5_hybrid/summary.json
$PY scripts/leftover_properties.py rp2   results/rp2_k3_cold/summary.json
$PY scripts/leftover_properties.py rp2   results/rp2_k11/summary.json
$PY scripts/leftover_properties.py klein results/klein_k9/summary.json
$PY scripts/leftover_properties.py klein results/klein_k8_r1/summary.json
$PY scripts/leftover_properties.py mobius results/mobius_seed_999/summary.json
# §4 full sweep — all eight runs stall 148/150 on two far-cell extreme paths:
for d in results/mobius_seed_* results/mobius_restart; do
    $PY scripts/leftover_properties.py mobius $d/summary.json
done
# plateau chains reduce to 5-8 links (same budget as the SAT 3-cover's 5-6):
$PY - <<'EOF'
import sys, json; import numpy as np; sys.path.insert(0, 'src')
import gpu_sc
P = gpu_sc.build('mobius')
for d in ['mobius_seed_999','mobius_seed_666','mobius_seed_523','mobius_seed_723']:
    s = json.load(open(f'results/{d}/summary.json'))
    ch = [np.array(c, np.int8) for c in s['best_chains']]
    doms = [sorted(int(f) for f in P.domain_of(c)) for c in ch]
    print(d, [len(gpu_sc.reduce_chain(P, c, dd)) - 1 for c, dd in zip(ch, doms)])
EOF
$PY scripts/klein_h1_absorption.py rp2   results/rp2_k5_hybrid/summary.json   # 60/60 dead
$PY scripts/klein_h1_absorption.py rp2   results/rp2_k3_cold/summary.json     # 200/202 dead
$PY scripts/klein_h1_absorption.py klein results/klein_k9/summary.json        # 15/17 dead
$PY scripts/klein_h1_absorption.py klein results/klein_cold666_seed_999_2/summary.json  # 330/338
$PY scripts/klein_h1_absorption.py mobius results/mobius_seed_999/summary.json # 0/2 dead
# rho: exact ILP over rect2_optimum.rect_masks cells (scipy.optimize.milp, §1)
```