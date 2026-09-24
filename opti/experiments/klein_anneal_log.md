# Klein bottle anneal log — no cover at k ≤ 8, and a proof the plateau cannot be extended

*2026-09-24. K = the vertex-minimal Klein bottle, Cervone's "242" triangulation
(8 vertices 0–7, 16 triangles, f-vector (8, 24, 16), χ = 0, no coherent
orientation; `gpu_sc.py --complex klein`, re-proved ‖K‖ ≅ K² by
`independent_check.check_klein`).  16 × 16 triangle pairs × 6 staircase paths
= **1536** facets of K × K, grid 8² = 64.  All statements at subdivision
level 0.*

TC(K²) = 4 (Cohen–Vandembroucq 2017; also Davis, and Iwase–Sakai–Tsutaya):
SC_strict⁰ ≥ 4 at **every** subdivision level, so **at least five domains**
always — the k = 4 control must never cover, and a 5-domain level-0 cover
would be sharp (SC_strict⁰ = 4 = TC).  The level-0 minimum is bracketed
**[5, 12] domains** (§1 gives the 12; TC gives the 5).

## 1. ρ(Klein) = 12 — the product family is nowhere near sharp at level 0

Exact set-cover ILP over the 64 candidate rectangles star(v) × star(w)
(scipy/HiGHS, binaries x_vw, one constraint per (s, t) pair): optimum **12**,
witness (v, w) = (0,0), (0,2), (1,3), (1,4), (2,0), (3,1), (3,5), (4,1),
(4,5), (5,3), (5,4), (6,6), bitmask-verified to cover all 256 pairs.  The
repo's own `rect_cover.py` branch-and-bound independently exhausted limits
1–8 (limit 8 alone: 4.38 M nodes, 5.9 s) before being stopped; it would have
had to exhaust 9–11 to see the 12.

```python
# reproduce: exact rho by ILP (src/ on path; P = gpu_sc.build("klein"))
pairs = [(s, t) for s in range(16) for t in range(16)]
rects = [(v, w) for v in range(8) for w in range(8)]
A[i, j] = 1 iff v in tris[s] and w in tris[t]   # scipy.optimize.milp, minimize sum x
# reproduce: independent lower bounds 1..8
python3 src/rect_cover.py klein
```

Consequences:

- Star covers certify only **SC_strict⁰ ≤ 11**.  The old `run_klein.sh` header
  line "k6 … covers trivially, gives SC ≤ 5" was wrong and is corrected; the
  recorded k = 6 run in fact reached 783/1536.
- The Császár torus is the same story (ρ = 11, `torus_starcover_opt`); the
  Klein bottle is the harder twin (one more rectangle, plus torsion).
- Like the cube (ρ = 8 against a true SC_strict⁰ = 2), the interesting
  question is how far below 12 the chain-first family can go.  Any cover at
  k ≤ 11 beats every product cover.

## 2. Campaign state

54 recorded runs, **13.3 GPU-hours**, no cover at any k.  Best coverage per k
(all `beta = 5`, `gamma = 0.5`, `cmax = 40`, S = 4096 unless noted):

| lineage | k | best | notes |
|---|---:|---:|---|
| `klein_k5` (run_klein.sh cold) → `_v2` → `_guided` | 5 | 806 → 926 → 1007 | 607–1231 s each |
| `klein_cold2`, `_cold3` | 5 | 1015, 954 | 1209 s each |
| `klein_cold666_repro` (3141 s) → `cold666_seed_999` (4714 s) → `_1` → `_2` → `_3` | 5 | 1170 → 1186 → 1195 → **1198** → 1198 | the best k = 5 lineage; short continuations gained +9…+12 then flat (78.0 %) |
| `klein_hybrid` (greedy 5-set seed) → `klein_hybrid_run` | 5 | 1162 → 1172 | 612 s |
| night ladder r1–r30 | 5 | cold rungs 384–430; warm rungs 1185–**1187** | ~620 s per rung; warm chain r15 → r17…r29 pinned at 1187 |
| `klein_rung_next` → rung2 → rung3 → … → `rung8_mu` | 5 | 1180 → 1185 → 1186 → (melt 1054) → 1125 → 1155 → 1164 → 1170 | seed_sweep rungs, 134–635 s; warm rungs are not monotone |
| `klein_k6` | 6 | 783 | 1228 s |
| `klein_k8` (cold, 6872 s) → `klein_k8_r1` (continuation, 1733 s) | 8 | 1470 → **1483/1536** | **53 uncovered**; +13 in epoch 29–49 then flat |
| `klein_k9` (cold, running at time of writing) | 9 | 1245 at epoch 66 | +3/epoch, decelerating |

The k = 8 plateau state `klein_k8_r1`: domain sizes [294, 294, 267, 270, 243,
255, 238, 229], all eight domains H₁-trivial (dim H₁(D_j; Z/2) = 0).  One
domain is a full rectangle — D_1 (294 = 7·7·6 facets, 49 base pairs) is exactly
star(0) × star(0) — and D_0 has the same size and pair count but is contained
in no rectangle, the same "one ceiling, several shapes" phenomenon as the S³
460-plateau (§12.2 of `optimized_algorithms.md`).

## 3. The H₁ absorption theorem for the k = 8 plateau

A contiguity chain on D is a homotopy π₁|D ≃ π₂|D; if (π₁ − π₂)_* is nonzero
on some cycle of D over Z/2, **no chain covers D, whatever the budget**
(`h1_obstruction.py`).  `scripts/klein_h1_absorption.py` applies this to the
plateau state and its leftovers L (the 53 uncovered facets):

| test | dim H₁ | rank (π₁−π₂)_* | verdict |
|---|---:|---:|---|
| all of K × K (reference) | 4 | 2 | obstructed, as it must be |
| each D_j alone (carries its chain) | 0 | 0 | passes — sanity |
| **L alone** | 4 | **2** | **obstructed: no fresh chain can take even just the leftovers** |
| D_j ∪ L, every j | 4–15 | 2 | obstructed, all eight |
| D_j ∪ {f}, all 53 f, all 8 j (424 tests) | — | > 0 | **obstructed, 424/424** |

The obstruction is monotone under inclusion (the witnessing cycle survives in
any superset and its image in K does not change), so the 424 single-facet
results kill **every nonempty absorption of L into any plateau domain**.
Together with L's own winding this proves:

> **Extension-only completion of the k = 8 plateau state requires at least
> two additional chains (k ≥ 10).  No growth of any subset of the eight
> domains can ever finish this state.  A 9-cover, if one is reachable from
> here at all, must restructure — hand whole bands of covered facets between
> chains — not extend.**

This is the proved mechanism behind the empirical behaviour: the
continuation run gained +13 and then sat flat for twenty epochs, and (as on
the dodecahedron, `gpu_pending.md`) more moves buy ever less because the
moves that remain are topologically forbidden, not merely hard.

The 53 dead facets are not scattered: they live on **14 base-facet pairs**
with |s ∩ t| ≤ 1 (far pairs; near-diagonal pairs are covered by the
star-shaped cores):

| pair (s, t) | paths dead | pair (s, t) | paths dead |
|---|---|---|---|
| (1, (1,2,6)) → (4, (0,3,6)) | 2 of 6 | (11, (2,3,6)) → (15, (1,4,6)) | **6 of 6** |
| (1, (1,2,6)) → (8, (0,3,7)) | 2 of 6 | (12, (2,4,5)) → (8, (0,3,7)) | 1 of 6 |
| (10, (2,3,5)) → (3, (0,4,6)) | 5 of 6 | (12, (2,4,5)) → (9, (1,3,7)) | 1 of 6 |
| (10, (2,3,5)) → (14, (1,4,7)) | 5 of 6 | (14, (1,4,7)) → (10, (2,3,5)) | **6 of 6** |
| (10, (2,3,5)) → (15, (1,4,6)) | **6 of 6** | (14, (1,4,7)) → (12, (2,4,5)) | 1 of 6 |
| (11, (2,3,6)) → (3, (0,4,6)) | **6 of 6** | (15, (1,4,6)) → (10, (2,3,5)) | **6 of 6** |
| (11, (2,3,6)) → (14, (1,4,7)) | 5 of 6 | (15, (1,4,6)) → (12, (2,4,5)) | 1 of 6 |

**Discipline.**  The theorem is about *this state*: it rules out completing
the recorded 8 domains, not the existence of a differently-shaped 8-cover.
No search failure and no state-level obstruction bounds SC_strict⁰ from below
(same status as C1, `REVIEW_NOTES.md`).  What it does give is a cheap
acceptance test for the ladder: before spending GPU on a k → k − 1 step, run
the absorption test on the dropped domain's private facets — extension-dead
steps need restructuring-aware search, not more epochs.

## 4. The Möbius band: the same plateau, one layer shallower

The 5-vertex Möbius band (|F_L| = 150) is the klein story in miniature:

- k = 5 covers in 36 s, k = 4 warm-started from it in 17 s
  (`mobius_k5`, `mobius_k4_from_k5`).
- **k = 3 pins at exactly 148/150** across six seeds, two betas (1.0, 5.0), a
  warm restart and a continuation — the same seed-independent plateau as the
  S³ 460 and the Klein 1483.
- A verified 3-cover **exists**: `mobius_k3_sat` (65 + 43 + 42, chains 5/6/6
  links, SC_strict⁰(Möbius) ≤ 2), produced outside the annealer — its
  summary records no annealer parameters, only the certificate.

The absorption test on the seed-666 plateau state (domains 60 + 60 + 48,
L = {23, 90}) splits the failure into two layers:

| test | verdict |
|---|---|
| D_j alone, all three | pass (carry chains) |
| L alone | passes — the two leftovers do not wind jointly |
| D_j ∪ L, every j | obstructed — no chain takes both at once |
| D_0 ∪ {90} | **passes** |
| D_1 ∪ {23} | **passes** |
| D_0 ∪ {23}, D_2 ∪ {either} | obstructed |

So on the Möbius band the topological door is **open**: an H₁-valid
completion exists (absorb 90 into chain 0 and 23 into chain 1 — a perfect
matching), and the annealer still never walks through it in six seeds of
~2 minutes each, nor warm-restarted from the state itself.  There the
binding constraint is purely the **move set** (single-vertex and track
mutations never produce the correlated chain edit that absorbs one facet),
while on the Klein k = 8 state it is **topology** (every absorption
forbidden).  Two depths of the same winding-family failure — and in both
cases the fix that worked was construction, not search (`mobius_k3_sat`
vs. the annealer's 148).

## 5. Why the simplex boundaries yield and this family stalls

The mechanism, as diagnosed on the circle (§12.3) and the torus (§13.1): a
chain is a homotopy, so a domain that wraps a non-contractible direction is
un-chainable at any budget — but the energy only sees per-link defects, and
winding looks locally optimal all the way in (the stuck circle states carried
35–41 distinct maps on 9- and 6-facet domains).  Escaping a wrap means
cutting a domain, a globally correlated move through a deep valley of
temporarily lost coverage.  The dodecahedron campaign measured the end state
of that dynamic: per-move yield down 50×, 73 % of warm restarts gaining
zero, "the domains do not move; they are being nudged" (`gpu_pending.md`).

∂Δⁿ is the opposite on every axis that matters: the 1-skeleton is complete
(`dist ∈ {0, 1}`, §10), the dual graph has diameter 1 (a 252-facet S³ domain
needs only 3 links), and H₁ = 0, so the H₁ tool is vacuous there (§12.2) —
**the energy is the whole problem**, and plateaus it finds are real
combinatorial facts (460 = exactly the 2-rectangle optimum, `rect2_optimum.py`).
The Möbius band and the Klein bottle put H₁ (Z, and Z ⊕ Z/2 with torsion —
the torsion is what lifts TC from the torus's 3 to the Klein bottle's 4) into
every candidate domain that crosses the core.

The whole family, one table:

| complex | H₁ | TC | \|F_L\| | annealer outcome |
|---|---|---:|---:|---|
| ∂Δ² (S¹) | Z | 1 | 18 | stalled 13–14/18 → corrected energy solves it, 15 s |
| wedge S¹∨S¹ | Z² | 2 | 50 | covered 41 s, **sharp** (SC_strict⁰ = 2) |
| barycenter Δ² | 0 | 0 | 54 | covered 3 s, sharp |
| ∂Δ³ (S²) | 0 | 2 | 96 | covered 22 s, sharp |
| cube boundary | 0 | 2 | 864 | covered 304 s guided, sharp — but ρ = 8 |
| dodecahedron | 0 | 2 | 7776 | failed at 23 % — sparsity, not topology |
| ∂Δ⁴ (S³) | 0 | 1 | 500 | covered at k = 3 (SC⁰ ≤ 2); **k = 2 pinned at exactly 460** through every attack (69 epochs, 5.65 × 10⁹ moves) — C1 |
| ∂Δ⁶ (S⁵) | 0 | 1 | 12348 | covered at k = 3 (SC⁰ ≤ 2); the TC = 1 gap persists a dimension up |
| ∂Δ⁵, ∂Δ⁷ (S⁴, S⁶) | 0 | 2 | 2520, 59136 | covered k = 3, **sharp** (SC⁰ = 2 = TC) |
| Császár torus | Z² | 3 | 1176 | **never covered**: best 733 (62 %); k = 4 → 8 → 11 goes 733 → 705 → 563 |
| Möbius band | Z | 1 | 150 | k = 5, 4 cover; k = 3 pins 148/150 × 6 while a 3-cover exists |
| **Klein bottle** | **Z ⊕ Z/2** | **4** | 1536 | **no cover at k ≤ 8 in 13.3 GPU-h; 424/424 extension moves provably dead** |

The pattern across the whole repository: **every completed annealer cover is
a level-0-sharp instance** — SC_strict⁰ = TC (circle, barycenter, cube, ∂Δ³,
∂Δ⁵, ∂Δ⁷, wedge).  **Every stall is an instance with a discretization gap**
(SC_strict⁰ > TC, proven or bracketed away from the floor): the odd spheres
at k = 2, the dodecahedron, the torus, the Möbius band at k = 3, the Klein
bottle.  Even the sphere wins are really *even-dimensional* simplex
boundaries; the odd ones (TC = 1) carry the same C1 gap the winding family
does, and their k = 2 searches stall identically.  Subdivision is what
closes the gap in general (Theorem 3.3) — and subdivision level 0 is exactly
the level the annealer cannot buy anything at.

## 6. Consequences for the campaign

1. **Level 1 is not a fallback.**  One barycentric subdivision gives 96
   triangles, |F_L| = 96² · 6 = **55,296** (∂Δ⁷-scale, whose k = 3 cover took
   the staged campaign ~2.5 h), grid 2304, and per-thread local state
   `bd[k][|F_L|] + ch[k][cmax+1][grid]` ≈ 750 KB at k = 5 / cmax = 40
   (≈ 564 KB even at cmax = 24) — past the same 512 KB CUDA per-thread limit
   that blocks ∂Δ⁸ outright (`s7_chain_first_wall.md`).  And it certifies
   nothing new: monotonicity is one-way (SC⁰ ≥ SC¹ ≥ … = TC), so a level-1
   cover bounds nothing at level 0, while TC(K²) = 4 is already a published
   theorem.  A level-1 5-domain cover would only date the collapse
   (SC_strict¹ = 4 = TC) — an auxiliary datum, not a result.
2. **The productive routes at level 0.**  (a) Restructure-aware search: moves
   that cut domains and hand bands of facets between chains — the one move
   class the plateau proofs say is missing; the μ penalty pushes chains apart
   but does not cut.  (b) External construction: the only method that has
   ever beaten a winding plateau in this repository is the
   `mobius_k3_sat`-style certificate — a SAT/ILP encoding of chains on the
   1536 facets is the Klein analog, and would settle k = 9…5 definitively
   where the annealer cannot.  (c) If any k ∈ [9, 12] covers (k = 12 exists
   by ρ), ladder down by warm start (`init_from_summary` keeps the k largest
   domains) with the absorption test as the gate before each step.
3. **What would count as a lower bound.**  None of the stalls bounds
   SC_strict⁰ from below.  The 53/53 theorem is state-relative.  A genuine
   "no 8-cover exists" needs a different instrument — the H₁ obstruction
   over *all* candidate 8-partitions (search over partitions, not chains), or
   the C1-style combination of exhaustive small cases with structural
   arguments.

## 7. Tooling response: `--crossover` and `--h1-guide` (2026-09-24)

The annealer gained two opt-in flags (both off by default; with defaults the
RNG stream and weight arithmetic are unchanged, so historical runs
bit-reproduce).  Both are host-side; the CUDA kernel is untouched.

- **`--crossover FRAC`** — the restructuring move.  At each tournament, after
  the noise slots, FRAC of the remaining replacement slots are filled with
  children of two distinct elite incumbents, each chain taken uniformly per
  chain from either parent (`ChainAnnealer.tournament`, gpu_sc.py).  Children
  keep valid π₁/π₂ endpoints (the kernel never mutates map 0 or map cmax) and
  get `bestscore = -1` (refreshed from the kernel's own snapshot next epoch).
- **`--h1-guide N`** — every N epochs, the incumbent's uncovered facets × k
  chains are tested with `h1_map_obstruction` (0.03 s/test); facets
  H₁-obstructed for **all** chains get weight 0.  Masked facets still count
  in coverage and ranking (the kernel's `cov` is unweighted); the mask is
  reapplied every guide epoch because the weight decay
  (`w = 1 + (w−1)·0.7`) would heal the zeros, and recomputed every N epochs
  (deferred when `|U|·k > 2048` — masking matters near plateaus, not during
  the early climb).

Verification (T4, alongside the running klein k=9 job):

| check | result |
|---|---|
| `test_contracts.py` plain + `-O` (extended with the new bounds) | 9/9 OK both |
| host smokes: π₁/π₂ endpoints intact over 6 crossover tournaments; `crossover=0.0` bit-identical to the old RNG stream | pass |
| circle k2, corrected energy, flags off / on | 18/18 (9+9) both; defaults still reproduce the old 13/18 stall |
| wedge k3, `--crossover 0.5 --h1-guide 2` | covered 50/50, VERIFIED SC ≤ 2 (97.7 s under contention; 41 s historical) |
| bary k1, flags on | 54/54 in 3.5 s |
| **Möbius k3 test A: warm from `mobius_seed_666`, `--crossover 0.5 --h1-guide 1`** | **148/150 — stall persists** (control also 148; nothing H₁-dead to mask: the two leftovers are H₁-open, §4) |
| Möbius k3 cold + `--crossover 0.5`, seeds 2001/2002 | **148/150, 148/150** — same shape family (58+60+48, 60+47+60) |
| klein k8 probe, warm from `klein_k8_r1`, `--crossover 0.25 --h1-guide 1`, 3 epochs | mask prints `53/53 … weight 0` each epoch (the offline theorem reproduced live); coverage 1483 as predicted |

**Interpretation of the negative Möbius A/B.**  Crossover did not move the
plateau because recombination needs raw material the population does not
contain: by the time the annealer stalls, every thread has converged to the
same shape family (~60+60+48), while the cover that exists (`mobius_k3_sat`,
65+43+42) is never visited by any thread — cold or warm.  Cloning and
recombining the converged population only reshuffles the family.  The still-
missing mechanism is the **targeted in-chain rewrite** (editing one chain so
as to absorb one H₁-feasible facet — the §6 "free certificate"/DP idea made
concrete, e.g. via `SubProduct` single-chain annealing:
`fixed_domain_test.py` machinery), proposed as Phase 3 in the plan.  The full
klein falsifiable test B (k=8 warm, `--crossover`, 20 epochs; success = any
epoch above 1483) is queued until the running k=9 job releases the GPU.

## Reproduce

```bash
python3 src/rect_cover.py klein                    # rho >= 9 by exhaustive B&B (stopped at limit 8)
# rho = 12: exact ILP as quoted in §1 (scipy.optimize.milp, HiGHS)
python3 scripts/klein_h1_absorption.py klein results/klein_k8_r1/summary.json
python3 scripts/klein_h1_absorption.py mobius results/mobius_seed_666/summary.json
```