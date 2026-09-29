# Surface campaign — run plan

## What we have now

| Surface | Product facets | TC (reduced) | Target k | Best result | Method |
|---|---|---|---|---|---|
| Möbius band | 150 | 1 | 2 (sharp at 3 domains) | **SC=2, done** | CP-SAT k=3 L=8 |
| Torus | 1176 | 2 | 3 | 579/1176 (49%) anneal; hybrid 6+4 misses 5 facets | Stalling |
| Klein bottle | 1536 | 4 | 5 | 1200/1536 (78%) | Stalling |
| RP² | 600 | 2 | 3 | **SC≤5** (anneal ladder k=11→6); k=5 stalls 540/600, k=4 stalls 469/600, k=3 stalls 398/600 | Anneal wall; CP-SAT next |
| S¹×(S¹∨S¹) | 5400 | 4 | 5 | **Not tried** | In code already |

## Full surface landscape, ranked by feasibility

### Tier 1 — Easy wins (k=3, small product, sharp at TC)

| Surface | V | Triangles | Product facets | TC (reduced) | Target k | Status |
|---|---|---|---|---|---|---|
| RP² | 6 | 10 | 600 | 2 | 3 | Ladder k=11→6 closed (SC≤5); k=5,4,3 all stall — CP-SAT pending |
| Torus (Császár) | 7 | 14 | 1176 | 2 | 3 | Tried but stalling — best pure run 579/1176 (49%), best hybrid 283/288 remainder uncovered with 6 star + 4 anneal chains |

**RP²** is the obvious next target: only 600 product facets (between S³
at 500 and cube at 864), k=3, and a sharp result would give
SC\_strict = 2 = TC(RP²). The 6-vertex/10-triangle hemi-icosahedron
triangulation is now in `gpu_sc.py` as `--complex rp2`.

**Torus** is already in the code and has been extensively attacked but is
not closing. The star cover needs 11 domains (ρ=11), and the pure
annealer at k=3 reaches only ~49%. Hybrid approaches (star-rectangles +
anneal remainder) get within 5 facets at 6+4=10 total domains, but can't
seal. This is itself a very interesting finding — the torus might have
SC\_strict much larger than TC at level 0.

### Tier 2 — Ambitious but doable at k=5

| Surface | V | Triangles | Product facets | TC (reduced) | Target k | Status |
|---|---|---|---|---|---|---|
| Klein bottle | 8 | 16 | 1536 | 4 | 5 | Stalling at ~78% |
| Genus-2 orientable | 10 | 24 | 3456 | 4 | 5 | Not tried |
| S¹ × (S¹ ∨ S¹) | 12 | 30 | 5400 | 4 | 5 | Not tried, already in code |

### Tier 3 — Interesting but diminishing returns

Higher non-orientable genera (N₃, N₄) — all TC=4, bigger triangulations.

## What's most interesting for the paper

1. **RP²** — should be the easiest closed non-orientable surface. 600
   facets, k=3. If the annealer solves it, you have the first-ever
   SC\_strict computation for a non-orientable closed surface. If it
   *doesn't*, that's also a striking finding: even at 600 facets with
   k=3 the annealer stalls on non-spheres.

2. **Torus** — you already have data showing it's stalling. This is
   news: the torus has TC=2 (same as S²) but the level-0 SC\_strict
   might be much higher. If the CP-SAT approach that cracked the Möbius
   can also close a 3-domain torus cover, you prove SC\_strict(torus)=2=TC.
   If CP-SAT also fails at k=3, you'd have evidence of a genuine
   subdivision gap on a closed orientable surface — that would be a
   theorem-level result.

3. **Klein bottle** — the flagship, but the hardest (k=5, 1536 facets,
   stalling at 78%). Worth continuing to attack, but maybe CP-SAT should
   be tried here too on the remaining ~336 uncovered facets after the
   annealer plateau.

4. **Möbius band** — already done (annealer gets 4, CP-SAT gets 3). The
   subdivision gap SC\_strict=2 > 1=TC is already in the paper.

## TC references for surfaces

| Surface | TC (reduced) | Source |
|---|---|---|
| S², torus T² | 2 | Farber 2003 |
| Orientable, genus ≥ 2 | 4 | Farber 2003 |
| RP² | 2 | Farber–Tabachnikov–Yuzvinsky |
| Klein bottle, non-orientable genus ≥ 2 | 4 | Dranishnikov; Cohen–Vandembroucq 2017 |

Nobody has computed level-0 SC\_strict for any surface beyond S¹ and
S¹∨S¹. Every result here is new.

## Phase 0 — prep (no GPU, done)

1. ✅ Added RP² (hemi-icosahedron) to `gpu_sc.py` as `--complex rp2`
   - 6 vertices, 10 triangles, 600 product facets
   - Verified: V=6, E=15, F=10, χ=1, 1-skeleton=K₆, non-orientable

2. ✅ Generalized `scs_sat.py` to accept `--complex` flag (rp2, torus, klein, etc.)
3. ✅ RP² ladder descent k=11→k=6 (SC_strict ≤ 5); k=5 stalls at 540/600, k=4 at 469/600, k=3 at 398/600

## Phase 1 — CPU (CP-SAT, no GPU needed)

```bash
# RP²: 600 facets, k=3, should be tractable
python scs_sat.py --complex rp2 --k 3 --L 8 --timeout 3600

# Torus: 1176 facets, k=3 — bigger, but worth trying
python scs_sat.py --complex torus --k 3 --L 10 --timeout 7200

# Klein bottle: 1536 facets, k=5 — may be too big for SAT, but try
python scs_sat.py --complex klein --k 5 --L 8 --timeout 14400
```

If RP² and torus are SAT at k=3, that's **two new SC=TC results** without
GPU. If UNSAT, that proves a subdivision gap — also a theorem.

## Phase 2 — GPU annealer runs

```bash
# RP² — should be trivial at 600 facets
python src/gpu_sc.py --complex rp2 --k 3 --cmax 40 --S 4096 --epochs 60 \
  --T0 0.9 --T1 0.02 --beta 5 --gamma 0.5 --seed 42

# Torus — retry with μ penalty (new parameter that helped Klein)
python src/gpu_sc.py --complex torus --k 3 --cmax 40 --S 4096 --epochs 60 \
  --T0 0.9 --T1 0.02 --beta 5 --gamma 0.5 --mu 0.5 --seed 42

# Torus at k=4 — fallback if k=3 still stalls
python src/gpu_sc.py --complex torus --k 4 --cmax 40 --S 4096 --epochs 60 \
  --T0 0.9 --T1 0.02 --beta 5 --gamma 0.5 --mu 0.5 --seed 42

# Klein bottle — continue from best incumbent (1200/1536), guided
python src/gpu_sc.py --complex klein --k 5 --cmax 40 --S 4096 --epochs 60 \
  --T0 0.9 --T1 0.02 --beta 5 --gamma 0.5 --mu 0.5 --guide 5 \
  --init results/klein_cold666_seed_999_2 --seed 999

# S¹×(S¹∨S¹) — 5400 facets, k=5, between S⁴ and S⁵ in size
python src/gpu_sc.py --complex s1xw --k 5 --cmax 40 --S 4096 --epochs 60 \
  --T0 0.9 --T1 0.02 --beta 5 --gamma 0.5 --mu 0.5 --seed 42
```

## Phase 3 — CP-SAT fallback for anything that stalls

For any surface where the annealer plateaus, try the constraint solver on
the remaining uncovered facets (warm-start from the annealer's best
partition).

## Stall analysis (2026-09-24)

ρ(RP²) = 6, ρ(Möbius) = 4 (exact ILP), so both surfaces stall at the first
rung below their rectangle-cover number (RP² k = 5, Möbius k = 3) while the
Klein bottle (ρ = 12) stalls far above it (k = 9, 8, 5). The RP² k = 5
plateau is literally the six diagonal rectangles star(v)×star(v) minus one:
its 60 leftovers are the private cells of star(5)×star(5), all far pairs,
all one defect short, and provably unabsorbable by any of the five plateau
domains (60/60 H₁-dead) though H₁-clean alone — completing means adding the
sixth domain, i.e. it is not a k = 5 completion at all. Klein k = 9: 15/17
dead; Klein k = 5: 330/338 dead and the leftover complex winds alone; Möbius
k = 3: the same two facets {23, 90} across seeds, H₁-open with a perfect
matching the move set never takes. Full battery, per-facet profiles and the
three-depths correlation (Möbius = move-set binding, RP² = product-family
trap, Klein = topology binding): `experiments/surface_stall_analysis.md`.


## Star-cover upper bounds (CPU, exact ρ) — 2026-09-28

Every vertex pair `(v,w)` gives the product domain `star(v) × star(w)`; `ρ` is the
minimum number of such rectangles covering all facet pairs — an exact set cover
over `|V|²` sets and `|F|²` pairs, solved in seconds by `rect_cover.py` (branch and
bound) or `scipy.optimize.milp` (HiGHS). Corollary 2 of
`experiments/optimized_algorithms.md` (§27 for the full write-up) gives
**`SC_strict(K) ≤ ρ − 1`** with no search, and `star_cover.py` is CPU-only.
On the hemi-icosa, hemicube and hemi-octa `ρ = V` exactly (6, 7, 9), so the bound is
`SC_strict ≤ V − 1 = 5, 6, 8`: the diagonal/grid covers of Lemma 7 meet the row–transversal
lower bound of Lemma 6 — proofs in `experiments/optimized_algorithms.md` **§28** and in the
paper (`main.tex` §5: `lem:rows`, `lem:covers`, `prop:hemi`).

| surface (`--complex`) | V: abstract → complex | Δ | `\|K×K\|` | ρ | star cover (domains → SC) | annealer / searched cover (domains → SC) |
|---|---|---|---|---|---|---|
| Möbius (`mobius`) | 5 → 5 | 5 | 150 | 4 | `results/mobius_starcover_opt` (4 → ≤ 3) | CP-SAT `results/mobius_k3_sat` (3 → **= 2**); annealer `results/mobius_k4_from_k5` (4 → ≤ 3) |
| hemi-icosa (`rp2`) | 6 → 6 | 10 | 600 | 6 | `results/rp2_starcover_opt` (6 → ≤ 5) | `results/rp2_k6` (6 → ≤ 5) |
| hemicube | 4 → 7 | 12 | 864 | 7 | `results/hemicube_starcover_opt` (7 → ≤ 6) | `results/hemicube_k7` (7 → ≤ 6) |
| hemi-octa (`hemi_octa`) | 3 → 9 | 16 | 1536 | 9 | `results/hemi_octa_starcover_opt` (9 → **≤ 8**) | `results/hemi_octa_k8` (8 → **≤ 7**, star-seeded, VERIFIED) |
| hemi-dodeca (`hemi_dodeca`) | 10 → 10 | 18 | 1944 | 15 | `results/hemi_dodeca_starcover_opt` (15 → **≤ 14**) | — (best 1777/1944, no cover) |
| Klein (`klein`) | 8 → 8 | 16 | 1536 | 12 | `results/klein_starcover_greedy` (12 → ≤ 11) | `results/klein_k10` (10 → ≤ 9) |
| torus | 7 → 7 | 14 | 1176 | 11 | `results/torus_starcover_opt` (11 → ≤ 10) | — (searches stall) |
| dodecahedron S² (`dodeca`) | 20 → 20 | 36 | 7776 | 36 | `results/dodeca_starcover_opt` (36 → ≤ 35) | — (k = 3 stagnates at 2602/7776 = 33.5 %) |
| icosahedron S² (`icosa`) | 12 → 12 | 20 | 2400 | 24 | `results/icosa_starcover_opt` (24 → **≤ 23**) | — (density-ladder control, not yet annealed) |

`V: abstract → complex` is the abstract polyhedron/regular map's vertex count versus the
simplicial complex actually built (a non-simplicial face must be subdivided: hemicube
4 → 7, squares coned from a centre; hemi-octahedron 3 → 9, edge midpoints; hemi-dodeca
10 → 10, pentagons fanned; Möbius/hemi-icosa/Klein/torus are triangulations already).

The **dodecahedron** row is the S² control from the density ladder: exact `ρ = 36`, a 6×6 grid
cover on the minimum facet-transversal `S = {0, 2, 8, 10, 12, 19}` (τ = 6, so `|S|² = 36`), with the
covering LP tight at 36 (HiGHS, 0.6 s). The row–transversal bound of Lemma 6 is weaker here
(`⌈6·36/8⌉ = 27`), so this is a grid/LP case, not a row–transversal one. Certificate
`results/dodeca_starcover_opt` (36 domains, sizes 384×4 + 288×8 + 216×4 + 192×8 + 144×8 + 96×4,
chains 2–5 links, verified) gives **`SC_strict(dodecahedron) ≤ 35`** search-free. With
`ρ(∂Δ³) = 3` and `ρ(cube) = 8`, three triangulations of S² now carry ρ = 3, 8, 36 — ρ is
combinatorial, not a homotopy invariant.

The **icosahedron** rung of the same ladder is now exact too: `ρ = 24`, and this one is
row–transversal tight (unlike the dodecahedron): τ = 6 (`{0,2,4,6,8,10}`), `d = 5`, `F = 20`,
`⌈6·20/5⌉ = 24` (Lemma 6), matched by the 24-rectangle cover of
`results/icosa_starcover_opt` (150×3 + 126×5 + 102×6 + 78×7 + 54×3 = 2400, chains 2–5 links,
verified) ⇒ **`SC_strict(icosa) ≤ 23`**. The covering LP is only 16 here; the double cover
`rp2` (hemi-icosahedron) has `ρ = 6 = V`, so the two S² triangulations of the same ladder
split the same way the RP² ones do.

The Möbius band was the first non-orientable case where the search provably beats
every product cover (`ρ − 1 = 3` vs `SC_strict = 2`); the hemi-octa star-seeded `k = 8`
run joins it (8 domains < `ρ = 9`, so the cover is non-product: **SC ≤ 7** vs product ≤ 8),
as do the wedge (§23) and cube (§26). The product bound is an
upper bound, not an optimality claim.

**Warm-starting the annealer from a star cover.** `star_cover.py` now writes the
chains as `best_chains`, so any certificate above can seed
`annealer_stretch.py --init` at `k = ρ − 1` (it keeps the k largest star domains
and drops the rest). For the hemi-octahedron, `k = 8` starts at 1440/1536 facets
(drops the 96-facet domain) and tries to absorb it — **and does**: 1536/1536 at epoch 106
(seed 8233, 6325 s), `VERIFIED system of 8 motion planners`, sizes
258/237/224/186/175/160/150/146, chains 4/5/2/4/5/3/4/5 links ⇒ **`SC_strict ≤ 7`**,
one below the product floor `ρ − 1 = 8` (§28):

```bash
PYTHONUNBUFFERED=1 $PY src/annealer_stretch.py --complex hemi_octa --k 8 --cmax 40 --S 4096 \
  --iters 5000 --epochs 200 --T0 0.6 --T1 0.02 --beta 5.0 --gamma 0.1 --mu 0 \
  --pcopy 0.5 --ptrack 0.3 --guide 5 --guide-max 8 --h1-guide 2 \
  --pstretch 0.5 --pcollapse 0.5 \
  --init results/hemi_octa_starcover_opt/summary.json \
  --seed 8233 --verbose \
  --outdir results/hemi_octa_k8 \
  2>&1 | tee logs/hemi_octa_k8.log
```

(With the v3 annealer add `--pself 0.9 --pseam 0.0 --reheat-every 12 --reheat-T 0.3
--stall-hook 4`; those flags are not in the current main copy of
`annealer_stretch.py`.) `hybrid_cover.py` is the complementary probe: keep the m
largest rectangles frozen and anneal only the remainder.


# NOTES:

What checks out
- Surface classification (answer 1): closed connected with H₁(;Z/2) = 0 ⟺ S²; with boundary ⟺ disjoint unions of disks and S². Möbius ≃ S¹, RP², torus, Klein all have H₁(;Z/2) ≠ 0 — correct. (Note H₁(;Z/2) = H₁(;Z)⊗Z/2, so the table is right.)
- All line refs verified: annealer_formal_properties.md:188-193 (Thm 3), :214-218 (winding/sparsity), klein_anneal_log.md:267 (Möbius --h1-guide masks nothing), s3_k2_push_log.md:163 ("no teeth"), dodeca_campaign_analysis.md:661-675 (cyclic-order vs K₅).
- nf = F²·C(4,2) = F²·6 for surface complexes ✓ (optimized_algorithms.md:564); 96/864/7776/1176/1536 all correct.
- Antipodal layer: deg A = det(−I) = (−1)^{n+1}; 1 − A_* = 0 (n odd) or 2 (n even) on H_n(−;Z); A_* = id over Z/2 at every n. The two-branch planner and "chainable ⟺ id ≃ A" are the classical proof of TC(S^n) = 1/2 (normalized). Degree-n (π₁−π₂)_* ≠ 0 obstructs a domain in any degree/coefficients — valid Theorem-2 analogue.
What's wrong
1. Torus TC row is a repo bug you inherited. klein_anneal_log.md:177,193 say torus TC = 3; but the repo's own convention is normalized — optimized_algorithms.md:571 and :952 say TC(T²) = 2, and Cohen–Vandembroucq's abstract literally says "(normalized) TC of the Klein bottle is 4". So "torsion lifts TC from the torus's 3 to Klein's 4" should read 2 → 4 (unshifted 3 → 5). The whole table is normalized except that one cell.
2. Torus runs mis-attributed. Best 733/1176 (62%) is torus_k4_c40_b5 (k = 4); k = 3 best is 579 (optimized_algorithms.md:665-672). "k=4 → 705, k=11 → 563" is really k = 4 → 8 → 11 = 733 → 705 → 563, and the k=8/k=11 runs only got 5 epochs (local-state/throughput), so the non-monotonicity is confounded.
3. Cube stall numbers. guided_sphere_k2_plan.md:49-52: cold k3 → 662/864 in 300 s; cube_k3_v2 ran 60 epochs (814 s) pinned at 820/864; cube_k3_guided closed 864/864 in 22 epochs. Not "662 for 60 epochs".
4. "SC⁰ = TC = 1 is the one value the level-0 annealer never attains" — false: the circle attains it (results/circle_probe3, 18/18 at k=2). True only for 2-dimensional instances at k=2 (that's C1).
5. "Every completed cover is level-0-sharp" — the Möbius k = 4, 5 covers are completed and not sharp (TC = 1). The repo's parenthetical list (circle, bary, cube, ∂Δ³/⁵/⁷, wedge) is the correct sharp set.
6. Punchline self-contradiction. "Every hard stall is invisible over Z/2 — and the surfaces are the lucky case where it is visible" contradicts its own table (row 1 says winding is visible). Correct: the Z/2 tool has teeth only on winding; every stall outside that family is out of its field of view.
7. "460-style ceiling on even spheres" — 460/500 is S³ (odd). Even-sphere k=2 ceilings are 84/96 (S²) and 2380/2520 (S⁴) (guided_sphere_k2_plan.md:30). All three are exactly the best 2-rectangle covers (rect2_optimum.py, C3) — the shared mechanism is the rectangle attractor, not parity; parity only decides whether a non-rectangle 2-cover exists.
8. "Dodeca failed at 23%" is stale: 23 % = escal-5 checkpoint 1831/7776; the same k=3 campaign later saturated at 2602/7776 = 33.5 % (dodeca_anneal_log.md "100 tries", 73 % of restarts gain 0).
9. Minor: "384 and 2400 facets — between Klein's 1536 and dodeca's 7776" — 384 < 1536. And "cup-length 3 → 4" for Klein: Z/2 zero-divisor cup-length gives 3 (over Z only 1 — the mod-2 structure is why), and the last step to 4 is Cohen–Vandembroucq's theorem, not cup-length.
Improved experiment plan
The density ladder idea is right; here it is with the missing invariants computed (I verified f-vectors, links, χ = 2, dual diameters, and ρ by the repo's own set-cover bounds):
K	V/F	density	dual diam	ρ
∂Δ³ (S²)	4/4	100 %	1	3
octahedron	6/8	40.0 %	3	4
cube	8/12	21.4 %	4	8
icosahedron	12/20	9.1 %	5	24
dodecahedron	20/36	3.2 %	7	36
Density and dual diameter are both monotone down the ladder, and ρ explodes — so octa/icosa at k = 3 cleanly separate "sparsity threshold" from "dodeca-specific". Run both; prediction if density is the disease: success at 40 %, failure or heavy stall at 9.1 %.
Second test — C1, stated correctly (optimized_algorithms.md:965): SC_strict⁰(∂Δ⁴) = 2 > 1 = TC(S³). The 16-cell (S³, 8 vertices, 16 tetrahedra, nf = 16²·20 = 5120) is the right probe: it has a simplicial antipodal map (unlike ∂Δ⁴, which has no A at all — 5 vertices), so the antipodal branch is an honest subcomplex and the only question is facet-respect. If it also pins at its 2-rectangle optimum, the gap is triangulation-independent; if it covers at k=2, C1 is a fact about ∂Δ⁴'s K₅ combinatorics (dodeca_campaign_analysis.md:661-675) and SC_strict⁰ depends on the triangulation.
Third — don't build a tool for the degree-n statement. The even-sphere k=2 wall is already certified by TC(S^{even}) = 2 ≤ SC_strict (González, Cor. 3); 1 − A_* = 2 is a 5-line lemma (deg A = (−1)^{n+1}) for the paper, and its only code-level payoff is per-domain certification (e.g. "the antipodal branch is not one domain"). Two implementation notes: the cube and dodeca facet lists in build() are not A-invariant (cube face diagonal 0–5 maps under (0 6)(1 7)(2 4)(3 5) to 3–6, absent from the opposite face's fan) — use cross-polytopes only; and the octahedron already serves both tests (density rung + A-simplicial S² control), so only the 16-cell and icosa need new facet lists. Optional genus-2: 10 vertices / 24 triangles (χ = −2), nf = 3456, H₁(;Z/2) rank 4 — fine as a rank test, but record TC in the normalized convention so it doesn't inherit the torus row's bug.



Short answer: not one disease — two, with the same dynamical signature. And "search limitation" gets a different verdict per family, which the repo already splits cleanly.
Topologically: no correlation (they're complementary)
family	homology
odd spheres (S³, S⁵)	H₁ = 0, and 1−A_* = 0 in every degree, over Z and Z/2
winding surfaces	H₁(;Z/2) ≠ 0
The two never overlap: odd dimension and H₁ ≠ 0 are mutually exclusive here. So there is no shared topological cause — if there were, the H₁ tool would fire on the spheres (it can't) or be vacuous on the surfaces (it isn't).
Dynamically: identical signature, one shared cause in the annealer
Both stall types show the same log shape: population collapses to one shape family, spread zero from epoch 1, --h1-guide/--crossover inert, extension moves dead, plateau at an exact combinatorial number. That number is always the ceiling of the reachable family:
- spheres at k=2: 84/96, 460/500, 2380/2520 = the proven best 2-rectangle covers (rect2_optimum.py);
- surfaces: shape-family optima (58+60+48 on Möbius, star rectangles on torus).
The shared cause is documented: single-entry + track moves cannot cut a domain (klein_anneal_log.md §6 — restructuring is the one missing move class), and on Möbius the barrier is coverage-structural, not weight-dependent (every repair costs ≥6 covered, swap nets −5). That part is a search limitation — with a proven reason, not a tuning problem.
The real correlation: SC_strict⁰ > TC
Every stall is a discretization gap; every completed cover is sharp. But the gap has two unrelated sources:
1. winding — topology survives discretization (Möbius: SC_strict⁰ ≥ 2 > 1 = TC, gap of exactly one unit, mobius_sc_strict.tex);
2. parity — the continuous planner isn't facet-respectable (C1: SC_strict⁰(∂Δ⁴) = 2 > 1 = TC; the cyclic-order/K₅ argument is the only positive evidence).
So: limitation or answer?
- Möbius k=3 — proven pure dynamics. A cover exists (mobius_k3_sat, CP-SAT, 42 s: 65+43+42) and the annealer never enters that family. annealer_formal_properties.md:140-142 says it outright: "the annealer's failure is purely its own dynamics, on a solvable instance."
- RP² k=5, Klein k≤8 — open, and either SAT answer is a theorem (SC_strict⁰(RP²) = 4 or 5).
- Odd spheres k=2 — probably the answer, not a failure. Pinning at exactly the 2-rectangle optimum is what the true level-0 answer looks like (C1), and the K₅/no-cyclic-order argument says the circle's mechanism doesn't lift. But it's a conjecture: nothing proves 460/500 is optimal among all 2-chain covers.
The one run that settles your question
python paper-overleaf/paper/scs_sat.py --complex s3 --k 2 --L 8 --timeout 7200
COVER(2, c, ∂Δ⁴) on 500 facets is the smallest open instance in the repo and the same instrument that settled the Möbius in 42 s. UNSAT ⇒ C1 becomes a theorem, the odd-sphere stall is the true answer, and the correlation you sense is purely the annealer's attractor; SAT ⇒ it's a search limitation after all, and the rectangle attractor (C2/C3) is the culprit on both families. Until that returns, the honest claim is: surfaces = certified search failure, odd spheres = conjectured discretization gap — same signature, different epistemic status.





Yes — there is a correlation, and it's sharper than "winding vs not". Your SAT run changes the picture first:
1. Your INFEASIBLE is the C1 theorem
∂Δ⁴, k=2, L=8: no two contiguity chains cover all 500 facets. Together with the known 3-cover (ρ=3, 2-link chains): SC_strict⁰(∂Δ⁴) = 2 > 1 = TC(S³) — a one-unit discretization gap, exactly like the Möbius band (mobius_sc_strict.tex). So the odd-sphere stall is the true level-0 answer, not a search failure — the annealer pinning at the proven 2-rectangle optimum (460) was it reporting the optimum. One caveat to close: the verified covers elsewhere use chains up to 9–10 links (e.g. klein_k13), so rerun --complex s3 --k 2 --L 16 to lift "within L≤8" to essentially unconditional (the model grows only in the map vars).
2. Your Klein ladder
k	result	evidence
15 → 10	covered (1536/1536), verified	SC_strict⁰(Klein) ≤ 9 — first bound below ρ−1 = 11
9	stall 1491/1536, 300 epochs, 6.1e9 moves	seeded from the k=10 cover minus its smallest domain, still stuck at 45 leftovers
8	stall 1483/1536 (53 leftovers)	old run
Sharp boundary: k≥10 covers in 4–28 epochs (minutes); k=9 is dead after 8.6 h. Also note the new moves don't fire there: stretch accepted 3%, collapse 0%.
3. The correlation — I ran the leftover analysis across all stalls
stall	|L|	rank(π₁−π₂)_* on L	rank(D_j ∪ L)	per-facet absorption
Klein k=8	53	2 (L winds, H₁=4)	2 for all 8	53/53 dead
RP² k=3	202	1 (L winds, H₁=2)	1 for all 3	200/202 dead
Klein k=9	45	0	2 for all 9	43/45 dead
RP² k=5	60	0	1 for all 5	60/60 dead
Möbius k=3	2	0	1 for all 3	0/2 dead (each f absorbable, pair not)
S³ k=2	40	vacuous (H₁=0)	—	tool silent
The rule that falls out (3/3 boundary predictions correct):
- rank(L) ≠ 0 — the leftover complex itself winds; no fresh chain can take it, even alone. Stall survives k→k+1.
- rank(L) = 0 — L is chainable; the plateau is exactly one domain short; k+1 covers immediately.
- Leftover geometry matches everywhere: far pairs (Klein k=9: \|s∩t\|≤1, dual dist 2–3; RP² k=5: all \|s∩t\|=1, dual 2), fully-dead cells (Klein 4 cells 6/6 dead; RP² 10), defect depth exactly 1 on every leftover, plateau domains near-exact star rectangles (RP² k=5: 4 of 5 are exact diagonal rectangles).
4. So: three stall species, one log signature
The stall signature is identical across all families (zero top10 spread, defect depth 1, dead extension, exact plateau number) — that's the annealer. The cause splits, and the arbiter is SAT + rank(L):
1. The answer is the plateau — S³ k=2 (your UNSAT), even spheres k=2 (TC floor).
2. One domain short, winding-locked — Klein k=9, RP² k=5: extension provably dead for the existing domains (43/45, 60/60), but L chainable; existence of the k-cover still open.
3. Pure dynamics — Möbius k=3: cover exists, leftovers H₁-open individually, only the joint D_j∪L is obstructed (the two facets must be split across domains — exactly what the SAT cover 65+43+42 does).
5. Next runs (all decisive, one per open question)
python paper-overleaf/paper/scs_sat.py --complex s3   --k 2 --L 16 --timeout 720000   # close the L-caveat
python paper-overleaf/paper/scs_sat.py --complex rp2  --k 5 --L 8  --timeout 86400    # cold (docs' open item)
python paper-overleaf/paper/scs_sat.py --complex klein --k 9 --L 8 --timeout 86400    # new boundary
Prediction from the rank(L) rule: k=9 UNSAT (so SC_strict⁰(Klein) = 9) and k=5 UNSAT (SC_strict⁰(RP²) = 5) — either would be the first exact SC_strict⁰ for a closed non-orientable surface. The klein model is ~40× the s3 one (16 triangles, 1536 facets); if it's too heavy, the hybrid mode (8 frozen plateau domains + 1 free) answers "can this plateau be completed" at ~1/9 the size. And log rank(L) at every plateau going forward — it's a seconds-cheap predictor of the ladder boundary.



L-caveat closed. L=16 is INFEASIBLE and subsumes all smaller L (shorter chains are special cases of ≤15-link chains), and every verified cover in the repo uses ≤10 links — so this is unconditional:
SC_strict⁰(∂Δ⁴) = 2 > 1 = TC(S³) — the s3 stall is the true answer, second exact level-0 value with a gap after the Möbius band. The s3 k=2 plateau at 460/500 was the annealer reporting the optimum, not failing.
Scaling across your four runs is clean and confirms the model formula y = k·NF·(L−2)·NT:
L
4
5
8
16
Near-linear in model size, trivially tractable. Extrapolating to the open boundary runs:
- rp2 k=5, L=8 — 5·600·6·NT y-vars, ~2–3× the s3 L=8 model. Should finish in minutes. Never run; the docs' open item (SC_strict⁰(RP²)=5 prediction).
- klein k=9, L=8 — 9·1536·6·16 ≈ 1.3M y-vars, ~40× s3 L=8. Hours, possibly overnight; if too heavy, hybrid mode with 8 frozen plateau domains + 1 free.


