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
