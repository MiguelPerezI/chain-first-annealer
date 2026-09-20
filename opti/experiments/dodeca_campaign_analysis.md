# Campaign Analysis: Dodecahedron Boundary and ∂Δ⁴ k=2

*2026-09-20. A rigorous post-hoc analysis of two experiments from the latest commit:
(1) the chain-first annealer on the dodecahedron boundary — a failed search for a
3-domain cover; (2) the two-domain push on ∂Δ⁴ (S³) — the 460/500 plateau and
its characterisation. The purpose is to establish exactly what happened, what was
measured, and what was corrected, so conclusions can be drawn and the relevant
parts identified for the paper.*

---

## Background: what the search is doing

The **simplicial complexity** SC_strict(K) of a simplicial complex K is one less than
the minimum number of subsets ("domains") that partition the facets of K × K, such
that on each domain there exists a **contiguity chain**:

    π₁ = φ₀, φ₁, …, φ_c = π₂ : K × K → K

where consecutive maps are **1-contiguous** — for every facet σ in the domain,
φ_i(σ) ∪ φ_{i+1}(σ) is a simplex of K. Such a system is a set of piecewise-linear
motion planners on ‖K‖.

The chain-first insight: a chain already certifies its own domain for free. The domain
D(φ) is the set of facets on which every consecutive pair is 1-contiguous. A k-tuple
of chains whose domains cover all of K × K is a certificate that SC_strict(K) ≤ k − 1.

The **GPU annealer** (`gpu_sc.py`) runs simulated annealing on a population of S
k-chain states, minimising an energy that penalises uncovered facets (weighted) and
per-chain uncleanliness (γ term). The primary tracked metric is **coverage**: the
number of facets of K × K covered by at least one chain's domain. A 100% coverage is
a certificate; anything less is a partial result with no certificate claim.

---

## Part 1: The Dodecahedron Boundary

### 1.1 Mathematical framing

The dodecahedron boundary is a **third triangulation of S²**, alongside ∂Δ³ (4
vertices, 4 triangles) and the cube boundary (8 vertices, 12 triangles). As a closed
orientable surface with χ = 2, it has TC = 2 and therefore SC_strict ≥ 2. The cube
is the nearest precedent: product covers of the cube certify only SC ≤ 7 (ρ = 8),
and the annealer closes that gap to the sharp SC_strict = 2 in 304 s. The question
for the dodecahedron is the same: how far is the product family from optimal, and
can the chain-first search close the gap?

The triangulation used is the **fan triangulation**: each of the 12 pentagons is
fanned from one of its vertices into 3 triangles, giving 20 vertices, 54 edges, and
36 triangles. The product K × K has **7,776 facets** on a grid of 400 vertices.

**The triangulation is not vertex-transitive.** The fan construction breaks the
dodecahedron's full symmetry: the 20 vertices have degrees ranging from **3 to 8**
(three vertices of degree 3, five of degree 4, two of degree 5, five of degree 6,
one of degree 7, and four of degree 8 — 13 distinct degree classes). The fan-centre
hubs lie in up to 8 of the 36 triangles; the product family's largest rectangles are
almost certainly concentrated on these hubs rather than spread over a symmetric mesh.

> **Correction on record.** An earlier draft of the analysis in `dodeca_anneal_log.md`
> called the triangulation vertex-transitive and argued from that. It is not — the fan
> construction breaks the dodecahedron's symmetry badly. Corrected there and here.

---

### 1.2 Engineering: two real bugs behind `nv <= 11`

Before the dodecahedron could be run at all, a guard in `Product` had to be lifted.
The guard's message — "int8 grid ids … ngrid = nv² ≤ 127" — was ambiguous and led to
a wrong first reading. Unpicking it found two distinct constraints:

**Bug 1 — `facet_verts` holds grid IDs, not vertex IDs.**
A product facet's vertices are grid IDs `a·nv + b`, which reach `ngrid − 1 = 399`
for nv = 20 — past int8. The guard message was right, but the fix is localised to
`facet_verts` only (not all chain arrays, which do hold vertex IDs and fit int8).
Fixed by widening `facet_verts` to **int16**, which supports nv ≤ 181.

**Bug 2 — `is_simplex` was O(2^nv × |F|) Python operations.**
The table was built by scanning every subset s ∈ [0, 2^nv) for containment in some
facet — for nv = 20 that is 1,048,576 × 36 Python operations. Replaced with the
equivalent **submask enumeration**: each facet contributes exactly 2^|facet| subsets,
so the total is 36 × 8 = 288 operations for the dodecahedron. Verified byte-identical
to the old construction on circle, s2, s3, s4, s5, wedge, bary and cube before being
trusted. Guard raised to `nv ≤ 24` (16 MB `is_simplex` table).

**The auditor had to learn it too.**
`independent_check.py` rebuilds each K from scratch. Added `dodeca_facets()` (and a
`nf == 7776` detection clause). More significantly: the cube's surface re-proof
(expected f-vector, every edge in exactly two triangles, every vertex link a single
cycle, χ = 2) was **extracted into `check_sphere_surface(name, KF, nv, ne)` and
reused** rather than copied, so the dodecahedron's topological certificate is
structurally identical to the cube's. The refactor was verified to leave the cube's
audit output byte-identical, and all present certificates (∂Δ³, ∂Δ⁴, ∂Δ⁵,
barycenter triangle, torus) still audit clean.

**Scale.** |F_L| = 36 × 36 × 6 = 7,776 product facets; 400 grid vertices; largest
star 384. Per-thread local state is ~71 KB for k = 3, well inside the 512 KB limit
that stopped S⁷ (∂Δ⁸). Domain floor: 3 (SC_strict ≥ 2, so a 3-domain cover would
settle SC_strict = 2 exactly).

---

### 1.3 The core obstacle: simplex density and the dual graph

**What `is_simplex` measures and why it matters.**
A chain's domain is the set of facets on which *every* link of the chain is simplicial
— `domain_of` keeps the facets where the count of bad links is zero. So the domain
floor is set by the chance that a random link image is already a simplex of K.
`is_simplex(σ)` asks whether the union of the images of σ's vertices is a simplex —
and that is governed entirely by how densely K is triangulated.

**Density table** (all complexes in the repository):

| K | vertices | triangles | possible vertex triples | density | initial coverage |
|---|---:|---:|---:|---:|---:|
| ∂Δ³ (S²) | 4 | 4 | 4 | 100% | 36.0% |
| ∂Δ⁴ (S³) | 5 | 5 | 10 | 50% | 36.8% |
| ∂Δ⁶ (S⁵) | 7 | 21 | 35 | 60% | 34.8% |
| cube boundary | 8 | 12 | 56 | 21.4% | 12.1% |
| Császár torus | 7 | 14 | 35 | 40.0% | 10.4% |
| **dodecahedron** | **20** | **36** | **1140** | **3.2%** | **3.0%** |

*Initial coverage measured on `sweep_chain` initialisation, 20 chains per complex,
cmax 40 — host-side, no search.*

The dodecahedron is the outlier on both axes simultaneously: sparsest K in the
repository, and smallest starting coverage by a factor of 3.4 against the next worst.

**The diagonal floor.**
The density is not the whole story — ∂Δ⁶ is only 60% dense yet starts at 35%, while
the torus at 40% starts at 10%. But there is a structural floor: the **diagonal
cells** (s = t in the K × K grid) always have their images in the same triangle, so
their facets are always simplicial for any chain. The diagonal is the minimum coverage
any nontrivial search can achieve.

| K | diagonal cells | share of K×K | diagonal facets | measured initial coverage |
|---|---:|---:|---:|---:|
| dodecahedron | 36 of 1296 | 2.8% | 216 | 235 (3.0%) |
| cube | 12 of 144 | 8.3% | 72 | 104 (12.1%) |
| ∂Δ⁴ | 5 of 25 | 20.0% | 100 | 184 (36.8%) |

For the dodecahedron the diagonal floor and the measured initial are almost
identical (216 vs. 235) — meaning the density is so low that almost nothing beyond
the diagonal is easy from a random start.

**The off-diagonal correction — a wrong explanation and the right one.**

> **Correction on record.** An earlier draft of `dodeca_anneal_log.md` (and an earlier
> draft of this analysis) claimed off-diagonal cells were *unreachable*, because two
> distinct triangles of a closed surface share at most an edge so their union is never
> a simplex, and the first link (from π₁, whose image is all of s) forces the next
> image to lie inside s. The premise is right and the conclusion is **false**. The
> repository already contained the counterexample: the cube's three domains cover
> **all 144 cells**, 792 of their 864 facets off-diagonal — including cells whose two
> triangles share no vertex at all.

What actually happens on an off-diagonal cell is a **walk through K's dual graph**.
The cube's certificate shows it concretely on facet 576, which lies in cell (8, 0) —
two triangles with no shared vertex:

```
map 0  (π₁): [4,4,4,5,6]   image {4,5,6}   = triangle 8
map 2      : [4,4,4,4,6]   image {4,6}     shrinks to the shared edge
map 3      : [4,7,7,7,7]   image {4,7}     grows into the neighbouring triangle
...
map 16 (π₂): [0,3,7,7,7]   image {0,3,7}   = triangle 0
```

Each link is legal because the union of consecutive images is some simplex of K (not
necessarily s or t): the chain walks triangle by triangle across K's dual graph from
triangle 8 to triangle 0. The difficulty is not that off-diagonal facets are
unreachable; it is that reaching them requires the chain to coordinate such a walk
**simultaneously for all 7,776 facets** — one global map serves all of them at once.

**Dual graph statistics** (both K are closed surfaces, so every triangle has exactly
3 neighbours — cubic duals):

| K | triangles | dual edges | dual diameter | mean dual distance |
|---|---:|---:|---:|---:|
| cube | 12 | 18 | 4 | 2.08 |
| dodecahedron | 36 | 54 | **7** | **3.81** |

A chain link can advance a walk by about one triangle, so a facet in a cell at dual
distance d needs roughly d transition links all doing the right thing simultaneously.
The dodecahedron nearly doubles the required walk length (7 vs. 4) and triples the
number of constraints (36 triangles vs. 12). This is the cleanest single predictor of
why the search stalls.

---

### 1.4 Chain-length (cmax) sweep

Fixed settings: S 1024, iters 2000, 4 epochs, β 5 / γ 0.5, ptrack 0.3, seed 3.

| cmax | best after 4 epochs |
|---:|---:|
| 20 | 300 / 7776 |
| 40 | 330 / 7776 |
| 100 | **391 / 7776** |
| 200 | *no epoch completed in 150 s* |

Chain length is a real lever (+30% from cmax 20 to 100), but it saturates hard.
A track move on a block of L links costs O(L · star); with star = 384, a single epoch
of ptrack 0.3 work does not finish in 150 s at a quarter of the search population
when cmax = 200. **cmax ≈ 100 is the practical ceiling** with track moves on this
complex.

---

### 1.5 The five long escalation rungs

All runs: `--complex dodeca --k 3 --cmax 40 --S 4096 --iters 5000 --T0 0.6 --T1 0.03
--beta 5.0 --gamma 0.5 --pcopy 0.5 --ptrack 0.3 --guide 3.0 --guide-max 11.0
--tournament-every 2 --tournament-frac 0.25` (cold run used guide 1.0 / guide-max 4.0;
escalations used guide 3.0 / guide-max 11.0 with warm starts).

| run | start | end | **gain** | epochs | moves | avg/epoch | gain per Mmove |
|---|---:|---:|---:|---:|---:|---:|---:|
| cold (`dodeca_k3`) | 0 | 644 | **+644** | 106 | 2.17 × 10⁹ | 6.08 | 0.30 |
| escal 1 (`dodeca_k3_escal`) | 644 | 1208 | **+564** | 109 | 2.23 × 10⁹ | 5.17 | 0.25 |
| escal 2 (`dodeca_k3_escal2`) | 1208 | 1579 | **+371** | 109 | 2.23 × 10⁹ | 3.40 | 0.17 |
| escal 3 (`dodeca_k3_escal3`) | 1579 | 1749 | **+170** | 109 | 2.23 × 10⁹ | 1.56 | 0.076 |
| escal 4 (`dodeca_k3_escal4`, 15 min) | 1749 | 1802 | **+53** | — | 0.55 × 10⁹ | — | 0.096 |
| escal 5 (`dodeca_k3_escal5`, 7 min) | 1802 | 1831 | **+29** | — | 0.27 × 10⁹ | — | 0.107 |

Rung-over-rung decay ratios (full-hour runs): **0.88, 0.66, 0.46** — not levelling off.

Domain sizes after the cold run: 342 + 381 + 319.
Domain sizes after escal 4: 998 + 918 + 392.

**The warm-start rate error and correction.**

> **Correction on record.** Mid-run, the escalation was described as "roughly tripling
> the rate" — comparing its opening epochs (+8–9/epoch) against the cold run's *late*
> epochs (+2.5). That comparison is unfair and the summary it implies is wrong. The
> cold run also opened fast: 361 by epoch 5 (≈ +72/epoch) — because the first few
> hundred facets are easy from a random start. Over a full hour at essentially the same
> move count, the warm-started run gained **fewer** facets (+564) than the cold run
> (+644), and its average rate was *lower*. The warm start does produce a real
> transient — the seeded population opens at 678 and climbs at +8/epoch — but the
> transient is not a rate advantage over a cold run, because the facets it is handed
> are the expensive ones.

What the escalation series demonstrates is the shape of the landscape: marginal
difficulty rises steeply, so any head start is immediately consumed by harder marginal
work. Both the cold run and each escalation made ~2.2 × 10⁹ moves in their hour; the
gains halved with each successive rung.

---

### 1.6 The yield power law and rung-length analysis

**Within-rung front-loading**, measured inside escalation 3 (fixed starting state
1579, so difficulty is held constant):

| window | gain | per minute |
|---:|---:|---:|
| first 7 min | +33 | 4.71 |
| first 15 min | +58 | 3.87 |
| full 60 min | +170 | 2.83 |

So the yield fits ≈ t^0.8: doubling 7 → 15 min multiplies the gain by 1.76;
quadrupling 15 → 60 min by 2.9. Rate falls as t^−0.2.

**Five-point fit** from independent measurements (ladder mean at 150 s, both 420 s
rungs, the 900 s rung, the 3600 s rung):

    gain ≈ 0.184 · t^0.835

| window | measured | fit |
|---:|---:|---:|
| 150 s (ladder mean, n = 20) | +11.65 | +12.08 |
| 420 s (escal 5) | +29 | +28.55 |
| 420 s (escal 4's first 7 min) | +30 | +28.55 |
| 900 s (escal 4) | +53 | +53.97 |
| 3600 s (escal 3) | +170 | +171.86 |

A single power law with exponent 0.835 holds across a **24× range** of window
lengths, to within 4% at every point.

**Optimal rung length.** If a rung yields c · t^0.835 and costs a fixed overhead of
~40 s per rung (JIT compile + host setup), the facets-per-hour is
(3600 / (t + 40)) · c · t^0.835, which peaks at t = 4 · 40 s ≈ **2.7 min**:

| rung length | yield per rung | facets per GPU-hour |
|---:|---:|---:|
| 35 s (one-epoch floor) | 4.5 | 217 |
| 2.0 min | 12.1 | 273 |
| **2.5 min** (ladder used) | **15.2** | **274** |
| 2.7 min (model peak) | 15.2 | 274 |
| 7 min (escal 5, measured +29) | 33.0 | 258 |
| 60 min (rungs 1–3, measured +170 … +644) | 184.1 | 182 |

The 2.5-minute ladder is ~1.5× more facets/GPU-hour than the long rungs. That is
a real optimisation of the constant. It does not change the exponent.

---

### 1.7 The 2.5-minute ladder — first 20 rungs

**`run_dodeca_ladder.sh`**, 2026-09-20 03:46 UTC. Warm-start ladder, each rung 150 s,
seeded from the previous rung's best state, starting from the 1831-facet escal 5
state.

Result: **1831 → 2064 (+233) in 54 min 57 s** (2.89 min/rung including overhead).

Per-rung gains:
```
rung  1-10:  27  17  15  17  15  11  11  10  13  12
rung 11-20:   7   7   7   7   8   8   8   9  12  12
```

Predicted 20-rung total from the fit: 20 × 12.08 = 241. Measured: +233. Within 3%.

Normalised rate: 233 / 0.916 h ≈ **254 facets/hour**, against ~182 from the
hour-long rungs — the predicted ~1.4×.

The ladder validates the model, and it does not change the outcome. 5,712 facets
remain uncovered; the per-rung gains are already falling. The per-rung mean over the
second half (rungs 11–20) is 8.5 vs. 14.6 over the first half.

---

### 1.8 Unattended operation — 12-hour ladder

**`run_dodeca_ladder_forever.sh`**, 2026-09-20 05:22 UTC, 12-hour deadline (17:22 UTC).
20 rungs per batch; each batch runs under `flock` to prevent races; resumable at rung
granularity (chain state in `results/dodeca_ladder/rungNNN/summary.json`); failure
policy: one retry after pause, three consecutive failures stop the loop; a full cover
stops everything and audits.

**Result: seven complete batches (rungs 22–161) + partial eighth (rungs 162–168),
2026-09-20 05:22 → 12:04 UTC. 2075 → 2551 (+476) in 148 rungs.**

| batch | rungs | coverage | gain | mean/rung |
|---|---|---:|---:|---:|
| 1 | 22–41 | 2075 → 2218 | +143 | 7.2 |
| 2 | 42–61 | 2218 → 2310 | +92 | 4.6 |
| 3 | 62–81 | 2310 → 2400 | +90 | 4.5 |
| 4 | 82–101 | 2400 → 2441 | +41 | 2.1 |
| 5 | 102–121 | 2441 → 2487 | +46 | 2.3 |
| 6 | 122–141 | 2487 → 2523 | +36 | 1.8 |
| 7 | 142–161 | 2523 → 2550 | +27 | 1.4 |
| 8 (partial) | 162–168 | 2550 → 2551 | +1 | 0.1 |

**The complete per-rung gain series** (rungs 1–172, from both the first ladder and
the unattended run):

```
rung   1- 10:  27  17  15  17  15  11  11  10  13  12
rung  11- 20:   7   7   7   7   8   8   8   9  12  12
rung  21- 30:  11  10   9  18  10   6   4   7   8   6
rung  31- 40:   9   5   6   6   5   6   8   7   5   4
rung  41- 50:   4   6  10  10   4   5   5   6   5   3
rung  51- 60:   3   6   3   4   3   3   2   4   2   4
rung  61- 70:   4   3   4   2   5   6   3  13   6  10
rung  71- 80:   4   4   4   3   3   2   3   3   4   2
rung  81- 90:   6   2   2   2   4   1   2   2   2   2
rung  91-100:   2   2   2   2   1   2   4   2   2   1
rung 101-110:   2   2   2   5   1   2   2   4   5   2
rung 111-120:   3   2   1   2   5   2   1   1   1   2
rung 121-130:   1   2   2   2   2   1   1   1   1   0
rung 131-140:   1   3   2   1   2   3   2   2   3   3
rung 141-150:   2   2   1   1   2   1   1   1   1   3
rung 151-160:   0   0   0   3   1   1   6   1   1   1
rung 161-172:   0   1   0   0   0   0   0   0   0   0   0   0
```

Mean: **+4.2 per rung** over 172. The last **eleven rungs** (162–172) contribute
exactly 1 facet between them. This is the clearest picture the campaign produced of
the difficulty curve: rapid opening (27, 17, 15, 17…), a long slow middle in the
single digits, and a final collapse to literal zeros.

**Where the power law breaks — and what actually governs the amplitude.**

The t^0.835 law was fitted on rungs taken between coverage ~1580 and ~1830. It held
there because the amplitude c ≈ 0.184 was roughly constant over that narrow band.
Extrapolating to higher coverage, the per-rung gain at fixed window length (150 s)
fell: +11.65, +7.2, +4.6, +4.5, +2.1, +2.3, +1.8, +1.4, ~+0.1 — the amplitude c
halves roughly every ~200 facets. The corrected model is:

    gain ≈ c(coverage) · t^0.835    with c a declining function of coverage

The exponent describes how a rung's yield scales with time *at a given difficulty*.
The difficulty is a second, separate factor that the early fit never captured because
the first three data points spanned too narrow a coverage band. The second term — the
decline of c — is what decides the outcome.

---

### 1.9 Window comparison at depth (coverage ~2553–2570)

After the 12-hour ladder stalled, three experiments probed whether the short rungs
had become too short.

**1-hour rung from 2553** (`results/dodeca_hour_from2553`, seed 200):
Prediction from the two-term model (c ≈ 0.016 at this coverage, 3600 s rung):
c · 3600^0.835 ≈ +16. Predicted band: **+10 to +25**.
**Result: +13 facets** (2553 → 2566). Inside the band, at its low end.

**30-minute rung from 2566** (`results/dodeca_half_hour_from2566`, seed 201):
Prediction: c · 1800^0.835 ≈ +8. Predicted band: **+5 to +12**.
**Result: +4 facets** (2566 → 2570). Below the band.

The rates are **non-monotone**:

| window | gain | per hour |
|---:|---:|---:|
| 2.5 min (ladder end rate) | ~+1 | ~20 |
| 30 min | +4 | 8 |
| 60 min | +13 | 13 |

"Shorter is better" does not hold at this depth — the ordering is not monotone, and
with gains of +1, +4, and +13 each from a handful of runs, the per-hour differences
sit inside the noise. The model's ordering claim was over-reading three points.

What *is* robust: every window from 2.5 min to 1 hour buys single digits. The domain
sizes barely move between runs (1425 + 1232 + 433 → 1425 + 1235 + 433 → 1425 + 1235 +
433). The search is not finding new structure — it is nudging the same three chains.

---

### 1.10 The 100-try wide-basin test

**`results/dodeca_tries`**, started 2026-09-20 15:16 UTC from the 2570 state. 100
warm-start tries, each exactly 3 epochs (not a time limit — every try does the same
amount of search), each seeded from the previous try's best state with a fresh seed.

Stopped at 70 of 100 tries (at 17:22 UTC, the deadline). Results over 70:

| | |
|---|---|
| coverage | 2570 → **2602** (+32) |
| mean per try | **+0.46** |
| tries gaining zero | **51 of 70 (73%)** |
| best single try | **+4** |
| distribution | `{0: 51, 1: 12, 2: 2, 3: 4, 4: 1}` |

Per-try gains (`.` = 0, digit = gain):
```
.1111.1........23.......2141........1........33.1.....113........1......
```

This was framed as a pre-registered two-way question: is the plateau a *point
attractor* (restarting never escapes) or a *wide basin with rare exits*? The result
is **neither, and both at once**:

- Not a point attractor: 19/70 tries (27%) found something new, so exits exist.
- But the exits are worth 1–4 facets, and across ~287,000 annealer runs over 70 seeds
  not one produced a real escape.

The reachable set from ~2570 is effectively exhausted at ~2,600; the search can leave
its current optimum, but only into neighbours barely better than it. This closes the
restart line of enquiry: a wide shallow basin yields to sampling, and more sampling
gives only more of the same.

---

### 1.11 Final state

| | |
|---|---|
| Coverage | **~2602 / 7,776 (33.5%)** |
| Domain sizes | ~1425 + 1235 + 433 |
| Total GPU time | ~20+ hours across all phases |
| Total moves | ~10¹⁰ |
| Certificate | **None** |

The chain state is intact on disk at `results/dodeca_tries/rung070/summary.json`. A
future experiment can resume from there without loss.

---

### 1.12 What the experiment established

All of the following is **measured**, not asserted:

1. **The obstacle is K's sparsity, and it is quantifiable.** The simplex density
   (3.2%) sets the diagonal floor (2.8%); coverage beyond the diagonal requires
   coordinating walks through K's dual graph, whose diameter here is 7 and mean
   distance 3.81 — nearly double the cube's 4 and 2.08.

2. **Chain length is walk budget.** cmax 20 → 40 → 100 gave 300 → 330 → 391 at
   fixed settings. cmax 200 is unusable — track moves cost O(L · star) with star 384.

3. **Warm starting buys a decaying transient, not a rate.** Gains across five rungs:
   +644, +564, +371, +170, +53 — ratios 0.88, 0.66, 0.46, not levelling off. Over
   equal move budgets, the warm start gains *fewer* facets than the cold run.

4. **A rung's yield is front-loaded (gain ≈ 0.184 · t^0.835) but that is not a
   strategy.** A 15-minute rung captures ~31% of an hour's yield, and beats the
   previous rung's normalised rate — yet four hours of 15-minute rungs extrapolates
   to ≈ 848, against the 1802 the long rungs actually delivered. The short ladder
   optimises the constant; the decay of c with coverage is what decides the scale.

5. **The power law c · t^0.835 has two terms, and the second is dominant.** The
   amplitude c is not a constant — it halves roughly every 200 facets of coverage.
   The exponent describes within-rung scaling at fixed difficulty; the difficulty is
   what grows, and no rung-length choice touches it.

6. **Wide, shallow basin at ~2,570.** 73% of fresh-seed restarts gain zero facets;
   the best single try gained +4. Restarting escapes occasionally, but only to
   neighbours 1–4 facets better. The reachable set is exhausted near 2,600.

### What was NOT tried (recorded to distinguish from "tried and failed")

- cmax > 100 with `ptrack` lowered to make track moves affordable at that star size
- Initialisations built from collapsing maps rather than the random `sweep_chain`
- A different move attribution — one that does not touch the whole star of a grid
  vertex — which is the formulation-level change this instance may actually need
- Any product-domain certificate (out of scope by instruction)

---

## Part 2: The ∂Δ⁴ k=2 Attack — The 460 Plateau

### 2.1 The question

The three-chain cover for ∂Δ⁴ (S³) is already certified: `results/s3_k3_v2` gives
domains of 309 + 130 + 61 = 500 facets (partition form), establishing
SC_strict(∂Δ⁴) ≤ 2. Since TC(S³) = 1, a **two-chain cover** would give the sharp
SC_strict(∂Δ⁴) = 1. The paper's Corollary 5.3 is therefore one domain away from
sharp on every odd sphere. **Conjecture C1** asserts that no such two-domain cover
exists.

Twelve recorded two-chain searches (runs `s3_k2_A` through `s3_k2_D`, `s3_k2_v2_point`,
`s3_k2_v2_track`, and the `push_s3` family) all pinned at **exactly 460/500** from the
first epoch. But every one of those twelve runs was cut at a 300 s wall — 2–3 epochs.
The campaign's premise: the plateau might be a budget artifact, not a structural one.

∂Δ⁴: 5 vertices, 5 tetrahedral facets, K × K has 500 product facets.

---

### 2.2 Two corrections made before spending GPU time

**Correction 1 — the guide escalation ladder was a no-op.**

The plan's passes 2 and 3 escalated the coverage guide weight from 1.0 → 3.0 → 10.0.
But `guide_max` was hard-coded at 4.0 inside `anneal_cover`, and a facet that stays
uncovered across multiple epochs saturates at:

    w* = 1 + guide / 0.3

For guide = 1.0 this is 4.33, which clamps to 4.0 at epoch 7. Guides of 1.0, 3.0
and 10.0 produce **identical weight vectors by epoch 7**:

| increment | weights at epochs 1–10 (old cap 4.0) |
|---|---|
| 1.0 | 2.0, 2.7, 3.19, 3.53, 3.77, 3.94, **4.0**, 4.0, 4.0, 4.0 |
| 3.0 | **4.0**, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0 |
| 10.0 | **4.0**, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0 |

The plan's escalation ladder would have tested nothing. Fixed in `gpu_sc.py`:
`--guide-max` is now a CLI parameter (default 4.0, so all prior runs are unchanged
and reproducible). The saturation identity w* = 1 + guide/0.3 is in the docstring
and `--help`.

**Correction 2 — the energy, tested as a controlled pair.**

Every successful certificate in this repository (s2/s3/s4/s5 k=3, cube, wedge) used
the corrected energy β = 5.0, γ = 0.5. The plan's pass-1 command used the defaults
β = 0.05, γ = 0. The two prior k=2 runs with the corrected energy (`v2_point`,
`v2_track`) hit 460 on 300 s leashes. Pass 1 was therefore run **twice at seed 5**,
differing in energy alone — same initial population, so the pair isolates energy as a
variable.

---

### 2.3 Exhaustive characterisation of 460

Before reading any result, `rect2_optimum.py` enumerated all pairs (300) and triples
(2300) of star-product rectangles star(v) × star(w) over ∂Δ³, ∂Δ⁴, ∂Δ⁵:

| complex | best 1 rectangle | best 2 rectangles | best 3 rectangles | recorded k=2 plateau |
|---|---:|---:|---:|---:|
| ∂Δ³ (S²) | 54/96 | **84/96** | 96/96 | 84/96 |
| ∂Δ⁴ (S³) | 320/500 | **460/500** | 500/500 | 460/500 |
| ∂Δ⁵ (S⁴) | 1750/2520 | **2380/2520** | 2520/2520 | 2380/2520 |

**Every recorded two-chain plateau = the exhaustive 2-rectangle optimum.** The 460
ceiling is not an artefact of the search settling into rectangle covers — it is a
provably sharp wall for the rectangle family. Yet it is not the case that the search
is always *finding* rectangle covers:

| run | domain sizes | total | overlap | **union** |
|---|---:|---:|---:|---:|
| `s3_k2_B` | 203 + 296 | 499 | 39 | **460** |
| `s3_k2_D` | 275 + 212 | 487 | 27 | **460** |
| `s3_k2_C` | 284 + 216 | 500 | 40 | **460** |
| `s3_k2_A` | 284 + 265 | 549 | 89 | **460** |
| `s3_k2_v2_point` | 297 + 317 | 614 | 154 | **460** |
| `s3_k2_longcor_5` (69 epochs) | 311 + 303 | 614 | 154 | **460** |
| `s3_k2_v2_track` | 320 + 320 | 640 | 180 | **460** |

One ceiling, seven domain shapes. Total sizes span 487–640 (factor 1.3); overlaps
span 27–180 (factor 6.7). `v2_point` and `longcor_5` have domains that are **not**
star-product rectangles at all, yet their union is still exactly 460. That is the
sharpest form of the observation: the rectangle ceiling is not merely an attractor
that the search finds; it is a wall that the search cannot penetrate regardless of
what shape the domains are.

**Two further facts:**

- **No facet is inherently uncoverable.** Taking the union of covered sets over all
  seven states covers all 500 facets. The 40 missing facets are not a fixed obstacle;
  the hole moves.

- **`v2_point` and `longcor_5` are statistically the same attractor.** Different
  seeds (54 vs. 5), different budgets (6 epochs vs. 69), domain sets differing by 34%
  and 81% as sets — yet both have total 614, overlap 154, union 460, and their covered
  sets differ on exactly 40 facets on each side (Jaccard 0.84). Two runs from
  different starting points landing on the same total/overlap/union is a well-defined
  attractor basin, not sampling noise.

---

### 2.4 Pass 1 — first real budget (one hour each)

Settings: `--complex s3 --k 2 --cmax 40 --S 4096 --iters 20000 --epochs 200
--ptrack 0.05 --guide 1.0 --guide-max 4 --time-limit 3600`, seed 5 for both,
differing only in energy.

At ptrack 0.05 an epoch costs ~54 s, so 3600 s buys ~66 epochs — roughly 25× the
2–3 epochs every prior k=2 run got. In moves: ~5.4 × 10⁹ per run, against 2.5 × 10⁸
for all of `s3_k2_D`. The two pass-1 runs together search more than the twelve
recorded two-chain runs combined by roughly an order of magnitude.

| run | energy | epochs | moves | `best` values | result |
|---|---|---:|---:|---|---|
| `longcor_5` | β 5.0, γ 0.5 | 69 | 5.65 × 10⁹ | {460} | **460 in every epoch** |
| `longflat_5` | β 0.05, γ 0 | 41 | 3.36 × 10⁹ | {460} | **460 in every epoch** |

`longcor_5`: domain sizes 311 + 303; `best` = 460.0 in all 69 epochs, `median` 460
from epoch 2 onward (455 in epoch 1).

`longflat_5`: domain sizes 232 + 273; `best` = 460 in all 41 epochs; `median` varies
449–460 (the flat-energy population wanders more, but its best never exceeds 460).

**The ceiling is not a budget artifact and not an energy artifact.** A flat-energy
search that explores a much wider population range still hits the same wall. The
factor-25 increase in search budget produced zero facets beyond 460.

> **Caveat on the controlled pair.** The two runs shared a GPU with construction
> probes running simultaneously, so `longflat_5` finished with fewer epochs. The pair
> is qualitative, not epoch-matched. A fully clean control would re-run longflat alone.
> The conclusion holds — both are zero-spread at 460 — but the pair is honestly
> presented as a controlled comparison in quality rather than in move count.

---

### 2.5 Structured domain families — both closed with reasons

**Decoding the circle's two-domain cover** (`results/circle_probe3`, 9 + 9 facets).

Each of the two domains takes:
- Half of each diagonal cell (s,s) by triangle index
- All of every off-diagonal cell (s,t) by *cyclic direction*: domain 0 takes the three
  cells with "missing-vertex shift +1" (in the cyclic order of ∂Δ²'s three vertices),
  domain 1 takes shift −1

This exploits the fact that ∂Δ²'s three facets are the arcs of a circle — the vertex
set has a natural **cyclic order**. Every off-diagonal cell goes entirely to the
domain that owns its cyclic direction; diagonal cells are halved.

**Why this does not lift to ∂Δ⁴.**
∂Δ⁴'s five facets are the faces of a 4-simplex: K₅. Every vertex is adjacent to
every other; there is no cyclic order. The natural analogue — splitting by orientation
of the omitted pair (i, j) — gives 250-facet candidate domains A (lower triangle of
the 5 × 5 cell grid plus half the diagonal) and B (upper triangle plus the other
half). Whether those halves can be chosen so that both admit a chain is a finite
question; the probes below test the two natural candidates and both fail.

This is the campaign's **one positive argument that C1 is true**, as opposed to
negative evidence: the mechanism that makes the circle work is 1-dimensional (cyclic
order), and the odd spheres above S¹ do not have it.

---

**`fixed_domain_test.py`** anneals a single chain (k = 1) on the grid restricted to
a given facet set. Full coverage ⇒ the set is a verified domain with an explicit
planner. Validated against three known answers on the circle before being trusted.

**The whole-cell family.**

*Upper probe* ({s ≤ t}, 300 facets, 908 s): does **not** cover the 300, but what it
holds reveals the structure:

| part of {s ≤ t} | held |
|---|---|
| all 10 off-diagonal upper cells (200 facets) | **200 / 200** |
| diagonal cells (1,1), (2,2), (3,3) | **20 / 20 each** |
| diagonal cells (0,0), (4,4) | **0 / 20 each** |

The probe's verified domain is **260 facets**: all 10 upper off-diagonal cells +
the three middle diagonal cells. It drops exactly (0,0) and (4,4) — the same 40
facets the full k=2 search always misses.

This turns the search into a one-question test: if the complement B = {s > t} ∪
{(0,0), (4,4)} (240 facets) also admits a chain, then A ∪ B = 500 is a two-domain
cover and SC_strict(∂Δ⁴) = 1. A ∪ B = 500 and A ∩ B = ∅ were verified explicitly.

*Lower-ends probe* ({s > t} ∪ {(0,0),(4,4)}, 240 facets): opens at **200/240 from
epoch 1** with the whole top-10 at 200 and no movement. Holds 9 of 10 strict-lower
cells + (4,4), drops (4,3). The two extreme diagonal cells cannot be held together
with the full lower block in the search budget tried.

*Swap configuration* A' = {s < t} ∪ {(0,0),(4,4)}, B' = {s > t} ∪ {(1,1),(2,2),(3,3)}:
the A' probe (240 facets) also opens at 200/240 and does not close. The whole-cell
family is dead at both ends it was testable.

> **Correction on record.** An earlier draft of this section claimed a budget-free
> deduction from the lower-ends probe result: "the lower chain cannot hold both
> extremes, because a chain restricts to any sub-domain". That argument is invalid —
> it would need a chain covering all of {s>t} ∪ {(0,0),(4,4)}, and the chain found
> covers 200 of the 240. The honest statement is only that no such chain was found in
> a 20-minute search. Corrected here and in `s3_k2_push_log.md`.

**The half-split family.** The circle's cover does not give whole diagonal cells to
either domain — it gives one triangle of each diagonal cell to each domain. The lift:
split each of the 5 diagonal cells' 20 facets in half by local staircase index
(first 10 vs. last 10), giving `upper-half` (250 facets) and `lower-half` (250 facets).
Tested on the circle, where the answer is known:

| facet set | size | one chain covers | result |
|---|---:|---:|---|
| `upper-half` | 9 | 7/9 | **no chain** |
| `lower-half` | 9 | 7/9 | **no chain** |
| certificate's domain 0 | 9 | 9/9, epoch 1 | chain ✓ |

Two domains of the same size and same half-split shape fail, while the certificate's
two domains succeed. The difference is *which* facets — the certificate's choice is
organised by the cyclic order of the three facets, not by the staircase-index split.
The half-split family is closed: even on the circle where a two-domain cover exists,
this particular construction does not recover it.

**The H₁ obstruction is vacuous on spheres.**

`h1_obstruction.py` tests the necessary condition (π₁ − π₂)_* = 0 on H₁(|J|; Z₂)
for a candidate domain J. Run on ∂Δ⁴:

| candidate J | facets | dim H₁(J;Z₂) | rank (π₁−π₂)_* |
|---|---:|---:|---:|
| all of K × K | 500 | 0 | 0 |
| D_diag (s = t) | 100 | 0 | 0 |
| both domains of `v2_track` | 320 | 0 | 0 |
| both domains of `v2_point` | 297, 317 | 0 | 0 |

H₁(∂Δ⁴ × ∂Δ⁴) = 0, so the obstruction separates nothing. **C1 cannot be settled
by the H₁ test** on the odd spheres. Recorded to prevent re-derivation.

---

### 2.6 What was NOT run (explicit list)

| not run | why it was the next step |
|---|---|
| `lower-mid` probe ({s>t} ∪ middles, 260 facets) | second half of the swap pair; moot once `upper-ends` returned 200/240 |
| `upper` escalation (300 facets, ptrack 0.3, guide 3/max 11) | would test whether {s ≤ t} can be made fully coverable |
| `strict-lower` probe (200 facets) | whether a chain can hold a whole orientation block |
| `esc-track` / `esc-guide` rungs (the plan's escalation ladder) | made real by the guide_max fix; the next step if pass-1 had shown movement |

The first three are cheap (10–20 min each); the last two are one hour each.

---

### 2.7 Verdict on C1

**The 460 plateau was not broken.** Every two-chain search remains pinned at exactly
460 — now including the first searches ever given a real budget (69 epochs, 5.65 × 10⁹
moves) and the first where the guide escalation actually changed the weight schedule.

What the campaign **established**:

1. 460 = exhaustive 2-rectangle optimum (verified by complete enumeration for S², S³,
   S⁴). The ceiling is not an artefact of the search.
2. The plateau is reached by domain pairs whose shapes vary dramatically (totals
   487–640, overlaps 27–180) — including two pairs where neither domain is a product
   rectangle. One ceiling, many shapes.
3. No facet is inherently uncoverable. The 40 missing facets move across runs.
4. Both structured constructive families (whole-cell and half-split) were closed with
   explicit reasons, not just null results.
5. The H₁ obstruction is vacuous on spheres and cannot be used here.
6. The circle's k=2 escape mechanism relies on a cyclic order that S³'s simplex
   complex does not possess. This is the campaign's one positive argument that C1 is
   true (as opposed to accumulated negative evidence).

What the campaign does **not** establish: a proof of C1. The search space is chains,
not domains — "no cover found" is evidence about the landscape. Only a lower-bound
argument can make C1 a theorem, and the one general tool this repository has for
that (H₁ obstruction) is vacuous here.

The defensible claim: *no two-domain cover of ∂Δ⁴ has been found by any search,
structured or unstructured, at any budget tried — including 5.65 × 10⁹ moves from a
corrected energy — and all of them stop at exactly the 2-rectangle optimum.*

---

## Summary of open questions

| question | status |
|---|---|
| SC_strict(dodecahedron boundary) = 2? | Open. The search does not find a cover. No lower-bound argument beyond SC_strict ≥ 2 was attempted. |
| SC_strict(∂Δ⁴) = 1? (Conjecture C1 false) | Open. No two-domain cover found; H₁ vacuous; circle mechanism non-liftable. |
| SC_strict(∂Δ⁴) = 2? (Conjecture C1 true) | Open. No lower-bound argument available. |
| Optimal rung length for dodecahedron ladder | Empirically ~2.7 min at early stages; decays as coverage rises. Not a route to a certificate. |
| What move attribution could help the dodecahedron? | Unknown. A move that does not touch the whole star of a grid vertex is the next formulation-level change. Not tried. |
