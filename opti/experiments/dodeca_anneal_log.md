# The dodecahedron boundary with the chain-first annealer

*2026-09-19. Target: the boundary of the regular dodecahedron, triangulated by
fanning each of the 12 pentagons from one vertex into 3 triangles — 20
vertices, 54 edges, 36 triangles, χ = 2. Like the cube boundary it is a
triangulated S² that is **not** a boundary of a simplex, so the question is
the cube question again: how far is the product family from optimal, and can
the chain-first search close the gap to the sharp SC_strict = 2?*

## What had to change to run it at all

The complex has 20 vertices, and `Product` asserted `nv <= 11`. Unpicking that
guard found two real constraints, only one of which was what the message said:

1. **`facet_verts` genuinely holds grid ids.** A product facet's vertices are
   grid ids `a*nv + b`, which reach `ngrid - 1 = 399` here — past int8. The
   guard's message ("int8 grid ids … ngrid = nv^2 <= 127") was *right*, and my
   first reading of it (that every int8 array in the chain path holds only
   K-vertex ids) was wrong: the chain arrays do, but `facet_verts` does not.
   Fixed by widening it to int16, which supports nv <= 181.
2. **The `is_simplex` table is `2^nv` bytes**, built by scanning *every* subset
   `s ∈ [0, 2^nv)` for containment in some facet — 1,048,576 × 36 Python
   operations here. Replaced with the equivalent submask enumeration (each
   facet contributes its `2^|facet|` subsets): 36 × 8 for the dodecahedron.
   Verified **byte-identical** to the old construction on circle, s2, s3, s4,
   s5, wedge, bary and cube before being trusted.

The assert is now `nv <= 24` (16 MB table), with the reasoning recorded in the
code rather than left as folklore.

Scale of the result: |F_L| = 36 × 36 × 6 = **7,776** product facets, 400 grid
vertices, largest star 384 — 9× the cube's 864 facets, and about two thirds of
the S⁵ complex that the search already handles. Per-thread local state is
~71 KB for k = 3, well inside the 512 KB limit that stopped S⁷.

## The auditor had to learn it too

`independent_check.py` rebuilds each K from scratch, so it needs the mesh, and
its facet list must match `gpu_sc.build`'s order exactly (the staircase
enumeration of K×K depends on the facet order and on each facet's vertex
order). Added `dodeca_facets()` plus a `nf == 7776` detection clause.

The cube's surface re-proof (expected f-vector, every edge in exactly two
triangles, every vertex link a single cycle, χ = 2) was **extracted into
`check_sphere_surface(name, KF, nv, ne)` and reused**, rather than copied, so
the dodecahedron gets the same argument. The refactor was checked to leave the
cube's audit output byte-identical, and all present certificates
(∂Δ³, ∂Δ⁴, ∂Δ⁵, barycenter triangle, torus) still audit clean.

Domain floor for the dodecahedron is **3**, by the same reasoning as the cube:
χ = 2 and closed force orientable, so TC = 2 and SC_strict ≥ 2.

## The obstacle, measured: simplex density of K

A chain's domain is the set of facets on which **every** link of the chain is
simplicial — `chain_bad_counts` returns, per facet, the *number* of links that
fail, and `domain_of` keeps the facets where that count is zero. (Not "some
link works": a chain whose every link but one is fine still has an empty domain
at the facets the bad link spoils.) So the chance a random chain has a large
domain falls off with the number of links *and* with how rarely a random image
is a simplex.

`is_simplex` asks whether the union of the images of a facet's vertices spans a
simplex of K. That is governed by how densely K is triangulated:

| K | vertices | triangles | possible triples | density | initial coverage of a random chain |
|---|---:|---:|---:|---:|---:|
| ∂Δ³ (S²) | 4 | 4 | 4 | 100% | 36.0% |
| ∂Δ⁴ (S³) | 5 | 5 | 10 | 50% | 36.8% |
| ∂Δ⁶ (S⁵) | 7 | 21 | 35 | 60% | 34.8% |
| cube boundary | 8 | 12 | 56 | 21.4% | 12.1% |
| Császár torus | 7 | 14 | 35 | 40.0% | 10.4% |
| **dodecahedron** | 20 | 36 | 1140 | **3.2%** | **3.0%** |

(Initial coverage measured on the annealer's own `sweep_chain` initialisation,
20 chains per complex, cmax 40 — host-side, no search.)

Density is not the whole story — ∂Δ⁶ is 20% dense by this count yet starts at
35%, and the torus at 40% starts at 10% — but the dodecahedron is the outlier
on both axes at once: the sparsest K, and the smallest starting coverage by a
factor of 3.4 against the next worst.

Where does the 3% *come from*? Cell by cell, it is almost exactly the diagonal.
A facet of the cell (s, t) has grid vertices (a_i, b_i); π₁ sends them to the
vertices of the triangle s, π₂ to those of t, so on a facet of an
**off-diagonal** cell the images have to travel from s to t. In the diagonal
cells (s = t) both projections land in the same triangle, and the facets are
easy:

> **Correction — the first explanation written here was wrong.** An earlier
> draft claimed off-diagonal cells are *unreachable*, because two distinct
> triangles of a surface share at most an edge so their union is never a
> simplex, and the first link (from π₁, whose image is all of s) forces the
> next image to lie inside s. The premise is right and the conclusion is
> false, and the repository already contained the counterexample: the cube's
> three domains cover **all 144 cells**, 792 of their 864 facets off-diagonal,
> including cells whose two triangles share no vertex at all.

| K | diagonal cells | share of K×K | diagonal facets | measured initial coverage |
|---|---:|---:|---:|---:|
| dodecahedron | 36 of 1296 | 2.8% | 216 | 235 (3.0%) |
| cube | 12 of 144 | 8.3% | 72 | 104 (12.1%) |
| ∂Δ⁴ | 5 of 25 | 20.0% | 100 | 184 (36.8%) |

The diagonal is the floor and the *density of K* is what buys coverage beyond
it: ∂Δ⁴ converts its 20% diagonal share into 37%, the dodecahedron converts
its 2.8% into 3.0%.

What actually carries a chain across an off-diagonal cell is a **walk through
the dual graph**, and the cube's own certificate shows it in the raw maps.
Facet 576 of the cube lies in cell (8, 0) — two triangles with no shared
vertex — and chain 0's images on it run:

```
map 0  (π₁): [4,4,4,5,6]   image {4,5,6}   = triangle 8
map 2      : [4,4,4,4,6]   image {4,6}     shrinks to the shared EDGE
map 3      : [4,7,7,7,7]   image {4,7}     grows into the NEIGHBOURING triangle
...
map 16 (π₂): [0,3,7,7,7]   image {0,3,7}   = triangle 0
```

Each link is legal because the union of consecutive images is always *some*
simplex of K, not necessarily s or t: {4,5,6} → {4,6} → {4,6,7} → … → {0,3,7}
walks triangle by triangle across K's dual graph. So the difficulty is not
that off-diagonal facets are unreachable but that reaching them requires the
chain to coordinate such a walk **for every facet at once** — one global map
serves all 7,776 of them.

How long those walks are is a property of K's dual graph, and it is the
cleanest predictor the campaign found. Both complexes have cubic duals (every
triangle has exactly 3 neighbours, as a closed surface forces):

| K | triangles | dual edges | dual diameter | mean dual distance |
|---|---:|---:|---:|---:|
| cube | 12 | 18 | 4 | 2.08 |
| dodecahedron | 36 | 54 | **7** | **3.81** |

A chain's link can advance a walk by about one triangle, so a facet in a cell
whose triangles sit at dual distance d needs a chain with roughly d transition
links **that happen to do the right thing in every other cell simultaneously**.
The dodecahedron nearly doubles the distance and triples the number of
constraints. That is the honest reading of why this instance stalls, and it
also explains the cmax sweep below: chain length is walk budget.

## Runs

| run | k | cmax | settings | budget | result |
|---|---:|---:|---|---|---|
| smoke | 3 | 20 | S 512, iters 2000 | 3 epochs, 17 s | 304/7776 (sanity only) |
| `dodeca_k3` | 3 | 40 | β 5, γ 0.5, ptrack 0.3, guide 1.0 | 3600 s | *(running — see below)* |

### The chain-length (cmax) sweep

At fixed settings (S 1024, iters 2000, 4 epochs, β 5 / γ 0.5, ptrack 0.3,
seed 3), varying only the chain length:

| cmax | best after 4 epochs |
|---:|---:|
| 20 | 300/7776 |
| 40 | 330/7776 |
| 100 | **391/7776** |
| 200 | *no epoch completed in 150 s* |

So chain length is a real lever (+30% from 20 to 100) but not a decisive one,
and it saturates hard: a track move on a block of L links costs O(L · star), so
at cmax 200 (with star = 384) a single epoch of ptrack 0.3 work does not finish
in 150 s at a *quarter* of the search population. cmax ≈ 100 is the practical
ceiling with track moves on this complex.

### Cold run (`dodeca_k3`) — a full hour

| | |
|---|---|
| epochs | 105 |
| final coverage | **644 / 7,776 (8.3%)** |
| domain sizes | 342 + 381 + 319 |
| moves | 2.17 × 10⁹ |
| kernel time | 3,572 s |

Trajectory: 320, 327, 337, 348, 361, …, 421 (epoch 19), 467 (37), 504 (50),
533 (61), 571 (81), 610 (94), 631 (100), **644** (105). The rate opened at
≈ +10/epoch, sagged to ~+2.5 through the middle, and the last twenty epochs
climbed at ~+3.5 again — a slow crawl throughout, never a stall, and never
remotely near 7,776. The cube, by contrast, opened at 21% (185/864) and
climbed ~+45/epoch to a guided finish.

### Escalation 1 (`dodeca_k3_escal`) — the cube's recipe, and a correction

Warm start from the cold run's own best state (half the population seeded with
its reduced chains), guide raised to 3 with the cap at 11 — the `--guide-max`
fix making the increment real for the first time. It opened at 678 and climbed
at +8/epoch over its first twenty epochs (678, 704, 732, 757, 790, 802).

| | cold (`dodeca_k3`) | escalation (`dodeca_k3_escal`) |
|---|---:|---:|
| epochs | 106 | 109 |
| coverage | 0 → **644** | 644 → **1208** |
| gain in the hour | +644 | **+564** |
| moves | 2.17 × 10⁹ | 2.23 × 10⁹ |
| average rate | +6.08 / epoch | **+5.17 / epoch** |

> **Correction.** Mid-run I reported the escalation "roughly tripling the
> rate", comparing its opening epochs (+8 to +9/epoch) against the cold run's
> *late* epochs (+2.5). That comparison is unfair and the summary it implies is
> wrong: the cold run also opened fast — 361 by epoch 5, ≈ +72/epoch — because
> the first few hundred facets are easy from a random start. Over a full hour
> at essentially the same move count, the warm-started run gained **fewer**
> facets (+564) than the cold run (+644), and its average rate was *lower*.

The warm start does produce a real transient — the seeded population opens at
678 and climbs at +8/epoch — but the transient is not a rate advantage over a
cold run, because the facets it is handed are the expensive ones. What the
escalation demonstrates is the shape of the landscape: marginal difficulty
rises steeply, so *any* head start is eaten by the harder marginal work. Both
runs made ~2.2 × 10⁹ moves in their hour; both gained ~600 facets; the second
hour started 644 facets in and ended 1208 facets in.

### Escalation 2 — the restart ladder, second rung (finished)

Final: **1579 / 7,776 (20.3%)**, domains 896 + 784 + 387, 109 epochs, 2.23 × 10⁹
moves. The three hours together:

| run | start | end | gain | epochs | avg/epoch | moves | gain per Mmove |
|---|---:|---:|---:|---:|---:|---|---:|
| cold | 0 | 644 | **+644** | 106 | 6.08 | 2.17 × 10⁹ | 0.30 |
| escalation 1 | 644 | 1208 | **+564** | 109 | 5.17 | 2.23 × 10⁹ | 0.25 |
| escalation 2 | 1208 | 1579 | **+371** | 109 | 3.40 | 2.23 × 10⁹ | 0.17 |

Started at the author's request at 2026-09-20 00:25 UTC, warm-started from
`results/dodeca_k3_escal/summary.json` (the 1208 state — which only became
available on disk once escalation 1 wrote its summary; that is why this could
not be done mid-run when asked). Same configuration as escalation 1, seed 85.
Its question — does the second restart produce the same transient, or a
smaller one? — is answered above: **smaller, by roughly half at the open and
by a third over the hour.**

### Escalation 3 — third rung (finished)

Started 2026-09-20 01:42 UTC at the author's request, warm-started from
`results/dodeca_k3_escal2/summary.json` (the 1579 state), same configuration,
seed 86, outdir `results/dodeca_k3_escal3`.

**Final: 1749 / 7,776 (22.5%)**, domains 956 + 888 + 401, 109 epochs,
2.23 × 10⁹ moves — a gain of **+170**.

Prediction, recorded before the run: *below rung 2's 371, on the decay
argument.* **Confirmed: +170.** One caveat the same data insists on — rung 3's
*opening* was +16, above rung 2's +14, so the opening rate is not what decays
monotonically; the full-hour yield is, and decisively.

### Escalation 4 — a 15-minute rung (finished)

Started 2026-09-20 03:16 UTC, warm-started from the 1749 state, same
configuration, seed 87, `--time-limit 900`.

The point is not coverage, it is the **shape of the yield within a rung**.
Escalation 3's hour was front-loaded: +58 of its +170 (34%) arrived in the
first ~15 minutes (epoch 25 at ~35 s/epoch). If that holds, a 15-minute rung
captures a third of an hour's gain for a quarter of the GPU time — i.e. short
rungs are more efficient per GPU-minute, and a ladder of many short rungs
would beat the four long ones. If instead the 15-minute rung yields much less
than ~58, the gain is spread evenly and rung length does not matter.

Prediction on record before the result: **+40 to +60 facets**, i.e. in the
neighbourhood of escalation 3's own first quarter-hour.

**Final: 1802 / 7,776 (23.2%)**, domains 998 + 918 + 392, 15 minutes,
5.53 × 10⁸ moves — a gain of **+53**. Prediction confirmed.

### What the 15-minute rung settles, and what it does not

Normalising every rung to facets per GPU-minute:

| rung | minutes | gain | gain/min | gain per hour |
|---|---:|---:|---:|---:|
| cold | 60 | +644 | 10.73 | 644 |
| escalation 1 | 60 | +564 | 9.40 | 564 |
| escalation 2 | 60 | +371 | 6.18 | 371 |
| escalation 3 | 60 | +170 | 2.83 | 170 |
| **escalation 4** | **15** | **+53** | **3.53** | **212** |

- **Front-loading is real.** Rung 4's +53 in a quarter-hour closely matches
  rung 3's own first-quarter-hour (+58), and its normalised rate (212/h) beats
  the hour-long rung it followed (170/h) — despite starting from a *higher*,
  therefore harder, state. So a rung does deliver a disproportionate share of
  its yield early.
- **It does not reopen the restart route**, which is what I provisionally said
  it might. The normalised rate still falls monotonically *down the ladder*
  (644, 564, 371, 170, 212 — rung 4 only exceeds its immediate predecessor).
  And the arithmetic is decisive: the four-hour ladder gained **+1802**, while
  extrapolating 4 hours of 15-minute rungs from rung 4's rate gives
  16 × 53 ≈ **848**. A short-rung ladder wins on the rate of the most decayed
  rung and loses badly on the total, because it throws away the high-yield
  early runs.

So the honest conclusion from all five rungs: front-loading is a real property
of a rung, and it is not a strategy — the rung-over-rung decay (0.88, 0.66,
0.46) dominates it.

### Escalation 5 — a 7-minute rung (finished)

Started 2026-09-20 03:34 UTC, warm-started from the 1802 state, same
configuration, seed 88, `--time-limit 420`.

Rung 4 sharpened the front-loading question rather than settling it: its first
7 minutes gave **+30** of its +53 (57% of the yield in 47% of the time), so the
curve steepens as the window shortens. Rung 5 halves the window again, and the
prediction on record is **+25 to +35**, from rung 4's own first seven minutes.

**Final: 1831 / 7,776 (23.5%)**, domains 1008 + 934 + 393, 7.4 minutes,
2.66 × 10⁸ moves — a gain of **+29**. Prediction confirmed.

### The yield curve, measured confound-free

Rungs started from different states (1579, 1749, 1802), so comparing them
across rungs confounds window length with starting difficulty. Escalation 3
answers it inside one rung, where the start state is fixed:

| window within escalation 3 | gain | per minute |
|---|---:|---:|
| first 7 min | **+33** | 4.71 |
| first 15 min | **+58** | 3.87 |
| full 60 min | **+170** | 2.83 |

So the within-rung yield fits ≈ **t^0.8**: doubling 7 → 15 minutes multiplies
the gain by 1.76, quadrupling 15 → 60 by 2.9. Short windows genuinely have
higher rates — the rate falls as t^-0.2 — and the three late rungs' own
first-7-minute gains (+33, +30, +29) are almost identical, which is the same
law seen three more times.

**What that is worth.** If a rung's yield is c·t^0.8 and each rung costs a
fixed ~40 s of kernel compile on top of t, the per-hour yield is maximised at
t = 4τ ≈ 160 s ≈ **2.7 minutes** — and at the measured c the optimum is ≈ 267
facets per hour, against the 170 the hour-long rungs actually delivered. So the
ladder *can* be tuned: roughly 1.5× more coverage per GPU-hour from
~3-minute rungs. It is a real optimisation of the method.

**And it does not matter.** 267 facets/hour against the 5,945 still uncovered
is ~22 hours of GPU — against ~30 for the long rungs. The tuning moves the
constant; the exponent is what makes the problem out of reach, and shortening
rungs cannot touch it. The power law also predicts a 3.5-minute rung yields
~+18, which is the next point that would confirm or break the fit, and it is
the only remaining rung worth running.

**Not tried** (recorded so that "not attempted" stays distinguishable from
"attempted and failed"): cmax > 100 with ptrack lowered to make track moves
affordable; initialisations built from collapsing maps rather than the random
sweep; and any product-domain certificate, which is out of scope by
instruction.

The instance itself is now supported end to end — `gpu_sc.py --complex dodeca`
anneals it, `independent_check.py` audits it, and the surface re-proof is
shared with the cube rather than duplicated — so a future attempt with a
different formulation starts from a working harness rather than from the
`nv <= 11` guard that blocked it.

## Why this instance is worth running

- ∂Δ³, the cube boundary and the dodecahedron are **three triangulations of the
  same space**. SC_strict is a topological invariant in the limit but SC_strict
  at subdivision level 0 is not, so the three are genuinely different
  problems — and the cube is precisely where the product family was shown to
  be far from optimal (ρ = 8 → products certify only SC ≤ 7, against the sharp
  2).
- The mesh is **highly irregular**, which is itself the interesting contrast
  with the cube. The *dodecahedron* is vertex-transitive, but the fan
  triangulation is not, and by a lot: fanning each pentagon from one vertex
  gives the 20 vertices degrees from **3 to 8** — three vertices of degree 3,
  five of 4, two of 5, five of 6, one of 7 and four of 8 (13 distinct degree
  classes in total once links are compared). The high-degree vertices are the
  fan centres, hubs that lie in up to 8 of the 36 triangles, so the product
  family's largest rectangles `star(v) x star(w)` are almost certainly
  concentrated on a few hubs rather than spread over a symmetric mesh. A
  search that finds 3 domains here does so on a mesh with no symmetry to
  exploit.

  > **Correction.** An earlier draft of this section called the triangulation
  > vertex-transitive and argued from that. It is not — the fan construction
  > breaks the dodecahedron's symmetry badly. Kept visible rather than
  > silently edited.
- It also exercises the chain-first formulation on a complex whose product is
  **not** of the form ∂Δⁿ × ∂Δⁿ: the cell grid is 36 × 36 with every cell a
  Δ² × Δ² block, which is the same local shape as the cube's but with a much
  larger, and far more irregular, gluing.

## The 2.5-minute ladder (`run_dodeca_ladder.sh`)

Started 2026-09-20 03:46 UTC at the author's request: a chained ladder of
**2.5-minute** warm-start rungs, each seeded from the previous rung's best
state, driven by `run_dodeca_ladder.sh`.

This is the optimisation the yield curve points to, applied. With gain(t) ≈
c·t^0.8 and ~40 s of fixed per-rung cost (JIT compile plus host setup), the
coverage per GPU-hour is (3600/(t+40))·c·t^0.8, which peaks at t = 4·40 s ≈
2.7 min:

| rung length | yield per rung | facets per GPU-hour |
|---:|---:|---:|
| 35 s (one-epoch floor) | 4.5 | 217 |
| 2.0 min | 12.1 | 273 |
| **2.5 min** (this ladder) | **15.2** | **274** |
| 2.7 min (model peak) | 15.2 | 274 |
| 7 min (rung 5, measured +29) | 33.0 | 258 |
| 60 min (rungs 1–3, measured +170…+644) | 184.1 | 182 |

Caveats attached to the fit, which the ladder is also a test of: the exponent
comes from three points of a single rung; the constant c drifts as coverage
rises (later rungs start on harder facets, so the true optimum moves down over
a campaign); and the model over-predicts absolute yields by ~15% (it says +33
where the 7-minute rung measured +29).

**What it changes, and what it does not.** At the modelled 274 facets/hour the
5,945 facets still uncovered need ~22 hours, against ~30 at the hour-rung rate.
The ladder therefore makes the search ~1.5× cheaper per facet and does not make
a cover reachable — it optimises the constant while the exponent sets the
scale.

The script is resumable (it continues from the highest-numbered rung on disk),
stops to audit if any rung produces `cover.txt`, and refuses to continue if a
rung fails to write its summary, so a ladder can be stopped and restarted
without losing the chain. Per-rung results are in `results/dodeca_ladder/` and
`logs/rung*.log`.

### Ladder result, and the yield law it confirms

20 rungs, 2026-09-20 03:45:56 → 04:40:53 UTC (54 min 57 s, 2.89 min per rung
including the ~24 s of per-process overhead), **1831 → 2064 facets (+233)**.

Per-rung gains, in order:

```
27 17 15 17 15 | 11 11 10 13 12 | 7 7 7 7 8 | 8 8 9 12 12
```

Fitting the five independent yield measurements — the ladder's 20-rung mean at
t = 150 s, both 420 s points, and the 900 s and 3600 s rungs — gives

    gain ≈ 0.184 · t^0.835

| window | measured | fit |
|---:|---:|---:|
| 150 s (ladder mean, n = 20) | +11.65 | +12.08 |
| 420 s (escalation 5) | +29 | +28.55 |
| 420 s (escalation 4's first 7 min) | +30 | +28.55 |
| 900 s (escalation 4) | +53 | +53.97 |
| 3600 s (escalation 3) | +170 | +171.86 |

A single power law with exponent 0.835 holds across a **24× range of window
lengths**, from 2.5 minutes to an hour, to within 4% at every point. That is
the campaign's cleanest quantitative result, and it validates the model the
ladder was built on: at 2.89 min per rung the ladder delivers
233/0.916 h ≈ **254 facets/hour**, against the ~182/hour the hour-long rungs
gave — the predicted ~1.4×, and the predicted absolute rate (20.7 rungs/hour ×
12.08 = 250) to within 2%.

**What the ladder does not do is change the outcome**, and its own numbers say
why: the per-rung gain *fell* as coverage rose — 18.2 mean over rungs 1–5, 11.4
over 6–10, 7.2 over 11–15, 9.8 over 16–20. The exponent 0.835 is a property of
a rung at a given difficulty; the difficulty itself is what grows, and no
choice of rung length touches that. At the ladder's own end rate (~10 per
2.89 min ≈ 208/hour, already down from 254) the 5,712 facets still uncovered
need **~27 hours** of GPU, and the rate is still falling.

So: the ladder is a genuine ~1.4× optimisation of the constant, obtained in one
55-minute experiment, and it moves the projected time to a certificate from
"~30 hours" to "~27 hours and climbing". Neither is a route.

## Unattended operation (`run_dodeca_ladder_forever.sh`)

Started 2026-09-20 05:22 UTC, 12-hour deadline (until 17:22 UTC), at the
author's request: the 2.5-minute ladder run as a **self-perpetuating loop**,
20 rungs per batch, resuming from the last completed rung at every batch
boundary.

Design decisions worth recording, since this is meant to run with nobody
watching:

- **Exclusive by construction.** Each batch runs under `flock`, so a second
  invocation waits for the lock instead of reading the same "highest rung on
  disk" and racing to write the same outdir. (The race is real: the first
  attempt at this handover did leave an orphaned rung-22 process writing to
  the same directory the wrapper was about to use. It was killed before the
  wrapper started; the fix is the lock, so it cannot recur.)
- **Resumable at rung granularity.** The chain lives in
  `results/dodeca_ladder/rungNNN/summary.json`; a crash or a `kill` costs at
  most the rung in flight.
- **Failure policy**: one failed batch is retried after a pause, three
  consecutive failures stop the loop so the cause can be inspected. A batch
  that reports a full cover stops everything and audits the certificate.
- **Cost**: 560 KB of state per rung (~140 MB over 12 h), ~2.9 min per rung,
  so ~12 facets per rung at the current difficulty and a declining rate.

The monitoring is best-effort and lives in the agent session; the run does not
depend on it. Everything lands in `logs/batchNNN.log`,
`logs/dodeca_ladder_forever.log` and the per-rung directories regardless.

### Unattended run, results — and where the yield law breaks

Seven complete batches (rungs 22–161) plus a partial eighth, 2026-09-20
05:22 → 12:04 UTC, **2075 → 2551 facets in 148 rungs**:

| batch | rungs | coverage | gain | mean/rung |
|---|---|---:|---:|---:|
| 1 | 22–41 | 2075 → 2218 | +143 | +7.2 |
| 2 | 42–61 | 2218 → 2310 | +92 | +4.6 |
| 3 | 62–81 | 2310 → 2400 | +90 | +4.5 |
| 4 | 82–101 | 2400 → 2441 | +41 | +2.1 |
| 5 | 102–121 | 2441 → 2487 | +46 | +2.3 |
| 6 | 122–141 | 2487 → 2523 | +36 | +1.8 |
| 7 | 142–161 | 2523 → 2550 | +27 | +1.4 |
| 8 (partial) | 162–168 | 2550 → 2551 | +1 | +0.1 |

**This is where the `t^0.835` law breaks, and the reason is worth recording.**
The law was fitted on rungs taken between coverage ~1580 and ~1830, where it
predicted the ladder's yield to within 2%. It does not extrapolate *down in
coverage* — the per-rung gain at fixed window length (150 s) fell from +11.65
(first ladder) to +7.2, +4.6, +4.5, +2.1, +2.3, +1.8, +1.4 and finally ~+0.1,
i.e. the amplitude `c` is not a constant at all:

    gain ≈ c(coverage) · t^0.835      with c halving roughly every ~200 facets

So the exponent describes how a rung's yield scales with *time* at a given
difficulty, and the difficulty is a second, separate factor that the fit never
captured because the early measurements spanned too narrow a coverage band to
reveal it. The honest model has two terms, and the second one is the one that
decides the outcome.

Consequence for the projection: at ~+1 per rung and 20.7 rungs/hour the run is
now adding ~20 facets/hour, against 254 when the ladder started. Reaching the
remaining 5,225 facets at that rate is ~260 hours, and `c` is still falling —
the series is not converging on a certificate, it is converging on a wall.

Coverage stands at **2551 / 7,776 = 32.8%**.

### The complete 2.5-minute series — every rung, 1831 → 2551

All 172 rungs run at the 2.5-minute budget (rungs 1–20 the first ladder, 21 the
handover run, 22–172 the unattended ladder), each value the facets that rung
added:

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
rung 161-170:   0   1   0   0   0   0   0   0   0   0
rung 171-172:   0   0
```

Mean +4.2 per rung over the 172. The series is the clearest picture the
campaign produced of the difficulty curve: a rapid opening (27, 17, 15, 17…),
a long slow middle in the single digits, and a final collapse to literal zeros
— the last **eleven** rungs contributing one facet between them.

### Leaving the ladder: one hour-long rung from the 2553 state

Stopped the unattended run at 2026-09-20 12:42 UTC (rung 173, **2553** facets),
at the author's request, and launched instead a **single 1-hour rung** from that
state — `results/dodeca_hour_from2553`, same configuration, seed 200,
`--time-limit 3600`.

The question is whether the 2.5-minute rungs stopped because the *window* had
become too short to make progress, or because the search has genuinely
saturated. The two-term model predicts the former is not enough: with the
amplitude now at c ≈ 0.016 (from the recent ~+1 per 150 s), a 3600 s rung should
yield c·3600^0.835 ≈ **+16 facets**. So the prediction on record, before the
result, is **+10 to +25** — a longer window buys something but not a return to
the early rate. A result far above that band (say +60 or more) would mean the
short rungs were window-limited and the ladder was mis-tuned; a result at or
below +10 would mean the search is saturated at ~2553 and no rung length helps.

**Result: 2566 / 7,776 (33.0%) — a gain of +13**, domains 1425 + 1232 + 433,
2.23 × 10⁹ moves. Inside the predicted band, at its low end.

**Verdict: the window was not the problem, and the long rung is the *worse*
buy.** Normalised to an hour:

| how the hour was spent | facets gained |
|---|---:|
| 20.7 rungs of 2.5 min (the ladder's end state) | ~**20** |
| one rung of 60 min (this run) | **13** |

Which is what the two-term model says should happen — the ladder's
1273·c per hour against the single rung's 996·c, a ratio of 1.28 — and it is
the front-loading result reasserting itself at depth: each fresh rung's opening
transient is worth more than the tail of a long run. So the short ladder was
not mis-tuned; it was already the better allocation of the same GPU hour, and
it still only bought ~20 facets.

The saturation reading therefore stands. Three hours of 2.5-minute ladder plus
one hour of a long rung moved coverage from 2553 to 2566 — **+13 facets in four
GPU-hours**, with the domain sizes (1425 + 1232 + 433) showing the chains are
large and the uncovered 5,210 facets are not reachable by any configuration of
this search that the campaign has tried.

### A 30-minute rung, to place the middle of the curve (running)

Started 2026-09-20 14:31 UTC from the 2566 state — `results/dodeca_half_hour_from2566`,
same configuration, seed 201, `--time-limit 1800`.

This fills the one unmeasured point between the two ends of the window
comparison, and it is the test that the two-term model either survives or does
not. At the current amplitude (c ≈ 0.0163, calibrated from the ladder's last
rungs at +1 per 150 s), the model predicts

    gain(1800 s) = c · 1800^0.835 ≈ +8

so the prediction on record is **+5 to +12**, i.e. a per-hour rate of ≈ 17,
sitting between the ladder's 20 and the hour-rung's 13 and preserving the
ordering that short rungs are the better buy at this depth. A result **well
above** that (say +25 or more) would mean the model's amplitude is not the
right description and that intermediate windows punch above their weight; a
result **at or below +5** would mean the curve has flattened completely and
window length has stopped mattering at all.

**Result: 2570 / 7,776 (33.1%) — a gain of +4**, domains 1425 + 1235 + 433,
1.13 × 10⁹ moves. Below the predicted band, and it **breaks the ordering the
model implied**:

| window | gain | per hour |
|---:|---:|---:|
| 2.5 min (ladder's end rate) | ~+1 | ~**20** |
| 30 min (this run) | **+4** | **8** |
| 60 min | +13 | 13 |

The rates are **not monotone** — 20, 8, 13 — so "shorter is better" is not a
resolvable claim at this depth after all. With gains of +1, +4 and +13 from a
handful of runs each, the per-hour differences sit inside the noise, and the
model's ordering claim was over-reading three points. What *is* robust is the
magnitude: every window from 2.5 minutes to an hour now buys single digits, and
the domain sizes barely move between runs (1425 + 1232 + 433 → 1425 + 1235 +
433), i.e. the search is not finding new structure, it is nudging the same
three chains.

The honest summary of the window comparison is therefore narrower than the
previous section claimed: **the ladder's ~20/hour edge over the hour-rung's 13
should not be treated as established** — it was one comparison of two noisy
estimates, and the middle point has now landed below both. The conclusion that
survives is the one all three agree on: at ~2,570 of 7,776 facets this search
has saturated, and no window length in the range tested changes that.

### 100 tries of 3 epochs each (running)

Started 2026-09-20 15:16 UTC from the 2570 state — `results/dodeca_tries`,
100 warm-start tries, each **exactly 3 epochs** (via `--epochs 3`, not a
wall-clock limit, so every try does the same amount of *search*; epoch duration
varies, a fixed `--time-limit` does not fix the work), each seeded from the
previous try's best state with a fresh seed. ~1.8 min per try, so ~3 hours.

This changes the unit of effort from *time* to *epochs*, which is the right
unit for asking the question the timed rungs could not answer cleanly: with
the seed varying over 100 tries, does any of them break the 2,570 plateau?
The five window lengths tested so far all used the same narrow seed range
(84–201) and the same starting structure; 100 fresh seeds is a sample of the
restart distribution rather than another point on the window curve.

The measurable, stated before the result: the **distribution** of per-try
gains, not just its mean. The timed rungs' mean was ~+1 per try with every try
positive; if this sample contains zeros and near-zeros with a few larger
outliers, the plateau is a wide basin with rare exits and the outliers are
worth chasing. If all 100 land in a tight band around +1, the plateau is a
point attractor and no amount of restarting moves it.

**Stopped at 70 of 100 tries** (2026-09-20 17:22 UTC, by the author, to be
continued later). Results over those 70:

| | |
|---|---|
| coverage | 2570 → **2602** (+32) |
| mean per try | **+0.46** |
| tries gaining zero | **51 of 70 (73%)** |
| best single try | **+4** |
| distribution | `{0: 51, 1: 12, 2: 2, 3: 4, 4: 1}` |

```
per-try gains:
.1111.1........23.......2141........1........33.1.....113........1......
```

**Answer to the pre-registered question: neither of the two options — the
basin is wide *and* shallow.** Restarting does escape: 19 of 70 tries (27%)
found something the previous state did not have, so the plateau is not a point
attractor. But the exits are worth **1 to 4 facets**, and across 70 fresh seeds
(~287,000 annealer runs at S = 4096) not one produced a real escape. The
reachable set from this state is effectively exhausted at ~2,600; the search can
leave its current optimum, but only into neighbours barely better than it.

That is a cleaner result than either branch of the prediction, and it is the
one that closes the restart line of enquiry: a wide shallow basin yields to
sampling but not to *more* sampling, which is what every restart strategy is.

Continuing later is trivial — the chain is intact:

```bash
EPOCHS=3 RUNGS=30 OUTDIR=results/dodeca_tries \
    INIT=results/dodeca_tries/rung070/summary.json ./run_dodeca_ladder.sh
```

### The state is saved, and stopping costs nothing

Verified at rung 171 rather than assumed: `results/dodeca_ladder/rung171/summary.json`
carries the three **chains**, not merely a coverage count, and recomputing the
covered set from those chains (`Product.domain_of` on each) reproduces exactly
**2551**. That is the same file `--init` consumes, and the resume path has
already run four times at batch boundaries (rung020 → 21, rung041 → 42,
rung061 → 62, …). So the ladder can be stopped at any rung and a different
strategy started from the 2551-facet state without losing anything; conversely
`./run_dodeca_ladder.sh` alone will pick the chain back up from the
highest-numbered rung on disk.

## Conclusion

**The chain-first annealer does not reach a cover on the dodecahedron
boundary.** Five runs — one cold and four warm-started rungs, 4 h 20 m of GPU
and ~9 × 10⁹ moves — left it at **1802 of 7,776 product facets (23.2%)** with
three domains of 998 + 918 + 392. No certificate, and no trajectory that
suggests one is reachable at this scale by this method.

What the experiment established, all of it measured rather than asserted:

1. **The obstacle is K's sparsity, and it is quantifiable.** A chain's domain
   needs *every* link simplicial; the domain floor is the diagonal cells (2.8%
   here, against the cube's 8.3% and ∂Δ⁴'s 20%); coverage beyond that comes
   from coordinating walks across K's dual graph, whose diameter here is 7 and
   mean distance 3.81 (the cube's are 4 and 2.08).
2. **Chain length is walk budget** — cmax 20 → 40 → 100 gave 300 → 330 → 391
   at fixed settings; cmax 200 is unusable, since track moves cost O(L·star).
3. **Warm starting buys a decaying transient, not a rate.** The rungs gained
   564, 371, 170 — ratio 0.88, 0.66, 0.46, not levelling off.
4. **A rung's yield is front-loaded but that is not a strategy.** A 15-minute
   rung captures ~31% of an hour's gain, and beats its own predecessor's
   normalised rate (212 vs 170 per hour) — yet extrapolating 4 hours of
   15-minute rungs gives ≈ 848 against the 1802 the long rungs actually
   delivered.

What it does **not** establish is anything about SC_strict of this complex: a
search that fails to find a cover says nothing about whether one exists, and
no lower-bound argument was attempted. The claim supported here is the
empirical one — *this method, at this scale, with these levers, does not reach
a cover*, with the causes identified.

**Not tried** (recorded so that "not attempted" stays distinguishable from
"attempted and failed"): cmax > 100 with `ptrack` lowered so track moves are
affordable; initialisations built from collapsing maps rather than the random
sweep; a genuinely different move attribution (a move that does not touch the
whole star of a grid vertex), which is the formulation-level change this
instance probably needs; and any product-domain certificate, out of scope by
instruction.

The instance is supported end to end — `gpu_sc.py --complex dodeca` anneals it,
`independent_check.py` audits it with the same surface re-proof the cube uses —
so a future attempt with a different formulation starts from a working harness
rather than from the `nv <= 11` guard that blocked this one.
