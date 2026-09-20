# The two-domain push on ∂Δ⁴ (S³) — campaign log

*Opened 2026-09-19. Executes `guided_sphere_k2_plan.md`. Every run below is in
`results/` with its raw log in `logs/`; the driver is `run_s3_k2_push.sh`.*

## The question, in one paragraph

Corollary 5.3 gives SC_strict(∂Δ⁴) ≤ 2 for ∂Δ⁴ = S³ (three domains, certified
in `results/s3_k3_v2` as the partition 309 + 130 + 61 = 500 — the raw annealed
domains there are 309 + 283 + 271, which overlap by 363 facets; `cover.txt`
and the README report the partition, `summary.json` the raw domains, and the
same distinction applies to every plateau table below). Since TC(S³) = 1, a **two-domain**
cover would give the sharp SC_strict = 1 — the paper's own corollary is one
domain away from sharp on every odd sphere. Twelve recorded two-chain searches
(all of `s3_k2_A`–`D`, `s3_k2_v2_point`, `s3_k2_v2_track`, the `push_s3` runs,
old and corrected energy) pinned at **exactly 460/500** from the first epoch,
which is precisely the best cover by ρ − 1 = 2 *rectangles*. Conjecture C1 says
that plateau is structural; the plan's objection is that none of those twelve
runs was ever given a real budget — the 300 s time limit cut every one of them
at 2–3 epochs. This campaign runs the untested variables.

## Two corrections made to the plan before spending GPU time

### 1. The `--guide` escalation ladder was a no-op

The plan's pass 2/3 escalate the coverage guide 1.0 → 3.0 → 10.0. But the
guided-local-search cap was **hard-coded** at `guide_max = 4.0` inside
`anneal_cover`, and a facet that stays uncovered keeps `w ← 1 + 0.7(w − 1)`
plus the increment, so it saturates at

    w* = 1 + guide/0.3     →     4.33 for guide = 1.0

i.e. **guide 1.0 already saturates the old cap of 4.0**. The recurrence, run
out by hand for a facet left uncovered every epoch:

| increment | weights, epochs 1–10 (old cap 4.0) |
|---|---|
| 1.0 | 2.0, 2.7, 3.19, 3.53, 3.77, 3.94, **4.0**, 4.0, 4.0, 4.0 |
| 3.0 | **4.0**, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0 |
| 10.0 | **4.0**, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0 |

By epoch 7, `--guide 1.0`, `3.0` and `10.0` are the same weight vector: the
plan's step 3 ("escalating guide weight") would have tested nothing. The cap
and the increment have to move **together** — but note that a cap *equal* to
the increment is degenerate in the other direction (it clamps on the first
epoch, never accumulating), so the rungs below raise the cap to the natural
saturation value 1 + guide/0.3, or a little under it.

Fixed in `gpu_sc.py` (2026-09-19): `--guide-max` is now a CLI parameter
(default 4.0, so all earlier runs are unchanged and reproducible), threaded
into `anneal_cover`, and recorded in `summary.json` under `params`. The
saturation identity is in the `anneal_cover` docstring and in `--help`.

### 2. The energy, as a controlled pair

`optimized_algorithms.md` §14.3 closes with "the 460 plateau there also
deserves a retry with the new energy", and every successful certificate in
this repo (s2/s3/s4/s5 k=3, the cube, the wedge) ran the corrected energy
β = 5.0, γ = 0.5 — but the plan's pass-1 command line uses the *defaults*
β = 0.05, γ = 0. The two runs that did use the corrected energy at k = 2
(`s3_k2_v2_point`, `s3_k2_v2_track`) still hit 460, but on a 300 s leash.

So pass 1 is run **twice at the same seed 5**, differing in the energy alone.
Same seed ⇒ same initial population ⇒ the pair isolates the energy as a
variable, and a null result under β 5 / γ 0.5 cannot be blamed on the flat
penalty slope that §14 diagnosed.

## First: what the 460 actually is (checked exhaustively, no search)

Before reading any result, `rect2_optimum.py` (new, pure combinatorics: 25
rectangles star(v) × star(w), all 300 pairs and all 2300 triples unioned
exhaustively) pins down what the plateau value means:

| complex | best 1 rectangle | best 2 rectangles | best 3 rectangles | recorded k=2 plateau |
|---|---:|---:|---:|---:|
| ∂Δ³ (S²) | 54/96 | **84/96** | 96/96 | 84/96 |
| ∂Δ⁴ (S³) | 320/500 | **460/500** | 500/500 | 460/500 |
| ∂Δ⁵ (S⁴) | 1750/2520 | **2380/2520** | 2520/2520 | 2380/2520 |

The 2-rectangle optimum is attained by star(0) × star(0) ∪ star(1) × star(1),
and the 3-rectangle optimum is the star cover of the paper — all three
column-3 values are exactly the sizes in `results/s*_starcover` (S³:
320 + 140 + 40 = 500). Every recorded two-chain plateau in the repo is
**exactly** the 2-rectangle optimum of its complex. So the plan's reading of
460 is confirmed by exhaustive enumeration, not just by the search agreeing
with itself.

The same script also classifies the domains the annealer actually reached,
which sharpens C2 rather than merely restating it:

| run | sizes | domains |
|---|---|---|
| `s3_k2_A` | 284 + 265 | one inside a star-rectangle, one **not** a rectangle |
| `s3_k2_B` | 203 + 296 | one **not** a rectangle, one inside a star-rectangle |
| `s3_k2_C` | 284 + 216 | one inside a star-rectangle, one **not** a rectangle |
| `s3_k2_D` | 275 + 212 | one inside a star-rectangle, one **not** a rectangle |
| `s3_k2_v2_point` | 297 + 317 | **neither** is a rectangle |
| `s3_k2_longcor_5` (69 epochs) | 311 + 303 | **neither** is a rectangle |
| `s3_k2_v2_track` | 320 + 320 | both are *exactly* rectangles: star0×star1 and star3×star3 |

And the union is pinned at 460 across *very* differently shaped domain pairs —
this is the sharpest form of the observation, and it is not in the plan:

| run | domain sizes | total | overlap | **union** |
|---|---:|---:|---:|---:|
| `s3_k2_B` | 203 + 296 | 499 | 39 | **460** |
| `s3_k2_D` | 275 + 212 | 487 | 27 | **460** |
| `s3_k2_C` | 284 + 216 | 500 | 40 | **460** |
| `s3_k2_A` | 284 + 265 | 549 | 89 | **460** |
| `s3_k2_v2_point` | 297 + 317 | 614 | 154 | **460** |
| `s3_k2_longcor_5` | 311 + 303 | 614 | 154 | **460** |
| `s3_k2_v2_track` | 320 + 320 | 640 | 180 | **460** |

The search reaches 460 with pairs whose total size varies by 150 facets and
whose overlap varies by a factor of 6.7 — and every time, the union lands on
exactly the 2-rectangle optimum. Six independent shapes, one ceiling. (What
the search *achieved*, not a proven bound — but the variance in every other
quantity makes the constancy of the union hard to read as coincidence.)

The 40 missing facets are never the same 40 (pairwise overlaps between the
uncovered sets run from 0/40 to 40/40), and they are sometimes *two whole
20-facet cells* of the grid (as in `v2_track`: cells (1,3) and (4,1); the
optimal rectangle pair leaves (3,4) and (4,3)) and sometimes smeared over up
to four cells.

Two further facts from the campaign runs (`s3_k2_longcor_5` added below),
which together say the ceiling is not a *fixed* hole anywhere on the grid:

- **No facet is inherently uncoverable.** Taking the union of the covered sets
  over all seven recorded two-domain states, every one of the 500 facets is
  covered by at least one state. The obstruction moves around; it is not a
  pinned set of untouchable facets.
- **`v2_point` and `longcor_5` are statistically the same attractor.** Different
  seeds (54 vs 5), different budgets (6 epochs vs 69), and their domain pairs
  agree only 34% and 81% as sets — yet both have total 614, overlap 154, union
  460, and their *unions* differ by exactly 40 facets on each side (|A∖B| =
  |B∖A| = 40, Jaccard 0.84). Two runs landing on the same total, the same
  overlap and the same union from different directions is a well-defined
  attractor, not six lucky samples.

So the plateau is **not** merely "the search falls into rectangle covers" —
`v2_point` reached 460 with two domains that are not products at all, and
`v2_track` reached it with two *perfect* rectangles, just not the optimal pair
(its union is 460, the optimum pair's is also 460). Overshooting 460 does not
look like a matter of finding the right rectangle pair; the non-product
plateaus are the evidence that matters for C1.

## A proof route that is closed: the H₁ obstruction is vacuous on spheres

`h1_obstruction.py` decides, for a facet set J, the necessary condition
(π₁ − π₂)_* = 0 on H₁(|J|; Z₂) — if it fails, J is not a contiguity domain
"whatever the budget", which is how the torus experiments (§16) pruned
candidates. Run on ∂Δ⁴, it separates nothing:

| candidate J | facets | dim H₁(J;Z₂) | rank (π₁−π₂)_* |
|---|---:|---:|---:|
| all of K × K | 500 | 0 | 0 |
| D_diag (s = t) | 100 | 0 | 0 |
| both domains of `s3_k2_v2_track` | 320 | 0 | 0 |
| both domains of `s3_k2_v2_point` | 297, 317 | 0 | 0 |

H₁(∂Δ⁴ × ∂Δ⁴) = 0, so the test is vacuous here — **C1 cannot be settled by
the H₁ obstruction**; on the odd spheres the H₁ tool has no teeth and search
evidence is what there is. (Recorded so nobody re-derives it.)

## Pass 1 — cold, one real hour each (was: 300 s)

| run | energy | seed | budget | result |
|---|---|---|---|---|
| `s3_k2_longcor_5` | β 5.0, γ 0.5 | 5 | 3600 s | **460/500 — 69 epochs, best 460 in *every* epoch** |
| `s3_k2_longflat_5` | β 0.05, γ 0 (plan-literal) | 5 | 3600 s | **460/500 — 41 epochs, best 460 in *every* epoch** |

Both runs, in full:

| run | epochs | moves | domain sizes | distinct `best` | distinct `median` |
|---|---:|---:|---|---|---|
| `longcor_5` | 69 | 5.65 × 10⁹ | 311 + 303 | {460} | {455, 460} |
| `longflat_5` | 41 | 3.36 × 10⁹ | 232 + 273 | {460} | {449, 457, 458, 460} |

The control is worth more than its one-line result suggests: under the flat
energy the *population* wanders far more (medians 449–460, against a nearly
frozen 455→460 for the corrected energy), yet its best never exceeds 460
either. The ceiling is therefore not an artefact of a collapsed population —
a wider-exploring search with the diagnosed-flat energy hits the same wall.

`s3_k2_longcor_5`, in full: 69 epochs, 5.65 × 10⁹ moves, 3648 s, domain sizes
311 + 303, `best` = 460.0 in all 69 epochs, `median` 460 in every epoch but
the first (455). The plan's "zero spread" signature, held for an hour.

> **Caveat on the controlled pair.** The construction probes below were run on
> the same GPU while the control was searching, so `s3_k2_longflat_5` will
> finish with roughly half the epochs of `s3_k2_longcor_5` (throughput dropped
> from ~1.54 to ~0.95 Mmoves/s). It still gets far more than the 2–3 epochs
> every historical k = 2 run got, so a plateau in it is still informative — but
> the pair is *not* epoch-matched, and the honest comparison is qualitative.
> A clean control would re-run p1flat alone; noted rather than quietly
> presented as matched.

Both: `--complex s3 --k 2 --cmax 40 --S 4096 --iters 20000 --epochs 200
--ptrack 0.05 --guide 1.0 --guide-max 4`, i.e. the plan's pass-1 command plus
the explicit energy. At `ptrack 0.05` an epoch costs ≈ 54 s, so 3600 s buys
**≈ 66 epochs** — versus the 2–3 epochs every earlier k = 2 run got. That
factor of ~25 in search budget is the whole point of the pass. In moves
(S = 4096 threads × 20000 iterations = 8.19 × 10⁷ per epoch): one pass-1 run
makes **≈ 5.4 × 10⁹ moves**, against 2.5 × 10⁸ for the whole of `s3_k2_D`.
Both pass-1 runs together therefore search more than the twelve recorded
two-chain runs combined by roughly an order of magnitude.

## Escalation rungs — the levers, and which of them have actually been tried

Every two-chain search on record ran **one move set at one guide setting on a
300 s leash**. The rungs below each change exactly one thing that has never
been tested at a real budget, warm-started from the previous rung's own best
state (half the population seeded, half random — `--init`):

| rung | what it changes | why |
|---|---|---|
| `esc-track` | `ptrack 0.3` (was 0.05), cmax 40 | the move set that solved the *circle* at k = 2 (§14.3: β 5, ptrack 0.3 → 18/18 in 9 s). On S³ it has had **3 epochs total** (`s3_k2_v2_track`), never a budget. |
| `esc-guide` | `guide 3 / max 11`, cmax 80 | a real guide escalation (see the cap discussion above) plus domains long enough to leave rectangle structure wholesale. |
| `esc-heavy` | `guide 10 / max 34`, cmax 80, `ptrack 0.3`, `T1 0.1` | the kitchen sink, only if the first two move. |

```bash
INIT=results/s3_k2_longcor_5/summary.json OUT=results/s3_k2_esc_track_6 \
    GUIDE=1 GMAX=4 PTRACK=0.3 CMAX=40 SEED=6 ./run_s3_k2_push.sh esc

INIT=results/s3_k2_esc_track_6/summary.json OUT=results/s3_k2_esc_guide_7 \
    GUIDE=3 GMAX=11 CMAX=80 PTRACK=0.05 SEED=7 ./run_s3_k2_push.sh esc
```

Each rung is one hour. A rung that stays at exactly 460 with zero top-10
spread for its full hour is the plan's decision criterion for "structural, not
budgetary" — and the *reason* it is worth running even when pass 1 already
looks pinned is the cube precedent: there, 60 full epochs (814 s) stayed at
820/864 and only a warm start + guide closed the last 44.

## The one precedent: how the circle's two-domain cover is actually built

The circle is the only place in this repo where a *two-domain* cover was found
by search (`results/circle_probe3`, 9 + 9 facets, `SC_strict(∂Δ²) = 1`). If the
recipe lifts to ∂Δ⁴, C1 is false by construction; if it provably cannot, that
is a reason to believe C1. Decoding the certificate (18 facets in 9 cells of
Δ¹ × Δ¹, two staircase triangles each):

| | domain 0 | domain 1 |
|---|---|---|
| diagonal cell (s,s), s = 0,1,2 | 1 of its 2 triangles | the other 1 |
| off-diagonal cells (s,t) | the 3 with **missing-vertex shift +1** | the 3 with shift **−1** |

where "missing vertex" identifies a facet of ∂Δ² with the vertex it omits
(facet s of K omits vertex 2 − s), and the shift is read in the cyclic order
of the 3 vertices of Δ². So the cover is: **each domain takes half of every
diagonal cell, and the off-diagonal cells are split between the two cyclic
directions.** Both domains get 3 half-diagonals + 3 whole cells = 9 facets.

That is the simplicial lift of the classical two-chart cover of S¹ × S¹ — and
it is exactly why it does not obviously lift to ∂Δ⁴. The recipe needs a
*cyclic order on the facets' complementary data*, and it uses the fact that
the vertex set of ∂Δ² **is** a cycle (its three vertices are cyclically
ordered). The vertex set of ∂Δ⁴ is the 1-skeleton of a 4-simplex: K₅, in which
every vertex is adjacent to every other and no cyclic order exists. The
natural analogue of "shift +1 vs −1" there is the orientation of the ordered
pair (i,j) of omitted vertices — i.e. the candidate cover

    A = {cells (i,j) : i < j} ∪ half of every diagonal cell (i,i)
    B = {cells (i,j) : i > j} ∪ the other half of every diagonal cell

250 facets each if the diagonal cells can be halved. Whether the halves can be
chosen so that *both* domains admit a chain is exactly the open question, and
it is a finite combinatorial question, not a matter of budget.

### The harness, and what it says on the circle

`fixed_domain_test.py` (new) turns the question around: give it a facet set and
it anneals **one** chain (k = 1) on the grid restricted to that set — through
`hybrid_cover.SubProduct`, the same machinery the hybrid experiments use. Full
coverage ⇒ that set is a verified domain. Validated against three known
answers on the circle:

| facet set | size | one chain covers | expected |
|---|---:|---:|---|
| `circle_probe3` domain 0 | 9 | **9/9, epoch 1** | a domain — it is in the certificate ✓ |
| all of ∂Δ² × ∂Δ² | 18 | 9/18, no chain in budget | not a domain (SC_strict = 1 needs two) ✓ |
| the "upper triangle" {s ≤ t} | 12 | 9/12, no chain in budget | *unknown a priori* |

The third row is the cautionary one, and it **kills the naive lift**: the
crude upper/lower decomposition does not give valid domains even on the
circle, where a two-domain cover exists. The 9 facets it *does* cover are
{1,2,3,4,8,9,10,11,17} — not the certificate's domain 0 (a different 9-facet
set), and not the "off-diagonal cells plus half of each diagonal" shape one
would guess: this chain took all of the diagonal cell (1,1) and only halves of
(0,0) and (2,2), plus half of the off-diagonal (0,2). (Written after checking —
the first guess, that it had reproduced the certificate's shape, was wrong.)

So the circle's cover is not recoverable from the naive construction, and the
lift to ∂Δ⁴ would have to choose the diagonal halves in a way that is not
"the upper triangle's share". The ∂Δ⁴ probe (`results/s3_upper_probe`: how
much of {s ≤ t}, 300 facets, one chain can hold) is the measurement that says
how far the crude version gets — it is a probe, not a certificate attempt.

## The constructive lead: a two-domain cover with one domain already verified

`results/s3_upper_probe` — one chain annealed on {s ≤ t} (300 facets), 908 s —
produced the most useful object of the campaign. It did **not** cover the 300,
but *what* it held is the structure:

| part of {s ≤ t} | held |
|---|---|
| all 10 off-diagonal cells (200 facets) | **200/200** |
| diagonal cells (1,1), (2,2), (3,3) | **20/20 each** |
| diagonal cells (0,0), (4,4) | **0/20 each** |

Look at what that chain's domain *is*: a verified domain of **260 facets**
(the chain is right there in the probe's `summary.json`), holding the entire
upper off-diagonal block and exactly the three middle diagonal cells. The two
it drops are 40 facets — the same 40 the whole k=2 campaign keeps missing.

That turns the loose "find a cover" into a finite, structured question. If

    A = the probe's 260-facet chain domain       (verified: the chain exists)
    B = {s > t} ∪ {(0,0), (4,4)}                 (240 facets)

then A ∪ B = 500 and A ∩ B = ∅ — *checked, exactly a partition of the product*.
So **one single-chain search decides it**: if B admits a chain, the two
domains cover ∂Δ⁴ × ∂Δ⁴ and SC_strict(∂Δ⁴) = 1 (sharp against TC(S³) = 1),
i.e. C1 is false and the corollary is one domain from sharp.

The general family this sits in: give each of the 5 diagonal cells wholly to A,
wholly to B, or split it — with the off-diagonal cells assigned by orientation
(upper to A, lower to B). The circle's own cover is a member of this family
(it splits every diagonal cell and assigns off-diagonals by cyclic direction),
which is what makes the family worth searching. The immediate test is the
single configuration above.

```bash
# B, the complement of the probe's domain — running since 21:25 UTC
$PY fixed_domain_test.py s3 --rule lower-ends --S 2048 --iters 10000 \
    --epochs 100 --time-limit 1200 --seed 11 --outdir results/s3_lower_ends_probe
```

### What B does, and one deduction that comes free

`results/s3_lower_ends_probe` opens at **200/240 from epoch 1 with the whole
top-10 at 200 and no movement** — the same signature as the A probe. It holds
the 200 lower off-diagonal facets and takes **nothing** from (0,0) or (4,4).

Its domain is worth reading exactly, because it is not what the coverage
number suggests: **200 facets = 9 of the 10 strict-lower cells + the whole
diagonal cell (4,4)**. It holds one of the two extremes in full, and pays for
it by dropping the off-diagonal cell (4,3).

> **Correction.** An earlier draft of this section claimed a budget-free
> deduction here — "the lower chain cannot hold both extremes, because a chain
> restricts to any sub-domain". That argument is **invalid**: it needs a chain
> covering *all* of {s>t} ∪ {(0,0),(4,4)}, and the chain found covers 200 of
> the 240 (it drops (4,3)), so the restriction argument has nothing to apply
> to. The honest statement is only that no chain covering all of that set was
> found in a 20-minute search. Also note it *does* hold the extreme (4,4), so
> "the extremes are hard" is not the right reading either. Left here rather
> than deleted, because the mistake is easy to repeat.

So the whole-cell family reduces to a question with the following shape —
**can a chain hold the extremes together with enough of the block?** — and to
the "swap" configuration:

    A' = {s < t} ∪ {(0,0), (4,4)}      (`--rule upper-ends`, 240 facets)
    B' = {s > t} ∪ {(1,1), (2,2), (3,3)}   (`--rule lower-mid`, 260 facets)

A' ∪ B' = 500 again (verified: 240 + 260, disjoint). Both have to admit a
chain for this family to give the cover; each is one single-chain search.

### The half-split family, and why it does not lift either

The circle's cover does **not** give whole diagonal cells to either domain — it
gives each domain *one triangle* of every diagonal cell. The natural lift of
that to ∂Δ⁴ is to cut each diagonal cell's 20 staircase facets in half (first
10 vs last 10 local indices, i.e. facets whose last shuffle step advances in
the first vs the second factor), giving `--rule upper-half` (= {s < t} plus
half of each diagonal cell, 250 facets) and `--rule lower-half` (250 facets),
which partition the 500. Checked on the circle, where the answer is known:

| facet set | size | one chain covers | |
|---|---:|---:|---|
| `upper-half` | 9 | 7/9 | **no chain** |
| `lower-half` | 9 | 7/9 | **no chain** |
| the certificate's own domain 0 | 9 | 9/9, epoch 1 | chain ✓ |

So the half-split shape is *not* what makes the circle work: two 9-facet
domains with exactly that shape fail, while the certificate's two 9-facet
domains succeed. The difference is which facets they take — and the
certificate's choice is organised by the **cyclic order of the three facets**
(off-diagonal cells go to the two cyclic directions t = s ± 1, diagonal cells
are halved). That is a genuinely 1-dimensional structure: ∂Δ²'s facets are the
three arcs of a circle. ∂Δ⁴'s five facets are the five faces of a 4-simplex —
pairwise adjacent, no cyclic order — so the mechanism that makes the circle
work has no obvious analogue one dimension up.

That is the campaign's most useful *positive* argument for C1 (as opposed to
the negative evidence of yet another plateau): the known escape from the
product bounds at k = 2 exploits a 1-dimensional cycle, and the odd spheres
above S¹ do not have one.

## Where the campaign stopped, and what was deliberately not run

**Stopped on 2026‑09‑19 ~21:55 UTC by decision of the author**: the line is
plateauing, so it stops and is documented rather than extended. Two runs were
killed or never launched, and they are listed here so that a later reader can
tell "not tried" from "tried and failed":

| not run | why it was the next step |
|---|---|
| `lower-mid` ({s>t} ∪ middles, 260) | second half of the swap pair; moot once `upper-ends` returned 200/240, since that family already cannot cover |
| `upper` escalation (300 facets, ptrack 0.3, guide 3/max 11, seed 23) | the one configuration that could still make {s ≤ t} fully coverable, hence a cover with {s > t} |
| `strict-lower` (200) | whether a chain can hold a whole orientation block |
| `esc-track` / `esc-guide` (the plan's rungs) | the plan's own escalation ladder, made real by the `--guide-max` fix |

The first three are cheap (10–20 min each) and the last two are an hour each;
none is a dead end in principle, they are simply unrun.

## Verdict

**The 460 plateau was not broken.** Every two-chain search still pins at
exactly 460 with zero spread — now including the first ones ever given a real
budget (69 epochs, 5.65 × 10⁹ moves). What the campaign added is that 460 is
now *characterised* rather than merely observed, and that the two structured
families which could plausibly have contained a cover were closed with reasons:

1. 460 is exactly the exhaustive 2-rectangle optimum (S² 84/96, S³ 460/500,
   S⁴ 2380/2520 — all three verified by enumerating all rectangle pairs).
2. …yet the plateau states are frequently not rectangle covers at all, and the
   union is 460 across domain pairs whose totals span 487–640 and overlaps
   27–180. One ceiling, many shapes.
3. No facet is inherently uncoverable — the hole moves.
4. The H₁ obstruction is vacuous on spheres, so the repo's one general tool for
   turning this into a theorem cannot be applied here.
5. Constructively: two verified single-chain domains were exhibited
   (260 and 200 facets) and their complements do not close; the whole-cell
   family is dead on the two ends it was testable at, and the half-split family
   was shown invalid *on the circle*, where the answer is known.
6. The circle's k = 2 escape mechanism is **1-dimensional** (it partitions the
   off-diagonal cells by the cyclic order of ∂Δ²'s three facets, and halves the
   diagonal cells). ∂Δ⁴'s five facets have no cyclic order. This is the
   campaign's one positive argument that C1 is true, as opposed to more
   negative evidence.

What this does **not** do is prove C1. The search space is chains, not domains;
"no cover found" is evidence about the landscape. The claim that now has
support is the empirical one — *no two-domain cover of ∂Δ⁴ has been found by
any search, structured or unstructured, at any budget tried, and all of them
stop at exactly the 2-rectangle optimum.*

## Pre-registration — what outcome counts as what

Fixed before any result was read, from the plan's decision criteria:

1. **Still exactly 460/500, zero top-10 spread, after the hour *and* after the
   escalation rungs** → the plateau is structural, not budgetary. C1 stands on
   much firmer ground, and the twelve recorded runs become eighteen-plus runs
   of evidence. The claim that gets written down is the defensible one:
   *no two-domain cover was found by searches totalling ~N hours and ~M
   epochs, all pinned at the exact 2-rectangle optimum*, not "no two-domain
   cover exists".
2. **Any movement past 460** → the attractor is escapable and the budget
   hypothesis was right; every facet past 460 is new information. Escalate
   harder (wider cmax, higher guide cap, another seed).
3. **500/500** → SC_strict(∂Δ⁴) = 1, sharp against TC(S³) = 1, beating
   Corollary 5.3's 2 with non-product domains — the circle phenomenon one
   dimension up, and C1 is false. Then audit, then repeat the ladder on ∂Δ⁵
   (plateau 2380/2520).

A note on what a null result can and cannot say: the search space is chains,
not domains, so "no cover found" is evidence about the landscape, never a
proof. Only a lower-bound argument can make C1 a theorem — and the one
general-purpose tool this repo has for that, the H₁ obstruction, is vacuous
on spheres (above).
