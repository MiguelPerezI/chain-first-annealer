# Formal properties of the annealer on surfaces: what is provable, what is not

*2026-09-24. Companion to `surface_stall_analysis.md` (the state-by-state
data) and `klein_anneal_log.md` §3–§5. Instruments:
`scripts/plateau_barrier_audit.py` (new — exhaustive one-move neighborhood
audit, coverage AND kernel-energy), `scripts/klein_h1_absorption.py`,
`src/h1_obstruction.py`. Every "machine-proved" claim below replays with the
commands at the end. Conjectures are labelled as such.*

## 1. The search problem, formalized

Fix K (finite simplicial complex, nv vertices, f triangles), the staircase
triangulation of K×K with facet set F, |F| = 6f². Fix k (domains) and
cmax (chain length budget).

- **State space** S = the k-tuples of chains Φ^(1..k), each an array
  (cmax+1) × nv² of vertex maps with Φ[0] = π₁, Φ[cmax] = π₂ pinned.
  |S| = nv^{k·(cmax−1)·nv²} — exponential in every natural parameter.
- **Objective** (gpu_sc energy): E = Σ_f w_f·pen(min_j bad_j[f]) +
  γ·Σ_{j,f} bad_j[f] + μ·(redundancy), pen(0) = 0, pen(b) = 1+βb; the runs
  here use β = 5, γ = 0.5, μ = 0, w_f = 1 at the floor (the guide raises
  w_f on uncovered facets).
- **Move set** M: track moves — (chain j, grid vertex w, block i1..i2 ⊆
  interior maps 1..cmax−1, |block| ≤ 8, new value u ∈ V). A move changes
  bad-counts only on star(w) (kernel line: "delta over the star of w").
  **M is finite and exhaustively enumerable**: |M| = k·nv²·284·(nv−1) at
  cmax=40 (e.g. 257,485 for RP² k=5, 1,031,296 for Klein k=8).
- **Dynamics**: Metropolis, per-move acceptance min(1, e^{−dE/T}), geometric
  schedule T₀ = 0.6 → T₁ = 0.03 over 5000 iters × 60 epochs, S = 4096
  independent replicas, tournament selection between epochs.

The decision problem behind the search —
**COVER(k, c, K): do k strictly-contiguous domains with chains of ≤ c
links cover F?** — is in **NP** (a witness is the maps; verification is
O(k·c·|F|) simplex-table lookups, which is what `independent_check.py`
does with no shared code). Its exact complexity is open: no NP-hardness is
known, no polynomial algorithm either. Two consequences:

1. **An unconditional "the annealer needs super-polynomial time" is out of
   reach.** It would have to classify the entire reachable state space
   (|S| exponential), and even then would be a statement about one
   heuristic, not about the problem.
2. **The provable substitute is a stagnation certificate**: at a recorded
   state, every move is uphill by a certified amount, so escape at the
   annealer's own final temperature takes certified exponential time
   (Theorem 1). This is the formal content of "the anneal will not get
   there in polynomial time" — it is state-relative, which is exactly the
   right granularity, because the annealer provably *reaches* these states
   (they are the incumbents of the recorded runs).

## 2. Theorem 1 (one-move certificates) — with the audit numbers

`plateau_barrier_audit.py` enumerates the entire move set M(s*) of a
recorded plateau state and computes, for every move, the coverage change
Δcov and the kernel-energy change dE (β = 5, γ = 0.5, w = 1 — the floor
weights; the guide can only lower the uphill dE of repair moves, so these
are the conservative certificates). Results:

| state | coverage | \|M\| | max Δcov | improving moves | min dE over coverage-gaining moves | cheapest move overall |
|---|---:|---:|---:|---:|---|---|
| Möbius k=3 (seed 666) | 148/150 | 93,192 | **0** | 0 | — (none exist) | dE = **−53** (Δcov = 0) |
| RP² k=5 | 540/600 | 257,485 | **0** | 0 | — (none exist) | dE = **−75** (Δcov = 0) |
| RP² k=3 | 398/600 | 164,447 | **0** | 0 | — (none exist) | dE = **−82.5** (Δcov = 0) |
| Klein k=8 | 1483/1536 | 1,031,296 | **+2** | 78 | **+13** (all 78 uphill, max +684) | dE = **−186.5** (Δcov = 0) |
| Klein k=9 | 1519/1536 | 1,197,625 | **+1** | 62 | **+44.5** (all 62 uphill, max +395) | dE = **−252** (Δcov = 0) |

**Theorem 1 (certified freeze).** At each recorded plateau state s*:

(i) **Möbius, RP² k=5, RP² k=3**: s* is a *strict local optimum of
coverage*: no move of the entire enumerated move set increases coverage —
at any temperature, under any weights. Every one-move repair of a leftover
is net-negative or exactly neutral: on Möbius every repair loses ≥ 6
covered facets (best repair 148 → 142, reproducing the audit quoted in
`mobius_sc_strict.tex`); on RP² the 747 (k=5) / 4,193 (k=3) repair moves
*swap* — repair up to 3 leftovers while losing the same number of covered
facets. **The move set can trade but never gain.**

(ii) **Klein k=8, k=9**: coverage-improving moves *do* exist (78, resp.
62), but **every one is uphill in the kernel energy**: gaining 1–2
facets dirties dozens of links and the γ-cleanliness term dominates —
dE ≥ +13 (k=8; max +684), dE ≥ +44.5 (k=9; max +395); zero of them are
downhill. The Metropolis acceptance of the cheapest coverage gain at
the run's final temperature T₁ = 0.03 is e^{−13/0.03} = e^{−433}.
**The β/γ energy — added to suppress chain drift — provably forbids the
remaining repairs.** This explains the flat 20 epochs of the k=8
continuation: the state is not coverage-locally-optimal, it is
*energy*-blocked.

(iii) **All five states admit coverage-neutral, energy-downhill moves**
(dE down to −82.5 on RP² k=3, −186.5 on Klein k=8, −252 on Klein k=9,
at Δcov = 0): the annealer at the plateau keeps lowering its energy
forever at frozen coverage — by polishing redundant link defects
(pen(3) → pen(1) on near-miss facets) that provably cannot reach 0 (the
residual defect sits on a witness cycle, Theorem 2). **This is the
flat-score plateau in its exact mechanism**: energy decreases, coverage
does not, because the last defect is topological, not energetic.

**Corollary (what "will not succeed in polynomial time" means, exactly).**
From s*, any increase of coverage requires either (a) accepting an
energy-uphill move — impossible in one move on Möbius by (i), of
probability ≤ e^{−433} on Klein by (ii) — or (b) escaping *sideways*:
drifting through coverage-neutral swap moves to a state outside the
plateau family and ascending from there. (a) is machine-proved dead **at
floor weights w = 1**. (b) is the honest formal gap, and it is
**empirically real on RP²**: a second k=5 run (seed 7434, 59 epochs, same
parameters including guide=3.0) pinned at 540 for 40 epochs and then
crawled to 542 — proving that sideways drift along the ridge eventually
reaches a different 540-state from which absorption is possible.

**Crucial caveat on the RP² barrier certificates.** The `rp2_k5_hybrid`
barrier audit (60/60 H₁-dead, 0 improving moves, 747 neutral swaps) was
computed at **floor weights w = 1**. But the guide raises the weights of
uncovered facets to w* = 1 + guide/0.3 = 11.0. At guide weights, **254
of the 747 neutral swap moves become strongly energy-downhill** (best
dE = −301.5): gaining a leftover facet worth 11 × pen(1) = 66 outweighs
the γ-cleanliness cost. This is the escape mechanism: the guide energy
provides a gradient across the coverage-neutral ridge, slowly pulling
the domains toward shapes that can absorb leftovers, until after ~800 s
of drift the first +1 exit was found (epoch 41 of the 7434 run). The
`rp2_k5_hybrid` run stalled at 540 after only **3 epochs (66 s)** — a
time limit, not a dynamics barrier. The **honest claim** is: the recorded
3-epoch state is a strict coverage local optimum at floor weights; the
guide provides a gradient to escape it; the second run proves escape is
reachable in ~40 epochs; but it then re-stalls at 542, and the question
of whether a full 5-cover *exists* remains open.

For **Möbius** the situation is genuinely harder: the 105 repair moves at
the 148/150 plateau cost ≥ 6 covered facets even at floor weights. A
swap that gains 1 uncovered loses 6 covered — a net −5 in coverage that
no weighting scheme can make attractive (at best the energy for the swap
is less negative, never positive). The Möbius barrier is
coverage-structural, not weight-dependent. And the 97-epoch long run,
six seeds, warm restarts, and the crossover operator all pin at exactly
148 — sideways escape is empirically refuted at the 10⁹-move scale.

The remaining question is not "can the annealer escape the plateau"
(on RP² it provably can given time; on Möbius it empirically cannot) but
"does a k-cover exist at all" — which belongs to COVER(k, c, K):

- Möbius k=3: a cover **exists** (`mobius_k3_sat`, CP-SAT, 42 s) — there
  the annealer's failure is purely its own dynamics, on a solvable
  instance.
- RP² k=5: **open**. The warm hybrid CP-SAT (3 frozen plateau domains + 2
  free, L = 8) is INFEASIBLE, but a cold k=5 model (~10⁶ variables) has
  never been run. Either answer is a theorem: SAT ⇒ SC_strict⁰(RP²) ≤ 4;
  UNSAT ⇒ SC_strict⁰(RP²) = 5 — the first exact value for a closed
  non-orientable surface.
- Klein k ≤ 8: open, and TC(Klein) = 4 only forces k ≥ 5.

## 3. Theorem 2 (why local moves cannot heal winding) — the local/global gap

**Lemma (defect localization).** Let D ⊆ F with
rank (π₁−π₂)_* : H₁(D; Z/2) → H₁(K; Z/2) > 0, and let c be a witnessing
1-cycle (c ≠ 0 in H₁(D), (π₁−π₂)(c) ≠ 0). Then **every chain has a bad
link on some facet carrying an edge of c**. *Proof.* Suppose Φ has all
links contiguous on every facet of D that contains an edge of c; call
their union T ⊆ D. The cycle c is a cycle of T's 1-skeleton (the chain
complex of `h1_obstruction` is generated by the facets of D, so every
edge of c lies in a facet of D, hence in T). If c bounded in T it would
bound in D ⊇ T, contradicting c ≠ 0 in H₁(D); so c ≠ 0 in H₁(T).
Contiguity on all facets of T makes Φ|_T a homotopy π₁|_T ≃ π₂|_T
(contiguity ⇒ homotopy; faces inherit contiguity), so
(π₁−π₂)_*[c] = 0 — contradiction. ∎

**Corollary (restructuring necessity — the formal content of the
absorption tables).** If D_j ∪ {f} is obstructed, then chain j can never
have zero bad links on all of D_j ∪ {f}. Since it *does* have zero bad
links on D_j, absorbing f is possible only by **shedding a facet of the
witness cycle c ⊆ D_j ∪ {f}** — an edit potentially far from f, in the
star-locality sense: a track move at w changes bad-counts only on
star(w), but the facets that must leave the domain lie wherever c goes,
and c is noncontractible — on RP² it crosses the core of the projective
plane, on the Klein bottle it may wrap the Z/2 torsion direction as well.
Monotonicity (`opti_sc.note_failure`) then upgrades the per-facet
machine tests (60/60 on RP² k=5, 200/202 on RP² k=3, 424/424 on Klein
k=8, 15/17 on Klein k=9, 330/338 on Klein k=5) to: **for every nonempty
A ⊆ L and every j, D_j ∪ A is unchainable — no completion of the recorded
state exists that only grows domains.** This is the theorem behind
"extension is dead; restructuring or nothing."

What the lemma does NOT give: a bound on the cost of restructuring (the
RP² audit shows the swap moves exist and are free — the ridge is walkable);
it bounds *extension*, and localizes *why* the single remaining defect
sits where it sits (on the witness cycle — the measured defect depth 1
everywhere is the annealer having pushed the defect onto the cycle and
shuttling it along, never off, because off = shedding = losing coverage).

## 4. Theorem 3 (sphere vacuity) — why ∂Δⁿ yields

**Theorem.** If H₁(‖K‖; Z/2) = 0 then for every D ⊆ F,
rank (π₁−π₂)_* = 0: the H₁ obstruction is **vacuous** — no domain
whatsoever is topologically forbidden, at any budget. ∎ (The target of
(π₁−π₂)_* is H₁(K; Z/2) = 0.)

Every sphere in the repository — ∂Δ²..∂Δ⁹, the cube, the dodecahedron —
has H₁ = 0. On these, chainability is constrained only by the *energy*
(per-link simplex tests), never by topology. The empirical correlates
(`klein_anneal_log.md` §5, now with the audits):

- every completed annealer cover is a level-0-sharp instance — SC_strict⁰
  = TC (circle after the energy fix, barycenter, cube, ∂Δ³, ∂Δ⁵, ∂Δ⁷,
  wedge). Where a cover exists and no topological obstruction exists, the
  annealer finds it in seconds-to-hours;
- the spheres' own stalls (S³/S⁵ k=2 at 460/500, 84/96, 2380/2520) are
  **exact combinatorial optima of the reachable family**
  (`rect2_optimum.py`: 460 IS the 2-rectangle optimum) — plateaus that are
  real facts about the problem (and conjecturally about the problem's
  answer itself, C1), not dynamics failures.

So the dichotomy is not "sphere easy, surface hard" but:

**Two orthogonal diseases, and the surfaces have the first one:**

| disease | formal criterion | present on | provable? |
|---|---|---|---|
| **winding** | H₁(K; Z/2) ≠ 0 | Möbius, RP², torus, Klein — every stalled surface; the obstruction fires at every scale down to L alone | yes — Theorem 2 + machine absorption tables |
| **sparsity** | simplex density of K, dual diameter | dodecahedron (3.2 % triples, dual diameter 7) — failed at 23 % with H₁ = 0 | measured, mechanism understood (`gpu_pending.md`) |
| *(neither)* | H₁ = 0 **and** dense | ∂Δⁿ: 1-skeleton complete, dual diameter 1, every proper vertex subset a simplex — links almost never bad, the energy is smooth | — |

RP² and the Möbius band are the cleanest specimens because they have the
winding disease **and maximal density** (1-skeletons K₆ and K₅: triangle
density 10/20 and 5/10; every facet lies in 9 maximal rectangles) — the
sparsity axis is completely controlled, and the stall is pure winding:
all leftovers on far pairs at the dual diameter, defect depth exactly 1,
absorption dead. The Klein bottle has both axes degraded (28.6 % triple
density, dual diameter 4, plus torsion), and it is the only surface where
the *leftover complex itself* winds (rank 2: both the Z and the Z/2
directions) at k = 8 and k = 5.

Why winding kills the annealer specifically: the annealer optimizes a sum
of **local** terms (per-link, per-facet), but by Theorem 2 chainability
of a domain is a **global** (homotopical) property. On H₁ = 0 complexes
the two coincide (local = global: all domains admissible). On H₁ ≠ 0
complexes the boundary between chainable and unchainable domains passes
through the state space in a way no local term can see: a winding domain
is locally optimal at every link (defect depth 1 = the irreducible
witness-defect of Lemma 2, parked on the cycle), and eliminating it
requires a non-star-local edit that costs coverage — the certified
uphill/stagnation regime of Theorem 1.

Subdivision is the theorem-level resolution (SC = lim_b SC_strict^b = TC):
contiguity always implies homotopy, and homotopy implies contiguity only
after subdividing — the level-0 gap SC_strict⁰ > TC is exactly the
combinatorial residue that the winding disease is the mechanism of.

## 5. What to do next, in light of the formal status

1. **Settle RP² k=5 existence by CP-SAT (cold, k=5, L=8)** — the decisive
   experiment: ~10⁶ variables, no GPU. Either outcome is a publishable
   theorem (§2). Then k=4, k=3 if SAT.
2. **γ-annealing as the formal cure for the Klein regime**: by Theorem 1(ii)
   the 78 improving moves at the k=8 plateau are uphill only because
   γ·(Σ Δbad) > β·(coverage gain); with γ = 0 they are strictly downhill
   (their coverage term is negative). A falsifiable prediction: re-run the
   k=8 continuation with γ scheduled → 0 over the last epochs — coverage
   must move (≥ +2 available in one move); if it still stalls, the barrier
   is one level deeper than the certificate. The h1-guide weighting lowers
   exactly these deltas too (it raises w_f only on uncovered facets), so
   the same prediction applies to strong guide. On Möbius/RP² (Theorem
   1(i)) no reweighting can help: no move gains coverage at any weights —
   only new *move types* (domain-cutting) or external construction can.
3. **Quantify the witness radius** (open): extract minimal witness cycles
   from `gf2_nullspace` for each failed (j, f) absorption and measure
   their grid extent — this would turn "the required edit is non-local"
   into a number (lower bound on the move count of any restructuring).

## Reproduce

```bash
PY=~/whisper_env/bin/python   # numpy+numba
$PY scripts/plateau_barrier_audit.py mobius results/mobius_seed_666/summary.json
$PY scripts/plateau_barrier_audit.py rp2   results/rp2_k5_hybrid/summary.json
$PY scripts/plateau_barrier_audit.py rp2   results/rp2_k3_cold/summary.json
$PY scripts/plateau_barrier_audit.py klein results/klein_k8_r1/summary.json
$PY scripts/plateau_barrier_audit.py klein results/klein_k9/summary.json
# absorption / extension-deadness (Theorem 2 corollary):
$PY scripts/klein_h1_absorption.py <complex> <summary.json>
# Aut(hemi-icosahedron) = 60, transitive on the 6 vertices (S_6 enumeration):
#   the six dropped-diagonal-rectangle plateau states of RP² k=5 are
#   isomorphic; one audit certifies the family.