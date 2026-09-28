# RP² k=5 by orbit decomposition and GPU contiguity tables

Date: 2026-09-27. Status: approach approved by Jaziel, spec under review (Mike
decides the orbit split, section 3).

## Problem

The decisive question of the cluster runbook is whether RP²×RP² (600 staircase
facets) has a level-0 cover by 5 planner domains. CP-SAT on the monolithic
model `P_{5,L}` did not finish on the laptop: at L = 4 the model is small
(63,546 variables, 636,604 constraints) but memory grew for 4.5 hours until it
hit 6 GB, with no solution and no bound (`rp2_klein_cp_sat_results.md`).
Two things drive that:

- Search memory, not model memory: 8 parallel subsolvers keep learned
  clauses for hours.
- Symmetry: CP-SAT detected 1 generator, but Aut(RP²) = A₅ has order 60 and
  acts transitively on the 10 triangles (`rp2_new_mathematics.md` section 2).
  The solver explores up to 60 equivalent copies of every partial assignment.

GPU cannot run CP-SAT (no production SAT/CP solver runs on GPU; clause
propagation is sequential and irregular). GPU can do the combinatorial
precomputation and the case enumeration. This spec uses it for exactly that.

## Route 1: orbit decomposition, many small CP-SAT cases

### Idea

Fix domain 0 up to symmetry. Any 5-cover has a domain 0; applying an element
of A₅ (acting diagonally on RP²×RP², so the order stays 60) yields another
5-cover. It is enough to search covers whose domain 0 is a chosen
representative of each orbit of "candidate domain 0". Each case is a smaller
CP-SAT: domain 0 frozen, 4 free domains, residual = facets not in domain 0.
UNSAT in every case is UNSAT globally, with the same rigor as the monolithic
model (the case split is exhaustive by construction).

### Two case families, in order

**Family A: the star family (cheap, structured).** Every observed RP² stall
re-converges to diagonal rectangles star(v)×star(v)
(`rp2_new_mathematics.md` section 6): four exact rectangles plus a fifth
missing 9 facets. Cases:

- A1: domain 0 = star(0)×star(0) exactly (150 facets; A₅ is vertex
  transitive, so one representative covers all 6). Residual 450 facets, 4 free
  domains.
- A2: domain 0 = star(0)×star(0) minus one A₅-orbit of private cells (under
  the stabilizer D₁₀ of vertex 0, order 10). The runbook table gives the
  orbits; Mike names them.

**Family B: exhaustive split by facet ownership (complete).** Enumerate which
of the 10 triangle pairs (s, t) with s ∩ t = ∅ or |s ∩ t| = 1 have all six
staircase facets in domain 0. Quotient by A₅ acting on pairs. Each orbit
representative fixes a subset S ⊆ F_L into domain 0; the case solves for the
rest with x[f, 0] = 1 for f ∈ S and x[f, 0] free elsewhere. This is the
family that makes the split exhaustive; Family A is the fast path that may
already produce SAT or the bulk of UNSAT certificates.

Exhaustiveness argument to write into the results file: for any 5-cover,
domain 0 contains some set S of full pairs; its orbit representative is one
of the enumerated cases with the same residual size, so the case's CP-SAT
would have found that cover (up to symmetry). The argument is checked by a
script that verifies orbit representatives are pairwise non-equivalent and
that their union of orbits is the full set of subsets considered.

### Where the GPU works

- Enumeration and pruning of cases: the annealer kernel already computes
  `domain_of(chain)` and the H₁ absorption tests
  (`scripts/klein_h1_absorption.py`). A case whose residual is H₁-obstructed
  for every single-facet absorption can be discarded before CP-SAT, with the
  obstruction recorded as its certificate.
- Orbit computation: A₅ on 600 facets is tiny; pure Python is fine. The GPU
  matters only for the pruning tests over thousands of cases.

### Execution

Each case is one CP-SAT run through `scripts/sat_rung.sh` in hybrid mode with
domain 0 frozen (`--warm case.json --free 4`), L from the ladder
(4, 6, 8, 12), 4 workers, 8 GB, on the Ibero KFP pipeline as a
`dsl.ParallelFor` over the case list. Each case fits the namespace quota; the
pipeline runs 2 at a time.

Outcome table per case: status, L, wall, RSS, certificate or obstruction.
Global verdict: SAT in any case (audited) proves `SC_strict⁰(RP²) ≤ 4`;
INFEASIBLE in all cases at L proves no 5-cover with chains of at most L
links.

## Route 2: contiguity tables instead of enforced constraints

### Idea

The 630,000 `kLinear1` constraints of `P_{5,4}` are the `y[d,f,i,T]`
variables: for each domain, facet, middle link and triangle T, "both maps
send the facet's vertices into T". That encodes 1-contiguity indirectly.
Directly: for each facet f, the set of pairs of maps (a, b) restricted to
verts(f) that are 1-contiguous on f is a finite table. Precompute it once,
give CP-SAT `AddAllowedAssignments` on the tuple
`(m[d,i,v] for v in verts(f)) + (m[d,i+1,v] for v in verts(f))`, enforced by
x[f, d]. No y variables.

### Size

A staircase facet of RP²×RP² has 5 grid vertices; each map sends a vertex to
one of 6 values, so a pair of restricted maps has 6^10 = 60 M candidates per
facet before filtering. That is too many to tabulate naively, so the table is
built per facet by the GPU kernel (`maps_contiguous_on`, already vectorised
in the annealer) and then compressed: only tuples where each map is itself
simplicial on f survive (a few thousand per facet), and 600 facets share
tables under A₅ (10 orbits of facets, so 10 distinct tables). Expected: tens
of thousands of allowed tuples per table, 10 tables, one
`AddAllowedAssignments` per (d, f, i). Measured before committing to it
(task in the plan).

### Why it may win

Table constraints propagate with a dedicated algorithm and carry no
auxiliary Booleans, so the model loses its 630,000 enforced constraints and
most of the clause growth that came with them. It is the same problem with a
different encoding; verdicts remain sound. The two routes compose: the
orbit cases can use the table encoding.

## Controls (before any UNSAT is reported)

- Route 2 must reproduce SAT on every known cover: Moebius k=3 L=6, RP² k=6
  L=8, Klein k=10 L=8, each audited.
- Route 1 must find the known 6-cover when run with k=6 and the same split.
- A cross-check: one RP² case solved both with the current encoding and with
  tables must give the same verdict.

## Deliverables

- `opti/scripts/orbits/rp2_aut.py`: A₅ as permutations of the 6 vertices,
  induced action on grid vertices, facets and pairs; orbit representatives
  with a proof-of-exhaustiveness check.
- `opti/scripts/orbits/make_cases.py`: writes one warm file per case (domain
  0 as a frozen chain when it is a rectangle, or as a facet set when only
  ownership is fixed) plus `cases.json`.
- `scs_sat.py`: `--fix-domain0 facets.json` (x[f,0] = 1 for the listed
  facets) and `--tables` (route 2 encoding). Both opt-in; defaults unchanged.
- `opti/scripts/tables/build_tables.py`: GPU-built contiguity tables per
  facet orbit, cached as `.npz`, with a CPU fallback for small complexes.
- KFP pipeline variant `cp_sat_cases` in `opti/cluster/kfp/`: ParallelFor
  over `cases.json`.
- Results in `opti/experiments/rp2_klein_cp_sat_results.md`.

## Decisions for Mike

1. Family A2: which D₁₀ orbits of private cells to treat as cases (the runbook
   table lists residual sizes; the choice fixes how many cases A2 has).
2. Whether Family B splits by full pairs (as written) or by facet membership
   of a smaller generating set, which changes the case count.
3. Whether route 2 should replace `y` variables entirely or keep them for the
   end links (m₁ ⊆ σ(f), m_{L-1} ⊆ τ(f) are already cheap).

## Out of scope

- A GPU SAT solver.
- Klein (same machinery applies later; Aut is smaller and the case count
  larger).
- Level-1 subdivision.
