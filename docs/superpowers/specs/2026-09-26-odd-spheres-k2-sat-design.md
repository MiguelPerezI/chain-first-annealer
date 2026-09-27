# Two-domain covers of odd spheres by CP-SAT (S^3, S^5, S^7)

Date: 2026-09-26. Status: approved approach, spec under review.

## Question

For the odd spheres S^n = boundary of Delta^{n+1} (n = 3, 5, 7), TC(S^n) = 1, so
a level-0 cover needs at least 2 domains. The explicit three-star construction
gives 3 domains (SC_strict^0 <= 2) at every n. Does a 2-domain cover exist?

This is conjecture C1 (`opti/REVIEW_NOTES.md`): "two domains do **not**
suffice if SC_strict(boundary of Delta^4) = 2". Evidence so far is only
search failure:

- S^3, k = 2: pinned at exactly 460/500 through 69 epochs and 5.65 x 10^9
  moves; 460 equals the 2-rectangle optimum (`rect2_optimum.py`).
- S^5, k = 2: the same gap reported one dimension up
  (`klein_anneal_log.md` section 5 table).
- S^7: the chain-first annealer cannot launch (per-thread state 764 KB against
  a 512 KB CUDA limit, `s7_chain_first_wall.md`).

A FEASIBLE, audited 2-domain cover at any n refutes C1 at that n and is sharp
(SC_strict^0 = 1 = TC).

## Why a ladder and not S^7 directly

Model size for `scs_sat.py solve_cold` at k = 2, L = 8, using its own
construction (y variables = k * |F| * (L - 2) * |facets of K|, two enforced
constraints per y per vertex of a product facet):

| sphere | complex | product facets | facets of K | vertices per product facet | y vars | enforced constraints |
|---|---|---:|---:|---:|---:|---:|
| S^3 | boundary Delta^4 | 500 | 5 | 7 | 30,000 | about 0.4 M |
| S^5 | boundary Delta^6 | 12,348 | 7 | 11 | about 1.0 M | about 23 M |
| S^7 | boundary Delta^8 | 277,992 | 9 | 15 | about 30 M | about 900 M |

CP-SAT runs on CPU and RAM; it does not use the GPU. Production runs execute
on the AI lab machine ([cores], [RAM], [OS], [how jobs are launched]: to be
filled in); the laptop (14 GB RAM, 24 cores) is used only for smoke tests.

S^3 is small. S^5 needs lab-scale RAM. S^7 is roughly 900 times S^3 in
constraints and may not fit even there. The ladder measures instead of
assuming: each rung records real model size and peak RSS, and the next rung
runs only if the measured growth fits the lab RAM.

## Approach

### Tooling

1. Shared with the Klein plan: `opti/requirements-sat.txt`
   (`ortools==9.15.6755`), venv `opti/.venv-sat`, and
   `opti/scripts/sat_witness_to_certificate.py` with audit by
   `opti/src/independent_check.py`.
2. `scs_sat.py`: add `s4`, `s5`, `s6`, `s7` to `--complex` choices.
   `gpu_sc.build` already supports them. No change to the model.

### Ladder

| rung | sphere | k | L | cap per run | gate to run |
|---|---|---:|---|---|---|
| A | S^3 | 2 | 3, 4, 6, 8, 10, 12 | 2 h | none |
| B | S^5 | 2 | 3, 4, 6, 8 | 8 h | the smoke-test RSS fits the lab RAM |
| C | S^7 | 2 | 3, 4 | 24 h | rung B built and solved at L = 4, and RSS extrapolated from A and B fits the lab RAM |

Rung A runs L upward and stops at the first FEASIBLE. L = 3 is the smallest
L whose model has a middle link (`scs_sat.py` adds y variables for links
1 to L - 2); the three-star construction uses chains of 2 links.

### Recording

- Results: `opti/results/s{n}_k2_sat_L{L}/` with witness, solver log and, when
  feasible, the audited certificate.
- Log: `opti/experiments/odd_spheres_k2_sat_log.md`, one row per run: sphere,
  L, status, wall time, variables, constraints, peak RSS.
- The log ends with a measured scaling table (A to B) and a written decision
  on rung C.
- One experiment plan per rung in `docs/superpowers/plans/`, with the template
  frontmatter.

## Interpretation rules

- FEASIBLE plus a clean audit refutes C1 at that n: SC_strict^0(S^n) = 1.
- INFEASIBLE at (k = 2, L) only rules out 2-domain covers with chains of at
  most L links. It is recorded as that bounded statement. A run of INFEASIBLE
  results over growing L is evidence for C1, not a proof.
- UNKNOWN (time cap) says nothing.

## Out of scope

- A new encoding or a domain-level formulation for S^7 (a later spec if rung C
  is ruled out).
- Even spheres (already sharp at k = 3).
- Changes to the annealer kernel.
