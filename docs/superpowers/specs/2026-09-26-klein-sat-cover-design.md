# Klein bottle cover by CP-SAT

Date: 2026-09-26. Status: approved approach, spec under review.

## Question

Does the vertex-minimal Klein bottle K (8 vertices, 16 triangles, 1536
staircase facets of K x K, grid 64) admit a level-0 cover with fewer domains
than the best known one?

What is known (`opti/experiments/klein_anneal_log.md`):

- TC(K) = 4, so every cover has at least 5 domains.
- The best product cover has rho = 12 rectangles, so SC_strict^0 <= 11.
- The annealer never covered: best 1483/1536 at k = 8 (`results/klein_k8_r1`),
  1519/1536 at k = 9 (`results/klein_k9`).
- Section 3 proves the k = 8 plateau cannot be finished by growing its 8
  domains: extension-only completion needs at least two more chains (k >= 10).
- Section 6 names SAT/ILP as the Klein analog of the method that produced the
  Moebius 3-cover.

## What exists

- `opti/paper-overleaf/paper/scs_sat.py`: CP-SAT model P_{k,L} (k domains,
  chains of L links, maps m[d,i,v], membership x[f,d], per-link triangle choice
  y[d,f,i,T]). `--complex klein` is accepted. Modes: cold (`solve_cold`) and
  hybrid (`--warm summary.json --free n`: freezes the k - n largest warm
  domains as constants and solves the rest).
- Output: `scs_witness_k{k}_L{L}.json` with `{chains, domains}`; domains may
  overlap, chains are not reduced.
- `opti/src/gpu_sc.py`: `partition_from_chains`, `reduce_chain`,
  `verify_system`, `write_certificate`.
- `opti/src/independent_check.py`: audits Klein certificates (1536 facets).

Missing:

- OR-Tools is not installed anywhere and no requirements file lists it.
- No converter from a witness to `cover.txt` / `planners.txt`.
- In hybrid mode the frozen chains are not part of the solver variables, so a
  converter must merge them back from the warm file.

## Approach

### Tooling

1. `opti/requirements-sat.txt` pinning `ortools==9.15.6755` (the version the
   Moebius certificate cites, 9.15; wheels exist for CPython 3.14). Installed
   into a dedicated venv, `opti/.venv-sat`, with `uv`.
2. `opti/scripts/sat_witness_to_certificate.py`: reads a witness and, for
   hybrid runs, the warm summary and the frozen domain ids; rebuilds the full
   chain list; runs `reduce_chain`, `partition_from_chains`, `verify_system`;
   writes a certificate directory with `write_certificate`; records
   provenance (`method`, `k`, `L`, `mode`, `warm`, `free`, `timeout`,
   `status`, `wall_s`, `ortools_version`) in `summary.json`.
3. Every certificate is accepted only after
   `python3 src/independent_check.py <dir>` exits 0.

### Ladder

Each rung has a fixed time cap and runs only if the previous rung finished.

| rung | mode | k | free | L | cap | success adds |
|---|---|---:|---:|---|---|---|
| 0 | hybrid from `klein_k8_r1` | 10 | 2 | 6, 8 | 1 h each | SC_strict^0 <= 9 |
| 1 | hybrid from `klein_k8_r1` | 9 | 3 | 6, 8 | 2 h each | SC_strict^0 <= 8 |
| 2 | hybrid from `klein_k8_r1` | 8 | 2 | 6, 8 | 2 h each | SC_strict^0 <= 7 |
| 3 | cold | 7, 6 | - | 6, 8, 10 | 4 h each | SC_strict^0 <= 6, 5 |
| 4 | cold | 5 | - | 6, 8, 10 | 8 h each | SC_strict^0 = 4 = TC (sharp) |

Rung 0 is the cheapest possible positive result: it keeps all eight plateau
domains and asks for the two extra chains the log's theorem says are
necessary. Rungs 1 and 2 restructure, which is the move class the log says the
annealer lacks.

### Resource limits

This laptop has 14 GB RAM (about 6 GB free), 24 cores and an RTX 5080 (unused
by CP-SAT). The cold model at k = 5, L = 8 has about 737,000 y variables and
about 7.4 million enforced domain constraints (estimate: k * 1536 * (L - 2) *
16 y variables, ten constraints each). Before each cold rung the model is built
once with a 60 s solve to measure variables, constraints and peak RSS. If peak
RSS exceeds 5 GB, that rung moves to the RTX PRO 6000 server used for the S^6
campaign.

### Recording

- Results: `opti/results/klein_sat_k{k}_L{L}_{cold|hyb}/` with witness,
  solver log and, when feasible, the audited certificate.
- Log: `opti/experiments/klein_sat_log.md`, one row per run: rung, k, L, mode,
  status (FEASIBLE, OPTIMAL, INFEASIBLE, UNKNOWN), wall time, model size, RSS.
- One experiment plan per rung group in `docs/superpowers/plans/`, with the
  template frontmatter.

## Interpretation rules

- FEASIBLE plus a clean audit is a theorem: SC_strict^0(K) <= k - 1.
- INFEASIBLE at (k, L) only rules out covers whose chains have at most L links.
  It is recorded as that bounded statement and never as a lower bound on
  SC_strict^0.
- UNKNOWN (time cap) says nothing and is recorded as such.

## Out of scope

- Adding Klein or Moebius support to the pinned auditor in
  `simplicial-complexity-certificates` (that belongs to the S^8 validation
  plan).
- Level-1 subdivision.
- Changes to the annealer.
