# RP² k=5 and Klein k=9: CP-SAT results

Runbook: `opti/experiments/cluster_runbook_rp2_k5_klein_k9.md`. Rows from
`scripts/sat_rung.sh`; cluster runs come from `opti/cluster/kfp/`.

## 2026-09-27, laptop head start (no verdict)

Laptop: 14 GB RAM, 24 threads. Each run capped at 6 GB RAM with no swap
(systemd scope, `MemoryMax=6G`), 8 CP-SAT workers, one run at a time.
Artifacts: `opti/results/cp_sat_rp2_k5_laptop/`.

| run | model | outcome | wall | peak RAM |
|---|---|---|---|---|
| control rp2 k=6 L=8 | 221,118 vars, 2,218,205 cons | killed by OOM at the cap | 73 s | 6 GB |
| rp2 k=5 L=4 | 63,546 vars, 636,604 cons | killed by OOM at the cap, no solution and no bound reported | 4 h 35 min (36.7 CPU-hours) | 6 GB |
| rp2 k=5 L=5 | 93,726 vars, 939,604 cons | killed by OOM at the cap | 45 s | 6 GB |

These rows prove nothing about the covers. What they measure:

- RAM grows during the search, not only while building: L = 4 built in
  18 s and then grew for four and a half hours until it reached 6 GB.
- RAM grows with workers: the control (2.2 M constraints) passed 6 GB with
  8 workers, while the runbook measured about 2.2 GB for rp2 k=5 L=8
  (1.85 M constraints) with 4 workers.
- CP-SAT's symmetry detection found 1 generator on rp2 k=5 L=4
  (`[Symmetry] #generators: 1`). Aut(RP²) has order 60 (A₅, vertex
  transitive), so the solver is not using that symmetry. Adding explicit
  symmetry breaking to the model is the likeliest way to shorten an UNSAT
  proof.

Consequence for the cluster: do not size rungs by model size alone. The KFP
defaults (8 CPU, 28 GiB, 6 workers) leave room for growth during the search;
record peak RSS per rung and lower `--workers` before raising RAM.

## 2026-09-27, orbit decomposition and encoding measurements (laptop)

Plan: `docs/superpowers/plans/2026-09-27-rp2-orbit-decomposition.md`.

**Symmetry.** Aut(RP²) = A₅ (60 elements, triangle transitive), but only the
identity maps staircase facets to staircase facets: a staircase is monotone
in the vertex order. The diagonal action lives on triangle pairs (s, t): 100
pairs, 3 orbits (|s ∩ t| = 1, 2, 3; two distinct triangles always meet). An
orbit split therefore needs a transport argument between the staircase
triangulations of equivalent pairs (Mike, spec Decision 2). Case A1
(domain 0 = star(0)×star(0)) uses no symmetry and is ready.

**Encodings on RP² k=5 L=4** (cold, 10 min, 4 workers, 6 GB cap):

| encoding | vars | cons | status at 600 s | peak RAM |
|---|---:|---:|---|---:|
| `y[d,f,i,T]` (scs_sat.py) | 63,546 | 636,604 | UNKNOWN | 3.33 GB |
| factored clauses (`scripts/tables/factored_model.py`) | 42,786 | 454,384 | UNKNOWN | 1.62 GB |

Same verdicts on Moebius L=6: k=3 OPTIMAL (15.4 s vs 7.2 s), k=2 INFEASIBLE
(3.0 s vs 1.5 s). The factored encoding halves the RAM in the same window.

**A1 hybrid on RP² k=5 L=4:** 50,838 vars, 509,403 cons (UNKNOWN at 20 s,
smoke only). Launch on the lab with
`python opti/cluster/kfp/lanzar.py --warm-file results/rp2_cases/A1_star0.json --free 4`.

**Control with domain 0 fixed** (rp2 k=6 L=8, 132 facets fixed, 4 workers):
OOM at 6 GB on the laptop. Must run on the lab.

## 2026-09-28, factored encoding control on the laptop

`scs_sat.py --complex rp2 --k 6 --L 8 --encoding factored`, 4 workers, 6 GB
cap: **OPTIMAL in 53.8 s**, 143,790 vars, 1,559,189 cons, peak RSS 2.98 GB.
The same control with the `y` encoding was killed by OOM at 6 GB (8 and 4
workers). Certificate `results/cp_sat_rp2_factored/control_k6_L8/certificate`:
6 domains, audited by `independent_check.py` (RP² support added in this
change; before it the auditor rejected every 600-facet cover, which would
also have kept the KFP control gate closed for RP²).

## 2026-09-28 to 2026-10-04, KFP ladder on the lab (factored, cold)

Revision `e3c25d8` (L=8, 12, 16: `eecb9b9`, see `revision.txt`), pod limit 7 CPU / 26 GiB, 5 CP-SAT workers, 24 h limit
per rung (`nproc.txt` says 32: that is the node, not the pod limit). Raw output in
`results/cp_sat_rp2_kfp/` (copied from `~/cp_sat_results`).
Encoding: `factored` for every rung (`SCS_ENCODING=factored` in the compiled
pipeline; each `solver.log` prints `middle-link encoding: factored`). `row.md`
and the certificate params do not record the encoding.

| rung | vars | cons | status | wall | peak RSS | certificate |
|---|---:|---:|---|---:|---:|---|
| control rp2 k=6 L=8 | 143,790 | 1,559,189 | OPTIMAL | 39.8 s | 3.14 GB | audited |
| rp2 k=5 L=4 | 42,786 | 454,384 | UNKNOWN (24 h limit) | 86,403 s | 3.61 GB | - |
| rp2 k=5 L=5 | 62,046 | 665,644 | UNKNOWN (24 h limit) | 86,403 s | 5.37 GB | - |
| rp2 k=5 L=6 | 81,306 | 876,904 | UNKNOWN (24 h limit) | 86,403 s | 7.38 GB | - |
| rp2 k=5 L=8 | 119,826 | 1,299,424 | UNKNOWN (24 h limit) | 86,405 s | 8.71 GB | - |
| rp2 k=5 L=12 | 196,866 | 2,144,464 | UNKNOWN (24 h limit) | 86,409 s | 11.14 GB | - |
| rp2 k=5 L=16 | 273,906 | 2,989,504 | UNKNOWN (24 h limit) | 86,408 s | 12.97 GB | - |

The factored encoding removes the OOM: every k=5 rung ran the full 24 h
well under the RAM limit (peak 12.97 GB at L=16). It does not settle them:
no cover and no infeasibility proof in 24 h at any L in 4..16 (L=6: 70.8 M
conflicts, 299 M branches). Larger L does not help the cold search. No
UNKNOWN row is evidence for or against a 5-cover of RP².
