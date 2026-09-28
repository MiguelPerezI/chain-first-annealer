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
