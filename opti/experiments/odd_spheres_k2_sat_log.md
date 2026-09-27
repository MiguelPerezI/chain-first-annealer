# Odd spheres, k = 2, CP-SAT log (conjecture C1)

Spec: `docs/superpowers/specs/2026-09-26-odd-spheres-k2-sat-design.md`.
Rows are appended by `scripts/sat_rung.sh`. An audited row refutes C1 at that
sphere. INFEASIBLE at L means no 2-domain cover with chains of at most L links.

Smoke notes (laptop, 14 GB RAM, capped with `ulimit -v 5000000`, 60 s):

- S^3, L = 12: **INFEASIBLE**, proved by CP-SAT in 60.2 s (51,555 variables,
  724,501 constraints, peak RSS 2.73 GB). This is a solver proof, not a time
  cap. A shorter chain pads to L links by repeating pi1 (allowed by the model),
  so the statement covers every L <= 12: no 2-domain level-0 cover of the
  boundary of Delta^4 with chains of at most 12 links. Evidence for C1, not a
  proof; the CP-SAT infeasibility proof is not independently checked.
- S^5, L = 3: model built (197,771 variables, 4,383,541 constraints; 2.98 GB
  while building), then `std::bad_alloc` at solve start under the 5 GB cap.
- S^5, L = 4: model built (370,741 variables, 8,211,421 constraints; 3.36 GB
  while building), then `std::bad_alloc` at solve start. Rung B needs the lab.

| date | complex | k | L | mode | free | status | wall_s | vars | cons | rss_gb | certificate | outdir |
|---|---|---:|---:|---|---:|---|---:|---:|---:|---:|---|---|
| 2026-09-27T18:00Z | s3 | 2 | 12 | cold | - | INFEASIBLE | 60.2 | 51555 | 724501 | 2.73 | - | results/s3_k2_sat_smoke_L12 |
| 2026-09-27T18:01Z | s5 | 2 | 3 | cold | - | CRASH | - | - | - | 2.98 | - | results/s5_k2_sat_smoke_L3 |
| 2026-09-27T18:01Z | s5 | 2 | 4 | cold | - | CRASH | - | - | - | 3.36 | - | results/s5_k2_sat_smoke_L4 |
