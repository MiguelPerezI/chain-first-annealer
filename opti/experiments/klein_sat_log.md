# Klein bottle CP-SAT log

Spec: `docs/superpowers/specs/2026-09-26-klein-sat-cover-design.md`.
Rows are appended by `scripts/sat_rung.sh`. INFEASIBLE at (k, L) means no cover
whose chains have at most L links; it is not a lower bound. Smoke rows (60 s,
laptop) measure model size and RSS only.

Smoke notes (laptop, 14 GB RAM, capped with `ulimit -v 5000000`):

- k = 10 hybrid from `klein_k8_r1`, 2 free, L = 6: 200,328 variables,
  2,010,625 constraints, peak RSS 2.25 GB; UNKNOWN at the 60 s cap, as expected.
- k = 5 cold, L = 10: the model built (993,608 variables, 9,970,180
  constraints; peak RSS 2.96 GB while building) and the solver then failed with
  `std::bad_alloc` under the 5 GB virtual-memory cap. Rung 4 at L = 10 needs
  the lab machine; its RAM must be measured there before the 8 h run.

| date | complex | k | L | mode | free | status | wall_s | vars | cons | rss_gb | certificate | outdir |
|---|---|---:|---:|---|---:|---|---:|---:|---:|---:|---|---|
| 2026-09-27T17:57Z | klein | 10 | 6 | hybrid | 2 | UNKNOWN | 60.7 | 200328 | 2010625 | 2.25 | - | results/klein_sat_smoke_k10_L6_hyb |
| 2026-09-27T17:57Z | klein | 5 | 10 | cold | - | CRASH | - | - | - | 2.96 | - | results/klein_sat_smoke_k5_L10_cold |
