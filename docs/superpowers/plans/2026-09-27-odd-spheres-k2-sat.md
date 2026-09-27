---
status: proposed
type: experiment
owner: ambos
repo: chain-first-annealer
related:
  - docs/superpowers/specs/2026-09-26-odd-spheres-k2-sat-design.md
  - docs/superpowers/plans/2026-09-27-klein-sat-cover.md
  - opti/REVIEW_NOTES.md
  - opti/experiments/s7_chain_first_wall.md
---

# Two-domain covers of odd spheres by CP-SAT: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test conjecture C1 with CP-SAT at k = 2 on S^3, then S^5, then S^7 only if measured model growth fits the lab machine.

**Architecture:** Reuses the Klein plan's SAT env, converter and `sat_rung.sh`. The only code change is exposing `s4` to `s7` in `scs_sat.py --complex`. Every rung appends to one log, which ends with a measured scaling table and a written decision on S^7.

**Tech Stack:** Python 3.14, OR-Tools 9.15.6755, bash, GNU time.

**Spec:** `docs/superpowers/specs/2026-09-26-odd-spheres-k2-sat-design.md`

## Global Constraints

- Precondition: Tasks 1 to 3 of `2026-09-27-klein-sat-cover.md` are complete (`opti/.venv-sat`, `scripts/sat_witness_to_certificate.py`, `scripts/sat_rung.sh`).
- Paths relative to `opti/` unless they start with `docs/`.
- k = 2 in every rung. A FEASIBLE, audited result refutes C1 at that sphere (SC_strict^0 = 1 = TC).
- INFEASIBLE at L means no 2-domain cover with chains of at most L links; it is evidence, not proof.
- Production rungs run on the AI lab machine: [LAB_HOST], [LAB_CORES] cores, [LAB_RAM_GB] GB RAM, via [LAB_LAUNCH]. The laptop only runs tests and 60 s smoke builds.
- No em-dash characters (U+2014) in new files.

## Review Focus

1. A sphere name the solver does not accept. Expected: `--complex s5` builds instead of failing argparse (Task 1 test).
2. A smoke build killed for memory. Expected: a CRASH row with RSS, and the rung is not launched on the lab until RSS fits (Task 2 gate).
3. Stopping too early or too late in L. Expected: each sphere's loop stops at the first `audited` row and otherwise runs every listed L (Tasks 3 and 4 loops).
4. Reading INFEASIBLE as a proof of C1. Expected: the conclusion states bounded results only (Task 5 text).
5. Launching S^7 on a guess. Expected: rung C runs only if the written extrapolation in Task 5 fits [LAB_RAM_GB] (Task 6 precondition).

---

### Task 1: Expose s4 to s7 in scs_sat.py

**Files:**
- Modify: `opti/paper-overleaf/paper/scs_sat.py:374-376` (the `--complex` choices)

**Interfaces:**
- Produces: `scs_sat.py --complex {…, s4, s5, s6, s7}`.

- [ ] **Step 1: Run the failing command**

Run: `cd opti && .venv-sat/bin/python paper-overleaf/paper/scs_sat.py --complex s4 --k 2 --L 3 --timeout 5 --workers 2`
Expected: `error: argument --complex: invalid choice: 's4'`.

- [ ] **Step 2: Add the choices**

In the `--complex` argument, append `"s4", "s5", "s6", "s7"` to the existing `choices` list. `gpu_sc.build` already builds them; nothing else changes.

- [ ] **Step 3: Run the command again**

Run: `cd opti && .venv-sat/bin/python paper-overleaf/paper/scs_sat.py --complex s4 --k 2 --L 3 --timeout 5 --workers 2`
Expected: a final line `k=2 L=3: <STATUS> in <t>s (vars <n>, cons <m>)`.

- [ ] **Step 4: Run the Klein plan tests for regressions**

Run: `cd opti && .venv-sat/bin/python -m unittest src/test_sat_witness.py && bash scripts/test_sat_rung.sh`
Expected: `OK` and `SAT_RUNG OK`.

- [ ] **Step 5: Commit**

```bash
git add opti/paper-overleaf/paper/scs_sat.py
git commit -m "sat: expose s4-s7 in scs_sat.py --complex"
```

---

### Task 2: Laptop smoke builds

**Files:**
- Create: `opti/experiments/odd_spheres_k2_sat_log.md` (rows written by the runner)

- [ ] **Step 1: Smoke S^3 at the largest L and S^5 at the two smallest**

```bash
cd opti
LOG=experiments/odd_spheres_k2_sat_log.md
bash scripts/sat_rung.sh s3 2 12 60 8 results/s3_k2_sat_smoke_L12 $LOG
bash scripts/sat_rung.sh s5 2 3 60 8 results/s5_k2_sat_smoke_L3 $LOG
bash scripts/sat_rung.sh s5 2 4 60 8 results/s5_k2_sat_smoke_L4 $LOG
```

Expected: three rows with `vars`, `cons`, `rss_gb`. A CRASH row for S^5 means its RSS exceeds the laptop; rung B then needs the lab and the smoke is repeated there with the same command.

- [ ] **Step 2: Header and commit**

Prepend to the log:

```markdown
# Odd spheres, k = 2, CP-SAT log (conjecture C1)

Spec: `docs/superpowers/specs/2026-09-26-odd-spheres-k2-sat-design.md`.
Rows are appended by `scripts/sat_rung.sh`. An audited row refutes C1 at that
sphere. INFEASIBLE at L means no 2-domain cover with chains of at most L links.
```

```bash
git add opti/experiments/odd_spheres_k2_sat_log.md
git commit -m "sat: odd-sphere smoke builds, model size and RSS"
```

---

### Task 3: Rung A, S^3 on the lab

- [ ] **Step 1: Run L upward, stop at the first audited row**

```bash
cd opti
LOG=experiments/odd_spheres_k2_sat_log.md
for L in 3 4 6 8 10 12; do
  bash scripts/sat_rung.sh s3 2 $L 7200 [LAB_CORES] results/s3_k2_sat_L$L $LOG
  tail -1 $LOG | grep -q '| audited |' && break
done
```

Expected: up to six rows. `audited` refutes C1 for S^3.

- [ ] **Step 2: Commit**

```bash
git add opti/experiments/odd_spheres_k2_sat_log.md opti/results/s3_k2_sat_L*
git commit -m "sat: S^3 k=2 rung A results"
```

---

### Task 4: Rung B, S^5 on the lab

Precondition: the S^5 smoke RSS (Task 2, laptop or lab) fits [LAB_RAM_GB].

- [ ] **Step 1: Run**

```bash
cd opti
LOG=experiments/odd_spheres_k2_sat_log.md
for L in 3 4 6 8; do
  bash scripts/sat_rung.sh s5 2 $L 28800 [LAB_CORES] results/s5_k2_sat_L$L $LOG
  tail -1 $LOG | grep -q '| audited |' && break
done
```

- [ ] **Step 2: Commit**

```bash
git add opti/experiments/odd_spheres_k2_sat_log.md opti/results/s5_k2_sat_L*
git commit -m "sat: S^5 k=2 rung B results"
```

---

### Task 5: Scaling table and decision on S^7

- [ ] **Step 1: Append the measured table**

Append to the log a table with one row per (sphere, L) taken from the S^3 and S^5 rows: `vars`, `cons`, `rss_gb`, `wall_s`, status.

- [ ] **Step 2: Extrapolate S^7 and decide**

For L = 3 and 4, estimate S^7 constraints with the spec's construction (y = 2 * 277,992 * (L - 2) * 9, fifteen vertices per product facet, two constraints per vertex), giving about 150 M constraints at L = 3 and about 300 M at L = 4. Scale the measured S^5 RSS at the same L by the constraint ratio. Write one of:

- "Rung C runs: extrapolated RSS X GB fits [LAB_RAM_GB]", or
- "Rung C ruled out: extrapolated RSS X GB exceeds [LAB_RAM_GB]; S^7 needs a different formulation (separate spec)."

- [ ] **Step 3: Commit**

```bash
git add opti/experiments/odd_spheres_k2_sat_log.md
git commit -m "sat: odd-sphere scaling table and S^7 decision"
```

---

### Task 6: Rung C, S^7 on the lab (only if Task 5 says it runs)

- [ ] **Step 1: Run**

```bash
cd opti
LOG=experiments/odd_spheres_k2_sat_log.md
for L in 3 4; do
  bash scripts/sat_rung.sh s7 2 $L 86400 [LAB_CORES] results/s7_k2_sat_L$L $LOG
  tail -1 $LOG | grep -q '| audited |' && break
done
```

- [ ] **Step 2: Conclusion and close**

Append a "Conclusion" section: which spheres were refuted (audited rows), which carry only bounded INFEASIBLE results and up to which L, and the S^7 decision. Set `status: done` in this plan's frontmatter.

```bash
git add opti/experiments/odd_spheres_k2_sat_log.md opti/results/s7_k2_sat_L* docs/superpowers/plans/2026-09-27-odd-spheres-k2-sat.md
git commit -m "sat: odd spheres k=2 conclusion"
```
