---
status: proposed
type: experiment
owner: ambos
repo: chain-first-annealer
related:
  - docs/superpowers/specs/2026-09-26-klein-sat-cover-design.md
  - opti/experiments/klein_anneal_log.md
  - opti/results/klein_k8_r1/summary.json
  - opti/paper-overleaf/paper/scs_sat.py
---

# Klein bottle cover by CP-SAT: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the CP-SAT tooling (env, witness converter, rung runner) and run the Klein ladder k = 10 to 5 on the AI lab machine, archiving every feasible result as an audited certificate.

**Architecture:** `scs_sat.py` stays unchanged. A converter turns its witness JSON into a certificate directory, restoring frozen chains from the warm file (hybrid mode truncates them). A shell runner executes one rung, measures time and RSS, converts, audits and appends a row to the log.

**Tech Stack:** Python 3.14, OR-Tools 9.15.6755 (CP-SAT), numpy/numba (for `gpu_sc`), uv, GNU time, bash.

**Spec:** `docs/superpowers/specs/2026-09-26-klein-sat-cover-design.md`

## Global Constraints

- All paths below are relative to `opti/` unless they start with `docs/`.
- `ortools==9.15.6755`; venv at `opti/.venv-sat` (git-ignored, see Task 1).
- `scs_sat.py` is not modified by this plan.
- A result counts only if `python3 src/independent_check.py <dir>` exits 0.
- INFEASIBLE at (k, L) is recorded as "no cover with chains of at most L links", never as a lower bound. UNKNOWN is recorded as UNKNOWN.
- Production rungs run on the AI lab machine: [LAB_HOST], [LAB_CORES] cores, [LAB_RAM_GB] GB RAM, launched via [LAB_LAUNCH]. Fill from the LAB_INVENTORY block before Task 5.
- The laptop only runs tests and 60 s smoke builds.
- No em-dash characters (U+2014) in new files.

## Review Focus

1. Hybrid witness with frozen chains longer than L: `scs_sat.py` copies only maps 0..L of each frozen chain (`src_i = min(i, chain_len)`), so a 41-map warm chain arrives truncated and not ending at pi2. Expected: the converter restores full warm chains and the certificate audits (Task 2, `test_hybrid_restores_truncated_frozen_chains`).
2. Witness that does not cover F_L or has a broken chain. Expected: exit code 2 and no certificate claimed (Task 2, `test_truncated_without_warm_is_rejected`).
3. Solver hits the time cap (UNKNOWN) or proves INFEASIBLE. Expected: a log row with that status and no converter call (Task 3, `status` branch; exercised by the INFEASIBLE smoke in Task 3 Step 4).
4. Model too large for the machine. Expected: the smoke build's RSS is logged and the rung is not launched on a machine whose RAM is below it (Task 4 gate).
5. Ties in warm domain sizes (klein_k8_r1 has two domains of 294). Expected: the converter freezes exactly the domains the solver froze, by reusing `scs_sat.load_warm` ordering (Task 2 uses `load_warm`, never its own sort).

---

### Task 1: SAT environment

**Files:**
- Create: `opti/requirements-sat.txt`
- Modify: `.gitignore` (append `opti/.venv-sat/`)

**Interfaces:**
- Produces: `opti/.venv-sat/bin/python` with ortools, numpy, numba, scipy.

- [ ] **Step 1: Verify the env is missing**

Run: `opti/.venv-sat/bin/python -c "import ortools"`
Expected: `No such file or directory`.

- [ ] **Step 2: Write requirements**

`opti/requirements-sat.txt`:

```
-r ../requirements.txt
ortools==9.15.6755
```

- [ ] **Step 3: Create the venv**

```bash
cd opti
uv venv .venv-sat --python 3.14
uv pip install --python .venv-sat/bin/python -r requirements-sat.txt
```

- [ ] **Step 4: Verify imports**

Run: `cd opti && .venv-sat/bin/python -c "import ortools, sys; sys.path.insert(0,'src'); import gpu_sc; print(ortools.__version__, gpu_sc.build('klein').nf)"`
Expected: `9.15.6755 1536`.

- [ ] **Step 5: Ignore the venv and commit**

```bash
echo 'opti/.venv-sat/' >> .gitignore
git add .gitignore opti/requirements-sat.txt
git commit -m "sat: pinned CP-SAT environment (ortools 9.15.6755)"
```

---

### Task 2: Witness to certificate converter

**Files:**
- Create: `opti/scripts/sat_witness_to_certificate.py`
- Test: `opti/src/test_sat_witness.py`

**Interfaces:**
- Consumes: `scs_sat.build_complex(name) -> (NV, NG, NF, FAC, TRIS, PI1, PI2, P)`, `scs_sat.load_warm(path, NV, NG, NF, FAC, TRIS) -> (raw_chains, raw_domains, covered, order)`, `gpu_sc.partition_from_chains(P, chains) -> parts | None`, `gpu_sc.write_certificate(P, parts, outdir, meta)`.
- Produces: CLI `sat_witness_to_certificate.py WITNESS OUTDIR --complex C --k K --L L [--warm W --free N] [--timeout T --status S --wall-s X]`; exit 0 audited, 2 not a certificate, 3 audit failed.

- [ ] **Step 1: Write the failing tests**

`opti/src/test_sat_witness.py`:

```python
"""Converter from scs_sat.py witnesses to audited certificates."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

OPTI = Path(__file__).resolve().parents[1]
CONV = OPTI / "scripts" / "sat_witness_to_certificate.py"
WARM = OPTI / "results" / "mobius_k5" / "summary.json"


def run_conv(args):
    return subprocess.run([sys.executable, str(CONV), *args],
                          capture_output=True, text=True)


def scs_order():
    sys.path.insert(0, str(OPTI / "paper-overleaf" / "paper"))
    import scs_sat
    NV, NG, NF, FAC, TRIS, _, _, _ = scs_sat.build_complex("mobius")
    return scs_sat.load_warm(str(WARM), NV, NG, NF, FAC, TRIS)


def truncated(chain, L):
    # what solve_hybrid puts in the witness for a frozen chain
    return [chain[min(i, len(chain) - 1)] for i in range(L + 1)]


class Converter(unittest.TestCase):
    def test_cold_witness_certifies(self):
        s = json.loads(WARM.read_text())
        with tempfile.TemporaryDirectory() as d:
            w = Path(d) / "w.json"
            w.write_text(json.dumps({"chains": s["best_chains"],
                                     "domains": s["domains"]}))
            r = run_conv([str(w), f"{d}/cert", "--complex", "mobius",
                          "--k", "5", "--L", "40"])
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertTrue((Path(d) / "cert" / "cover.txt").is_file())

    def test_hybrid_restores_truncated_frozen_chains(self):
        chains, _, _, order = scs_order()
        L = 4
        witness = [truncated(chains[i], L) for i in order[:4]] + [chains[order[4]]]
        with tempfile.TemporaryDirectory() as d:
            w = Path(d) / "w.json"
            w.write_text(json.dumps({"chains": witness, "domains": []}))
            r = run_conv([str(w), f"{d}/cert", "--complex", "mobius", "--k", "5",
                          "--L", str(L), "--warm", str(WARM), "--free", "1"])
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_truncated_without_warm_is_rejected(self):
        chains, _, _, order = scs_order()
        witness = [truncated(chains[i], 4) for i in order]
        with tempfile.TemporaryDirectory() as d:
            w = Path(d) / "w.json"
            w.write_text(json.dumps({"chains": witness, "domains": []}))
            r = run_conv([str(w), f"{d}/cert", "--complex", "mobius",
                          "--k", "5", "--L", "4"])
            self.assertEqual(r.returncode, 2, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd opti && .venv-sat/bin/python -m unittest src/test_sat_witness.py -v`
Expected: 3 failures, converter file not found (return code 2 from python for a missing script in the first two; the third may pass by accident, which is fine at this step).

- [ ] **Step 3: Write the converter**

`opti/scripts/sat_witness_to_certificate.py`:

```python
#!/usr/bin/env python3
"""Turn a scs_sat.py witness into an audited certificate directory.

Hybrid witnesses carry frozen chains cut to L + 1 maps; the full chains are
restored from the warm summary, in the order scs_sat.load_warm froze them.
Exit codes: 0 audited, 2 not a certificate, 3 independent audit failed.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

OPTI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OPTI / "src"))
sys.path.insert(0, str(OPTI / "paper-overleaf" / "paper"))
import gpu_sc  # noqa: E402
import scs_sat  # noqa: E402


def full_chains(witness, warm, k, n_free, dims):
    chains = [list(c) for c in witness["chains"]]
    if warm:
        raw_chains, raw_domains, _, order = scs_sat.load_warm(warm, *dims)
        n_frozen = min(k - n_free, len(raw_domains))
        chains[:n_frozen] = [raw_chains[i] for i in order[:n_frozen]]
    return [np.array(c, np.int8) for c in chains]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("witness")
    ap.add_argument("outdir")
    ap.add_argument("--complex", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--warm")
    ap.add_argument("--free", type=int, default=0)
    ap.add_argument("--timeout", type=int)
    ap.add_argument("--status")
    ap.add_argument("--wall-s", type=float)
    a = ap.parse_args(argv)

    NV, NG, NF, FAC, TRIS, _, _, P = scs_sat.build_complex(a.complex)
    witness = json.loads(Path(a.witness).read_text())
    chains = full_chains(witness, a.warm, a.k, a.free, (NV, NG, NF, FAC, TRIS))
    parts = gpu_sc.partition_from_chains(P, chains)
    if parts is None:
        print("witness chains do not cover F_L", file=sys.stderr)
        return 2
    import ortools
    meta = {"params": {
        "complex": a.complex, "method": "CP-SAT witness (scs_sat.py)",
        "ortools": ortools.__version__, "k": a.k, "L": a.L,
        "mode": "hybrid" if a.warm else "cold", "init": a.warm,
        "free": a.free, "timeout": a.timeout, "status": a.status,
        "wall_s": a.wall_s, "witness": str(a.witness)}}
    try:
        gpu_sc.write_certificate(P, parts, a.outdir, meta)
    except ValueError as e:
        print(f"certificate rejected: {e}", file=sys.stderr)
        return 2
    audit = subprocess.run([sys.executable, str(OPTI / "src" / "independent_check.py"),
                            a.outdir])
    return 0 if audit.returncode == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests**

Run: `cd opti && .venv-sat/bin/python -m unittest src/test_sat_witness.py -v`
Expected: `Ran 3 tests ... OK`.

- [ ] **Step 5: Run the existing contract tests**

Run: `cd opti/src && ../.venv-sat/bin/python -m unittest test_contracts.py`
Expected: `OK` (no regressions; the converter touches no existing file).

- [ ] **Step 6: Commit**

```bash
git add opti/scripts/sat_witness_to_certificate.py opti/src/test_sat_witness.py
git commit -m "sat: witness to audited certificate converter, restores frozen hybrid chains"
```

---

### Task 3: Rung runner

**Files:**
- Create: `opti/scripts/sat_rung.sh`
- Test: `opti/scripts/test_sat_rung.sh`

**Interfaces:**
- Consumes: Task 2 CLI and exit codes.
- Produces: `sat_rung.sh COMPLEX K L TIMEOUT WORKERS OUTDIR LOG_MD [WARM FREE]`; writes `OUTDIR/{solver.log,time.txt,scs_witness_k{K}_L{L}.json,certificate/}` and appends one row to `LOG_MD`. Always exits 0 after logging; nonzero only on its own usage errors.

- [ ] **Step 1: Write the failing test**

`opti/scripts/test_sat_rung.sh`:

```bash
#!/usr/bin/env bash
# Smoke test for sat_rung.sh on the Moebius band (fast).
set -euo pipefail
OPTI=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d)
bash "$OPTI/scripts/sat_rung.sh" mobius 5 4 120 4 "$tmp/hyb" "$tmp/log.md" \
     "$OPTI/results/mobius_k5/summary.json" 1
bash "$OPTI/scripts/sat_rung.sh" mobius 1 3 120 4 "$tmp/infeas" "$tmp/log.md"
cat "$tmp/log.md"
grep -qE '\| mobius \| 5 \| 4 \| hybrid \| 1 \| (OPTIMAL|FEASIBLE) \|.*\| audited \|' "$tmp/log.md"
grep -qE '\| mobius \| 1 \| 3 \| cold \| - \| INFEASIBLE \|.*\| - \|' "$tmp/log.md"
echo "SAT_RUNG OK"
```

(k = 1 on the Moebius band is infeasible because TC = 1 needs two domains.)

- [ ] **Step 2: Run it to see it fail**

Run: `bash opti/scripts/test_sat_rung.sh`
Expected: fails with `sat_rung.sh: No such file or directory`.

- [ ] **Step 3: Write the runner**

`opti/scripts/sat_rung.sh`:

```bash
#!/usr/bin/env bash
# One CP-SAT rung: solve, measure, convert, audit, log.
# Uso: sat_rung.sh COMPLEX K L TIMEOUT WORKERS OUTDIR LOG_MD [WARM FREE]
set -euo pipefail
[ $# -ge 7 ] || { echo "usage: $0 COMPLEX K L TIMEOUT WORKERS OUTDIR LOG_MD [WARM FREE]" >&2; exit 64; }
cx=$1 k=$2 L=$3 to=$4 wk=$5 out=$6 log=$7 warm=${8:-} free=${9:-0}
OPTI=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$OPTI/.venv-sat/bin/python}
mkdir -p "$out"
args=(--complex "$cx" --k "$k" --L "$L" --timeout "$to" --workers "$wk" --outdir "$out" --witness)
mode=cold; freecol=-
if [ -n "$warm" ]; then args+=(--warm "$warm" --free "$free"); mode=hybrid; freecol=$free; fi

/usr/bin/time -v -o "$out/time.txt" "$PY" -u "$OPTI/paper-overleaf/paper/scs_sat.py" "${args[@]}" \
    > "$out/solver.log" 2>&1 || true

line=$(grep -oE '(OPTIMAL|FEASIBLE|INFEASIBLE|UNKNOWN|MODEL_INVALID) in [0-9.]+s \(vars [0-9]+, cons [0-9]+\)' "$out/solver.log" | tail -1 || true)
status=$(echo "$line" | awk '{print $1}'); status=${status:-CRASH}
wall=$(echo "$line" | sed -nE 's/.* in ([0-9.]+)s.*/\1/p')
nvars=$(echo "$line" | sed -nE 's/.*vars ([0-9]+).*/\1/p')
ncons=$(echo "$line" | sed -nE 's/.*cons ([0-9]+).*/\1/p')
rss_kb=$(sed -nE 's/.*Maximum resident set size \(kbytes\): ([0-9]+).*/\1/p' "$out/time.txt")
rss_gb=$(awk -v kb="${rss_kb:-0}" 'BEGIN{printf "%.2f", kb/1048576}')

cert=-
if [ "$status" = OPTIMAL ] || [ "$status" = FEASIBLE ]; then
  conv=("$OPTI/scripts/sat_witness_to_certificate.py" "$out/scs_witness_k${k}_L${L}.json" "$out/certificate"
        --complex "$cx" --k "$k" --L "$L" --timeout "$to" --status "$status" --wall-s "${wall:-0}")
  [ -n "$warm" ] && conv+=(--warm "$warm" --free "$free")
  set +e; "$PY" "${conv[@]}" > "$out/convert.log" 2>&1; rc=$?; set -e
  case $rc in 0) cert=audited ;; 2) cert=not-a-cover ;; *) cert=audit-failed ;; esac
fi

if [ ! -f "$log" ]; then
  printf '| date | complex | k | L | mode | free | status | wall_s | vars | cons | rss_gb | certificate | outdir |\n|---|---|---:|---:|---|---:|---|---:|---:|---:|---:|---|---|\n' > "$log"
fi
printf '| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n' \
  "$(date -u +%Y-%m-%dT%H:%MZ)" "$cx" "$k" "$L" "$mode" "$freecol" "$status" \
  "${wall:--}" "${nvars:--}" "${ncons:--}" "$rss_gb" "$cert" "$out" >> "$log"
echo "$cx k=$k L=$L $mode: $status, certificate=$cert"
```

- [ ] **Step 4: Run the test**

Run: `bash opti/scripts/test_sat_rung.sh`
Expected: two log rows printed and `SAT_RUNG OK`. If the hybrid rung comes back INFEASIBLE at L = 4, raise L to 6 in the test and ledger it as a ruling.

- [ ] **Step 5: Commit**

```bash
chmod +x opti/scripts/sat_rung.sh opti/scripts/test_sat_rung.sh
git add opti/scripts/sat_rung.sh opti/scripts/test_sat_rung.sh
git commit -m "sat: rung runner with RSS, status log and audited conversion"
```

---

### Task 4: Laptop smoke builds for Klein

**Files:**
- Create: `opti/experiments/klein_sat_log.md` (written by the runner)

- [ ] **Step 1: Smoke rung 0 and the largest cold model**

```bash
cd opti
LOG=experiments/klein_sat_log.md
bash scripts/sat_rung.sh klein 10 6 60 8 results/klein_sat_smoke_k10_L6_hyb $LOG results/klein_k8_r1/summary.json 2
bash scripts/sat_rung.sh klein 5 10 60 8 results/klein_sat_smoke_k5_L10_cold $LOG
```

Expected: two rows. Each has a status (UNKNOWN is fine at 60 s), `vars`, `cons` and `rss_gb`. If the second one is killed for memory, its row shows CRASH; record that as the RSS gate for rung 4.

- [ ] **Step 2: Add the header note and commit the log**

Prepend to `opti/experiments/klein_sat_log.md`:

```markdown
# Klein bottle CP-SAT log

Spec: `docs/superpowers/specs/2026-09-26-klein-sat-cover-design.md`.
Rows are appended by `scripts/sat_rung.sh`. INFEASIBLE at (k, L) means no cover
whose chains have at most L links; it is not a lower bound. Smoke rows (60 s,
laptop) measure model size and RSS only.
```

```bash
git add opti/experiments/klein_sat_log.md
git commit -m "sat: Klein smoke builds, model size and RSS"
```

---

### Task 5: Rung 0 on the lab (k = 10, hybrid, 2 free)

Precondition: LAB_INVENTORY filled into Global Constraints; smoke RSS for k = 10 fits [LAB_RAM_GB].

- [ ] **Step 1: Run on the lab**

```bash
cd opti
for L in 6 8; do
  bash scripts/sat_rung.sh klein 10 $L 3600 [LAB_CORES] results/klein_sat_k10_L${L}_hyb \
       experiments/klein_sat_log.md results/klein_k8_r1/summary.json 2
done
```

Expected: two rows. Success = any row with `certificate = audited`, which proves SC_strict^0(K) <= 9.

- [ ] **Step 2: Commit results and log**

```bash
git add opti/experiments/klein_sat_log.md opti/results/klein_sat_k10_*
git commit -m "sat: Klein rung 0 (k=10 hybrid) results"
```

---

### Task 6: Rungs 1 and 2 on the lab (hybrid restructuring)

- [ ] **Step 1: Run**

```bash
cd opti
for L in 6 8; do
  bash scripts/sat_rung.sh klein 9 $L 7200 [LAB_CORES] results/klein_sat_k9_L${L}_hyb \
       experiments/klein_sat_log.md results/klein_k8_r1/summary.json 3
done
for L in 6 8; do
  bash scripts/sat_rung.sh klein 8 $L 7200 [LAB_CORES] results/klein_sat_k8_L${L}_hyb \
       experiments/klein_sat_log.md results/klein_k8_r1/summary.json 2
done
```

Expected: four rows. `audited` at k = 9 proves SC_strict^0 <= 8; at k = 8, <= 7.

- [ ] **Step 2: Commit**

```bash
git add opti/experiments/klein_sat_log.md opti/results/klein_sat_k9_* opti/results/klein_sat_k8_*
git commit -m "sat: Klein rungs 1-2 (hybrid k=9, k=8) results"
```

---

### Task 7: Rungs 3 and 4 on the lab (cold descent)

Precondition: smoke RSS for k = 5, L = 10 (Task 4) fits [LAB_RAM_GB].

- [ ] **Step 1: Run k = 7 and 6, then k = 5**

```bash
cd opti
for k in 7 6; do for L in 6 8 10; do
  bash scripts/sat_rung.sh klein $k $L 14400 [LAB_CORES] results/klein_sat_k${k}_L${L}_cold experiments/klein_sat_log.md
done; done
for L in 6 8 10; do
  bash scripts/sat_rung.sh klein 5 $L 28800 [LAB_CORES] results/klein_sat_k5_L${L}_cold experiments/klein_sat_log.md
done
```

Stop a k loop at its first `audited` row. Expected: rows for each run; `audited` at k = 5 proves SC_strict^0(K) = 4 = TC.

- [ ] **Step 2: Write the conclusion and close the plan**

Append a "Conclusion" section to `opti/experiments/klein_sat_log.md` with the best audited k (or none), and the list of INFEASIBLE (k, L) pairs as bounded statements. Set `status: done` in this plan's frontmatter.

```bash
git add opti/experiments/klein_sat_log.md opti/results/klein_sat_k[765]_* docs/superpowers/plans/2026-09-27-klein-sat-cover.md
git commit -m "sat: Klein cold descent results and conclusion"
```

---

## Note for Mike

`solve_hybrid` builds frozen domains from `chain[min(i, chain_len)]` for i in 0..L, so a frozen chain longer than L links is cut and its last map is not pi2. The solver is still sound (frozen x are constants), but the witness chains for frozen domains are not valid planners. The converter works around it; fixing it in `scs_sat.py` (pad or keep full frozen chains in the witness) is left to you.
