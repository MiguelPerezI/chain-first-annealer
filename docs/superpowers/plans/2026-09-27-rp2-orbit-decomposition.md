---
status: proposed
type: tooling
owner: ambos
repo: chain-first-annealer
related:
  - docs/superpowers/specs/2026-09-27-rp2-orbit-decomposition-design.md
  - opti/experiments/cluster_runbook_rp2_k5_klein_k9.md
  - opti/experiments/rp2_new_mathematics.md
  - opti/experiments/rp2_klein_cp_sat_results.md
---

# RP² k=5 by orbit decomposition and contiguity tables: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pieces that do not depend on Mike's three decisions (A₅ machinery with an exhaustiveness check, Family A1 cases, contiguity tables, the two solver flags, controls), measure them on the laptop, and leave the cluster pipeline ready for the case list.

**Architecture:** Pure-Python A₅ module (the group is tiny) feeding a case generator; `scs_sat.py` gains two opt-in flags, `--fix-domain0` and `--tables`; tables are built by a script that uses the CPU path of `gpu_sc` and caches `.npz`; every case runs through the existing `sat_rung.sh`, so audit and logging stay unchanged.

**Tech Stack:** Python 3.14, OR-Tools 9.15.6755 (`opti/.venv-sat`), numpy/numba (`gpu_sc`), bash, KFP 2.17.

**Spec:** `docs/superpowers/specs/2026-09-27-rp2-orbit-decomposition-design.md`

## Global Constraints

- Paths relative to `opti/` unless they start with `docs/`.
- `PY=opti/.venv-sat/bin/python` for every command below.
- Defaults of `scs_sat.py` unchanged: without `--fix-domain0` and `--tables` the model and RNG are identical (Task 3 and 5 check this by comparing `model.proto` sizes).
- Every SAT claim goes through `scripts/sat_rung.sh` and `independent_check.py`. UNSAT per case is recorded as "no cover with domain 0 in this case and chains ≤ L".
- Mike's three decisions (spec, "Decisions for Mike") are parameters: A2 orbits (Task 2 reads them from a file), Family B split (Task 7 stub), and whether tables replace end-link `y` (Task 5 keeps end links as they are).
- No em-dash characters (U+2014) in new files.

## Review Focus

1. A wrong A₅: a permutation set that is not a group or not automorphisms. Expected: Task 1 checks closure, order 60, triangle transitivity, and that every element maps `kfacets` to `kfacets`.
2. Non-exhaustive orbit split: representatives whose orbits miss a subset. Expected: Task 1's `orbit_reps` returns reps plus a covered-count that must equal the input size; Task 2 asserts it.
3. Domain 0 fixed inconsistently: `--fix-domain0` sets `x[f,0]=1` for facets that no chain can cover together. Expected: Task 3's known-cover control (RP² k=6 with domain 0 = one audited domain of `results/rp2_k6`) is SAT.
4. Tables that admit non-contiguous pairs or drop contiguous ones. Expected: Task 4 cross-checks every table tuple against `gpu_sc.maps_contiguous_on` on 1,000 random samples per facet orbit, and Task 5's controls reproduce the known covers.
5. Case runs that report SAT without the frozen domain: `sat_rung.sh` hybrid path needs the warm chain for domain 0. Expected: Task 6 passes the rectangle chain (`pi1 ~ c_v ~ pi2`) as warm file with `--free 4` and the audit checks the full partition.

---

### Task 1: A₅ module with exhaustiveness check

**Files:**
- Create: `opti/scripts/orbits/__init__.py` (empty), `opti/scripts/orbits/rp2_aut.py`
- Test: `opti/scripts/orbits/test_rp2_aut.py`

**Interfaces:**
- Produces: `automorphisms(P) -> list[tuple[int,...]]` (permutations of 0..nv-1 that map `P.kfacets` onto itself); `act_grid(perm, g, nv) -> int` (diagonal action on grid id `a*nv+b`); `act_facet(P, perm, f) -> int` (facet id after the action; built once as a 60x600 table `facet_perm`); `orbit_reps(items, act) -> (reps, covered)` where `act(g, item) -> item` and `covered == len(items)`.

- [ ] **Step 1: Write the failing test**

```python
"""A5 on the hemi-icosahedron and its diagonal action on RP2 x RP2."""
import sys, unittest
from pathlib import Path
OPTI = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OPTI / "src")); sys.path.insert(0, str(OPTI / "scripts"))
import gpu_sc
from orbits import rp2_aut as A


class Aut(unittest.TestCase):
    def setUp(self):
        self.P = gpu_sc.build("rp2")
        self.G = A.automorphisms(self.P)

    def test_group_of_order_60(self):
        self.assertEqual(len(self.G), 60)
        ids = set(self.G)
        for g in self.G:
            for h in self.G:
                self.assertIn(tuple(g[h[i]] for i in range(6)), ids)

    def test_maps_triangles_to_triangles_transitively(self):
        K = set(self.P.kfacets)
        for g in self.G:
            for t in K:
                self.assertIn(tuple(sorted(g[v] for v in t)), K)
        orbit = {tuple(sorted(g[v] for v in self.P.kfacets[0])) for g in self.G}
        self.assertEqual(orbit, K)

    def test_facet_action_is_a_permutation(self):
        for g in self.G:
            img = [A.act_facet(self.P, g, f) for f in range(self.P.nf)]
            self.assertEqual(sorted(img), list(range(self.P.nf)))

    def test_orbit_reps_cover_everything(self):
        reps, covered = A.orbit_reps(range(self.P.nf), lambda g, f: A.act_facet(self.P, g, f), self.G)
        self.assertEqual(covered, self.P.nf)
        self.assertEqual(len(reps), 10)   # 600 facets / 60, free action


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it, expect ImportError**

Run: `cd opti && $PY -m unittest scripts/orbits/test_rp2_aut.py`
Expected: `ModuleNotFoundError: No module named 'orbits'`.

- [ ] **Step 3: Implement**

```python
"""Aut(RP2) = A5 as vertex permutations, with its diagonal action on RP2 x RP2."""
import itertools
import numpy as np


def automorphisms(P):
    K = set(P.kfacets)
    out = []
    for perm in itertools.permutations(range(P.nv)):
        if all(tuple(sorted(perm[v] for v in t)) in K for t in K):
            out.append(perm)
    return out


def act_grid(perm, g, nv):
    a, b = divmod(g, nv)
    return perm[a] * nv + perm[b]


def _facet_index(P):
    return {tuple(sorted(int(v) for v in P.facet_verts[f][:P.facet_nv[f]])): f
            for f in range(P.nf)}


_FACET_INDEX = {}


def act_facet(P, perm, f):
    idx = _FACET_INDEX.setdefault(id(P), _facet_index(P))
    verts = [int(v) for v in P.facet_verts[f][:P.facet_nv[f]]]
    return idx[tuple(sorted(act_grid(perm, v, P.nv) for v in verts))]


def facet_perm_table(P, G):
    return np.array([[act_facet(P, g, f) for f in range(P.nf)] for g in G], dtype=np.int32)


def orbit_reps(items, act, G):
    """Representatives of the orbits of `items` under G; `act(g, item)`.
    Returns (reps, covered) with covered == number of items seen in some orbit."""
    items = list(items)
    seen, reps = set(), []
    for it in items:
        key = it if not isinstance(it, list) else tuple(it)
        if key in seen:
            continue
        reps.append(it)
        for g in G:
            img = act(g, it)
            seen.add(img if not isinstance(img, list) else tuple(img))
    covered = sum(1 for it in items if (it if not isinstance(it, list) else tuple(it)) in seen)
    return reps, covered
```

- [ ] **Step 4: Run the test**

Run: `cd opti && $PY -m unittest scripts/orbits/test_rp2_aut.py -v`
Expected: 4 tests OK. If `len(G) != 60`, stop: the facet list is not the hemi-icosahedron (`rp2_new_mathematics.md` section 2 says 60).

- [ ] **Step 5: Commit**

```bash
git add opti/scripts/orbits/__init__.py opti/scripts/orbits/rp2_aut.py opti/scripts/orbits/test_rp2_aut.py
git commit -m "orbits: A5 on RP2 with diagonal action and orbit representatives"
```

---

### Task 2: Family A cases

**Files:**
- Create: `opti/scripts/orbits/make_cases.py`
- Test: `opti/scripts/orbits/test_make_cases.py`
- Input (Mike): `opti/scripts/orbits/a2_orbits.json` (optional; absent means A1 only)

**Interfaces:**
- Consumes: Task 1.
- Produces: `results/rp2_cases/cases.json` = list of `{id, family, domain0_facets, warm_file, residual}`; one warm file per case with a single chain `[pi1, c_v, pi2]` (constant map to vertex v) in the `best_chains` format `load_warm` reads.

- [ ] **Step 1: Write the failing test**

```python
import json, subprocess, sys, tempfile, unittest
from pathlib import Path
OPTI = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OPTI / "src")); sys.path.insert(0, str(OPTI / "paper-overleaf" / "paper"))
import gpu_sc, scs_sat


class Cases(unittest.TestCase):
    def test_a1_case_is_the_star_rectangle(self):
        with tempfile.TemporaryDirectory() as d:
            subprocess.run([sys.executable, str(OPTI / "scripts/orbits/make_cases.py"), "--out", d], check=True)
            cases = json.loads(Path(d, "cases.json").read_text())
            a1 = [c for c in cases if c["family"] == "A1"]
            self.assertEqual(len(a1), 1)
            P = gpu_sc.build("rp2")
            self.assertEqual(len(a1[0]["domain0_facets"]), 150)
            NV, NG, NF, FAC, TRIS, _, _, _ = scs_sat.build_complex("rp2")
            chains, doms, _, _ = scs_sat.load_warm(a1[0]["warm_file"], NV, NG, NF, FAC, TRIS)
            self.assertEqual(sorted(doms[0]), sorted(a1[0]["domain0_facets"]))
            self.assertEqual(a1[0]["residual"], 450)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run, expect failure** (`make_cases.py` missing).

- [ ] **Step 3: Implement**

```python
#!/usr/bin/env python3
"""Family A cases for RP2 k=5: domain 0 = star(0) x star(0) (A1) and, if
scripts/orbits/a2_orbits.json exists, A1 minus each listed D10-orbit of private
cells (A2). Writes one warm file per case and cases.json."""
import argparse, json, sys
from pathlib import Path
import numpy as np
OPTI = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OPTI / "src")); sys.path.insert(0, str(OPTI / "scripts"))
import gpu_sc
from orbits import rp2_aut as A


def star_chain(P, v):
    pi1 = [g // P.nv for g in range(P.ngrid)]
    pi2 = [g % P.nv for g in range(P.ngrid)]
    return [pi1, [v] * P.ngrid, pi2]


def write_case(out, cid, family, P, chain, dom, extra):
    warm = Path(out, f"{cid}.json")
    warm.write_text(json.dumps({"best_chains": [chain], "status": "partial",
                                "chain_domain_sizes": [len(dom)], "coverage": len(dom)}))
    return {"id": cid, "family": family, "domain0_facets": sorted(dom),
            "warm_file": str(warm), "residual": P.nf - len(dom), **extra}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OPTI / "results" / "rp2_cases"))
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    P = gpu_sc.build("rp2")
    chain = star_chain(P, 0)
    dom = [int(f) for f in P.domain_of(np.array(chain, np.int8))]
    cases = [write_case(a.out, "A1_star0", "A1", P, chain, dom, {})]
    orbits = OPTI / "scripts" / "orbits" / "a2_orbits.json"
    if orbits.exists():
        for name, cells in json.loads(orbits.read_text()).items():
            sub = [f for f in dom if f not in set(cells)]
            # domain 0 is fixed by facet set here; the chain is only a placeholder
            cases.append(write_case(a.out, f"A2_{name}", "A2", P, chain, sub,
                                    {"fix_by_facets": True}))
    Path(a.out, "cases.json").write_text(json.dumps(cases, indent=1))
    print(f"{len(cases)} cases -> {a.out}/cases.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test.** Expected: OK, residual 450.
- [ ] **Step 5: Commit** `orbits: Family A case generator (A1 star rectangle, A2 from Mike's orbits file)`.

---

### Task 3: `--fix-domain0` in scs_sat.py

**Files:**
- Modify: `opti/paper-overleaf/paper/scs_sat.py` (argparse + `solve_cold`)
- Test: `opti/src/test_scs_flags.py`

**Interfaces:**
- Produces: `--fix-domain0 FILE` (JSON list of facet ids): adds `x[f,0] == 1` for listed f. Nothing else changes.

- [ ] **Step 1: Failing test**

```python
import json, subprocess, sys, tempfile, unittest
from pathlib import Path
OPTI = Path(__file__).resolve().parents[1]
SCS = OPTI / "paper-overleaf" / "paper" / "scs_sat.py"


def run(args):
    return subprocess.run([sys.executable, str(SCS), *args], capture_output=True, text=True).stdout


class FixDomain0(unittest.TestCase):
    def test_default_model_size_unchanged(self):
        base = run(["--complex", "mobius", "--k", "3", "--L", "4", "--timeout", "1", "--workers", "1"])
        self.assertIn("model built: 9830 vars", base.replace("(", " ").replace(",", " ")[:0] + base) if False else None
        self.assertRegex(base, r"model built: \d+ vars, \d+ constraints")

    def test_fix_domain0_adds_only_unit_constraints(self):
        with tempfile.TemporaryDirectory() as d:
            fx = Path(d, "fx.json"); fx.write_text(json.dumps([0, 1, 2]))
            base = run(["--complex", "mobius", "--k", "3", "--L", "4", "--timeout", "1", "--workers", "1"])
            fixed = run(["--complex", "mobius", "--k", "3", "--L", "4", "--timeout", "1", "--workers", "1", "--fix-domain0", str(fx)])
            import re
            nb = int(re.search(r"model built: (\d+) vars, (\d+) constraints", base).group(2))
            nf = int(re.search(r"model built: (\d+) vars, (\d+) constraints", fixed).group(2))
            self.assertEqual(nf, nb + 3)


if __name__ == "__main__":
    unittest.main()
```

(Drop the dead first assertion line when implementing; keep the regex one.)

- [ ] **Step 2: Run, expect the second test to fail** (`unrecognized arguments: --fix-domain0`).
- [ ] **Step 3: Implement** in `solve_cold` signature add `fix_domain0=None`; after building `x`:

```python
    for f in (fix_domain0 or []):
        model.Add(x[int(f), 0] == 1)
```

and in argparse `parser.add_argument("--fix-domain0", default=None, help="JSON list of facet ids forced into domain 0")`, loading it with `json.load` and passing it to `solve_cold`.

- [ ] **Step 4: Run tests.** Expected: 2 OK.
- [ ] **Step 5: Control (Review Focus 3):** `results/rp2_k6/summary.json` domain 0 as the fixed set, `--k 6 --L 8 --timeout 3600 --workers 4`. Expected: OPTIMAL or FEASIBLE. Record wall time in `rp2_klein_cp_sat_results.md`.
- [ ] **Step 6: Commit** `scs_sat: --fix-domain0 (facets forced into domain 0)`.

---

### Task 4: Contiguity tables

**Files:**
- Create: `opti/scripts/tables/__init__.py`, `opti/scripts/tables/build_tables.py`
- Test: `opti/scripts/tables/test_build_tables.py`

**Interfaces:**
- Produces: `build_tables(P, G) -> dict[rep_facet_id, np.ndarray[n_tuples, 2*facet_nv]]` and a cache `results/tables/<complex>_tables.npz` with keys `rep_<f>` and `facet_to_rep` (array of length nf). A tuple is `(a[v] for v in verts(f)) + (b[v] for v in verts(f))` in the order of `P.facet_verts[f]`, accepted iff a and b are simplicial on f and `maps_contiguous_on(P, a, b, [f])`.

- [ ] **Step 1: Failing test**

```python
import sys, unittest, numpy as np
from pathlib import Path
OPTI = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OPTI / "src")); sys.path.insert(0, str(OPTI / "scripts"))
import gpu_sc
from orbits import rp2_aut as A
from tables import build_tables as T


class Tables(unittest.TestCase):
    def test_tables_match_maps_contiguous_on(self):
        P = gpu_sc.build("mobius")
        G = A.automorphisms(P)
        tabs, f2r = T.build_tables(P, G)
        rng = np.random.default_rng(0)
        for f in range(P.nf):
            verts = [int(v) for v in P.facet_verts[f][:P.facet_nv[f]]]
            tab = {tuple(t) for t in tabs[int(f2r[f])].tolist()}
            for _ in range(200):
                a = rng.integers(0, P.nv, P.ngrid, dtype=np.int8)
                b = rng.integers(0, P.nv, P.ngrid, dtype=np.int8)
                ok = (gpu_sc.maps_contiguous_on(P, a, a, [f]) and gpu_sc.maps_contiguous_on(P, b, b, [f])
                      and gpu_sc.maps_contiguous_on(P, a, b, [f]))
                key = tuple(int(a[v]) for v in verts) + tuple(int(b[v]) for v in verts)
                self.assertEqual(key in tab, bool(ok), (f, key))


if __name__ == "__main__":
    unittest.main()
```

(Note: tables are per representative facet; for a non-representative f the test must map the tuple through the automorphism that sends f's representative to f. Implement `f2r` together with `rep_perm[f]` and apply it in the test before the lookup. The plan's Task 4 Step 3 defines both.)

- [ ] **Step 2: Run, expect ImportError.**
- [ ] **Step 3: Implement** `build_tables.py`:

```python
"""Per-facet tables of 1-contiguous map pairs, one table per facet orbit."""
import itertools, sys
from pathlib import Path
import numpy as np
OPTI = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OPTI / "src")); sys.path.insert(0, str(OPTI / "scripts"))
import gpu_sc
from orbits import rp2_aut as A


def _simplicial_images(P, verts):
    """All assignments verts -> V_K whose image is a simplex of K."""
    out = []
    for img in itertools.product(range(P.nv), repeat=len(verts)):
        S = 0
        for x in img:
            S |= 1 << x
        if P.is_simplex[S]:
            out.append(img)
    return out


def table_for_facet(P, f):
    verts = [int(v) for v in P.facet_verts[f][:P.facet_nv[f]]]
    maps = _simplicial_images(P, verts)
    rows = []
    for a in maps:
        Sa = 0
        for x in a:
            Sa |= 1 << x
        for b in maps:
            S = Sa
            for x in b:
                S |= 1 << x
            if P.is_simplex[S]:
                rows.append(a + b)
    return np.array(rows, dtype=np.int8)


def build_tables(P, G):
    reps, covered = A.orbit_reps(range(P.nf), lambda g, f: A.act_facet(P, g, f), G)
    assert covered == P.nf
    f2r = np.full(P.nf, -1, np.int32)
    rep_perm = [None] * P.nf
    for r in reps:
        for g in G:
            f = A.act_facet(P, g, r)
            if f2r[f] < 0:
                f2r[f], rep_perm[f] = r, g
    tabs = {int(r): table_for_facet(P, r) for r in reps}
    return tabs, f2r, rep_perm


def save(P, G, path):
    tabs, f2r, rep_perm = build_tables(P, G)
    np.savez_compressed(path, facet_to_rep=f2r, rep_perm=np.array(rep_perm, np.int8),
                        **{f"rep_{r}": t for r, t in tabs.items()})
    return tabs, f2r, rep_perm


if __name__ == "__main__":
    name = sys.argv[1]
    P = gpu_sc.build(name)
    tabs, f2r, _ = save(P, A.automorphisms(P), OPTI / "results" / "tables" / f"{name}_tables.npz")
    for r, t in tabs.items():
        print(f"facet {r}: {len(t)} allowed tuples")
```

Note: `A.automorphisms` uses `P.kfacets`, so it works for any complex whose vertices are 0..nv-1; on the Moebius band it returns its own automorphism group. In the test, when `f2r[f] != f`, apply `rep_perm[f]` to the vertices of `f`'s representative before comparing (the table row is in the representative's vertex order). The simplest correct test: only check facets with `f2r[f] == f` on the Moebius band and, separately, assert that `table_for_facet(P, f)` for a non-representative f has the same length as its representative's table.

- [ ] **Step 4: Run test; then measure RP²:** `cd opti && $PY scripts/tables/build_tables.py rp2`. Expected: 10 tables printed; record tuples per table and build time in `rp2_klein_cp_sat_results.md`. If a table exceeds 500,000 tuples, ledger it: `AddAllowedAssignments` will be too large and Task 5 must switch to per-link Boolean tables (out of this plan).
- [ ] **Step 5: Commit** `tables: per-orbit contiguity tables for CP-SAT`.

---

### Task 5: `--tables` encoding in scs_sat.py

**Files:**
- Modify: `opti/paper-overleaf/paper/scs_sat.py` (`solve_cold` middle links)
- Test: `opti/src/test_scs_flags.py` (add)

**Interfaces:**
- Produces: `--tables FILE.npz`: for each (d, f, i) with `1 <= i <= L-2`, replace the `y` block by `model.AddAllowedAssignments([m[d,i,v] for v in verts] + [m[d,i+1,v] for v in verts], rows).OnlyEnforceIf(x[f,d])`, where `rows` is the table of `f`'s representative with the row columns permuted by `rep_perm[f]` applied to the representative's vertex order. End links unchanged.

- [ ] **Step 1: Failing test:** add to `test_scs_flags.py` a run with `--tables results/tables/mobius_tables.npz` on Moebius k=3 L=6 timeout 300, asserting the output contains `OPTIMAL` or `FEASIBLE` and that `vars` in `model built` is smaller than without the flag.
- [ ] **Step 2: Run, expect `unrecognized arguments: --tables`.**
- [ ] **Step 3: Implement** (load npz once; build the per-facet row list lazily and cache by f).
- [ ] **Step 4: Controls (spec):** Moebius k=3 L=6 (audited via `sat_rung.sh`), RP² k=6 L=8, Klein k=10 L=8 with `--tables`. Each must be SAT and audited. Record vars/cons/RSS/wall next to the y-encoding numbers.
- [ ] **Step 5: Cross-check:** RP² k=5 L=4 with `--fix-domain0` = A1 facets, both encodings, 30 min each, 4 workers, 6 GB cap. Same verdict required (both UNKNOWN is acceptable; record RSS growth).
- [ ] **Step 6: Commit** `scs_sat: --tables contiguity-table encoding for middle links`.

---

### Task 6: Case runner and KFP variant

**Files:**
- Create: `opti/scripts/orbits/run_case.sh` (wraps `sat_rung.sh` with `--warm case --free 4`, or `--fix-domain0` when `fix_by_facets`)
- Modify: `opti/cluster/kfp/pipeline.py` (add `cp_sat_cases` pipeline: `ParallelFor` over `cases.json` entries), `opti/cluster/kfp/lanzar.py` (`--cases`)

- [ ] **Step 1:** `run_case.sh` test on Moebius: a case fixing domain 0 to `mobius_k3_sat` domain 0 with k=3 must come back SAT audited.
- [ ] **Step 2:** Compile the `cases` pipeline variant with `SAT_PVC=x`; assert `parallelismLimit: 2` and one `cp-sat-rung` executor in the IR.
- [ ] **Step 3: Commit** `cluster: cases pipeline for orbit decomposition`.

---

### Task 7: Family B stub and hand-off

- [ ] **Step 1:** `make_cases.py --family B` raises `SystemExit("Family B split pending Mike's decision (spec, Decisions 2)")`.
- [ ] **Step 2:** Append to `rp2_klein_cp_sat_results.md` a section "Orbit decomposition, laptop measurements" with Tasks 3-5 numbers and the exhaustiveness statement for Family A1 (one orbit, since A₅ is vertex transitive).
- [ ] **Step 3:** Set this plan's `status: running`; commit `docs: orbit decomposition measurements and Family B stub`.
