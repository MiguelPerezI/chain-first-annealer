"""C2: compression bound for s3 -- min links on grown domains, fast checks.

chainable(D, L) = one SAT call at exactly L (padding argument covers all
shorter chains).  Growth: add a random facet while chainable(D, 24).
Measure exact min-L only on grown domains, by scanning L from 2 with a
small timeout and escalating.
"""
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import gpu_sc
from ortools.sat.python import cp_model

P = gpu_sc.build("s3")
NV, NG, NF = P.nv, P.ngrid, P.nf
FAC = [list(map(int, f)) for f in P.facets]
TRIS = [tuple(sorted(t)) for t in P.kfacets]
NT = len(TRIS)

def chainable(D, L, timeout=60):
    model = cp_model.CpModel()
    m = {}
    for i in range(L + 1):
        for v in range(NG):
            if i == 0:
                m[i, v] = model.NewConstant(int(P.pi1[v]))
            elif i == L:
                m[i, v] = model.NewConstant(int(P.pi2[v]))
            else:
                m[i, v] = model.NewIntVar(0, NV - 1, "")
    for f in D:
        verts = FAC[f]
        sig = {g // NV for g in verts}
        tau = {g % NV for g in verts}
        for v in verts:
            model.AddLinearExpressionInDomain(m[1, v], cp_model.Domain.FromValues(sig))
            model.AddLinearExpressionInDomain(m[L - 1, v], cp_model.Domain.FromValues(tau))
        for i in range(1, L - 1):
            ys = [model.NewBoolVar("") for _ in range(NT)]
            model.Add(sum(ys) >= 1)
            for t, T in enumerate(TRIS):
                for v in verts:
                    model.AddLinearExpressionInDomain(
                        m[i, v], cp_model.Domain.FromValues(T)).OnlyEnforceIf(ys[t])
                    model.AddLinearExpressionInDomain(
                        m[i + 1, v], cp_model.Domain.FromValues(T)).OnlyEnforceIf(ys[t])
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    solver.parameters.num_workers = 2
    st = solver.StatusName(solver.Solve(model))
    return st in ("OPTIMAL", "FEASIBLE")

def min_links(D, hi=24, timeout=60):
    for L in range(2, hi + 1):
        if chainable(D, L, timeout):
            return L
    return None

if __name__ == "__main__":
    rng = np.random.default_rng(11)
    s = json.load(open("results/s3_starcover/summary.json"))
    seeds = [set(int(x) for x in d) for d in s["domains"]]
    s2 = json.load(open("results/s3_k3_v2/summary.json"))
    seeds += [set(int(x) for x in d) for d in s2["domains"]]

    best = max(seeds, key=len)
    D = set(best)
    L0 = min_links(D)
    print(f"grow start: |D|={len(D)} min-L={L0}", flush=True)
    peak = L0
    rejected = 0
    for step in range(400):
        f = int(rng.integers(0, NF))
        if f in D:
            continue
        trial = D | {f}
        if chainable(trial, 24, timeout=45):
            D = trial
            if step % 20 == 0:
                r = min_links(D)
                if isinstance(r, int) and isinstance(peak, int) and r > peak:
                    peak = r
                print(f"  step {step}: |D|={len(D)} min-L={r} peak={peak}", flush=True)
        else:
            rejected += 1
    r = min_links(D)
    print(f"GROWN |D|={len(D)} min-L={r} peak={peak} rejected={rejected}", flush=True)

    # also probe near-full random subsets: how large can chainable domains get?
    for j in range(6):
        size = 400 + j * 20
        S = set(int(x) for x in rng.choice(NF, size=size, replace=False))
        ok24 = chainable(S, 24, timeout=90)
        r = min_links(S) if ok24 else None
        print(f"random |D|={size}: chainable(24)={ok24} min-L={r}", flush=True)
