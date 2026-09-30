#!/usr/bin/env python3
"""Prototype: cold P_{k,L} with a clause encoding of middle-link contiguity.

Replaces scs_sat.py's y[d,f,i,T] block (one Boolean per maximal simplex T and
ten enforced domain constraints each) by:
  b[d,i,v,c]      one-hot of m[d,i,v]
  h[d,f,i,c]      "colour c appears on facet f in map i or map i+1" (lit -> h)
  for every minimal non-face N of K and every (d,f,i):  OR(not x[f,d], not h[c] for c in N)
The implications only force h up; clauses only push h down, so the model is
equisatisfiable with the y encoding. End links (m1 in sigma(f), m_{L-1} in tau(f))
are identical to scs_sat.py. Prints the same status line format as scs_sat.py
so scripts/sat_rung.sh can parse it.

Usage: factored_model.py --complex rp2 --k 5 --L 4 --timeout 600 --workers 4 [--fix-domain0 F.json]
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

from ortools.sat.python import cp_model

OPTI = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OPTI / "paper-overleaf" / "paper"))
import scs_sat  # noqa: E402


def minimal_non_faces(NV, TRIS):
    faces = {frozenset(s) for T in TRIS for r in range(1, len(T) + 1)
             for s in itertools.combinations(T, r)}
    out = []
    for r in range(2, NV + 1):
        for s in itertools.combinations(range(NV), r):
            fs = frozenset(s)
            if fs in faces:
                continue
            if all(frozenset(t) in faces for t in itertools.combinations(s, r - 1)):
                out.append(s)
    return out


def build(k, L, NV, NG, NF, FAC, TRIS, PI1, PI2, fix_domain0=None):
    model = cp_model.CpModel()
    nonfaces = minimal_non_faces(NV, TRIS)
    m, b = {}, {}
    for d in range(k):
        for i in range(L + 1):
            for v in range(NG):
                if i == 0 or i == L:
                    c0 = PI1[v] if i == 0 else PI2[v]
                    m[d, i, v] = model.NewConstant(c0)
                    for c in range(NV):
                        b[d, i, v, c] = model.NewConstant(1 if c == c0 else 0)
                    continue
                m[d, i, v] = model.NewIntVar(0, NV - 1, f"m{d}_{i}_{v}")
                bs = [model.NewBoolVar(f"b{d}_{i}_{v}_{c}") for c in range(NV)]
                model.AddExactlyOne(bs)
                for c in range(NV):
                    b[d, i, v, c] = bs[c]
                    model.Add(m[d, i, v] == c).OnlyEnforceIf(bs[c])
    x = {}
    for f in range(NF):
        for d in range(k):
            x[f, d] = model.NewBoolVar(f"x{f}_{d}")
        model.AddBoolOr([x[f, d] for d in range(k)])
    for f in (fix_domain0 or []):
        model.Add(x[int(f), 0] == 1)
    for d in range(k):
        for f in range(NF):
            verts = FAC[f]
            sig = sorted({g // NV for g in verts})
            tau = sorted({g % NV for g in verts})
            for v in verts:
                model.AddLinearExpressionInDomain(
                    m[d, 1, v], cp_model.Domain.FromValues(sig)).OnlyEnforceIf(x[f, d])
                model.AddLinearExpressionInDomain(
                    m[d, L - 1, v], cp_model.Domain.FromValues(tau)).OnlyEnforceIf(x[f, d])
            for i in range(1, L - 1):
                h = [model.NewBoolVar(f"h{d}_{f}_{i}_{c}") for c in range(NV)]
                for c in range(NV):
                    for v in verts:
                        model.AddImplication(b[d, i, v, c], h[c])
                        model.AddImplication(b[d, i + 1, v, c], h[c])
                for N in nonfaces:
                    model.AddBoolOr([x[f, d].Not()] + [h[c].Not() for c in N])
    scs_sat.order_by_size(model, x, NF, range(1 if fix_domain0 else 0, k))
    return model, m, x, nonfaces


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--complex", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fix-domain0")
    a = ap.parse_args()
    NV, NG, NF, FAC, TRIS, PI1, PI2, P = scs_sat.build_complex(a.complex)
    fix = json.load(open(a.fix_domain0)) if a.fix_domain0 else None
    model, m, x, nonfaces = build(a.k, a.L, NV, NG, NF, FAC, TRIS, PI1, PI2, fix)
    nv, nc = len(model.proto.variables), len(model.proto.constraints)
    print(f"  minimal non-faces: {nonfaces}")
    print(f"  model built: {nv} vars, {nc} constraints", flush=True)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = a.timeout
    solver.parameters.num_workers = a.workers
    t0 = time.time()
    st = solver.StatusName(solver.Solve(model))
    print(f"k={a.k} L={a.L}: {st} in {time.time() - t0:.1f}s (vars {nv}, cons {nc})")


if __name__ == "__main__":
    main()
