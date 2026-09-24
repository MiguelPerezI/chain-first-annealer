"""CP-SAT decision procedure: does K x K admit a cover by <= k
strictly-contiguous planner domains with chains of <= L links?

A domain is a chain phi_0 = pi1 -> ... -> phi_L = pi2 of maps
grid -> V(K); facet f is covered by the domain iff every link is
1-contiguous AND simplicial on f: img(phi_i) | img(phi_{i+1}) on f lies
in one triangle of K.

Shorter chains embed into L links by stuttering (consecutive equal maps
are allowed), so UNSAT at L covers every L' <= L; empty domains are
allowed, so k = 3 covers every k' <= 3.

Encoding:
  m[d][i][v]  IntVar 0..NV-1   value of phi_i(v), d in {0..k-1}, i in 0..L
  m[d][0][v] fixed = pi1(v) ;  m[d][L][v] fixed = pi2(v)
  x[f][d]     Bool              domain d covers facet f ;  sum_d x[f,d] >= 1
  y[d][f][i][T] Bool            pair (i,i+1) of domain d on facet f is good
                                inside triangle T ; sum_T y >= x[f,d]
  y => m[d][i][v] in T and m[d][i+1][v] in T for all grid vertices v of f
  pair (0,1): pi1's image on f IS sigma(f), so goodness forces
  img(phi_1) <= sigma(f):   x[f,d] => m[d][1][v] in sigma(f)   (no y needed)
  pair (L-1,L): symmetric:  x[f,d] => m[d][L-1][v] in tau(f)
  symmetry breaking: domain sizes sum_f x[f,0] >= sum_f x[f,1] >= ...

UNSAT  => no such cover exists (machine-verifiable).
SAT    => prints the domains + chains.

Usage:
  python scs_sat.py --complex mobius --k 3 --L 8 --timeout 600
  python scs_sat.py --complex rp2 --k 3 --L 8 --timeout 3600
  python scs_sat.py --complex torus --k 3 --L 10 --timeout 7200
"""
import argparse
import sys
import time
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "opti"))
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from opti.src import gpu_sc as _gs
    gpu_sc = _gs
except Exception:
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "..", "src"))
        import gpu_sc
    except Exception:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "src"))
        import gpu_sc

from ortools.sat.python import cp_model


def build_complex(name):
    P = gpu_sc.build(name)
    NV = P.nv
    NG = P.ngrid
    NF = P.nf
    FAC = [sorted(int(v) for v in P.facet_verts[i][:P.facet_nv[i]])
           for i in range(NF)]
    TRIS = [tuple(sorted(t)) for t in P.kfacets]
    PI1 = [v // NV for v in range(NG)]
    PI2 = [v % NV for v in range(NG)]
    return NV, NG, NF, FAC, TRIS, PI1, PI2, P.name


def solve(k, L, timeout_s, workers, NV, NG, NF, FAC, TRIS, PI1, PI2,
          want_solution=False, outdir=None):
    NT = len(TRIS)
    model = cp_model.CpModel()
    m = {}
    for d in range(k):
        for i in range(L + 1):
            for v in range(NG):
                if i == 0:
                    m[d, i, v] = model.NewConstant(PI1[v])
                elif i == L:
                    m[d, i, v] = model.NewConstant(PI2[v])
                else:
                    m[d, i, v] = model.NewIntVar(0, NV - 1, f"m{d}_{i}_{v}")
    x = {}
    for f in range(NF):
        for d in range(k):
            x[f, d] = model.NewBoolVar(f"x{f}_{d}")
        model.Add(sum(x[f, d] for d in range(k)) >= 1)
    for d in range(k):
        for f in range(NF):
            verts = FAC[f]
            sig = sorted({g // NV for g in FAC[f]})
            tau = sorted({g % NV for g in FAC[f]})
            for v in verts:
                model.AddLinearExpressionInDomain(
                    m[d, 1, v], cp_model.Domain.FromValues(sig)
                ).OnlyEnforceIf(x[f, d])
                model.AddLinearExpressionInDomain(
                    m[d, L - 1, v], cp_model.Domain.FromValues(tau)
                ).OnlyEnforceIf(x[f, d])
            for i in range(1, L - 1):
                ys = [model.NewBoolVar(f"y{d}_{f}_{i}_{t}") for t in range(NT)]
                model.Add(sum(ys) >= x[f, d])
                for t, T in enumerate(TRIS):
                    for v in verts:
                        model.AddLinearExpressionInDomain(
                            m[d, i, v], cp_model.Domain.FromValues(T)
                        ).OnlyEnforceIf(ys[t])
                        model.AddLinearExpressionInDomain(
                            m[d, i + 1, v], cp_model.Domain.FromValues(T)
                        ).OnlyEnforceIf(ys[t])
    for d in range(k - 1):
        model.Add(sum(x[f, d] for f in range(NF))
                  >= sum(x[f, d + 1] for f in range(NF)))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_s
    solver.parameters.num_workers = workers
    solver.parameters.log_search_progress = True
    t0 = time.time()
    status = solver.Solve(model)
    dt = time.time() - t0
    st = solver.StatusName(status)
    print(f"k={k} L={L}: {st} in {dt:.1f}s "
          f"(vars {len(model.proto.variables)}, cons {len(model.proto.constraints)})")
    if st in ("OPTIMAL", "FEASIBLE"):
        chains = []
        for d in range(k):
            ch = [[solver.Value(m[d, i, v]) for v in range(NG)]
                  for i in range(L + 1)]
            chains.append(ch)
        doms = [[f for f in range(NF) if solver.Value(x[f, d]) == 1]
                for d in range(k)]
        for d in range(k):
            print(f"  domain {d}: {len(doms[d])} facets")
        if want_solution and outdir:
            os.makedirs(outdir, exist_ok=True)
            import json
            path = os.path.join(outdir, f"scs_witness_k{k}_L{L}.json")
            json.dump({"chains": chains, "domains": doms}, open(path, "w"))
            print(f"  witness written to {path}")
        elif want_solution:
            import json
            path = f"/tmp/scs_witness_k{k}_L{L}.json"
            json.dump({"chains": chains, "domains": doms}, open(path, "w"))
            print(f"  witness written to {path}")
    return st, dt


if __name__ == "__main__":
    import multiprocessing

    parser = argparse.ArgumentParser(description="CP-SAT cover decision for K×K")
    parser.add_argument("--complex", default="mobius",
                        choices=["mobius", "rp2", "torus", "klein", "s1xw",
                                 "circle", "s2", "s3", "cube", "dodeca"],
                        help="base simplicial complex K")
    parser.add_argument("--k", type=int, required=True,
                        help="number of domains")
    parser.add_argument("--L", type=int, required=True,
                        help="chain length (number of links)")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="timeout in seconds (default 3600)")
    parser.add_argument("--workers", type=int, default=0,
                        help="CP-SAT workers (0 = auto)")
    parser.add_argument("--outdir", default=None,
                        help="directory for witness JSON output")
    parser.add_argument("--witness", action="store_true",
                        help="save witness JSON on SAT")
    args = parser.parse_args()

    workers = args.workers if args.workers > 0 else min(multiprocessing.cpu_count(), 16)
    NV, NG, NF, FAC, TRIS, PI1, PI2, cname = build_complex(args.complex)

    print(f"{cname}: {NF} facets, {NG} grid vertices, "
          f"{len(TRIS)} triangles, NV={NV}; workers={workers}")
    print(f"Trying k={args.k}, L={args.L}, timeout={args.timeout}s")

    solve(args.k, args.L, args.timeout, workers,
          NV, NG, NF, FAC, TRIS, PI1, PI2,
          want_solution=args.witness, outdir=args.outdir)
