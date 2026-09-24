"""CP-SAT decision procedure: does K x K (K = minimal 5-vertex Moebius band)
admit a cover by <= k strictly-contiguous planner domains with chains of
<= L links?  A domain is a chain phi_0 = pi1 -> ... -> phi_L = pi2 of maps
grid -> V(K); facet f is covered by the domain iff every link is
1-contiguous AND simplicial on f: img(phi_i) | img(phi_{i+1}) on f lies in
one triangle of K.

Shorter chains embed into L links by stuttering (consecutive equal maps are
allowed), so UNSAT at L covers every L' <= L; empty domains are allowed, so
k = 3 covers every k' <= 3.

Encoding:
  m[d][i][v]  IntVar 0..4     value of phi_i(v), d in {0..k-1}, i in 0..L
  m[d][0][v] fixed = pi1(v) = v // 5 ;  m[d][L][v] fixed = pi2(v) = v % 5
  x[f][d]     Bool            domain d covers facet f ;  sum_d x[f,d] >= 1
  y[d][f][i][T] Bool          pair (i,i+1) of domain d on facet f is good
                              inside triangle T ; sum_T y >= x[f,d]
  y => m[d][i][v] in T and m[d][i+1][v] in T for all 5 grid vertices v of f
  pair (0,1): pi1's image on f IS the triangle sigma(f), so goodness forces
  img(phi_1) <= sigma(f):   x[f,d] => m[d][1][v] in sigma(f)   (no y needed)
  pair (L-1,L): symmetric:  x[f,d] => m[d][L-1][v] in tau(f)
  symmetry breaking: domain sizes sum_f x[f,0] >= sum_f x[f,1] >= ...

UNSAT  => no such cover exists (machine-verifiable).
SAT    => prints the domains + chains.
"""
import sys
import time

sys.path.insert(0, "/home/ubuntu/Documents/chain-first-annealer/opti")
from ortools.sat.python import cp_model
import gpu_sc

P = gpu_sc.build("mobius")
NV, NG, NF = P.nv, P.ngrid, P.nf
FAC = [sorted(int(v) for v in P.facet_verts[i][:P.facet_nv[i]])
       for i in range(NF)]
TRIS = [(0, 1, 2), (1, 2, 3), (2, 3, 4), (0, 3, 4), (0, 1, 4)]
PI1 = [v // NV for v in range(NG)]
PI2 = [v % NV for v in range(NG)]


def solve(k, L, timeout_s, workers, want_solution=False):
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
            sig = sorted({g // NV for g in FAC[f]})   # pi1-image triangle
            tau = sorted({g % NV for g in FAC[f]})   # pi2-image triangle
            # pair (0,1): pi1's image on f IS sigma(f), so goodness forces
            # img(phi_1) <= sigma(f)  (no y needed)
            for v in verts:
                model.AddLinearExpressionInDomain(
                    m[d, 1, v], cp_model.Domain.FromValues(sig)
                ).OnlyEnforceIf(x[f, d])
                # pair (L-1, L): symmetric
                model.AddLinearExpressionInDomain(
                    m[d, L - 1, v], cp_model.Domain.FromValues(tau)
                ).OnlyEnforceIf(x[f, d])
            for i in range(1, L - 1):
                ys = [model.NewBoolVar(f"y{d}_{f}_{i}_{t}") for t in range(5)]
                model.Add(sum(ys) >= x[f, d])
                for t, T in enumerate(TRIS):
                    for v in verts:
                        model.AddLinearExpressionInDomain(
                            m[d, i, v], cp_model.Domain.FromValues(T)
                        ).OnlyEnforceIf(ys[t])
                        model.AddLinearExpressionInDomain(
                            m[d, i + 1, v], cp_model.Domain.FromValues(T)
                        ).OnlyEnforceIf(ys[t])
    # symmetry breaking: domains ordered by coverage size
    for d in range(k - 1):
        model.Add(sum(x[f, d] for f in range(NF))
                  >= sum(x[f, d + 1] for f in range(NF)))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_s
    solver.parameters.num_workers = workers
    solver.parameters.log_search_progress = False
    t0 = time.time()
    status = solver.Solve(model)
    dt = time.time() - t0
    st = solver.StatusName(status)
    print(f"k={k} L={L}: {st} in {dt:.1f}s "
          f"(vars {len(model.proto.variables)}, cons {len(model.proto.constraints)})")
    if st == "OPTIMAL" or st == "FEASIBLE":
        chains = []
        for d in range(k):
            ch = [[solver.Value(m[d, i, v]) for v in range(NG)]
                  for i in range(L + 1)]
            chains.append(ch)
        doms = [[f for f in range(NF) if solver.Value(x[f, d]) == 1]
                for d in range(k)]
        for d in range(k):
            print(f"  domain {d}: {len(doms[d])} facets")
        if want_solution:
            import json
            json.dump({"chains": chains, "domains": doms},
                      open(f"/tmp/scs_witness_k{k}_L{L}.json", "w"))
            print(f"  witness written to /tmp/scs_witness_k{k}_L{L}.json")
    return st, dt


if __name__ == "__main__":
    import multiprocessing
    workers = min(multiprocessing.cpu_count(), 16)
    print(f"Moebius K x K: {NF} facets, {NG} grid vertices, "
          f"tris {TRIS}; workers={workers}")
    for k, L, tmo in [(4, 6, 300), (3, 6, 300), (3, 8, 600)]:
        solve(k, L, tmo, workers, want_solution=(k == 4))