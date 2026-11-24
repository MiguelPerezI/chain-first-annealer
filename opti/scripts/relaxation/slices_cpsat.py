"""Cross-check of 'NOT k-colourable' claims for the rigid-slice-triangle relaxation with OR-tools CP-SAT
(independent of the pysat solvers used to produce them)."""
import sys, time
from ortools.sat.python import cp_model
import slices

def check(name, k, tl=600, workers=2):
    H, empties, cyc = slices.slice_family(name)
    nf, ne = H.nf, len(H.E)
    M = cp_model.CpModel()
    x = [[M.NewBoolVar(f"x{f}_{d}") for d in range(k)] for f in range(nf)]
    p = [[M.NewBoolVar(f"p{e}_{d}") for d in range(k)] for e in range(ne)]
    for f in range(nf):
        M.AddBoolOr(x[f])
        for e in H.fedges[f]:
            for d in range(k): M.AddImplication(x[f][d], p[e][d])
    for cy in cyc:
        for d in range(k): M.AddBoolOr([p[e][d].Not() for e in cy])
    S = cp_model.CpSolver(); S.parameters.max_time_in_seconds = tl; S.parameters.num_workers = workers
    t0 = time.time(); st = S.Solve(M)
    print(f"{name} k={k}: CP-SAT {S.StatusName(st)} ({time.time()-t0:.1f}s)  [{len(cyc)} rigid slice triangles]", flush=True)

if __name__ == "__main__":
    for spec in sys.argv[1:]:
        n, k = spec.split(":"); check(n, int(k))
