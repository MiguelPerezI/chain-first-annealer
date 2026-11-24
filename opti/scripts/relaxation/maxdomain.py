"""Largest single set of facets containing none of the forbidden (rigid) cycles / spheres: k=1 relaxation optimum.
Gives the counting bound  k >= ceil(#facets / max)."""
import sys, time, math
from ortools.sat.python import cp_model
import relax, h1cegar, h1pre, slices

def solve(name, mode, tl=600, workers=2):
    C = relax.load_complexes()
    H = h1cegar.H1(C[name])
    if mode == "slices":
        _, _, cyc = slices.slice_family(name); cycles = list(cyc)
    else:
        cycles, _ = h1pre.short_bad_cycles(H, 4)
    nf, ne = H.nf, len(H.E)
    M = cp_model.CpModel()
    x = [M.NewBoolVar(f"x{f}") for f in range(nf)]
    p = [M.NewBoolVar(f"p{e}") for e in range(ne)]
    for f in range(nf):
        for e in H.fedges[f]: M.AddImplication(x[f], p[e])
    for cy in cycles: M.AddBoolOr([p[e].Not() for e in cy])
    M.Maximize(sum(x))
    S = cp_model.CpSolver(); S.parameters.max_time_in_seconds = tl; S.parameters.num_workers = workers
    t0 = time.time(); st = S.Solve(M)
    best = int(S.ObjectiveValue()) if st in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None
    bound = int(S.BestObjectiveBound())
    print(f"{name} [{mode}, {len(cycles)} forbidden cycles]: {S.StatusName(st)} best={best} upper bound={bound} "
          f"-> k >= {math.ceil(nf / bound)} (counting)  ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    solve(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 600)
