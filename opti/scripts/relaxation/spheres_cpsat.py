"""Independent re-check of the rigid-sphere 2-colouring infeasibility with OR-tools CP-SAT (different solver from pysat)."""
import sys, time
from ortools.sat.python import cp_model
import spheres
N = int(sys.argv[1]); k = 2
F, T, sph = spheres.build(N, verbose=True)
M = cp_model.CpModel()
x = [[M.NewBoolVar(f"x{f}_{d}") for d in range(k)] for f in range(len(F))]
for f in range(len(F)): M.AddBoolOr(x[f])
avoid = {}
def av(z, d):
    if (z, d) not in avoid:
        v = M.NewBoolVar(f"a{len(avoid)}"); avoid[z, d] = v
        for f in T[z]: M.AddImplication(v, x[f][d].Not())
    return avoid[z, d]
for sp in sph:
    for d in range(k): M.AddBoolOr([av(z, d) for z in sp])
S = cp_model.CpSolver(); S.parameters.num_workers = 4; S.parameters.max_time_in_seconds = 600
t0 = time.time(); st = S.Solve(M)
print(f"N={N}: CP-SAT {S.StatusName(st)} in {time.time()-t0:.1f}s")
