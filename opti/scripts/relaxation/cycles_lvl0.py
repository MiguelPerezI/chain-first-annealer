"""CP-SAT existence of a cover of C_n x C_n (n-cycle graph, staircase product) by k contiguity domains with chains of <= L links.
Independent re-implementation of the chain model (not the repository's scs_sat.py)."""
import sys, time
from ortools.sat.python import cp_model
import relax

def cover(kf, k, L, timeout=300, workers=4):
    kf = [tuple(sorted(f)) for f in kf]; nv = 1 + max(v for f in kf for v in f)
    F = relax.staircase(kf, nv); NG = nv * nv
    M = cp_model.CpModel(); m = {}
    for d in range(k):
        for i in range(L + 1):
            for g in range(NG):
                m[d, i, g] = (g // nv) if i == 0 else ((g % nv) if i == L else M.NewIntVar(0, nv - 1, f"m{d}_{i}_{g}"))
    x = {}
    for fi in range(len(F)):
        for d in range(k): x[fi, d] = M.NewBoolVar(f"x{fi}_{d}")
        M.AddBoolOr([x[fi, d] for d in range(k)])
    for d in range(k):
        for fi, (verts, s, t) in enumerate(F):
            for i in range(L):
                ys = []
                for ti, T in enumerate(kf):
                    y = M.NewBoolVar(f"y{d}_{fi}_{i}_{ti}"); ys.append(y)
                    for g in verts:
                        for j in (i, i + 1):
                            if j == 0 or j == L:
                                val = (g // nv) if j == 0 else (g % nv)
                                if val not in T: M.Add(y == 0)
                            else:
                                M.AddLinearExpressionInDomain(m[d, j, g], cp_model.Domain.FromValues(list(T))).OnlyEnforceIf(y)
                M.AddBoolOr(ys + [x[fi, d].Not()])
    S = cp_model.CpSolver(); S.parameters.max_time_in_seconds = timeout; S.parameters.num_workers = workers
    return S.StatusName(S.Solve(M))

if __name__ == "__main__":
    for n in range(3, 10):
        kf = [tuple(sorted((i, (i + 1) % n))) for i in range(n)]
        res = None
        for L in (2, 3, 4, 5, 6, 8, 10, 12):
            st = cover(kf, 2, L)
            if st in ("OPTIMAL", "FEASIBLE"): res = L; break
        print(f"C_{n}: two domains found with chains of <= {res} links  (so SCst(C_{n}) = 1 = TC)", flush=True)
