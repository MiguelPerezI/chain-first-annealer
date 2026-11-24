"""Run the rigid-sphere 2-colouring instance (N=5,7) through several independent SAT solvers."""
import sys, time
from pysat.solvers import Solver
import spheres

def cnf(N, k=2):
    F, T, sph = spheres.build(N, verbose=False)
    nf = len(F); x = lambda f, d: 1 + f * k + d
    clauses = [[x(f, d) for d in range(k)] for f in range(nf)]
    nxt = [1 + nf * k]; avoid = {}
    def av(z, d):
        if (z, d) not in avoid:
            v = nxt[0]; nxt[0] += 1; avoid[z, d] = v
            for f in T[z]: clauses.append([-v, -x(f, d)])
        return avoid[z, d]
    for sp in sph:
        for d in range(k): clauses.append([av(z, d) for z in sp])
    return clauses

for N in (5, 7):
    cl = cnf(N)
    for name in ("cadical153", "glucose4", "maplechrono", "lingeling", "minisat22"):
        t0 = time.time()
        with Solver(name=name, bootstrap_with=cl) as s:
            r = s.solve()
        print(f"N={N} {name:12s}: {'SAT' if r else 'UNSAT'} ({time.time()-t0:.2f}s)   [{len(cl)} clauses]", flush=True)
