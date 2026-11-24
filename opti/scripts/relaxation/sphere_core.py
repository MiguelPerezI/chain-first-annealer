import itertools, sys, time, random
from pysat.solvers import Cadical153
import spheres

def sat_with(F, T, sph_subset, k=2):
    nf = len(F)
    S = Cadical153()
    x = lambda f, d: 1 + f * k + d
    nxt = [1 + nf * k]
    def newv():
        nxt[0] += 1; return nxt[0] - 1
    for f in range(nf): S.add_clause([x(f, d) for d in range(k)])
    S.add_clause([x(0, 0)])
    avoid = {}
    def av(z, d):
        if (z, d) not in avoid:
            v = newv(); avoid[z, d] = v
            for f in T[z]: S.add_clause([-v, -x(f, d)])
        return avoid[z, d]
    for sp in sph_subset:
        for d in range(k): S.add_clause([av(z, d) for z in sp])
    r = S.solve(); S.delete(); return r

if __name__ == "__main__":
    N = int(sys.argv[1])
    F, T, sph = spheres.build(N)
    assert not sat_with(F, T, sph)
    cur = list(sph); random.seed(1); random.shuffle(cur)
    # deletion-based minimisation of the sphere family
    i = 0
    while i < len(cur):
        trial = cur[:i] + cur[i + 1:]
        if not sat_with(F, T, trial): cur = trial
        else: i += 1
    print("minimal unsatisfiable sphere family for N=%d: %d spheres" % (N, len(cur)))
    def desc(sp):
        pts = sorted({p for z in sp for p in z})
        return [(p // N, p % N) for p in pts]
    for sp in cur: print(desc(sp))
