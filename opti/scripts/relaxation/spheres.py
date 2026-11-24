"""Rigid-sphere obstruction for K = boundary of the (N-1)-simplex (N vertices, an (N-2)-sphere).
Z_w = {(i, w(i))}, w non-decreasing and not the identity, is a hollow (N-2)-sphere inside the staircase K x K
on which pr1 = id has degree 1 and pr2 = w has degree 0; so no contiguity domain (indeed no set on which
pr1 ~ pr2 homotopically) contains Z_w.  Question: can two sets D1, D2 (unions of facets) cover K x K
with neither containing any Z_w (or its transpose)?"""
import itertools, sys, time
from pysat.solvers import Cadical153
import relax

def boundary_simplex(N): return [tuple(v for v in range(N) if v != a) for a in range(N)]

def nondecr_maps(N):
    for w in itertools.combinations_with_replacement(range(N), N):
        if w != tuple(range(N)): yield w

def build(N, k=2, transpose=True, verbose=True):
    kf = boundary_simplex(N)
    F = relax.staircase(kf, N)
    nf = len(F)
    # tetrahedra (N-1 points) -> facets containing them
    T = {}
    for fi, (verts, s, t) in enumerate(F):
        for z in itertools.combinations(verts, N - 1):
            T.setdefault(z, []).append(fi)
    spheres = []
    for w in nondecr_maps(N):
        pts = [i * N + w[i] for i in range(N)]            # (i, w_i) as grid points, increasing chain
        spheres.append(tuple(tuple(pts[:i] + pts[i + 1:]) for i in range(N)))
        if transpose:
            pts = [w[i] * N + i for i in range(N)]
            spheres.append(tuple(tuple(pts[:i] + pts[i + 1:]) for i in range(N)))
    for sp in spheres:
        for z in sp: assert z in T, ("face not a simplex of S", z)
    if verbose: print(f"N={N}: facets {nf}, spheres {len(spheres)}, tetra-faces {len(T)}", flush=True)
    return F, T, spheres

def solve(N, k=2, transpose=True, timeout=None, verbose=True):
    F, T, spheres = build(N, k, transpose, verbose)
    nf = len(F)
    S = Cadical153()
    x = lambda f, d: 1 + f * k + d
    nxt = [1 + nf * k]
    def newv():
        nxt[0] += 1; return nxt[0] - 1
    for f in range(nf):
        S.add_clause([x(f, d) for d in range(k)])
    S.add_clause([x(0, 0)])
    avoid = {}          # (zeta, d) -> var meaning "no facet containing zeta is in D_d"
    def av(z, d):
        if (z, d) not in avoid:
            v = newv(); avoid[z, d] = v
            for f in T[z]: S.add_clause([-v, -x(f, d)])
        return avoid[z, d]
    for sp in spheres:
        for d in range(k):
            S.add_clause([av(z, d) for z in sp])
    t0 = time.time()
    r = S.solve()
    if verbose: print(f"N={N} k={k}: {'SAT' if r else 'UNSAT'}  ({time.time()-t0:.1f}s)", flush=True)
    return r

if __name__ == "__main__":
    for N in map(int, sys.argv[1:]): solve(N)
