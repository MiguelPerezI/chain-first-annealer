"""Rigid-sphere 2-colouring test for K = boundary of the (N-1)-simplex with facets generated lazily.
Only facets lying over a face of a rigid sphere Z_w (w non-decreasing, <= maxvals distinct values) are created;
all other facets are unconstrained and can be coloured freely, so the test is exact for the chosen sphere family.
Spheres: A-type {(j,w_j)} and its transpose (B-type)."""
import itertools, sys, time
from pysat.solvers import Solver

def lattice_paths(r1, c1, r2, c2):
    """all monotone lattice paths from (r1,c1) to (r2,c2) as lists of (r,c) points (inclusive)."""
    if r1 == r2 and c1 == c2: yield [(r1, c1)]; return
    if r1 < r2:
        for p in lattice_paths(r1 + 1, c1, r2, c2): yield [(r1, c1)] + p
    if c1 < c2:
        for p in lattice_paths(r1, c1 + 1, r2, c2): yield [(r1, c1)] + p

def facets_through(N, pts):
    """pts: chain of grid points (v,w) sorted in product order.  yields facets (a,b,path of grid points) through all of them.
    A facet is a monotone path in (V\\a) x (V\\b) from its min corner to its max corner."""
    vs = {v for v, w in pts}; ws = {w for v, w in pts}
    for a in range(N):
        if a in vs: continue
        s = [v for v in range(N) if v != a]
        for b in range(N):
            if b in ws: continue
            t = [w for w in range(N) if w != b]
            rk_s = {v: i for i, v in enumerate(s)}; rk_t = {w: i for i, w in enumerate(t)}
            rpts = [(rk_s[v], rk_t[w]) for v, w in pts]
            nodes = [(0, 0)] + rpts + [(len(s) - 1, len(t) - 1)]
            segs = []
            ok = True
            for (r1, c1), (r2, c2) in zip(nodes, nodes[1:]):
                if r2 < r1 or c2 < c1: ok = False; break
                segs.append(list(lattice_paths(r1, c1, r2, c2)))
            if not ok: continue
            for combo in itertools.product(*segs):
                path = []
                for seg in combo:
                    path.extend(seg if not path else seg[1:])
                yield (a, b, tuple(s[r] * N + t[c] for r, c in path))

def transpose_facet(N, f):
    a, b, path = f
    return (b, a, tuple((p % N) * N + p // N for p in path))

def spheres_family(N, maxvals):
    out = []
    for w in itertools.combinations_with_replacement(range(N), N):
        if w == tuple(range(N)) or len(set(w)) > maxvals: continue
        out.append(w)
    return out

def solve(N, maxvals, k=2, verbose=True, solver='cadical153'):
    fid = {}
    def fi(f):
        if f not in fid: fid[f] = len(fid)
        return fid[f]
    sphere_faces = []      # each sphere: list of faces; each face: list of facet ids
    for w in spheres_family(N, maxvals):
        for typ in 'AB':
            faces = []
            for i in range(N):
                if typ == 'A':
                    pts = [(j, w[j]) for j in range(N) if j != i]
                    fs = [fi(f) for f in facets_through(N, pts)]
                else:
                    ptsT = [(j, w[j]) for j in range(N) if j != i]          # transposed chain, A-type picture
                    fs = [fi(transpose_facet(N, f)) for f in facets_through(N, ptsT)]
                faces.append(fs)
            sphere_faces.append(faces)
    nf = len(fid)
    if verbose: print(f"N={N} maxvals={maxvals}: spheres {len(sphere_faces)}, facets involved {nf}", flush=True)
    S = Solver(name=solver)
    x = lambda f, d: 1 + f * k + d
    nxt = [1 + nf * k]
    def newv():
        nxt[0] += 1; return nxt[0] - 1
    for f in range(nf): S.add_clause([x(f, d) for d in range(k)])
    avoid = {}
    for faces in sphere_faces:
        for d in range(k):
            lits = []
            for fs in faces:
                key = (tuple(fs), d)
                if key not in avoid:
                    v = newv(); avoid[key] = v
                    for f in fs: S.add_clause([-v, -x(f, d)])
                lits.append(avoid[key])
            S.add_clause(lits)
    t0 = time.time(); r = S.solve()
    if verbose: print(f"  [{solver}] -> {'SAT (2-colouring exists for this sphere family)' if r else 'UNSAT: no 2-cover'}  ({time.time()-t0:.1f}s)", flush=True)
    return r

if __name__ == "__main__":
    N = int(sys.argv[1]); mv = int(sys.argv[2])
    solve(N, mv, solver=(sys.argv[3] if len(sys.argv) > 3 else 'cadical153'))
