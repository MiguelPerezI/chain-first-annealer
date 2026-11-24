"""Rigid slice triangles tau x {c}, {c} x tau over the empty triangles tau of K whose boundary is homologically
essential (Z2 or Z); k-colourability of the facets with no colour class containing one."""
import itertools, sys, time
from pysat.solvers import Cadical153, Glucose4
import relax, h1cegar

def slice_family(name):
    C = relax.load_complexes(); kf = C[name]
    H = h1cegar.H1(kf); nv = H.nv; E = H.E
    faces = {s for f in kf for r in range(1, len(f) + 1) for s in itertools.combinations(sorted(f), r)}
    empties = [s for s in itertools.combinations(range(nv), 3)
               if s not in faces and all(t in faces for t in itertools.combinations(s, 2))]
    cycles = {}
    for tau in empties:
        for c in range(nv):
            for typ in 'AB':
                pts = [t * nv + c if typ == 'A' else c * nv + t for t in tau]
                key = (E[(pts[0], pts[1])], E[(pts[1], pts[2])], E[(pts[0], pts[2])])
                # omega of the cycle p0->p1->p2->p0
                z = [0] * len(H.om_z2); s = 0
                for (a, b) in ((pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[0])):
                    ei = E[(a, b)] if (a, b) in E else E[(b, a)]
                    sg = 1 if (a, b) in E else -1
                    for i, om in enumerate(H.om_z2): z[i] ^= om[ei]
                    if H.om_int is not None: s += sg * H.om_int[ei]
                if any(z) or s != 0: cycles[tuple(sorted(key))] = (tau, c, typ)
    return H, empties, cycles

def colourable(H, cycles, k, partition=True, budget=None):
    nf, ne = H.nf, len(H.E)
    xv = lambda f, d: 1 + f * k + d
    pv = lambda e, d: 1 + nf * k + e * k + d
    S = Glucose4() if budget else Cadical153()
    for f in range(nf): S.add_clause([xv(f, d) for d in range(k)])
    S.add_clause([xv(0, 0)])
    for f in range(nf):
        for e in H.fedges[f]:
            for d in range(k): S.add_clause([-xv(f, d), pv(e, d)])
    for cy in cycles:
        for d in range(k): S.add_clause([-pv(e, d) for e in cy])
    if budget:
        S.conf_budget(budget)
        return S.solve_limited()      # True / False / None (unknown)
    return S.solve()

if __name__ == "__main__":
    name = sys.argv[1]
    H, empties, cyc = slice_family(name)
    print(name, "empty triangles", len(empties), "rigid slice triangles", len(cyc), flush=True)
    for k in range(2, int(sys.argv[2]) + 1):
        t0 = time.time(); r = colourable(H, list(cyc), k, budget=int(float(sys.argv[3])) if len(sys.argv) > 3 else None)
        print(f"  k={k}: {'colourable' if r else ('UNKNOWN (budget)' if r is None else 'NOT colourable')}  ({time.time()-t0:.1f}s)", flush=True)
        if r or r is None: break
