import itertools, json
from pysat.solvers import Cadical153
import relax, h1cegar
C = relax.load_complexes()
kf = C['mobius']; nv = 5
H = h1cegar.H1(kf)
faces = {s for f in kf for r in range(1, 4) for s in itertools.combinations(sorted(f), r)}
empties = [s for s in itertools.combinations(range(nv), 3) if s not in faces and all(t in faces for t in itertools.combinations(s, 2))]
print("empty triangles:", empties)
# integral winding of each empty triangle (cocycle g found earlier: g=1 on (0,3),(0,4),(1,4))
g = {(a, b): 0 for a in range(5) for b in range(a + 1, 5)}
for e in [(0, 3), (0, 4), (1, 4)]: g[e] = 1
for (a, b, c) in empties:
    print("  winding of", (a, b, c), "=", g[(a, b)] + g[(b, c)] - g[(a, c)])
# slice triangles tau x {c} and {c} x tau as 3-cycles of S (edge indices)
E = H.E
def slice_cycle(tau, c, typ):
    pts = [t * nv + c if typ == 'A' else c * nv + t for t in tau]
    return tuple(sorted((E[(pts[0], pts[1])], E[(pts[1], pts[2])], E[(pts[0], pts[2])])))
cycles = set()
for tau in empties:
    for c in range(nv):
        for typ in 'AB': cycles.add(slice_cycle(tau, c, typ))
print("slice triangles:", len(cycles))
def feasible(cycles, k=2):
    nf, ne = H.nf, len(E)
    xv = lambda f, d: 1 + f * k + d
    pv = lambda e, d: 1 + nf * k + e * k + d
    S = Cadical153()
    for f in range(nf): S.add_clause([xv(f, d) for d in range(k)])
    for f in range(nf):
        for e in H.fedges[f]:
            for d in range(k): S.add_clause([-xv(f, d), pv(e, d)])
    for cy in cycles:
        for d in range(k): S.add_clause([-pv(e, d) for e in cy])
    return S.solve()
print("k=2 feasible with slice triangles only?", feasible(cycles))
print("k=3 feasible with slice triangles only?", feasible(cycles, 3))
# minimal sub-family
cur = sorted(cycles)
i = 0
while i < len(cur):
    t = cur[:i] + cur[i + 1:]
    if not feasible(t): cur = t
    else: i += 1
print("minimal unsatisfiable slice family:", len(cur))
nvv = nv
for cy in cur:
    pts = sorted({p for ei in cy for p in H.edges[ei]})
    print("  ", [(p // nvv, p % nvv) for p in pts])
