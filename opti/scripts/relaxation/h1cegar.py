"""Edge-level CEGAR for the H^1 relaxation (Z2 classes of H^1(K;Z2) plus an optional integral class).

D (a set of facets of the staircase K x K) is H^1-balanced for a class u if every closed edge walk in the
1-skeleton of D has pr1^*u + pr2^*u = 0 (resp. pr1^*u - pr2^*u = 0 for an integral u).  Balanced is hereditary,
so every contiguity domain is balanced.  Bad cycles become clauses over edge-presence variables."""
import itertools, json, os, random, sys, time
from collections import deque
from pysat.solvers import Cadical153
import relax

def edge_data(kf):
    kf = [tuple(sorted(f)) for f in kf]
    nv = 1 + max(v for f in kf for v in f)
    F = relax.staircase(kf, nv)
    E = {}                                  # edge (p,q) with p<q in chain order -> index
    fedges = []
    for verts, s, t in F:
        es = []
        for p, q in itertools.combinations(verts, 2):
            if (p, q) not in E: E[(p, q)] = len(E)
            es.append(E[(p, q)])
        fedges.append(es)
    return kf, nv, F, E, fedges

def integral_h1_class(kf, nv):
    """A primitive integral cocycle generating H^1(K;Z) if b1 == 1, as dict (a,b)->int on sorted edges, else None."""
    import sympy as sp
    edges = sorted({e for f in kf for e in itertools.combinations(sorted(f), 2)})
    adj = {}
    for a, b in edges: adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)
    seen = {0}; tree = set(); dq = deque([0])
    while dq:
        a = dq.popleft()
        for b in adj[a]:
            if b not in seen: seen.add(b); tree.add(tuple(sorted((a, b)))); dq.append(b)
    free = [e for e in edges if e not in tree]
    fi = {e: i for i, e in enumerate(free)}
    rows = []
    for f in kf:
        for (a, b, c) in itertools.combinations(sorted(f), 3):
            r = [0] * len(free)
            for e, sgn in (((a, b), 1), ((b, c), 1), ((a, c), -1)):
                if e in fi: r[fi[e]] += sgn
            rows.append(r)
    A = sp.Matrix(rows)
    ns = A.nullspace()
    if len(ns) != 1: return None, len(ns)
    v = ns[0]; den = sp.ilcm(*[x.q for x in v]); v = v * den
    g = sp.igcd(*[int(x) for x in v if x != 0]); v = v / g
    u = {e: 0 for e in edges}
    for e, i in fi.items(): u[e] = int(v[i])
    return u, 1

class H1:
    def __init__(self, kf, use_int=True):
        self.kf, self.nv, self.F, self.E, self.fedges = edge_data(kf)
        self.nf = len(self.F)
        nv = self.nv
        H = relax.cohomology_basis(self.kf)
        self.z2 = [cl for cl in H.get(1, [])]
        self.integral = None
        if use_int:
            u, r = integral_h1_class(self.kf, nv)
            self.integral = u
        self.edges = [None] * len(self.E)
        for (p, q), i in self.E.items(): self.edges[i] = (p, q)
        # omega values per edge
        self.om_z2 = []
        for cl in self.z2:
            om = []
            for (p, q) in self.edges:
                v1, w1, v2, w2 = p // nv, p % nv, q // nv, q % nv
                a = cl[tuple(sorted((v1, v2)))] if v1 != v2 else 0
                b = cl[tuple(sorted((w1, w2)))] if w1 != w2 else 0
                om.append((a + b) % 2)
            self.om_z2.append(om)
        self.om_int = None
        if self.integral is not None:
            om = []
            for (p, q) in self.edges:            # oriented p -> q (increasing in the chain order => v1<=v2, w1<=w2)
                v1, w1, v2, w2 = p // nv, p % nv, q // nv, q % nv
                a = self.integral[(v1, v2)] if v1 != v2 else 0
                b = self.integral[(w1, w2)] if w1 != w2 else 0
                om.append(a - b)
            self.om_int = om

    def bad_cycle(self, eset, rng):
        """eset: set of edge indices present.  Returns a short cycle (list of edge indices) with nonzero omega, or None."""
        adj = {}
        for ei in eset:
            p, q = self.edges[ei]
            adj.setdefault(p, []).append((q, ei, +1)); adj.setdefault(q, []).append((p, ei, -1))
        verts = list(adj); rng.shuffle(verts)
        classes = [("z2", om) for om in self.om_z2] + ([("int", self.om_int)] if self.om_int is not None else [])
        best = None
        for kind, om in classes:
            for root in verts[:12]:
                # BFS over (vertex, sum) ; sum mod 2 or integer in [-3,3]
                start = (root, 0)
                par = {start: None}
                dq = deque([start])
                found = None
                while dq and found is None:
                    cur = dq.popleft()
                    v, s = cur
                    for (w, ei, sg) in adj[v]:
                        if par[cur] is not None and par[cur][1] == ei: continue
                        ns = (s + (sg * om[ei] if kind == "int" else om[ei]))
                        ns = ns % 2 if kind == "z2" else ns
                        if abs(ns) > 3 and kind == "int": continue
                        nxt = (w, ns)
                        if w == root and ns != 0:
                            par[nxt] = (cur, ei); found = nxt; break
                        if nxt not in par:
                            par[nxt] = (cur, ei); dq.append(nxt)
                if found is not None:
                    cyc = []; cur = found
                    while par[cur] is not None:
                        cur, ei = par[cur][0], par[cur][1]; cyc.append(ei)
                    if best is None or len(cyc) < len(best): best = cyc
                if best is not None and len(best) <= 3: return best
        return best

def run(kf, k, time_limit=1800, seed=0, use_int=True, verbose=True, rounds=8):
    H = H1(kf, use_int)
    rng = random.Random(seed)
    nf, ne = H.nf, len(H.E)
    xv = lambda f, d: 1 + f * k + d
    pv = lambda e, d: 1 + nf * k + e * k + d
    S = Cadical153()
    for f in range(nf): S.add_clause([xv(f, d) for d in range(k)])
    S.add_clause([xv(0, 0)])
    for f in range(nf):
        for e in H.fedges[f]:
            for d in range(k): S.add_clause([-xv(f, d), pv(e, d)])
    cycles = set()
    def add_cycle(c):
        key = tuple(sorted(set(c)))
        if key in cycles: return False
        cycles.add(key)
        for d in range(k): S.add_clause([-pv(e, d) for e in key])
        return True
    t0 = time.time(); it = 0
    while time.time() - t0 < time_limit:
        it += 1
        if not S.solve(): return "UNSAT", sorted(cycles)
        model = set(l for l in S.get_model() if l > 0)
        added = 0
        for d in range(k):
            eset = set()
            for f in range(nf):
                if xv(f, d) in model: eset.update(H.fedges[f])
            r = 0
            while r < rounds:
                c = H.bad_cycle(eset, rng)
                if c is None: break
                if add_cycle(c): added += 1
                r += 1
                eset.discard(rng.choice(c))
        if verbose and (it % 50 == 0 or added == 0):
            L = [len(c) for c in cycles] or [0]
            print(f"  it {it} cycles {len(cycles)} (+{added}) len min/avg/max {min(L)}/{sum(L)/len(L):.1f}/{max(L)} t={time.time()-t0:.0f}s", flush=True)
        if added == 0:
            return "SAT", [[f for f in range(nf) if xv(f, d) in model] for d in range(k)]
    return "TIMEOUT", sorted(cycles)

if __name__ == "__main__":
    name, kmin, kmax = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    tl = float(sys.argv[4]) if len(sys.argv) > 4 else 1800
    use_int = (len(sys.argv) <= 5) or sys.argv[5] != "noint"
    C = relax.load_complexes()
    for k in range(kmin, kmax + 1):
        st, res = run(C[name], k, tl, use_int=use_int)
        print(f"RESULT {name} k={k} {st}" + (f" cycles={len(res)}" if st != 'SAT' else ''), flush=True)
        json.dump(res, open(os.path.join(relax.HERE, f"h1_{name}_k{k}_{st}.json"), "w"))
        if st == "SAT": break
