#!/usr/bin/env python3
"""rigid.py -- rigid subcomplexes of the staircase product K x K, lower bounds, and stall audits.

Families of rigid subcomplexes (every one is PROVED rigid, no homology approximation):
  slices  : tau x {c}, {c} x tau over empty triangles tau -- rigid iff the two projected closed
            walks are in different contiguity classes of maps C_3 -> K (exact BFS); this contains
            the note's essential-empty-triangle slices.
  walks   : every closed walk of length m <= M in the 1-skeleton of K x K whose two projected
            walks lie in different contiguity classes of maps C_m -> K (exact BFS over V^m).
  graphs  : Gamma_w = {(v, w(v)) : v in sigma}, sigma in K, for every non-decreasing simplicial
            self-map w != id of K (and transposes); rigid when K has no dominated vertex
            (Theorem A: id is contiguous only to itself on such K, so a chain id ~ ... ~ w forces w = id).

Lower bound: if the k-class avoiding problem (every facet in some class, no class's generated
subcomplex contains a rigid member) is UNSAT then SC_strict(K) + 1 > k.

Usage:
  rigid.py bound  <complex> <kmax> [--walks M] [--no-graphs]
  rigid.py audit  <complex> <summary.json> [--walks M]
"""
import argparse, itertools, json, os, sys, time
from collections import defaultdict, deque

HERE = os.path.dirname(os.path.abspath(__file__))
RELAX = os.path.join(HERE, '..', 'relaxation')
sys.path.insert(0, RELAX)
import relax  # staircase(), load_complexes()


# ----------------------------------------------------------------------------- the complex K
class Complex:
    def __init__(self, kf):
        self.kf = [tuple(sorted(f)) for f in kf]
        self.nv = 1 + max(v for f in self.kf for v in f)
        self.faces = set()
        for f in self.kf:
            for r in range(1, len(f) + 1):
                for s in itertools.combinations(f, r):
                    self.faces.add(frozenset(s))
        self.edges = sorted({s for s in self.faces if len(s) == 2}, key=sorted)
        self.adj = defaultdict(set)
        for e in self.edges:
            a, b = sorted(e); self.adj[a].add(b); self.adj[b].add(a)

    def is_simplex(self, S):
        return frozenset(S) in self.faces

    def dominated(self):
        """vertices v dominated by some v' != v (every facet containing v contains v')."""
        out = {}
        for v in range(self.nv):
            inter = None
            for f in self.kf:
                if v in f:
                    inter = set(f) if inter is None else inter & set(f)
            if inter is not None and len(inter) > 1:
                out[v] = sorted(inter - {v})
        return out

    def empty_triangles(self):
        out = []
        for tau in itertools.combinations(range(self.nv), 3):
            if frozenset(tau) not in self.faces and all(frozenset(e) in self.faces for e in itertools.combinations(tau, 2)):
                out.append(tau)
        return out

    def nondecreasing_simplicial_maps(self):
        """all non-decreasing w: V -> V with w(sigma) in K for every facet sigma, w != id."""
        nv = self.nv
        out = []
        # DFS over values with pruning on the facets completed so far
        order = list(range(nv))
        fac_by_last = defaultdict(list)
        for f in self.kf:
            fac_by_last[max(f)].append(f)
        w = [0] * nv

        def rec(i, lo):
            if i == nv:
                if any(w[v] != v for v in range(nv)): out.append(tuple(w))
                return
            for val in range(lo, nv):
                w[i] = val
                ok = all(self.is_simplex({w[v] for v in f}) for f in fac_by_last[i])
                if ok: rec(i + 1, val)
        rec(0, 0)
        return out

    # contiguity classes of closed walks C_m -> K (exact) ------------------------------------
    def walk_classes(self, m):
        cache = self.__dict__.setdefault('_wc', {})
        key = m
        if key in cache: return cache[key]
        nv = self.nv
        allowed4 = {}  # (a,b) -> set of (a',b') with {a,b,a',b'} a simplex
        verts = range(nv)
        pairs = [(a, b) for a in verts for b in verts if a == b or frozenset((a, b)) in self.faces]
        for (a, b) in pairs:
            allowed4[(a, b)] = [(a2, b2) for (a2, b2) in pairs if self.is_simplex({a, b, a2, b2})]
        # closed walks: tuples of length m with consecutive pairs allowed (cyclically)
        walks = []
        def gen(prefix):
            if len(prefix) == m:
                if (prefix[-1], prefix[0]) in allowed4: walks.append(tuple(prefix))
                return
            last = prefix[-1]
            for b in verts:
                if b == last or frozenset((last, b)) in self.faces:
                    gen(prefix + [b])
        for a in verts: gen([a])
        index = {wk: i for i, wk in enumerate(walks)}
        parent = list(range(len(walks)))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        # neighbours: phi' with (phi(y_i),phi(y_{i+1}),phi'(y_i),phi'(y_{i+1})) simplex for all i
        for wk in walks:
            r = find(index[wk])
            # backtracking enumeration of contiguous phi'
            def rec(i, cur):
                if i == m:
                    if (cur[-1], cur[0]) in allowed4 and (cur[-1], cur[0]) in set(allowed4[(wk[-1], wk[0])]):
                        j = index.get(tuple(cur))
                        if j is not None:
                            rj = find(j)
                            if rj != find(index[wk]): parent[rj] = find(index[wk])
                    return
                a, b = wk[i - 1], wk[i] if i < m else wk[0]
                if i == 0:
                    for v in verts:
                        rec(1, [v])
                    return
                prev = cur[-1]
                for (a2, b2) in allowed4[(wk[i - 1], wk[i % m])]:
                    if a2 == prev:
                        rec(i + 1, cur + [b2])
            # cheaper: build positions sequentially
            def rec2(i, cur):
                if i == m:
                    # closing edge (y_{m-1}, y_0): phi = (wk[m-1], wk[0]), phi' = (cur[m-1], cur[0])
                    if self.is_simplex({wk[m - 1], wk[0], cur[m - 1], cur[0]}):
                        j = index.get(tuple(cur))
                        if j is not None:
                            rj = find(j); ri = find(index[wk])
                            if rj != ri: parent[rj] = ri
                    return
                if i == 0:
                    for v in verts: rec2(1, [v])
                    return
                prev = cur[-1]
                a, b = wk[i - 1], wk[i]
                for v in verts:
                    if self.is_simplex({a, b, prev, v}):
                        rec2(i + 1, cur + [v])
            rec2(0, [])
        classes = {wk: find(index[wk]) for wk in walks}
        cache[key] = classes
        return classes


# ----------------------------------------------------------------------------- the product K x K
class Product:
    def __init__(self, K):
        self.K = K
        nv = K.nv
        self.F = relax.staircase(K.kf, nv)           # (verts, s, t)
        self.nf = len(self.F)
        self.facet_verts = [frozenset(v) for v, s, t in self.F]
        # simplex -> facets containing it, for simplices up to dim 2 (edges, triangles) and (N-2)-simplices later
        self.by_simplex = defaultdict(list)
        for fi, verts in enumerate(self.facet_verts):
            vs = sorted(verts)
            for r in (2, 3):
                for z in itertools.combinations(vs, r):
                    self.by_simplex[frozenset(z)].append(fi)
        self.grid_edges = sorted(z for z in self.by_simplex if len(z) == 2)
        self.padj = defaultdict(set)
        for z in self.grid_edges:
            p, q = tuple(z); self.padj[p].add(q); self.padj[q].add(p)

    def pr(self, p): return p // self.K.nv, p % self.K.nv

    def facets_containing(self, z):
        z = frozenset(z)
        if z in self.by_simplex: return self.by_simplex[z]
        if len(z) > 3:
            out = [fi for fi, verts in enumerate(self.facet_verts) if z <= verts]
            self.by_simplex[z] = out
            return out
        return []


# ----------------------------------------------------------------------------- rigid families
def rigid_walks(P, M, verbose=True):
    """closed walks of length 3..M in the 1-skeleton of KxK with projections in different classes.
    Returns dict: frozenset(edges) -> (m, walk, kind)."""
    K = P.K; nv = K.nv
    out = {}
    for m in range(3, M + 1):
        t0 = time.time()
        cls = K.walk_classes(m)
        n_cls = len(set(cls.values()))
        cnt = 0; seen = set()
        pts = sorted(P.padj)
        def gen(path):
            nonlocal cnt
            if len(path) == m:
                if path[0] in P.padj[path[-1]]:
                    w1 = tuple(p // nv for p in path); w2 = tuple(p % nv for p in path)
                    if cls[w1] != cls[w2]:
                        edges = frozenset(frozenset((path[i], path[(i + 1) % m])) for i in range(m))
                        if edges not in out:
                            out[edges] = (m, tuple(path), 'walk')
                            cnt += 1
                return
            for q in P.padj[path[-1]]:
                if q < path[0]: continue      # canonical: smallest point first
                gen(path + [q])
        for p in pts: gen([p])
        if verbose:
            print(f"  walks m={m}: {len(cls)} maps C_m->K in {n_cls} contiguity classes; "
                  f"{cnt} rigid closed walks (distinct edge sets)  [{time.time()-t0:.1f}s]", flush=True)
    return out


def rigid_graphs(P, verbose=True):
    """Gamma_w and transposes for non-decreasing simplicial w != id; valid when K has no dominated vertex.
    Returns list of (maximal simplices as frozensets, label)."""
    K = P.K; nv = K.nv
    dom = K.dominated()
    if dom:
        if verbose: print(f"  graphs: SKIPPED, dominated vertices {dom}")
        return []
    maps = K.nondecreasing_simplicial_maps()
    out = []
    for w in maps:
        tris = [frozenset(v * nv + w[v] for v in f) for f in K.kf]
        out.append((tris, ('graph', w)))
        tris_t = [frozenset(w[v] * nv + v for v in f) for f in K.kf]
        out.append((tris_t, ('graph^T', w)))
    if verbose:
        nconst = sum(1 for w in maps if len(set(w)) == 1)
        print(f"  graphs: {len(maps)} non-decreasing simplicial self-maps != id "
              f"({nconst} constant, {len(maps)-nconst} other) -> {len(out)} rigid copies of K", flush=True)
    return out


def all_rigid(P, M=4, graphs=True, verbose=True):
    fam = []   # list of (list of maximal simplices (frozensets of grid points), label)
    W = rigid_walks(P, M, verbose)
    for edges, (m, path, kind) in W.items():
        fam.append((list(edges), (kind, m, path)))
    if graphs:
        fam += rigid_graphs(P, verbose)
    return fam


# ----------------------------------------------------------------------------- SAT: k-class avoidance
def avoid_sat(P, fam, k, verbose=True, time_limit=None):
    from pysat.solvers import Cadical153
    nf = P.nf
    S = Cadical153()
    x = lambda f, d: 1 + f * k + d
    nxt = [1 + nf * k]
    yv = {}
    def y(z, d):
        key = (z, d)
        if key not in yv:
            v = nxt[0]; nxt[0] += 1; yv[key] = v
            for f in P.facets_containing(z):
                S.add_clause([-v, -x(f, d)])
        return yv[key]
    for f in range(nf): S.add_clause([x(f, d) for d in range(k)])
    S.add_clause([x(0, 0)])
    for sims, label in fam:
        for d in range(k):
            S.add_clause([y(z, d) for z in sims])
    t0 = time.time()
    r = S.solve()
    if verbose:
        print(f"  k={k}: {'SAT (no bound)' if r else 'UNSAT  => SC_strict(K) >= %d' % k}  "
              f"[{len(fam)} rigid subcomplexes, {nxt[0]} vars, {time.time()-t0:.1f}s]", flush=True)
    if r:
        model = set(l for l in S.get_model() if l > 0)
        classes = [[f for f in range(nf) if x(f, d) in model] for d in range(k)]
        return False, classes
    return True, None


# ----------------------------------------------------------------------------- stall audit
def domains_from_chains(P, chains):
    """facets on which every link of the chain is contiguous (chain = list of maps over grid points)."""
    K = P.K
    doms = []
    for ch in chains:
        D = []
        for fi, (verts, s, t) in enumerate(P.F):
            ok = True
            for i in range(len(ch) - 1):
                U = {ch[i][g] for g in verts} | {ch[i + 1][g] for g in verts}
                if not K.is_simplex(U): ok = False; break
            if ok: D.append(fi)
        doms.append(D)
    return doms


def fam_index(P, fam):
    """simplex -> list of family ids containing it."""
    idx = defaultdict(list)
    for i, (sims, label) in enumerate(fam):
        for z in sims: idx[z].append(i)
    return idx


def contains_rigid(P, facet_set, fam, idx=None, new_facet=None):
    """does <facet_set> contain some rigid member of fam?  If new_facet is given, only members using a
    simplex of new_facet are candidates (sufficient when facet_set minus new_facet contains none)."""
    fs = set(facet_set)
    present = {}
    def has(z):
        if z not in present:
            present[z] = any(f in fs for f in P.facets_containing(z))
        return present[z]
    if new_facet is None or idx is None:
        cand = range(len(fam))
    else:
        vs = sorted(P.facet_verts[new_facet]); c = set()
        sizes = sorted({len(z) for z in idx})
        for r in sizes:
            for z in itertools.combinations(vs, r):
                c.update(idx.get(frozenset(z), ()))
        cand = sorted(c)
    for i in cand:
        sims, label = fam[i]
        if all(has(z) for z in sims): return label
    return None


def describe(P, label):
    if label is None: return None
    if label[0] == 'walk':
        m, path = label[1], label[2]
        return f"{m}-walk pr1={tuple(p // P.K.nv for p in path)} pr2={tuple(p % P.K.nv for p in path)}"
    return str(label)


def audit(P, fam, chains, verbose=True, show=6):
    doms = domains_from_chains(P, chains)
    covered = set().union(*map(set, doms))
    L = sorted(set(range(P.nf)) - covered)
    k = len(doms)
    idx = fam_index(P, fam)
    print(f"  {k} domains, sizes {[len(d) for d in doms]}, covered {len(covered)}/{P.nf}, leftovers {len(L)}")
    for j, D in enumerate(doms):
        lab = contains_rigid(P, D, fam)
        if lab is not None: print(f"  !! domain {j} contains rigid {lab} -- it is NOT a domain (bug?)")
    dead = 0; table = []
    for f in L:
        row = [contains_rigid(P, D + [f], fam, idx, f) for D in doms]
        table.append((f, row))
        if all(r is not None for r in row): dead += 1
    print(f"  leftovers whose addition to EVERY domain closes a rigid subcomplex: {dead}/{len(L)}")
    open_doors = [(f, [j for j, r in enumerate(row) if r is None]) for f, row in table if any(r is None for r in row)]
    if open_doors:
        print(f"  open doors (facet -> domains with no rigid witness): {open_doors[:20]}{' ...' if len(open_doors) > 20 else ''}")
    kinds = defaultdict(int); lens = defaultdict(int)
    for f, row in table:
        for r in row:
            if r is not None:
                kinds[r[0]] += 1
                if r[0] == 'walk': lens[r[1]] += 1
    print(f"  witness kinds: {dict(kinds)}  walk lengths: {dict(lens)}")
    labL = contains_rigid(P, L, fam)
    print(f"  leftover set alone contains a rigid subcomplex: {describe(P, labL)}")
    if show:
        for f, row in table[:show]:
            verts, s, t = P.F[f]
            print(f"   leftover {f} over cell s={s} t={t}:")
            for j, r in enumerate(row):
                print(f"      + domain {j}: {describe(P, r)}")
    return table, doms, L


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['bound', 'audit', 'info'])
    ap.add_argument('complex')
    ap.add_argument('arg', nargs='?')
    ap.add_argument('--walks', type=int, default=4)
    ap.add_argument('--no-graphs', action='store_true')
    ap.add_argument('--kmin', type=int, default=2)
    a = ap.parse_args()
    C = relax.load_complexes()
    K = Complex(C[a.complex])
    print(f"== {a.complex}: |V|={K.nv}, facets {len(K.kf)}, dominated vertices: {K.dominated() or 'none'}, "
          f"empty triangles: {len(K.empty_triangles())}", flush=True)
    P = Product(K)
    print(f"   K x K: {P.nf} facets, {len(P.grid_edges)} edges")
    if a.mode == 'info':
        return
    fam = all_rigid(P, a.walks, graphs=not a.no_graphs)
    print(f"   rigid family size {len(fam)}")
    if a.mode == 'bound':
        kmax = int(a.arg)
        for k in range(a.kmin, kmax + 1):
            unsat, classes = avoid_sat(P, fam, k)
            if not unsat: break
    else:
        s = json.load(open(a.arg))
        chains = s['best_chains']
        audit(P, fam, chains)


if __name__ == '__main__':
    main()
