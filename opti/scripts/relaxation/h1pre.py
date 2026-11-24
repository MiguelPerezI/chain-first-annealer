"""H^1 relaxation with all short bad cycles (length 3 and 4) preloaded, then CEGAR for longer ones."""
import itertools, json, os, random, sys, time
from pysat.solvers import Cadical153
import relax, h1cegar

def short_bad_cycles(H, maxlen=4):
    nv = H.nv
    nb = {}
    eidx = H.E
    for (p, q), ei in eidx.items():
        nb.setdefault(p, {})[q] = (ei, +1)      # traversing p->q is increasing
        nb.setdefault(q, {})[p] = (ei, -1)
    def wsum(cyc):   # cyc: list of vertices (closed), returns (z2 sums list, int sum)
        z = [0] * len(H.om_z2); s = 0
        for a, b in zip(cyc, cyc[1:] + cyc[:1]):
            ei, sg = nb[a][b]
            for i, om in enumerate(H.om_z2): z[i] ^= om[ei]
            if H.om_int is not None: s += sg * H.om_int[ei]
        return z, s
    out = set()
    verts = sorted(nb)
    # triangles
    for p in verts:
        for q in nb[p]:
            if q <= p: continue
            for r in nb[p]:
                if r <= q or r not in nb[q]: continue
                z, s = wsum([p, q, r])
                if any(z) or s != 0:
                    out.add(tuple(sorted((nb[p][q][0], nb[q][r][0], nb[p][r][0]))))
    n3 = len(out)
    if maxlen >= 4:
        for p in verts:
            for r in verts:
                if r <= p: continue
                common = [q for q in nb[p] if q in nb[r]]
                for q, s_ in itertools.combinations(common, 2):
                    z, s = wsum([p, q, r, s_])
                    if any(z) or s != 0:
                        out.add(tuple(sorted({nb[p][q][0], nb[q][r][0], nb[r][s_][0], nb[s_][p][0]})))
    return sorted(out), n3

def run(kf, k, time_limit=3000, use_int=True, maxlen=4, seed=0, verbose=True, rounds=8):
    H = h1cegar.H1(kf, use_int)
    rng = random.Random(seed)
    nf, ne = H.nf, len(H.E)
    xv = lambda f, d: 1 + f * k + d
    pv = lambda e, d: 1 + nf * k + e * k + d
    S = Cadical153()
    for f in range(nf): S.add_clause([xv(f, d) for d in range(k)])
    # WLOG a partition (balanced is hereditary): at most one colour per facet
    for f in range(nf):
        for d1 in range(k):
            for d2 in range(d1 + 1, k): S.add_clause([-xv(f, d1), -xv(f, d2)])
    # value-precedence symmetry breaking: colour d may be used at facet f only if colour d-1 is used at a facet < f
    base = 1 + nf * k + ne * k
    uv = lambda f, d: base + f * k + d          # colour d is used among facets 0..f
    for d in range(k):
        S.add_clause([-uv(0, d), xv(0, d)]); S.add_clause([uv(0, d), -xv(0, d)])
        for f in range(1, nf):
            S.add_clause([-uv(f, d), uv(f - 1, d), xv(f, d)])
            S.add_clause([uv(f, d), -uv(f - 1, d)]); S.add_clause([uv(f, d), -xv(f, d)])
    for d in range(1, k):
        S.add_clause([-xv(0, d)])
        for f in range(1, nf): S.add_clause([-xv(f, d), uv(f - 1, d - 1)])
    for f in range(nf):
        for e in H.fedges[f]:
            for d in range(k): S.add_clause([-xv(f, d), pv(e, d)])
    t0 = time.time()
    pre, n3 = short_bad_cycles(H, maxlen)
    if verbose: print(f"  preloaded {len(pre)} bad cycles ({n3} triangles) in {time.time()-t0:.1f}s", flush=True)
    cycles = set(pre)
    for key in pre:
        for d in range(k): S.add_clause([-pv(e, d) for e in key])
    def add_cycle(c):
        key = tuple(sorted(set(c)))
        if key in cycles: return False
        cycles.add(key)
        for d in range(k): S.add_clause([-pv(e, d) for e in key])
        return True
    it = 0
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
        if verbose and (it % 20 == 0 or added == 0):
            print(f"  it {it} cycles {len(cycles)} (+{added}) t={time.time()-t0:.0f}s", flush=True)
        if added == 0:
            return "SAT", [[f for f in range(nf) if xv(f, d) in model] for d in range(k)]
    return "TIMEOUT", sorted(cycles)

if __name__ == "__main__":
    name, kmin, kmax = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    tl = float(sys.argv[4]) if len(sys.argv) > 4 else 3000
    maxlen = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    C = relax.load_complexes()
    for k in range(kmin, kmax + 1):
        st, res = run(C[name], k, tl, maxlen=maxlen)
        print(f"RESULT {name} k={k} {st}" + (f" cycles={len(res)}" if st != 'SAT' else ''), flush=True)
        json.dump(res, open(os.path.join(relax.HERE, f"h1pre_{name}_k{k}_{st}.json"), "w"))
        if st == "SAT": break
