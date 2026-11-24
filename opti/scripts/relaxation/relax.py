"""Z2-cohomological relaxation of strict simplicial complexity at subdivision level 0.

A subcomplex D of the staircase product K x K is *balanced* if, for every class u in H^j(K;Z2), j>=1,
the cocycle pr1^*u + pr2^*u restricts to a coboundary on D.  Every contiguity domain is balanced
(contiguous maps are homotopic), so
        (least k with k balanced subcomplexes covering K x K)  <=  SC_strict(K) + 1.
The covering problem is solved by CEGAR: a SAT solver proposes k facet sets, a GF(2) elimination
checks each, and minimal unbalanced facet sets ("bad sets") are learned as clauses.  An UNSAT answer
is a proof, and the learned bad sets are an independently checkable certificate."""
import ctypes, itertools, json, os, random, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_SO = os.path.join(HERE, "libbal.so")
if not os.path.exists(_SO):
    subprocess.check_call(["g++", "-O2", "-shared", "-fPIC", "-o", _SO, os.path.join(HERE, "bal.cpp")])
LIB = ctypes.CDLL(_SO)
LIB.add_class.restype = ctypes.c_int
LIB.balanced.restype = ctypes.c_int


# ---------------------------------------------------------------- complexes and the staircase product
def staircase(kf, nv):
    """Facets of the ordered product K x K: monotone lattice paths in s x t for every pair of facets."""
    out = []
    for s in kf:
        for t in kf:
            ds, dt = len(s) - 1, len(t) - 1
            for steps in itertools.combinations(range(ds + dt), ds):
                st = set(steps)
                i = j = 0
                path = [(s[0], t[0])]
                for pos in range(ds + dt):
                    if pos in st: i += 1
                    else: j += 1
                    path.append((s[i], t[j]))
                out.append((tuple(v * nv + w for v, w in path), s, t))
    return out


def faces_by_dim(kf):
    S = {}
    for f in kf:
        for r in range(1, len(f) + 1):
            for s in itertools.combinations(sorted(f), r):
                S.setdefault(r - 1, set()).add(s)
    return {d: sorted(v) for d, v in S.items()}


def staircase_simplices(Fst, j):
    S = set()
    for verts, s, t in Fst:
        for c in itertools.combinations(verts, j + 1): S.add(c)
    return sorted(S)


# ---------------------------------------------------------------- GF(2) linear algebra / cohomology of K
def _echelon_add(red, v):
    x = v
    for c, pr in red:
        if x >> c & 1: x ^= pr
    if x:
        c = x.bit_length() - 1
        red = [(cc, (pp ^ x) if (pp >> c & 1) else pp) for cc, pp in red]
        red.append((c, x))
    return red, x


def gf2_nullspace(rows, ncols):
    red = []
    for r in rows: red, _ = _echelon_add(red, r)
    pivcols = {c for c, _ in red}
    basis = []
    for fcol in range(ncols):
        if fcol in pivcols: continue
        v = 1 << fcol
        for c, pr in red:
            if pr >> fcol & 1: v |= 1 << c
        basis.append(v)
    return basis


def cohomology_basis(kf):
    """{j: [cocycle as dict simplex->0/1]} : a basis of H^j(K;Z2) for j >= 1."""
    F = faces_by_dim(kf)
    top = max(F)
    idx = {d: {s: i for i, s in enumerate(F[d])} for d in F}
    out = {}
    for j in range(1, top + 1):
        n = len(F[j])
        rows = []
        if j + 1 in F:
            for t in F[j + 1]:
                m = 0
                for s in itertools.combinations(t, j + 1): m |= 1 << idx[j][s]
                rows.append(m)
        Z = gf2_nullspace(rows, n)
        red = []
        for s0 in F[j - 1]:
            m = 0
            for t in F[j]:
                if set(s0) <= set(t): m |= 1 << idx[j][t]
            red, _ = _echelon_add(red, m)
        classes = []
        for z in Z:
            red, r = _echelon_add(red, z)
            if r: classes.append(z)
        out[j] = [{s: (z >> idx[j][s]) & 1 for s in F[j]} for z in classes]
    return out


# ---------------------------------------------------------------- the relaxation object
class Relax:
    def __init__(self, kf, classes=None):
        self.kf = [tuple(sorted(f)) for f in kf]
        self.nv = 1 + max(v for f in self.kf for v in f)
        self.F = staircase(self.kf, self.nv)
        self.nf = len(self.F)
        self.H = cohomology_basis(self.kf)
        self.cls = []
        nv = self.nv
        for j, cl in self.H.items():
            for ci, cobj in enumerate(cl):
                if classes is not None and (j, ci) not in classes: continue
                sj = staircase_simplices(self.F, j)
                sm = staircase_simplices(self.F, j - 1)
                jid = {s: i for i, s in enumerate(sj)}
                mid = {s: i for i, s in enumerate(sm)}

                def pull(z, coord):
                    vs = [p // nv if coord == 0 else p % nv for p in z]
                    if len(set(vs)) < len(vs): return 0
                    return cobj[tuple(sorted(vs))]
                omega = np.array([(pull(z, 0) + pull(z, 1)) % 2 for z in sj], dtype=np.uint8)
                bd = np.array([[mid[z[:i] + z[i + 1:]] for i in range(j + 1)] for z in sj], dtype=np.int32).reshape(-1)
                off, fs = [0], []
                for verts, s, t in self.F:
                    for z in itertools.combinations(verts, j + 1): fs.append(jid[z])
                    off.append(len(fs))
                off = np.array(off, dtype=np.int32); fs = np.array(fs, dtype=np.int32)
                cid = LIB.add_class(j, len(sj), len(sm), omega.ctypes.data_as(ctypes.c_void_p),
                                    bd.ctypes.data_as(ctypes.c_void_p), self.nf,
                                    off.ctypes.data_as(ctypes.c_void_p), fs.ctypes.data_as(ctypes.c_void_p))
                self.cls.append((j, ci, cid))
        self.calls = 0

    def balanced(self, facets):
        """facets: iterable of facet indices (or a uint8 mask of length nf)."""
        self.calls += 1
        if isinstance(facets, np.ndarray) and facets.dtype == np.uint8 and len(facets) == self.nf:
            mask = facets
        else:
            mask = np.zeros(self.nf, dtype=np.uint8); mask[list(facets)] = 1
        return all(LIB.balanced(cid, mask.ctypes.data_as(ctypes.c_void_p), self.nf) for (_, _, cid) in self.cls)

    def minimize(self, facets, rng):
        """Deletion-minimal unbalanced subset of an unbalanced facet set."""
        cur = list(facets)
        order = list(cur); rng.shuffle(order)
        mask = np.zeros(self.nf, dtype=np.uint8); mask[cur] = 1
        for f in order:
            mask[f] = 0
            if self.balanced(mask): mask[f] = 1
            else: cur.remove(f)
        return sorted(cur)


def load_complexes(path=None):
    path = path or os.path.join(HERE, "complexes.json")
    return {k: [tuple(f) for f in v] for k, v in json.load(open(path)).items()}


# ---------------------------------------------------------------- CEGAR covering search
def cegar(R, k, time_limit=3600, seed=0, verbose=True, bad_init=None, rounds=6, save=None):
    from pysat.solvers import Cadical153
    rng = random.Random(seed)
    nf = R.nf
    var = lambda f, d: 1 + f * k + d
    S = Cadical153()
    for f in range(nf): S.add_clause([var(f, d) for d in range(k)])
    S.add_clause([var(0, 0)])          # symmetry breaking: domains are interchangeable
    bad = []

    def add_bad(B):
        bad.append(B)
        for d in range(k): S.add_clause([-var(f, d) for f in B])
    for B in (bad_init or []): add_bad(B)
    t0 = time.time(); it = 0
    while time.time() - t0 < time_limit:
        it += 1
        if not S.solve(): return "UNSAT", bad
        model = set(l for l in S.get_model() if l > 0)
        added = 0
        for d in range(k):
            D = [f for f in range(nf) if var(f, d) in model]
            mask = np.zeros(nf, dtype=np.uint8); mask[D] = 1
            r = 0
            while D and r < rounds and not R.balanced(mask):
                B = R.minimize(D, rng)
                add_bad(B); added += 1; r += 1
                g = rng.choice(B); D.remove(g); mask[g] = 0
        if verbose and (it % 25 == 0 or added == 0):
            sz = [len(b) for b in bad] or [0]
            print(f"  it {it:5d} bad {len(bad):6d} (+{added}) size min/avg/max {min(sz)}/{sum(sz)/len(sz):.1f}/{max(sz)}"
                  f"  t={time.time()-t0:.0f}s checks={R.calls}", flush=True)
        if added == 0:
            return "SAT", [[f for f in range(nf) if var(f, d) in model] for d in range(k)]
        if save and it % 50 == 0: json.dump(bad, open(save, "w"))
    return "TIMEOUT", bad


if __name__ == "__main__":
    name, k = sys.argv[1], int(sys.argv[2])
    tl = float(sys.argv[3]) if len(sys.argv) > 3 else 3600
    C = load_complexes()
    R = Relax(C[name])
    print(name, "facets", R.nf, "H dims", {j: len(v) for j, v in R.H.items()}, flush=True)
    st, res = cegar(R, k, tl, save=os.path.join(HERE, f"out_{name}_k{k}.partial.json"))
    print(name, f"k={k}", st, flush=True)
    json.dump(res, open(os.path.join(HERE, f"out_{name}_k{k}_{st}.json"), "w"))
