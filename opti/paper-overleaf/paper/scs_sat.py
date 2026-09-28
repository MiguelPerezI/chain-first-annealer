"""CP-SAT decision procedure: does K x K admit a cover by <= k
strictly-contiguous planner domains with chains of <= L links?

A domain is a chain phi_0 = pi1 -> ... -> phi_L = pi2 of maps
grid -> V(K); facet f is covered by the domain iff every link is
1-contiguous AND simplicial on f: img(phi_i) | img(phi_{i+1}) on f lies
in one maximal facet of K (called TRIS below; for surfaces these are
triangles, for s3 the 5 tetrahedra of the 4-simplex boundary).

Modes:
  Cold (default):  solve from scratch — full CP-SAT encoding for k domains.
  Hybrid (--warm): load annealer's best chains, freeze (k - free) domains,
                   let CP-SAT re-solve the freed domains to cover the residual.
                   Massive variable reduction for large-k stalls.

Usage:
  # Cold — RP² at k=3 (600 facets, should take minutes to hours)
  python scs_sat.py --complex rp2 --k 3 --L 8 --timeout 3600 --witness

  # Hybrid — use annealer's k=5 stall, free 2 worst domains for CP-SAT
  python scs_sat.py --complex rp2 --k 5 --L 8 --timeout 3600 --witness \
      --warm ../../results/rp2_k5/summary.json --free 2

  # Torus — might need longer chains and more time
  python scs_sat.py --complex torus --k 3 --L 10 --timeout 7200
"""
import argparse
import json
import sys
import time
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "opti"))
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))
    from opti.src import gpu_sc as _gs
    gpu_sc = _gs
except Exception:
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "..", "src"))
        import gpu_sc
    except Exception:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "src"))
        import gpu_sc

from ortools.sat.python import cp_model


class VerboseCallback(cp_model.CpSolverSolutionCallback):
    """Prints progress each time CP-SAT finds a feasible solution."""
    def __init__(self, NF, k, x):
        super().__init__()
        self._NF = NF
        self._k = k
        self._x = x
        self._count = 0
        self._t0 = time.time()

    def on_solution_callback(self):
        self._count += 1
        dt = time.time() - self._t0
        sizes = []
        for d in range(self._k):
            s = sum(self.Value(self._x[f, d]) for f in range(self._NF))
            sizes.append(s)
        total = len(set(f for d in range(self._k)
                        for f in range(self._NF) if self.Value(self._x[f, d])))
        print(f"  [SAT] solution #{self._count}  {dt:7.1f}s  "
              f"covered {total}/{self._NF}  domains {sizes}",
              flush=True)


def build_complex(name):
    P = gpu_sc.build(name)
    NV = P.nv
    NG = P.ngrid
    NF = P.nf
    FAC = [sorted(int(v) for v in P.facet_verts[i][:P.facet_nv[i]])
           for i in range(NF)]
    TRIS = [tuple(sorted(t)) for t in P.kfacets]
    PI1 = [v // NV for v in range(NG)]
    PI2 = [v % NV for v in range(NG)]
    return NV, NG, NF, FAC, TRIS, PI1, PI2, P


def _recompute_domains(chains, NV, NF, FAC, TRIS):
    """Given chains [k][c+1][ngrid], recompute which facets each covers."""
    tri_set = [set(t) for t in TRIS]
    domains = []
    for chain in chains:
        clen = len(chain)
        dom = []
        for f in range(NF):
            verts = FAC[f]
            ok = True
            for link in range(clen - 1):
                img_a = {chain[link][v] for v in verts}
                img_b = {chain[link + 1][v] for v in verts}
                union = img_a | img_b
                if not any(union <= ts for ts in tri_set):
                    ok = False
                    break
            if ok:
                dom.append(f)
        domains.append(dom)
    return domains


def load_warm(path, NV, NG, NF, FAC, TRIS):
    """Load annealer summary.json → (chains, domains, covered_set).

    Handles both complete covers (have 'domains' key) and partial stalls
    (only have 'best_chains' + 'chain_domain_sizes').
    """
    data = json.load(open(path))
    raw_chains = data["best_chains"]
    k_warm = len(raw_chains)

    # Complete summaries store "domains" sorted by size, not in best_chains
    # order, so pairing them by index freezes a domain under the wrong chain.
    # Recompute each chain's own domain in both cases.
    print(f"  warm summary (status={data.get('status','complete')}), "
          f"recomputing domains from chains...", flush=True)
    raw_domains = _recompute_domains(raw_chains, NV, NF, FAC, TRIS)

    covered = set()
    for dom in raw_domains:
        covered.update(dom)
    print(f"  warm-start: {k_warm} domains, {len(covered)}/{NF} covered, "
          f"{NF - len(covered)} residual", flush=True)
    sizes = [len(d) for d in raw_domains]
    for i, s in enumerate(sizes):
        print(f"    domain {i}: {s} facets", flush=True)
    order = sorted(range(k_warm), key=lambda i: sizes[i], reverse=True)
    return raw_chains, raw_domains, covered, order


def minimal_non_faces(NV, TRIS):
    """Minimal vertex sets of K that are not simplices (every proper subset is)."""
    import itertools
    faces = {frozenset(s) for T in TRIS for r in range(1, len(T) + 1)
             for s in itertools.combinations(T, r)}
    out = []
    for r in range(2, NV + 1):
        for s in itertools.combinations(range(NV), r):
            if frozenset(s) not in faces and all(
                    frozenset(t) in faces for t in itertools.combinations(s, r - 1)):
                out.append(s)
    return out


class MiddleLinks:
    """1-contiguity of the middle links m_i ~ m_{i+1} (1 <= i <= L-2) on facet f,
    enforced by x[f, d].

    encoding "y" (default): one Boolean per maximal simplex T of K, both maps
        inside T on every vertex of f (the original scs_sat.py model).
    encoding "factored": one-hot b[d,i,v,c] of the middle maps, colour
        indicators h[c] ("c appears on f in m_i or m_{i+1}", forced up by
        implications) and one clause per minimal non-face N of K:
        not x[f,d] or not all h[c], c in N. Equisatisfiable with "y";
        about half the peak RAM on RP2 k=5 L=4 (rp2_klein_cp_sat_results.md).
    """

    def __init__(self, model, m, x, L, NV, TRIS, encoding="y"):
        if encoding not in ("y", "factored"):
            raise ValueError(f"unknown encoding {encoding!r}")
        self.model, self.m, self.x, self.L, self.NV = model, m, x, L, NV
        self.TRIS, self.encoding = TRIS, encoding
        self.b = {}
        self.nonfaces = minimal_non_faces(NV, TRIS) if encoding == "factored" else None

    def _onehot(self, d, i, v):
        key = (d, i, v)
        if key not in self.b:
            bs = [self.model.NewBoolVar(f"b{d}_{i}_{v}_{c}") for c in range(self.NV)]
            self.model.AddExactlyOne(bs)
            for c in range(self.NV):
                self.model.Add(self.m[d, i, v] == c).OnlyEnforceIf(bs[c])
            self.b[key] = bs
        return self.b[key]

    def add(self, d, f, verts):
        model, m, x = self.model, self.m, self.x
        for i in range(1, self.L - 1):
            if self.encoding == "y":
                ys = [model.NewBoolVar(f"y{d}_{f}_{i}_{t}") for t in range(len(self.TRIS))]
                model.Add(sum(ys) >= x[f, d])
                for t, T in enumerate(self.TRIS):
                    for v in verts:
                        model.AddLinearExpressionInDomain(
                            m[d, i, v], cp_model.Domain.FromValues(T)
                        ).OnlyEnforceIf(ys[t])
                        model.AddLinearExpressionInDomain(
                            m[d, i + 1, v], cp_model.Domain.FromValues(T)
                        ).OnlyEnforceIf(ys[t])
                continue
            h = [model.NewBoolVar(f"h{d}_{f}_{i}_{c}") for c in range(self.NV)]
            for v in verts:
                for c, (a, b2) in enumerate(zip(self._onehot(d, i, v),
                                                self._onehot(d, i + 1, v))):
                    model.AddImplication(a, h[c])
                    model.AddImplication(b2, h[c])
            for N in self.nonfaces:
                model.AddBoolOr([x[f, d].Not()] + [h[c].Not() for c in N])


def solve_cold(k, L, timeout_s, workers, NV, NG, NF, FAC, TRIS, PI1, PI2,
               want_solution=False, outdir=None, fix_domain0=None, encoding="y"):
    """Full CP-SAT solve from scratch.

    fix_domain0: optional list of facet ids forced into domain 0 (x[f,0]=1).
    Used by the orbit-decomposition cases; None leaves the model unchanged."""
    assert L >= 2, "L must be >= 2 (chain of >= 2 links; L=1 is not encoded)"
    NT = len(TRIS)
    model = cp_model.CpModel()
    print(f"  building model: {k} domains × {L} links × {NG} grid vertices...",
          flush=True)
    m = {}
    for d in range(k):
        for i in range(L + 1):
            for v in range(NG):
                if i == 0:
                    m[d, i, v] = model.NewConstant(PI1[v])
                elif i == L:
                    m[d, i, v] = model.NewConstant(PI2[v])
                else:
                    m[d, i, v] = model.NewIntVar(0, NV - 1, f"m{d}_{i}_{v}")
    x = {}
    for f in range(NF):
        for d in range(k):
            x[f, d] = model.NewBoolVar(f"x{f}_{d}")
        model.Add(sum(x[f, d] for d in range(k)) >= 1)
    for f in (fix_domain0 or []):
        model.Add(x[int(f), 0] == 1)
    if fix_domain0:
        print(f"  domain 0 fixed on {len(fix_domain0)} facets", flush=True)
    print(f"  adding contiguity constraints ({k}×{NF}×{L-2}×{NT} y-vars)...",
          flush=True)
    links = MiddleLinks(model, m, x, L, NV, TRIS, encoding)
    print(f"  middle-link encoding: {encoding}", flush=True)
    for d in range(k):
        for f in range(NF):
            verts = FAC[f]
            sig = sorted({g // NV for g in FAC[f]})
            tau = sorted({g % NV for g in FAC[f]})
            for v in verts:
                model.AddLinearExpressionInDomain(
                    m[d, 1, v], cp_model.Domain.FromValues(sig)
                ).OnlyEnforceIf(x[f, d])
                model.AddLinearExpressionInDomain(
                    m[d, L - 1, v], cp_model.Domain.FromValues(tau)
                ).OnlyEnforceIf(x[f, d])
            links.add(d, f, verts)
    for d in range(k - 1):
        model.Add(sum(x[f, d] for f in range(NF))
                  >= sum(x[f, d + 1] for f in range(NF)))

    n_vars = len(model.proto.variables)
    n_cons = len(model.proto.constraints)
    print(f"  model built: {n_vars} vars, {n_cons} constraints", flush=True)
    print(f"  launching solver with {workers} workers, timeout {timeout_s}s...",
          flush=True)

    cb = VerboseCallback(NF, k, x)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_s
    solver.parameters.num_workers = workers
    solver.parameters.log_search_progress = True
    t0 = time.time()
    status = solver.Solve(model, cb)
    dt = time.time() - t0
    st = solver.StatusName(status)
    print(f"k={k} L={L}: {st} in {dt:.1f}s "
          f"(vars {len(model.proto.variables)}, "
          f"cons {len(model.proto.constraints)})")
    _report_solution(st, k, L, NG, NF, m, x, solver, want_solution, outdir)
    return st, dt


def solve_hybrid(k, L, timeout_s, workers, NV, NG, NF, FAC, TRIS, PI1, PI2,
                 warm_path, n_free,
                 want_solution=False, outdir=None, encoding="y"):
    """Hybrid solve: freeze (k - n_free) best annealer domains, free the rest.

    Frozen domains have their chains fixed as constants — zero CP-SAT variables.
    Only the freed domains get m/x/y variables.  Coverage constraint: every facet
    must be covered by either a frozen domain OR a freed domain.
    """
    assert L >= 2, "L must be >= 2 (chain of >= 2 links; L=1 is not encoded)"
    NT = len(TRIS)
    raw_chains, raw_domains, covered, order = load_warm(
        warm_path, NV, NG, NF, FAC, TRIS)
    k_warm = len(raw_domains)

    n_frozen = k - n_free
    if n_frozen > k_warm:
        print(f"  WARNING: asked to freeze {n_frozen} but warm file has {k_warm} "
              f"domains; freezing all {k_warm}")
        n_frozen = k_warm
        n_free = k - n_frozen

    frozen_ids = order[:n_frozen]
    frozen_covered = set()
    for idx in frozen_ids:
        frozen_covered.update(raw_domains[idx])
    residual = [f for f in range(NF) if f not in frozen_covered]

    print(f"  freezing {n_frozen} domains ({len(frozen_covered)} facets covered)")
    print(f"  freeing {n_free} domains to cover {len(residual)} residual facets")
    if not residual:
        print("  nothing to solve — frozen domains already cover everything!")
        return "FEASIBLE", 0.0

    model = cp_model.CpModel()
    print(f"  building hybrid model: {n_frozen} frozen + {n_free} free domains, "
          f"L={L}...", flush=True)

    # --- Frozen domains: chains become constants, coverage is automatic ---
    # We still need m/x variables for frozen domains so the output format
    # is uniform, but all m's are constants and x's are forced.
    m = {}
    x = {}
    for di, frozen_idx in enumerate(frozen_ids):
        chain = raw_chains[frozen_idx]
        chain_len = len(chain) - 1
        if chain_len > L:
            raise ValueError(
                f"warm chain {frozen_idx} has {chain_len} links > L={L}; "
                f"truncating would break m[d,L]=pi2. Re-run with --L {chain_len}.")
        for i in range(L + 1):
            src_i = min(i, chain_len)   # pad with repeated pi2 at the tail
            for v in range(NG):
                m[di, i, v] = model.NewConstant(int(chain[src_i][v]))
        for f in range(NF):
            if f in set(raw_domains[frozen_idx]):
                x[f, di] = model.NewConstant(1)
            else:
                x[f, di] = model.NewConstant(0)

    # --- Free domains: full CP-SAT variables ---
    free_start = n_frozen
    for fd in range(n_free):
        d = free_start + fd
        for i in range(L + 1):
            for v in range(NG):
                if i == 0:
                    m[d, i, v] = model.NewConstant(PI1[v])
                elif i == L:
                    m[d, i, v] = model.NewConstant(PI2[v])
                else:
                    m[d, i, v] = model.NewIntVar(0, NV - 1, f"m{d}_{i}_{v}")
        for f in range(NF):
            x[f, d] = model.NewBoolVar(f"x{f}_{d}")

    # --- Coverage: every facet covered by at least one domain ---
    for f in range(NF):
        all_x = [x[f, d] for d in range(k)]
        model.Add(sum(all_x) >= 1)

    # --- Contiguity constraints for free domains only ---
    links = MiddleLinks(model, m, x, L, NV, TRIS, encoding)
    print(f"  middle-link encoding: {encoding}", flush=True)
    for fd in range(n_free):
        d = free_start + fd
        for f in range(NF):
            verts = FAC[f]
            sig = sorted({g // NV for g in FAC[f]})
            tau = sorted({g % NV for g in FAC[f]})
            for v in verts:
                model.AddLinearExpressionInDomain(
                    m[d, 1, v], cp_model.Domain.FromValues(sig)
                ).OnlyEnforceIf(x[f, d])
                model.AddLinearExpressionInDomain(
                    m[d, L - 1, v], cp_model.Domain.FromValues(tau)
                ).OnlyEnforceIf(x[f, d])
            links.add(d, f, verts)

    # Symmetry breaking among free domains only
    for fd in range(n_free - 1):
        d1 = free_start + fd
        d2 = free_start + fd + 1
        model.Add(sum(x[f, d1] for f in range(NF))
                  >= sum(x[f, d2] for f in range(NF)))

    n_vars = len(model.proto.variables)
    n_cons = len(model.proto.constraints)
    print(f"  model built: {n_vars} vars, {n_cons} constraints", flush=True)
    print(f"  launching solver with {workers} workers, timeout {timeout_s}s...",
          flush=True)

    cb = VerboseCallback(NF, k, x)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_s
    solver.parameters.num_workers = workers
    solver.parameters.log_search_progress = True
    t0 = time.time()
    status = solver.Solve(model, cb)
    dt = time.time() - t0
    st = solver.StatusName(status)

    n_vars = len(model.proto.variables)
    n_cons = len(model.proto.constraints)
    print(f"hybrid k={k} (frozen={n_frozen}, free={n_free}) L={L}: "
          f"{st} in {dt:.1f}s (vars {n_vars}, cons {n_cons})")
    _report_solution(st, k, L, NG, NF, m, x, solver, want_solution, outdir)
    return st, dt


def _report_solution(st, k, L, NG, NF, m, x, solver, want_solution, outdir):
    if st not in ("OPTIMAL", "FEASIBLE"):
        return
    chains = []
    for d in range(k):
        ch = [[solver.Value(m[d, i, v]) for v in range(NG)]
              for i in range(L + 1)]
        chains.append(ch)
    doms = [[f for f in range(NF) if solver.Value(x[f, d]) == 1]
            for d in range(k)]
    for d in range(k):
        print(f"  domain {d}: {len(doms[d])} facets")
    total = len(set(f for dom in doms for f in dom))
    print(f"  total covered: {total}/{NF}")
    if want_solution:
        dest = outdir or "/tmp"
        os.makedirs(dest, exist_ok=True)
        path = os.path.join(dest, f"scs_witness_k{k}_L{L}.json")
        json.dump({"chains": chains, "domains": doms}, open(path, "w"))
        print(f"  witness written to {path}")


if __name__ == "__main__":
    import multiprocessing

    parser = argparse.ArgumentParser(
        description="CP-SAT cover decision for K×K (cold or hybrid warm-start)")
    parser.add_argument("--complex", default="mobius",
                        choices=["mobius", "rp2", "torus", "klein", "s1xw",
                                 "circle", "s2", "s3", "s4", "s5", "s6", "s7",
                                 "cube", "dodeca"],
                        help="base simplicial complex K")
    parser.add_argument("--k", type=int, required=True,
                        help="number of domains")
    parser.add_argument("--L", type=int, required=True,
                        help="chain length (number of links)")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="timeout in seconds (default 3600)")
    parser.add_argument("--workers", type=int, default=0,
                        help="CP-SAT workers (0 = auto)")
    parser.add_argument("--outdir", default=None,
                        help="directory for witness JSON output")
    parser.add_argument("--witness", action="store_true",
                        help="save witness JSON on SAT")
    parser.add_argument("--warm", default=None,
                        help="path to annealer summary.json for hybrid mode")
    parser.add_argument("--free", type=int, default=0,
                        help="number of domains to free in hybrid mode "
                             "(0 = free all k, i.e. cold solve with hints)")
    parser.add_argument("--fix-domain0", default=None,
                        help="JSON list of facet ids forced into domain 0 (cold mode)")
    parser.add_argument("--encoding", default="y", choices=["y", "factored"],
                        help="middle-link contiguity encoding (factored: ~half the RAM)")
    args = parser.parse_args()
    fix_domain0 = json.load(open(args.fix_domain0)) if args.fix_domain0 else None

    workers = args.workers if args.workers > 0 else min(multiprocessing.cpu_count(), 16)
    NV, NG, NF, FAC, TRIS, PI1, PI2, P = build_complex(args.complex)

    print(f"{P.name}: {NF} facets, {NG} grid vertices, "
          f"{len(TRIS)} maximal facets of K, NV={NV}; workers={workers}")

    if args.warm and args.free > 0:
        print(f"HYBRID mode: warm={args.warm}, free={args.free}/{args.k}")
        solve_hybrid(args.k, args.L, args.timeout, workers,
                     NV, NG, NF, FAC, TRIS, PI1, PI2,
                     args.warm, args.free,
                     want_solution=args.witness, outdir=args.outdir,
                     encoding=args.encoding)
    else:
        print(f"COLD mode: k={args.k}, L={args.L}, timeout={args.timeout}s")
        solve_cold(args.k, args.L, args.timeout, workers,
                   NV, NG, NF, FAC, TRIS, PI1, PI2,
                   want_solution=args.witness, outdir=args.outdir,
                   fix_domain0=fix_domain0, encoding=args.encoding)
