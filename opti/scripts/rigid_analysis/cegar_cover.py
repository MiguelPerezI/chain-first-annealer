#!/usr/bin/env python3
"""cegar_cover.py -- lazy exact cover search: does K x K admit a cover by k contiguity domains with chains of
at most L links?

Outer problem (CaDiCaL, incremental): x[f,d] = "facet f is in class d"; every facet in some class; the classes
must avoid every known rigid subcomplex (rigid.py families: rigid closed walks of length <= M and rigid graphs).
Inner oracle (domain_chain_sat.decide with selector literals): is class d a contiguity domain with <= L links?
If not, an UNSAT core of facets is extracted, deletion-minimised to a minimal non-domain facet set B, and the
clause "no class contains all of B" is added for every class.  Iterate.

SAT  -> k domains with explicit chains: written as a certificate (cover.txt / planners.txt / summary.json via
        src/gpu_sc.write_certificate) and audited with src/independent_check.py.
UNSAT-> no k-cover with chains of <= L links; the learned minimal non-domain sets are the certificate (each is
        re-checkable with domain_chain_sat.decide).  Conditional on L, like the paper's CP-SAT results.

usage: cegar_cover.py <complex> <k> <L> [--walks M] [--no-graphs] [--time T] [--outdir DIR] [--seed-summary summary.json]
"""
import argparse, itertools, json, os, random, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
OPTI = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, os.path.join(OPTI, 'src'))
import rigid, relax
from pysat.solvers import Cadical153


def chain_oracle(P, D, L, minimal_nonfaces, want_core=False):
    """exact: does facet set D admit a chain of <= L links?  Returns (sat, chain_or_None, core_or_None).
    Each facet's constraints are guarded by a selector literal so an UNSAT core names facets."""
    K = P.K; nv = K.nv; ng = nv * nv
    cnt = [0]
    def new():
        cnt[0] += 1; return cnt[0]
    x = {}
    for i in range(1, L):
        for g in range(ng):
            for v in range(nv): x[i, g, v] = new()
    S = Cadical153()
    for i in range(1, L):
        for g in range(ng):
            lits = [x[i, g, v] for v in range(nv)]
            S.add_clause(lits)
            for a, b in itertools.combinations(lits, 2): S.add_clause([-a, -b])
    def val_lit(i, g, v):
        if i == 0: return (g // nv == v)
        if i == L: return (g % nv == v)
        return x[i, g, v]
    sel = {}
    for f in D:
        sf = new(); sel[sf] = f
        verts = sorted(P.facet_verts[f])
        for i in range(L):
            p = [new() for v in range(nv)]
            fixed = set()
            for v in range(nv):
                for j in (i, i + 1):
                    for g in verts:
                        lit = val_lit(j, g, v)
                        if lit is True: fixed.add(v)
                        elif lit is False: continue
                        else: S.add_clause([-sf, -lit, p[v]])
            for v in fixed: S.add_clause([-sf, p[v]])
            for S_ in minimal_nonfaces: S.add_clause([-sf] + [-p[v] for v in S_])
    r = S.solve(assumptions=list(sel))
    if r:
        model = set(l for l in S.get_model() if l > 0)
        chain = [[g // nv for g in range(ng)]]
        for i in range(1, L):
            chain.append([next(v for v in range(nv) if x[i, g, v] in model) for g in range(ng)])
        chain.append([g % nv for g in range(ng)])
        return True, chain, None
    core = sorted(sel[abs(l)] for l in S.get_core())
    return False, None, core


def minimal_nonfaces(K):
    dim = max(len(f) for f in K.kf) - 1
    out = []
    for r in range(2, dim + 3):
        for S_ in itertools.combinations(range(K.nv), r):
            if not K.is_simplex(set(S_)) and all(K.is_simplex(set(T)) for T in itertools.combinations(S_, r - 1)):
                out.append(S_)
    return out


def minimise(P, B, L, mnf, rng):
    """deletion-minimal non-domain subset of B (chains <= L), using cores to shrink fast."""
    cur = list(B)
    changed = True
    while changed:
        changed = False
        order = list(cur); rng.shuffle(order)
        for f in order:
            trial = [g for g in cur if g != f]
            sat, _, core = chain_oracle(P, trial, L, mnf)
            if not sat:
                cur = core if core and len(core) < len(trial) else trial
                changed = True
                break
    return sorted(cur)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('complex'); ap.add_argument('k', type=int); ap.add_argument('L', type=int)
    ap.add_argument('--walks', type=int, default=4); ap.add_argument('--no-graphs', action='store_true')
    ap.add_argument('--time', type=float, default=3600); ap.add_argument('--outdir', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--learned', default=None, help='JSON file of previously learned non-domain sets to reuse')
    ap.add_argument('--partition', action='store_true', help='require disjoint classes (WLOG for existence, by heredity)')
    ap.add_argument('--cap', type=int, default=0, help='max facets per class (0 = none); a domain-size cap, valid only if a cover with such domains exists')
    a = ap.parse_args()
    C = relax.load_complexes(); K = rigid.Complex(C[a.complex]); P = rigid.Product(K)
    mnf = minimal_nonfaces(K)
    print(f"== {a.complex}: {P.nf} facets, k={a.k}, L={a.L}; minimal non-faces of K: {len(mnf)}", flush=True)
    fam = rigid.all_rigid(P, a.walks, graphs=not a.no_graphs, verbose=True)
    k = a.k; nf = P.nf
    S = Cadical153()
    x = lambda f, d: 1 + f * k + d
    nxt = [1 + nf * k]
    yv = {}
    def y(z, d):
        key = (z, d)
        if key not in yv:
            v = nxt[0]; nxt[0] += 1; yv[key] = v
            for f in P.facets_containing(z): S.add_clause([-v, -x(f, d)])
        return yv[key]
    for f in range(nf): S.add_clause([x(f, d) for d in range(k)])
    S.add_clause([x(0, 0)])
    if a.partition:
        for f in range(nf):
            for d1, d2 in itertools.combinations(range(k), 2): S.add_clause([-x(f, d1), -x(f, d2)])
    if a.cap:
        from pysat.card import CardEnc, EncType
        top = nxt[0]
        for d in range(k):
            cnf = CardEnc.atmost(lits=[x(f, d) for f in range(nf)], bound=a.cap, top_id=top, encoding=EncType.seqcounter)
            for cl in cnf.clauses: S.add_clause(cl)
            top = max(top, cnf.nv)
        nxt[0] = top + 1
    for sims, label in fam:
        for d in range(k): S.add_clause([y(z, d) for z in sims])
    learned = []
    def learn(B):
        learned.append(B)
        for d in range(k): S.add_clause([-x(f, d) for f in B])
    if a.learned:
        for B in json.load(open(a.learned)): learn(B)
    rng = random.Random(a.seed)
    t0 = time.time(); it = 0
    outdir = a.outdir or os.path.join(OPTI, 'results', f'{a.complex}_k{k}_cegar_L{a.L}')
    learned_path = os.path.join(HERE, 'certificates', f'learned_{a.complex}_k{k}_L{a.L}.json')
    while time.time() - t0 < a.time:
        it += 1
        if not S.solve():
            json.dump(learned, open(learned_path, 'w'))
            print(f"RESULT {a.complex} k={k} L={a.L}: UNSAT -- no cover by {k} domains with chains of <= {a.L} links "
                  f"({len(learned)} learned minimal non-domain sets, {it} iterations, {time.time()-t0:.0f}s) -> {learned_path}", flush=True)
            return 0
        model = set(l for l in S.get_model() if l > 0)
        classes = [[f for f in range(nf) if x(f, d) in model] for d in range(k)]
        chains = []; bad = 0
        for d, D in enumerate(classes):
            sat, chain, core = chain_oracle(P, D, a.L, mnf)
            if sat:
                chains.append(chain); continue
            bad += 1
            B = minimise(P, core, a.L, mnf, rng)
            learn(B)
            # learn a few more from the same class: remove one facet of B from D and retry (cheap extra cuts)
            D2 = [f for f in D if f != rng.choice(B)]
            for _ in range(2):
                sat2, _, core2 = chain_oracle(P, D2, a.L, mnf)
                if sat2: break
                B2 = minimise(P, core2, a.L, mnf, rng); learn(B2)
                D2 = [f for f in D2 if f != rng.choice(B2)]
        sizes = [len(B) for B in learned] or [0]
        print(f"  it {it:4d}: classes {[len(D) for D in classes]}, non-domains {bad}/{k}, learned {len(learned)} "
              f"(sizes min/med/max {min(sizes)}/{sorted(sizes)[len(sizes)//2]}/{max(sizes)})  t={time.time()-t0:.0f}s", flush=True)
        if it % 20 == 0: json.dump(learned, open(learned_path, 'w'))
        if bad == 0:
            import numpy as np, gpu_sc
            G = gpu_sc.build(a.complex)
            parts = gpu_sc.partition_from_chains(G, [[np.array(m, np.int8) for m in c] for c in chains])
            assert parts is not None
            meta = {'params': {'complex': a.complex, 'k': k, 'L': a.L, 'method': 'cegar_cover.py: lazy exact CEGAR (rigid families + exact chain oracle with learned minimal non-domain sets)'},
                    'elapsed_s': time.time() - t0, 'coverage': nf, 'best_chains': chains, 'learned_non_domain_sets': learned}
            summary = gpu_sc.write_certificate(G, parts, outdir, meta)
            print(f"RESULT {a.complex} k={k} L={a.L}: SAT -- cover by {len(parts)} domains, sizes {summary['domain_sizes']}, "
                  f"chains {summary['chain_lengths']} links; SC_strict <= {summary['sc_strict_bound']}; certificate {outdir}  ({time.time()-t0:.0f}s)", flush=True)
            return 0
    json.dump(learned, open(learned_path, 'w'))
    print(f"RESULT {a.complex} k={k} L={a.L}: TIMEOUT after {it} iterations, {len(learned)} learned sets -> {learned_path}", flush=True)
    return 2


if __name__ == '__main__':
    sys.exit(main())
