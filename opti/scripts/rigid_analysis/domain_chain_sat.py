#!/usr/bin/env python3
"""domain_chain_sat.py -- exact decision: does a given facet set D of K x K admit a contiguity chain
pi1 = phi_0 ~ phi_1 ~ ... ~ phi_L = pi2 of at most L links (as maps D -> K)?

Encoding (CNF, CaDiCaL via pysat): one-hot x[i,g,v] for interior maps i = 1..L-1, grid points g, values v.
For every facet f in D and link i (maps i, i+1) the union of the 14 images must be a face of K:
presence variables p[f,i,v] with x[i,g,v] -> p[f,i,v] (g in f) and x[i+1,g,v] -> p[f,i,v]; then forbid every
minimal non-face: non-edges {u,v}, empty triangles {a,b,d}, and (for 2-dimensional K) more than 3 values present.
Padding: a chain shorter than L is padded by pi2, so "no chain with exactly L links" = "no chain with <= L links".
"""
import itertools, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
from pysat.solvers import Cadical153
from pysat.card import CardEnc, EncType

sys.path.insert(0, HERE)
import rigid, relax


def decide(P, D, L, verbose=True, return_chain=False, budget=0):
    K = P.K; nv = K.nv; ng = nv * nv
    dim = max(len(f) for f in K.kf) - 1
    # minimal non-faces of K (every proper subset is a face): the forbidden value sets
    minimal_nonfaces = []
    for r in range(2, dim + 3):
        for S_ in itertools.combinations(range(nv), r):
            if not K.is_simplex(set(S_)) and all(K.is_simplex(set(T)) for T in itertools.combinations(S_, r - 1)):
                minimal_nonfaces.append(S_)
    cnt = [0]
    def new():
        cnt[0] += 1; return cnt[0]
    # maps 0..L ; map 0 = pi1, map L = pi2 fixed
    x = {}
    for i in range(1, L):
        for g in range(ng):
            for v in range(nv): x[i, g, v] = new()
    S = Cadical153()
    ncl = 0
    for i in range(1, L):
        for g in range(ng):
            lits = [x[i, g, v] for v in range(nv)]
            S.add_clause(lits); ncl += 1
            for a, b in itertools.combinations(lits, 2): S.add_clause([-a, -b]); ncl += 1
    def val_lit(i, g, v):
        """literal (or True/False) for 'map i sends g to v'."""
        if i == 0: return (g // nv == v)
        if i == L: return (g % nv == v)
        return x[i, g, v]
    for f in D:
        verts = sorted(P.facet_verts[f])
        for i in range(L):
            p = [new() for v in range(nv)]
            fixed_present = set()
            for v in range(nv):
                for j in (i, i + 1):
                    for g in verts:
                        lit = val_lit(j, g, v)
                        if lit is True: fixed_present.add(v)
                        elif lit is False: continue
                        else: S.add_clause([-lit, p[v]]); ncl += 1
            for v in fixed_present: S.add_clause([p[v]]); ncl += 1
            for S_ in minimal_nonfaces: S.add_clause([-p[v] for v in S_]); ncl += 1
    t0 = time.time()
    if budget:
        S.conf_budget(budget); r = S.solve_limited()
    else:
        r = S.solve()
    if verbose:
        print(f"    |D|={len(D)} L={L}: {'SAT (chain exists)' if r else ('UNKNOWN (budget)' if r is None else 'UNSAT (no chain with <= %d links)' % L)}  "
              f"[{cnt[0]} vars, {ncl} clauses, {time.time()-t0:.1f}s]", flush=True)
    if r and return_chain:
        model = set(l for l in S.get_model() if l > 0)
        chain = [[g // nv for g in range(ng)]]
        for i in range(1, L):
            chain.append([next(v for v in range(nv) if x[i, g, v] in model) for g in range(ng)])
        chain.append([g % nv for g in range(ng)])
        return r, chain
    return r, None


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('complex'); ap.add_argument('summary'); ap.add_argument('--L', type=int, default=6)
    ap.add_argument('--domain', type=int, default=None, help='index of the chain/domain to test (default: all)')
    ap.add_argument('--add', type=int, nargs='*', default=[], help='facet ids to add to the domain')
    a = ap.parse_args()
    C = relax.load_complexes(); K = rigid.Complex(C[a.complex]); P = rigid.Product(K)
    s = json.load(open(a.summary))
    doms = rigid.domains_from_chains(P, s['best_chains'])
    idxs = [a.domain] if a.domain is not None else range(len(doms))
    for j in idxs:
        print(f"  domain {j} (recorded chain {len(s['best_chains'][j]) - 1} links)" + (f" + {a.add}" if a.add else ''))
        decide(P, sorted(set(doms[j]) | set(a.add)), a.L)
