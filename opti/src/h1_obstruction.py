#!/usr/bin/env python3
"""h1_obstruction.py — necessary condition for a facet set J of K x K to be a
contiguity domain: pi1|J and pi2|J are homotopic, hence induce the same map
H_1(|J|) -> H_1(|K|).  Computed over GF(2): if (pi1 - pi2)_* != 0 on some
cycle of J, J is NOT a domain (no chain exists, whatever the budget)."""
import sys, json, itertools, numpy as np
# (gpu_sc is imported only in __main__: the test itself takes the complex from
#  the caller, so wedge_paper_check.py can reuse h1_map_obstruction with its
#  own stand-in P)

def gf2_rank(M):
    M = M.copy() % 2; r = 0; rows, cols = M.shape
    for c in range(cols):
        piv = np.nonzero(M[r:, c])[0]
        if len(piv) == 0: continue
        p = r + piv[0]; M[[r, p]] = M[[p, r]]
        others = np.nonzero(M[:, c])[0]; others = others[others != r]
        M[others] ^= M[r]; r += 1
        if r == rows: break
    return r

def gf2_nullspace(M):
    """basis of {x : M x = 0} over GF(2), columns of the returned matrix."""
    M = M.copy() % 2; rows, cols = M.shape; pivcols = []; r = 0
    for c in range(cols):
        piv = np.nonzero(M[r:, c])[0]
        if len(piv) == 0: continue
        p = r + piv[0]; M[[r, p]] = M[[p, r]]
        others = np.nonzero(M[:, c])[0]; others = others[others != r]
        M[others] ^= M[r]; pivcols.append(c); r += 1
        if r == rows: break
    free = [c for c in range(cols) if c not in pivcols]
    basis = []
    for f in free:
        x = np.zeros(cols, np.uint8); x[f] = 1
        for i, pc in enumerate(pivcols):
            if M[i, f]: x[pc] = 1
        basis.append(x)
    return np.array(basis, np.uint8).T if basis else np.zeros((cols, 0), np.uint8)

def simplices(facets, dim):
    S = set()
    for f in facets:
        for s in itertools.combinations(sorted(f), dim + 1): S.add(s)
    return sorted(S)

def boundary(hi, lo):
    idx = {s: i for i, s in enumerate(lo)}
    B = np.zeros((len(lo), len(hi)), np.uint8)
    for j, s in enumerate(hi):
        for k in range(len(s)):
            B[idx[s[:k] + s[k+1:]], j] = 1
    return B

def h1_map_obstruction(P, J):
    """returns (dim H1(J;Z2), rank of (pi1-pi2)_* : H1(J) -> H1(K;Z2))"""
    F = [P.facets[f] for f in J]
    V1 = simplices(F, 1); V2 = simplices(F, 2)
    d1 = boundary(V1, simplices(F, 0)); d2 = boundary(V2, V1)
    Z = gf2_nullspace(d1)                       # cycles (columns)
    # K: edges and triangles
    KE = simplices(P.kfacets, 1); KT = simplices(P.kfacets, 2); KV = simplices(P.kfacets, 0)
    ke = {e: i for i, e in enumerate(KE)}
    dK1 = boundary(KE, KV); dK2 = boundary(KT, KE)
    def push(z, proj):                          # chain map on 1-chains, GF(2)
        out = np.zeros(len(KE), np.uint8)
        for i in np.nonzero(z)[0]:
            a, b = V1[i]; pa, pb = proj(a), proj(b)
            if pa != pb: out[ke[tuple(sorted((pa, pb)))]] ^= 1
        return out
    nv = P.nv
    diffs = []
    for c in range(Z.shape[1]):
        z = Z[:, c]
        diffs.append(push(z, lambda g: g // nv) ^ push(z, lambda g: g % nv))
    D = np.array(diffs, np.uint8).T if diffs else np.zeros((len(KE), 0), np.uint8)
    # rank of D modulo boundaries of K: rank([dK2 | D]) - rank(dK2)
    rk = gf2_rank(np.concatenate([dK2, D], 1)) - gf2_rank(dK2)
    h1J = Z.shape[1] - gf2_rank(d2)
    return h1J, rk

if __name__ == "__main__":
    import gpu_sc
    name = sys.argv[1] if len(sys.argv) > 1 else "torus"
    P = gpu_sc.build(name); tris = [set(f) for f in P.kfacets]; nt = len(tris); npath = P.nf // (nt * nt)
    band = lambda pred: [f for f in range(P.nf) if pred(tris[f // npath // nt], tris[f // npath % nt])]
    tests = {"D_diag (s=t)": band(lambda s, t: s == t), "D_edge (|s∩t|>=2)": band(lambda s, t: len(s & t) >= 2),
             "D_vert (s∩t≠∅)": band(lambda s, t: len(s & t) >= 1), "all of K x K": list(range(P.nf))}
    for arg in sys.argv[2:]:
        s = json.load(open(arg)); tests[arg] = sorted(int(f) for d in s["domains"] for f in d) if "domains" in s else None
        if "best_chains" in s:
            for j, c in enumerate(s["best_chains"]):
                tests[f"{arg} chain {j} domain"] = P.domain_of(np.array(c, np.int8)).tolist()
    for nm, J in tests.items():
        if not J: continue
        h1, rk = h1_map_obstruction(P, J)
        print(f"{P.name}: {nm}: {len(J)} facets, dim H1(J;Z2) = {h1}, rank (pi1-pi2)_* = {rk}  -> "
              f"{'OBSTRUCTED (not a domain)' if rk else 'passes the H1 test'}", flush=True)
