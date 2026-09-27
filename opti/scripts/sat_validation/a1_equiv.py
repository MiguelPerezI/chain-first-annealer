"""A1: predicate equivalence for scs_sat.py's model semantics.

Compiles three independent notions of "chain covers facet f" and demands
agreement on random chains:
  (1) sat  - the exact constraint semantics of solve_cold()
  (2) gsc  - gpu_sc.chain_bad_counts == 0 (repo's own domain_of)
  (3) ref  - direct definition: every link's union of vertex images
             is contained in some maximal facet of K
"""
import sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import gpu_sc
from gpu_sc import Product

# ---- (3) reference: straight from the paper's definition --------------------
def ref_cover(P, chain):
    """facet f covered iff for every link i: img(phi_i)|f U img(phi_{i+1})|f
    lies in some kfacet of K.  chain: list of L+1 arrays over the grid."""
    ksets = [set(f) for f in P.kfacets]
    dom = []
    for f in range(P.nf):
        verts = list(P.facets[f])
        ok = True
        for i in range(len(chain) - 1):
            un = {int(chain[i][v]) for v in verts} | {int(chain[i + 1][v]) for v in verts}
            if not any(un <= ks for ks in ksets):
                ok = False
                break
        if ok:
            dom.append(f)
    return dom

# ---- (1) sat: exact semantics of the CP-SAT constraints ---------------------
def sat_cover(P, chain, L):
    """Emulates: m[0]=pi1, m[L]=pi2 constants; x[f,d]=1 requires
    m[1][v] in sig(f), m[L-1][v] in tau(f) (for v in verts(f)), and
    for each i in 1..L-2 exists t: m[i][v],m[i+1][v] in TRIS[t]."""
    assert len(chain) == L + 1
    ksets = [set(f) for f in P.kfacets]
    dom = []
    for f in range(P.nf):
        verts = list(P.facets[f])
        sig = {g // P.nv for g in P.facets[f]}
        tau = {g % P.nv for g in P.facets[f]}
        ok = all(int(chain[1][v]) in sig for v in verts)
        ok = ok and all(int(chain[L - 1][v]) in tau for v in verts)
        if ok:
            for i in range(1, L - 1):
                if not any(all(int(chain[i][v]) in T and int(chain[i + 1][v]) in T
                               for v in verts) for T in ksets):
                    ok = False
                    break
        if ok:
            dom.append(f)
    return dom

def check(name, n_random=400, lens=(2, 3, 4, 5, 6, 7)):
    P = gpu_sc.build(name)
    rng = random.Random(12345)
    np_rng = np.random.default_rng(12345)
    bad = 0
    total = 0
    for _ in range(n_random):
        L = rng.choice(lens)
        chain = [P.pi1.copy()]
        for _i in range(L - 1):
            chain.append(np_rng.integers(0, P.nv, size=P.ngrid).astype(np.int8))
        chain.append(P.pi2.copy())
        ch = np.stack(chain)
        d_ref = set(ref_cover(P, chain))
        d_sat = set(sat_cover(P, chain, L))
        d_gsc = set(int(f) for f in P.domain_of(ch))
        total += 1
        if not (d_ref == d_sat == d_gsc):
            bad += 1
            if bad <= 5:
                print(f"  MISMATCH {name} L={L}: ref={len(d_ref)} sat={len(d_sat)} "
                      f"gsc={len(d_gsc)}; symdiff ref^sat={len(d_ref ^ d_sat)} "
                      f"ref^gsc={len(d_ref ^ d_gsc)}")
    print(f"{name:8s} random chains: {total - bad}/{total} agree across all three predicates")
    return bad

# ---- A1b: structural checks for s3 ----------------------------------------
def structural():
    P = gpu_sc.build("s3")
    print(f"\n== A1b structural (s3) ==")
    print(f"name={P.name} nv={P.nv} ngrid={P.ngrid} nf={P.nf} "
          f"kfacets={len(P.kfacets)} maxfv={P.maxfv}")
    ok = True
    # kfacets = all 4-subsets of 5 vertices -> the boundary of the 4-simplex
    ksets = sorted(tuple(sorted(f)) for f in P.kfacets)
    expect = sorted(itertools_combinations_4())
    if ksets != expect:
        ok = False; print("  FAIL kfacets != all 4-subsets of {0..4}")
    else:
        print("  OK kfacets = all C(5,4)=5 four-subsets (boundary of the 4-simplex)")
    # NF = NV^2 * C(6,3) = 25 * 20
    from math import comb
    if P.nf != P.nv * P.nv * comb(6, 3):
        ok = False; print(f"  FAIL nf {P.nf} != 25*20")
    else:
        print("  OK nf = 25 * C(6,3) = 500")
    # every facet: 7 grid vertices, sig/tau are kfacets, sig==s0 tau==s1
    sig_ok = tau_ok = fv_ok = inter_ok = exclusive_ok = True
    pair_recs = []
    n_diag = n_off = 0
    for fi, f in enumerate(P.facets):
        if len(f) != 7:
            fv_ok = False
        sig = tuple(sorted({g // P.nv for g in f}))
        tau = tuple(sorted({g % P.nv for g in f}))
        if sig not in ksets:
            sig_ok = False
        if tau not in ksets:
            tau_ok = False
        inter = set(sig) & set(tau)
        ex1 = set(sig) - set(tau)
        ex2 = set(tau) - set(sig)
        if sig == tau:
            n_diag += 1
            if not (len(inter) == 4 and not ex1 and not ex2):
                inter_ok = False
        else:
            n_off += 1
            if len(inter) != 3:
                inter_ok = False
            if len(ex1) != 1 or len(ex2) != 1:
                exclusive_ok = False
        pair_recs.append((sig, tau, tuple(sorted(inter)), tuple(sorted(ex1)), tuple(sorted(ex2))))
    print(f"  {'OK' if fv_ok else 'FAIL'} every facet has 7 grid vertices")
    print(f"  {'OK' if sig_ok else 'FAIL'} sig(f) is a kfacet for all 500 facets")
    print(f"  {'OK' if tau_ok else 'FAIL'} tau(f) is a kfacet for all 500 facets")
    print(f"  {'OK' if inter_ok else 'FAIL'} diagonal (s=s): |s n t|=4 ({n_diag} facets), "
          f"off-diagonal: |s n t|=3 ({n_off} facets)")
    print(f"  {'OK' if exclusive_ok else 'FAIL'} off-diagonal: |s\\t| = |t\\s| = 1")
    ok = ok and fv_ok and sig_ok and tau_ok and inter_ok and exclusive_ok
    # pi1 on facet verts = sig, pi2 = tau  (the endpoint constraints' soundness)
    ep_ok = True
    for f in P.facets:
        if {int(P.pi1[g]) for g in f} != {g // P.nv for g in f}:
            ep_ok = False
        if {int(P.pi2[g]) for g in f} != {g % P.nv for g in f}:
            ep_ok = False
    print(f"  {'OK' if ep_ok else 'FAIL'} pi1(img of facet) == sig, pi2 == tau")
    ok = ok and ep_ok
    # every pair (s,t) of kfacets occurs with C(6,3)=20 facets
    from collections import Counter
    c = Counter((s, t) for s, t, _, _, _ in pair_recs)
    uniform = all(v == 20 for v in c.values()) and len(c) == 25
    print(f"  {'OK' if uniform else 'FAIL'} 25 (s,t) pairs x 20 staircase paths = 500")
    ok = ok and uniform
    return ok

def itertools_combinations_4():
    import itertools
    return [tuple(c) for c in itertools.combinations(range(5), 4)]

if __name__ == "__main__":
    bad = 0
    bad += check("s3", n_random=500)
    bad += check("circle", n_random=300)
    bad += check("klein", n_random=300)
    ok_s = structural()
    print(f"\nRESULT: predicate mismatches={bad}, structural={'PASS' if ok_s else 'FAIL'}")
    sys.exit(0 if (bad == 0 and ok_s) else 1)
