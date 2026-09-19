#!/usr/bin/env python3
"""band_test.py — are the 'diagonal bands' of K x K contiguity domains?
D_diag  = facets over pairs (s, s)            (union of images is s: trivially a domain)
D_edge  = facets over pairs with |s ∩ t| >= 2
D_vert  = facets over pairs with  s ∩ t != ∅
Tested with the batched optimized LocalSearch (gpu_sc.LocalSearchGPU), W walks x M steps."""
import sys, numpy as np, gpu_sc
name = sys.argv[1] if len(sys.argv) > 1 else "torus"
M = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
W = int(sys.argv[3]) if len(sys.argv) > 3 else 128
P = gpu_sc.build(name); tris = [set(f) for f in P.kfacets]; nt = len(tris); npath = P.nf // (nt * nt)
def band(pred):
    return sorted(f for f in range(P.nf) if pred(tris[f // npath // nt], tris[f // npath % nt]))
bands = {"D_diag": band(lambda s, t: s == t),
         "D_edge": band(lambda s, t: len(s & t) >= 2),
         "D_vert": band(lambda s, t: len(s & t) >= 1)}
ls = gpu_sc.LocalSearchGPU(P, walks=W)
for nm, D in bands.items():
    tests = [dict(facets=D, seed=1000 + i) for i in range(4)]
    res = ls.run(tests, M, 0.1, early_exit=1)
    found = [(i, w, st) for i, (f, w, st) in enumerate(res) if f]
    print(f"{P.name}: {nm}: {len(D)} facets ({len(D)//npath} pairs) — chain found in {len(found)}/4 tests (W={W}, M={M})"
          + (f"; first: test {found[0][0]} walk {found[0][1]} after {found[0][2]} steps" if found else ""), flush=True)
    if found:
        i, w, st = found[0]
        chain = ls.replay(tests[i], w, M, 0.1)
        red = gpu_sc.reduce_chain(P, chain, D)
        assert gpu_sc.verify_system(P, [(D, red)]) if len(D) == P.nf else all(gpu_sc.maps_contiguous_on(P, a, b, D) for a, b in zip(red, red[1:]))
        print(f"   verified; reduced chain {len(red)-1} links", flush=True)
