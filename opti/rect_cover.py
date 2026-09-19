#!/usr/bin/env python3
"""rect_cover.py — exact minimum number rho(K) of rectangles star(v) x star(w)
covering all pairs of facets of K (bitmask branch-and-bound)."""
import sys, time, gpu_sc

def rect_cover(name, maxlimit=14):
    P = gpu_sc.build(name); tris = [set(f) for f in P.kfacets]; nt = len(tris); nv = P.nv
    rects = [(v, w) for v in range(nv) for w in range(nv)]
    cov = []
    for v, w in rects:
        m = 0
        for s in range(nt):
            if v in tris[s]:
                for t in range(nt):
                    if w in tris[t]: m |= 1 << (s * nt + t)
        cov.append(m)
    ALL = (1 << (nt * nt)) - 1
    maxcov = max(bin(c).count("1") for c in cov)
    bypair = {p: [r for r in range(len(rects)) if cov[r] >> p & 1] for p in range(nt * nt)}
    nodes = [0]
    def dfs(covered, chosen, limit):
        nodes[0] += 1
        if covered == ALL: return list(chosen)
        rem = bin(ALL & ~covered).count("1")
        if len(chosen) + (rem + maxcov - 1) // maxcov > limit: return None
        r0 = ALL & ~covered
        p = (r0 & -r0).bit_length() - 1
        for r in bypair[p]:
            chosen.append(r)
            out = dfs(covered | cov[r], chosen, limit)
            chosen.pop()
            if out: return out
        return None
    for limit in range(1, maxlimit + 1):
        t0 = time.time(); nodes[0] = 0; sol = dfs(0, [], limit)
        print(f"{name}: limit {limit}: {'FOUND' if sol else 'none'} ({time.time()-t0:.1f}s, {nodes[0]} nodes)", flush=True)
        if sol:
            print(f"{name}: rho = {limit}, rectangles (v,w) = {[rects[r] for r in sol]}", flush=True)
            return [rects[r] for r in sol]

if __name__ == "__main__":
    for n in sys.argv[1:] or ["circle", "s2", "s3", "torus"]:
        rect_cover(n)
