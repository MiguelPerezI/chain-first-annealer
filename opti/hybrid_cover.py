#!/usr/bin/env python3
"""hybrid_cover.py — keep the m largest product rectangles star(v) x star(w) of
a star_cover.py certificate as fixed domains and let the chain-first
annealer (gpu_sc.anneal_cover) cover ONLY the remaining facets with k chains.
The annealer runs on a SubProduct (the same 7x7 grid, facet list restricted
to the remainder), so 'full coverage' means the remainder is covered; the
result is combined with the fixed rectangles, verified on the full product
with the plain Definition 2.3 check and written as an audit-ready certificate.

usage: hybrid_cover.py <complex> <certificate summary.json> <m kept> <k> <outdir> [gpu_sc anneal options...]
"""
import sys, json, argparse, time
from pathlib import Path
import numpy as np
import gpu_sc, star_cover


class SubProduct:
    """gpu_sc.Product restricted to a subset of its facets (same grid)."""
    def __init__(self, P, facet_ids):
        self.P, self.sub = P, np.array(sorted(facet_ids), np.int64)
        self.name, self.nv, self.ngrid, self.kfacets = P.name + " (remainder)", P.nv, P.ngrid, P.kfacets
        self.is_simplex, self.dk, self.pi1, self.pi2, self.maxfv = P.is_simplex, P.dk, P.pi1, P.pi2, P.maxfv
        self.facets = [P.facets[f] for f in self.sub]
        self.nf = len(self.facets)
        self.facet_verts, self.facet_nv = P.facet_verts[self.sub].copy(), P.facet_nv[self.sub].copy()
        counts = np.zeros(self.ngrid, np.int32)
        for f in self.facets:
            for v in f: counts[v] += 1
        off = np.zeros(self.ngrid + 1, np.int32); np.cumsum(counts, out=off[1:])
        inc = np.zeros(off[-1], np.int32); cur = off[:-1].copy()
        for fi, f in enumerate(self.facets):
            for v in f:
                inc[cur[v]] = fi; cur[v] += 1
        self.incid_off, self.incid_fac, self.starmax = off, inc, int(counts.max())
        self.maskw = (self.nf + 63) // 64
    verts_of = gpu_sc.Product.verts_of
    chain_bad_counts = gpu_sc.Product.chain_bad_counts
    domain_of = gpu_sc.Product.domain_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("complex"); ap.add_argument("certificate"); ap.add_argument("m", type=int); ap.add_argument("k", type=int)
    ap.add_argument("outdir")
    ap.add_argument("--cmax", type=int, default=20); ap.add_argument("--S", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=20000); ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--beta", type=float, default=5.0); ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--ptrack", type=float, default=0.3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-limit", type=float, default=600)
    ap.add_argument("--init", default=None, help="summary.json of a previous hybrid run (same m, k): warm-start half the population")
    a = ap.parse_args()
    P = gpu_sc.build(a.complex)
    cert = json.load(open(a.certificate))
    rects = [tuple(r) for r in cert["rectangles"]]
    # domains of the certificate are sorted by size and correspond to rectangles in the order used by star_cover.build;
    # recompute them here to be safe (facet -> first rectangle containing it, then keep the m largest parts)
    parts = star_cover.build(P, rects)                       # list of (facets, chain), largest first
    kept = parts[:a.m]
    fixed = np.zeros(P.nf, bool)
    for facs, _ in kept: fixed[facs] = True
    rem = np.flatnonzero(~fixed)
    print(f"{P.name}: keeping {a.m} rectangles ({[len(f) for f, _ in kept]} facets) -> remainder {len(rem)} facets, k={a.k} chains", flush=True)
    SP = SubProduct(P, rem)
    t0 = time.perf_counter()
    init = None
    if a.init:
        prev = json.load(open(a.init))
        bc = np.array(prev["best_chains"], np.int8)                  # [k, C1, ngrid]
        assert bc.shape[0] == a.k and bc.shape[1] == a.cmax + 1, "init run must have the same k and cmax"
        init = np.repeat(bc[None], a.S, axis=0)
        print(f"warm start from {a.init} (remainder coverage {prev.get('coverage')})", flush=True)
    chains, cov, hist, ann = gpu_sc.anneal_cover(SP, a.k, a.cmax, a.S, a.iters, a.epochs, beta=a.beta, gamma=a.gamma,
                                                 ptrack=a.ptrack, seed=a.seed, time_limit=a.time_limit, init_chains=init)
    dt = time.perf_counter() - t0
    print(f"remainder coverage {int(cov)}/{SP.nf}  time {dt:.1f}s", flush=True)
    sub_parts = gpu_sc.partition_from_chains(SP, [chains[j] for j in range(a.k)])
    outdir = Path(a.outdir)
    if sub_parts is None:
        covered = np.zeros(SP.nf, bool)
        for j in range(a.k): covered[SP.domain_of(chains[j])] = True
        print(f"NOT a covering of the remainder: {int((~covered).sum())} facets uncovered")
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "summary.json").write_text(json.dumps(dict(kept=a.m, k=a.k, remainder=int(SP.nf), coverage=int(covered.sum()),
            best_chains=[chains[j].tolist() for j in range(a.k)], history=hist), indent=2, default=str))
        return
    all_parts = list(kept) + [([int(SP.sub[f]) for f in facs], chain) for facs, chain in sub_parts]
    all_parts.sort(key=lambda t: -len(t[0]))
    gpu_sc.verify_system(P, all_parts)
    summ = gpu_sc.write_certificate(P, all_parts, outdir, dict(method=f"hybrid: {a.m} star(v)xstar(w) rectangles + {a.k} annealed chains",
                                    rectangles=[list(r) for r in rects], kept_domains=[len(f) for f, _ in kept], params=vars(a), elapsed_s=dt, history=hist))
    print(f"VERIFIED: {len(all_parts)} domains {summ['domain_sizes']}, chain lengths {summ['chain_lengths']} => SC_strict({P.name}) <= {len(all_parts)-1}; written to {outdir}")


if __name__ == "__main__":
    main()
