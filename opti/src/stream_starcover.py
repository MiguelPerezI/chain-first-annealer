#!/usr/bin/env python3
r"""stream_starcover.py — star cover of ∂Δ^{n+1} (S^n) for n where the
facet-by-facet certificate is not representable.

Above S⁹ the full certificate stops being writable: ngrid = (n+2)^2 > 127
overflows the int8 grid ids of gpu_sc.Product, the facet table needs 13.3 GB
even as raw int8 vertices at n = 12, and cover.txt would be ≈4.2 GB (n = 10),
21 GB (n = 11), 106 GB (n = 12) — extrapolated from the measured 80–118 bytes
per facet of the audited certificates for n = 5..8.  This tool therefore
verifies the
construction by streaming K x K in path-chunks, never materializing it:

  1. facet enumeration — for every pair (s, t) of facets of K the emitted
     facets are monotone staircase paths (unit steps in the (a, b) grid, from
     (s[0], t[0]) to (s[-1], t[-1])), pairwise distinct within the pair, and
     there are exactly C(2n, n) of them.  Validity + distinctness + count say
     the enumeration *is* the staircase triangulation of that pair.
  2. cover — each facet is assigned to the first domain D(v) = star(v) x
     star(v) (v = 0, 1, 2) with v in s and v in t; the per-domain counts are
     compared with the closed form |D(0)| = (n+1)^2 C(2n,n),
     |D(1) \ D(0)| = (2n+1) C(2n,n), |D(2) \ D(0) u D(1)| = 2 C(2n,n).
  3. contiguity — on every facet of its domain the chain pi1 ~ c_v ~ pi2 is
     simplicial and 1-contiguous: each map's image is a simplex of ∂Δ^{n+1}
     (size <= n+1) and so is each union of consecutive images.  The images are
     recomputed from the emitted vertex list of the facet, not assumed.

The verification is a direct brute-force instance of Theorem 1 (product
domains) at scale; it shares no code with star_cover.py or independent_check.py.

usage: stream_starcover.py <N> <outdir> [--sample K] [--chunk M]
       stream_starcover.py --cross-check      # n = 5..8 against the audited
                                              # full certificates in results/
"""
import sys, json, time, hashlib
from math import comb
from pathlib import Path
import numpy as np


def k_facets(nv):
    """Facets of K = ∂Δ^{nv-1}: the (nv-1)-subsets of {0..nv-1}, itertools order."""
    import itertools
    return [np.array(f, np.int64) for f in itertools.combinations(range(nv), nv - 1)]


def path_index_tables(n):
    """(A, B) of shape (npath, 2n+1): A[k] = number of 0-bits among the first
    k bits of the mask, B[k] = number of 1-bits.  Same bit convention as
    gpu_sc.Product (masks increasing, popcount = n, bit 0 advances the first
    factor), so facet ids agree with the full certificates for n <= 9."""
    masks = np.arange(1 << (2 * n), dtype=np.int64)
    pc = np.zeros(1 << (2 * n), np.int8)
    m = masks.copy()
    for _ in range(2 * n):                     # popcount by shifting
        pc += (m & 1).astype(np.int8)
        m >>= 1
    masks = masks[pc == n]
    bits = ((masks[:, None] >> np.arange(2 * n, dtype=np.int64)) & 1).astype(np.int8)  # 1 = advance 2nd factor
    npath = masks.size
    A = np.zeros((npath, 2 * n + 1), np.int16)
    B = np.zeros((npath, 2 * n + 1), np.int16)
    np.cumsum(bits == 0, axis=1, dtype=np.int16, out=A[:, 1:])
    np.cumsum(bits == 1, axis=1, dtype=np.int16, out=B[:, 1:])
    return A, B, masks


def hash_paths(verts):
    """Two independent rolling hashes of the vertex lists (mod 2^64), used to
    prove distinctness of the emitted paths within a pair."""
    h1 = np.zeros(verts.shape[0], np.uint64)
    h2 = np.zeros(verts.shape[0], np.uint64)
    for k in range(verts.shape[1]):
        h1 = h1 * np.uint64(1000003) + verts[:, k].astype(np.uint64)
        h2 = h2 * np.uint64(1000033) + (verts[:, k].astype(np.uint64) + np.uint64(k))
    return h1, h2


def verify(n, chunk=1 << 20, sample=3, log=print):
    nv = n + 2
    kf = k_facets(nv)
    npath = comb(2 * n, n)
    nf_exp = nv * nv * npath
    sizes_exp = [(n + 1) ** 2 * npath, (2 * n + 1) * npath, 2 * npath]
    t0 = time.perf_counter()
    A, B, masks = path_index_tables(n)
    assert A.shape[0] == npath, (A.shape, npath)
    log(f"  K = ∂Δ{nv-1} (S^{n}): nv = {nv}, {nv} facets of {nv-1} vertices")
    log(f"  K x K: {nv}x{nv} pairs x {npath} staircase paths = {nf_exp:,} facets, "
        f"{2*n+1} vertices each; per-facet check, chunk = {chunk:,} paths")
    cnt = [0, 0, 0]
    seen = 0
    hashes = []
    samples = {0: [], 1: [], 2: []}
    for si, s in enumerate(kf):
        for ti, t in enumerate(kf):
            # the domain rule: first v in {0,1,2} with v in s and v in t
            v = next((x for x in (0, 1, 2) if (s == x).any() and (t == x).any()), None)
            v_in_s = v is not None and bool((s == v).any())
            v_in_t = v is not None and bool((t == v).any())
            for lo in range(0, npath, chunk):
                hi = min(lo + chunk, npath)
                m = hi - lo
                aa = s[A[lo:hi]]                     # first coords   [m, 2n+1]
                bb = t[B[lo:hi]]                     # second coords
                # positions of the emitted vertices inside the two K-facets
                # (recovered from the emitted list, not taken from the generator)
                pa = np.searchsorted(s, aa)
                pb = np.searchsorted(t, bb)
                assert (s[pa] == aa).all(), f"pair ({si},{ti}): first coord not in s"
                assert (t[pb] == bb).all(), f"pair ({si},{ti}): second coord not in t"
                # (1a) staircase path: unit steps in the (position, position)
                # grid, from (0,0) to (n,n) — so the facet lies over (s,t) and
                # its first/second coordinate sets are exactly s and t
                dpa, dpb = np.diff(pa, axis=1), np.diff(pb, axis=1)
                ok_step = (((dpa == 1) & (dpb == 0)) | ((dpa == 0) & (dpb == 1))).all()
                assert ok_step, f"pair ({si},{ti}): not a unit-step staircase path"
                assert (pa[:, 0] == 0).all() and (pb[:, 0] == 0).all() and \
                       (pa[:, -1] == n).all() and (pb[:, -1] == n).all(), \
                    f"pair ({si},{ti}): path does not start at (s[0],t[0]) and end at (s[-1],t[-1])"
                # (1b) distinctness (two independent hashes, no collision seen)
                h1, h2 = hash_paths(aa * nv + bb)
                assert len(np.unique(h1)) == m and len(np.unique(h2)) == m, \
                    f"pair ({si},{ti}): repeated staircase path"
                hashes.append((int(h1.sum()), int(h2.sum()), m))
                # (2) cover: this pair must lie in some domain
                assert v is not None, f"pair ({si},{ti}) covered by no D(v), v = 0,1,2"
                cnt[v] += m
                # (3) contiguity of pi1 ~ c_v ~ pi2 on every facet of the chunk.
                # Image sizes recomputed from the emitted vertex list: the
                # distinct first coordinates (n1) and second coordinates (n2)
                # are the images of pi1 and pi2; img(c_v) = {v}.  Each image is
                # a simplex of ∂Δ^{n+1} iff its size is <= n+1, and a link is
                # 1-contiguous iff the union of the two images is a simplex.
                n1 = (dpa == 1).sum(axis=1) + 1      # |img(pi1)| per facet
                n2 = (dpb == 1).sum(axis=1) + 1      # |img(pi2)| per facet
                assert (n1 == nv - 1).all() and (n2 == nv - 1).all(), \
                    f"pair ({si},{ti}): a map image is not a simplex"
                budget = nv - 1 - (0 if v_in_s else 1)   # |img(pi1) u img(c_v)| <= n+1
                assert (n1 <= budget).all(), \
                    f"pair ({si},{ti}): pi1 u c_{v} not simplicial"
                budget = nv - 1 - (0 if v_in_t else 1)   # |img(c_v) u img(pi2)| <= n+1
                assert (n2 <= budget).all(), \
                    f"pair ({si},{ti}): c_{v} u pi2 not simplicial"
                seen += m
                if len(samples[v]) < sample:
                    for r in range(min(sample - len(samples[v]), m)):
                        samples[v].append([int(x) for x in (aa[r] * nv + bb[r])])
    dt = time.perf_counter() - t0
    assert seen == nf_exp, (seen, nf_exp)
    assert cnt == sizes_exp, (cnt, sizes_exp)
    dig = hashlib.sha256()
    for h1, h2, m in hashes:
        dig.update(f"{h1}:{h2}:{m}\n".encode())
    log(f"  [1] enumeration: {nf_exp:,} facets, all unit-step staircase paths, "
        f"pairwise distinct, {npath:,} per pair — the staircase triangulation")
    log(f"  [2] cover: 3 disjoint domains, sizes {cnt[0]:,} + {cnt[1]:,} + {cnt[2]:,} "
        f"= {nf_exp:,} (closed form {sizes_exp[0]:,} + {sizes_exp[1]:,} + {sizes_exp[2]:,})")
    log(f"  [3] contiguity: pi1 ~ c_v ~ pi2 simplicial and 1-contiguous on all "
        f"{nf_exp:,} facets, by recomputed image sets")
    log(f"  enumeration digest (sha256 of the per-pair path hashes): {dig.hexdigest()}")
    log(f"  {dt:.1f} s, {2*n+1} vertex entries per facet = {(2*n+1)*nf_exp/1e9:.1f} G checked")
    return dict(n=n, nv=nv, n_facets=nf_exp, n_paths=npath, domain_sizes=cnt,
                closed_form=sizes_exp, paths_per_pair=npath, elapsed_s=dt,
                digest=dig.hexdigest(), samples=samples, facets_per_vertex=2 * n + 1)


def sz(b):
    """Byte count with a readable unit (the tool is generic in n)."""
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if b >= div:
            return f"{b/div:.1f} {unit}"
    return f"{b:.0f} B"


def write_artifacts(res, outdir, log=print):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    n, nv = res["n"], res["nv"]
    nf = res["n_facets"]
    # planners.txt: the chains are tiny at any n (3 maps, nv x nv entries)
    with open(outdir / "planners.txt", "w") as fh:
        fh.write("Planner chains (explicit product construction, Theorem 1 / Corollary 3)\n\n")
        for v in range(3):
            fh.write(f"=== domain {v}: D({v},{v}) = star({v}) x star({v}): "
                     f"{res['domain_sizes'][v]:,} facets ===\n")
            fh.write("  chain: pi1 ~ c_%d ~ pi2  (2 links, 3 maps)\n" % v)
            for mi, lab in enumerate(("pi1", f"c_{v}", "pi2")):
                fh.write(f"  map {mi} ({lab}):\n")
                for a in range(nv):
                    fh.write("    " + " ".join(str(a if mi != 1 else v) for _ in range(nv)) + "\n")
            fh.write("\n")
    est_bytes = int(nf * (res["facets_per_vertex"] * 8 + 20))   # upper bound: the
    # audited certificates measure 80-118 bytes per facet for n = 5..8
    if nv * nv > 127:
        wall = ("ngrid = %d > 127 overflows the int8 grid ids of gpu_sc.Product" % (nv * nv))
    else:
        wall = ("the shipped auditor materializes the facet table as Python objects "
                "(≈2.8 KB per facet ⇒ ≈%s here)" % sz(nf * 2800))
    summ = dict(complex=dict(name=f"∂Δ{nv-1} (S^{n})", nv=nv, ngrid=nv * nv,
                             num_facets=nf),
                method="explicit product domains D(v,v) = star(v) x star(v) (Corollary 3), "
                       "verified by streaming brute force (stream_starcover.py)",
                n_domains=3, sc_strict_bound=2,
                domain_sizes=res["domain_sizes"], chain_lengths=[2, 2, 2],
                paths_per_pair=res["n_paths"], facets_per_vertex=res["facets_per_vertex"],
                enumeration_digest=res["digest"], elapsed_s=res["elapsed_s"],
                sample_facets=res["samples"],
                cover_txt_bytes_estimate=est_bytes,
                cover_txt_bytes_estimate_kind="upper bound",
                cover_txt_written=False,
                note=("%s, and cover.txt would be %s, so the facet table is not "
                      "written; the cover is verified facet by facet by stream_starcover.py instead"
                      % (wall, sz(est_bytes))))
    (outdir / "summary.json").write_text(json.dumps(summ, indent=2) + "\n")
    with open(outdir / "certificate.md", "w") as fh:
        fh.write(f"# Star cover of {summ['complex']['name']} — streamed certificate\n\n")
        fh.write(f"`SC_strict(∂Δ{nv-1}) ≤ 2` by 3 explicit product domains "
                 f"`D(v) = star(v) x star(v)`, `v = 0, 1, 2`, chains "
                 f"`pi1 ~ c_v ~ pi2` (Corollary 3 of `optimized_algorithms.md` §19).\n\n")
        fh.write(f"| quantity | value |\n|---|---|\n")
        fh.write(f"| complex | ∂Δ{nv-1} (S^{n}): {nv} vertices, {nv} facets |\n")
        fh.write(f"| facets of K x K | {nf:,} ({nv}x{nv} pairs x {res['n_paths']:,} staircase paths) |\n")
        fh.write(f"| domain sizes | {res['domain_sizes'][0]:,} + {res['domain_sizes'][1]:,} "
                 f"+ {res['domain_sizes'][2]:,} |\n")
        fh.write(f"| chain lengths | 2 links (3 maps) each |\n")
        fh.write(f"| verification | every facet: staircase validity, distinctness, "
                 f"domain membership, simpliciality + 1-contiguity of both links |\n")
        fh.write(f"| enumeration digest | `{res['digest']}` |\n")
        fh.write(f"| runtime | {res['elapsed_s']:.1f} s (streamed, chunked) |\n")
        fh.write(f"| cover.txt | *not written*: {sz(summ['cover_txt_bytes_estimate'])} |\n\n")
        fh.write("## Why there is no cover.txt\n\n")
        fh.write(f"At n = {n} the grid has {nv*nv} vertices and the facet table would be "
                 f"{sz(summ['cover_txt_bytes_estimate'])} of text\n")
        fh.write(f"(an upper bound — the audited certificates measure 80–118 bytes per facet at "
                 f"n = 5..8), and\n")
        fh.write(f"≈{sz(nf*2800)} of Python objects for the shipped auditor: {wall}.\n")
        fh.write("The domains and chains are given by the rule above and are tiny (`planners.txt`); "
                 "what is verified here\n")
        fh.write("is the part that would have been the file: the facet enumeration and the contiguity on\n")
        fh.write("every facet.  Re-run: `python3 stream_starcover.py %d <dir>`.\n\n" % n)
        fh.write("## Sample facets (first of each domain, vertex ids = (a,b) with id = %d a + b)\n\n" % nv)
        for v in range(3):
            fh.write(f"* `D({v})`, witness facet: " + " ".join(f"({x//nv},{x%nv})" for x in res["samples"][v][0]) + "\n")
    log(f"  wrote {outdir}/planners.txt summary.json certificate.md "
        f"(no cover.txt: {sz(summ['cover_txt_bytes_estimate'])})")


def cross_check(check_dirs=("s5", "s6", "s7", "s8"), root="results", log=print):
    """Re-derive the small certificates with the streaming verifier and compare
    the domain sizes with the audited full certificates (planners/cover path)."""
    ok = True
    for d in check_dirs:
        s = json.loads(Path(root, f"{d}_starcover", "summary.json").read_text())
        n = int(d[1:])
        res = verify(n, chunk=1 << 18, sample=0, log=lambda *a: None)
        match = res["domain_sizes"] == s["domain_sizes"] and res["n_facets"] == s["complex"]["num_facets"]
        ok &= match
        log(f"  {d}: streaming {res['domain_sizes']} vs audited {s['domain_sizes']} — "
            f"{'MATCH' if match else 'MISMATCH'}")
    return ok


if __name__ == "__main__":
    if sys.argv[1:2] == ["--cross-check"]:
        sys.exit(0 if cross_check() else 1)
    N = int(sys.argv[1])
    outdir = sys.argv[2]
    args = sys.argv[3:]
    chunk = 1 << 20
    if "--chunk" in args:
        chunk = int(args[args.index("--chunk") + 1])
    print(f"=== stream_starcover.py  S^{N} = ∂Δ{N+1}  -> {outdir} ===")
    res = verify(N, chunk=chunk)
    write_artifacts(res, outdir)
    print("STREAM_STARCOVER DONE")
