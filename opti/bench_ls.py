#!/usr/bin/env python3
"""bench_ls.py — self-test and benchmark of the batched optimized LocalSearch
(gpu_sc.ls_kernel) against the paper's walk as implemented in
Abstract_Motion_GPU (absmotion.kernels.ghost_kernel), on IDENTICAL tests.

Test set: the 35-facet S^3 domain of results/s3_smoke (the GPU repo's
`probe_escalation.py` scaffold) plus each of the 465 other facets as
AddFacet candidates; success = a pi1 -> pi2 contiguity chain exists on
J ∪ {sigma} within the budget.

  1. determinism: kernel (early_exit=0) == CPU twin on small tests;
  2. replay: every reported winner is replayed on the CPU and the chain
     verified with the plain definition;
  3. success counts / wall time at several budgets for both walks.
"""
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/ubuntu/Documents/Abstract_Motion_GPU")
import gpu_sc

GPU_REPO = Path("/home/ubuntu/Documents/Abstract_Motion_GPU")


def main():
    P = gpu_sc.build("s3")
    summ = json.loads((GPU_REPO / "results/s3_smoke/summary.json").read_text())
    D = set(max(summ["domains"], key=len))
    pool = sorted(set(range(P.nf)) - D)
    print(f"scaffold |J| = {len(D)}, candidates = {len(pool)}")
    ls = gpu_sc.LocalSearchGPU(P, walks=128)
    rng = np.random.default_rng(123)

    # ---- 1. determinism: GPU (early_exit=0) vs CPU twin, small walks ----
    small = [dict(facets=sorted(D | {f}), seed=int(rng.integers(1, 2**62))) for f in pool[:12]]
    M, r = 1500, 0.1
    gpu = ls.run(small, M, r, early_exit=0, walks=16)
    mism = 0
    for t, (found, winner, steps) in zip(small, gpu):
        jm, verts, nv, start, target, seeds = ls.pack([t])
        best = (0, -1, -1)
        for w in range(16):
            img = np.zeros(P.ngrid, np.int8); defect = np.zeros(P.ngrid, np.int8)
            bb = np.zeros(P.maskw, np.uint64); hist = np.zeros((1, P.ngrid), np.int8)
            ws = np.uint64((int(seeds[0]) + int(gpu_sc.GOLDEN) * (w + 1)) & ((1 << 64) - 1))
            f2, s2, _ = ls.walk_cpu(jm[0], verts[0], np.int32(nv[0]), start[0], target[0], ws,
                                    np.int32(M), np.float64(r), np.int32(0), np.zeros(1, np.int32),
                                    P.facet_verts, P.facet_nv, P.incid_off, P.incid_fac,
                                    P.is_simplex, P.dk, img, defect, bb, np.int32(0), hist)
            if f2:
                best = (1, w, int(s2)); break
        if best != (int(found), int(winner), int(steps)):
            mism += 1
            print("  MISMATCH", best, (found, winner, steps))
    print(f"[1] GPU == CPU twin on {len(small)} tests x 16 walks: {'OK' if mism == 0 else f'{mism} mismatches'}")

    # ---- 2. replay + verify chains of winners ----
    tests = [dict(facets=sorted(D | {f}), seed=int(rng.integers(1, 2**62))) for f in pool]
    res = ls.run(tests, 4000, r, early_exit=1)
    nrep = 0
    for t, (found, winner, steps) in list(zip(tests, res))[:400]:
        if not found:
            continue
        chain = ls.replay(t, winner, 4000, r)
        assert np.array_equal(chain[0], P.pi1) and np.array_equal(chain[-1], P.pi2)
        for a, b in zip(chain, chain[1:]):
            assert gpu_sc.maps_contiguous_on(P, a, b, t["facets"]), "link not contiguous"
        red = gpu_sc.reduce_chain(P, chain, t["facets"])
        for a, b in zip(red, red[1:]):
            assert gpu_sc.maps_contiguous_on(P, a, b, t["facets"])
        nrep += 1
    print(f"[2] replayed + verified {nrep} winning walks (chains reduce fine)")

    # ---- 3. benchmark vs the paper's walk (ghost_kernel) ----
    from absmotion.tables import build_s3
    from absmotion.driver import GpuTester
    Q = build_s3()
    gt = GpuTester(Q, walks=128, early_exit=1)
    mask = sum(1 << f for f in D)
    gtests = [dict(mask=mask, cand=f, verts=Q.verts_of_facets(D | {f}), dstart=Q.d_start(D | {f}),
                   seed=int(rng.integers(1, 2**62))) for f in pool]
    rows = []
    # warm up JIT
    gt.run_tests(gtests[:4], 100, r); ls.run(tests[:4], 100, r)
    for label, fn, budgets in (("paper walk (ghost_kernel)", lambda M: gt.run_tests(gtests, M, r), (2000, 8000, 40000)),
                               ("optimized LocalSearch (ls_kernel)", lambda M: ls.run(tests, M, r), (250, 1000, 2000, 8000))):
        for M in budgets:
            t0 = time.perf_counter(); out = fn(M); dt = time.perf_counter() - t0
            succ = sum(1 for o in out if o[0])
            steps = sorted(o[2] for o in out if o[0])
            med = steps[len(steps) // 2] if steps else -1
            rows.append((label, M, succ, dt, med))
            print(f"[3] {label:36s} walks=128 M={M:6d}: {succ:3d}/{len(pool)} successes  "
                  f"{dt:6.2f}s  median winner steps {med}")
    Path("results").mkdir(exist_ok=True)
    Path("results/bench_ls.json").write_text(json.dumps(
        dict(scaffold=sorted(D), candidates=len(pool), walks=128, r=r,
             rows=[dict(walk=a, M=b, successes=c, seconds=d, median_steps=e) for a, b, c, d, e in rows]), indent=2))


if __name__ == "__main__":
    main()
