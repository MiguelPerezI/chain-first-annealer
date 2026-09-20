#!/usr/bin/env python3
"""fixed_domain_test.py — does a *given* facet set admit a chain (a motion
planner π₁ ~ π₂)?  The constructive counterpart to the k-chain cover search.

Every search in this repo asks "can some pair of domains cover K × K?" and
anneals both domains at once, in a space where the plateau at the 2-rectangle
optimum is total.  This script asks the sharper question instead: here is a
domain, *find the chain*.  With `hybrid_cover.SubProduct` (the grid restricted
to a facet subset) a single chain (k = 1) either covers the whole subset —
which is a verified domain, hence a motion planner — or it does not.

That matters because domains may **overlap**: `write_certificate` partitions
them, but nothing requires the domains themselves to be disjoint (the k=3
certificate of ∂Δ⁴ has raw domains 309 + 283 + 271 = 863 facets over a
500-facet complex).  So a two-domain cover needs only dom₁ ∪ dom₂ = K × K,
and the circle's own solution (`results/circle_probe3`) is built this way —
see s3_k2_push_log.md for the decoding of it.

Domains are named by their cells: cell (s, t) is the product of facet s of K
and facet t of K, holding the same 20 staircase facets for every pair on
∂Δ⁴.  `--rule upper` is {s ≤ t}, `--rule lower` {s ≥ t}, whose union is all
25 cells = 500 facets.  `--facets` takes a JSON file with a facet-id list
(any certificate's `summary.json` has one under "domains").

usage:
  fixed_domain_test.py s3 --rule upper --rule lower --outdir results/union_test
  fixed_domain_test.py circle --facets results/circle_probe3/summary.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import gpu_sc
from hybrid_cover import SubProduct


def cell_of(P, f):
    """(s, t): the pair of K-facets whose product cell holds product facet f."""
    nt = len(P.kfacets)
    per = P.nf // (nt * nt)
    assert P.nf % (nt * nt) == 0, "ragged cells: facets of K must be uniform"
    return divmod(f // per, nt)


def facets_for_rule(P, rule):
    c = np.array([cell_of(P, f) for f in range(P.nf)])
    if rule == "upper":
        m = c[:, 0] <= c[:, 1]
    elif rule == "lower":
        m = c[:, 0] >= c[:, 1]
    elif rule == "strict-upper":
        m = c[:, 0] < c[:, 1]
    elif rule == "strict-lower":
        m = c[:, 0] > c[:, 1]
    elif rule == "lower-ends":
        # the complement of the domain a single chain actually reached on
        # `--rule upper` (s3_upper_probe): that chain held all 10 off-diagonal
        # cells of {s <= t} and the three diagonal cells (1,1), (2,2), (3,3),
        # dropping (0,0) and (4,4).  `lower-ends` is {s > t} plus the two
        # dropped diagonal cells — the second domain a 2-cover would need.
        last = len(P.kfacets) - 1
        m = (c[:, 0] > c[:, 1]) | ((c[:, 0] == c[:, 1]) & ((c[:, 0] == 0) | (c[:, 0] == last)))
    elif rule == "upper-ends":
        last = len(P.kfacets) - 1
        m = (c[:, 0] < c[:, 1]) | ((c[:, 0] == c[:, 1]) & ((c[:, 0] == 0) | (c[:, 0] == last)))
    elif rule == "lower-mid":
        # the swap scenario: the lower block plus the three diagonal cells the
        # upper chain held in s3_upper_probe — tests whether the *lower* chain
        # can hold the middles, so that the upper chain could take the extremes.
        last = len(P.kfacets) - 1
        m = (c[:, 0] > c[:, 1]) | ((c[:, 0] == c[:, 1]) & (c[:, 0] != 0) & (c[:, 0] != last))
    elif rule == "upper-mid":
        last = len(P.kfacets) - 1
        m = (c[:, 0] < c[:, 1]) | ((c[:, 0] == c[:, 1]) & (c[:, 0] != 0) & (c[:, 0] != last))
    elif rule in ("upper-half", "lower-half"):
        # the circle's structure, lifted: split every diagonal cell in half
        # instead of giving it whole to one side.  The halves are the two
        # halves of the staircase (first per/2 vs last per/2 local indices),
        # i.e. facets whose last shuffle step advances in the first vs the
        # second factor.  union of upper-half and lower-half = all 500.
        per = P.nf // (len(P.kfacets) ** 2)
        half = np.array([f % per for f in range(P.nf)]) < per // 2
        if rule == "upper-half":
            m = (c[:, 0] < c[:, 1]) | ((c[:, 0] == c[:, 1]) & half)
        else:
            m = (c[:, 0] > c[:, 1]) | ((c[:, 0] == c[:, 1]) & ~half)
    elif rule == "diag":
        m = c[:, 0] == c[:, 1]
    elif rule == "all":
        m = np.ones(P.nf, bool)
    else:
        raise SystemExit(f"unknown rule {rule!r}")
    return np.flatnonzero(m).tolist()


def facets_from_summary(path, index=None):
    """Facets listed in a certificate summary: domain `index` alone, or their
    union when no index is given."""
    s = json.loads(Path(path).read_text())
    out = set()
    if "domains" in s:                       # star_cover / write_certificate form
        ds = [d for d in s["domains"]] if index is None else [s["domains"][index]]
        for d in ds:
            out.update(int(f) for f in d)
    elif "best_chains" in s:                 # gpu_sc annealer form
        raise SystemExit(f"{path}: best_chains needs the complex's grid to decode; "
                         f"pass a --rule or a summary with explicit 'domains'")
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("complex")
    ap.add_argument("--rule", action="append", default=[], help="upper|lower|strict-upper|diag|all (repeatable)")
    ap.add_argument("--facets", default=None, help="summary.json with an explicit 'domains' list")
    ap.add_argument("--domain-index", type=int, default=None, help="test only this domain of --facets (default: their union)")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--cmax", type=int, default=40)
    ap.add_argument("--S", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--beta", type=float, default=5.0)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--ptrack", type=float, default=0.05)
    ap.add_argument("--guide", type=float, default=1.0)
    ap.add_argument("--guide-max", type=float, default=4.0, dest="guide_max")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-limit", type=float, default=900)
    ap.add_argument("--init", default=None, help="summary.json of a previous run: warm-start half the population")
    a = ap.parse_args()

    P = gpu_sc.build(a.complex)
    domains = []
    if a.facets:
        nm = Path(a.facets).stem + (f"[{a.domain_index}]" if a.domain_index is not None else "")
        domains.append((nm, facets_from_summary(a.facets, a.domain_index)))
    for r in a.rule:
        domains.append((r, facets_for_rule(P, r)))
    if not domains:
        raise SystemExit("give at least one --rule or --facets")

    print(f"{P.name}: {P.nf} facets, {P.nv} vertices, cells {len(P.kfacets)}x{len(P.kfacets)}"
          f" ({P.nf // (len(P.kfacets) ** 2)} facets per cell)")
    results = []
    for name, facs in domains:
        union = len(set(facs))
        print(f"\n--- domain {name!r}: {union} facets of {P.nf} ---", flush=True)
        SP = SubProduct(P, facs)
        init = None
        if a.init:
            prev = json.loads(Path(a.init).read_text())
            bc = np.array(prev["best_chains"], np.int8)
            init = np.repeat(bc[:1][None], a.S, axis=0)   # k=1: the single chain
            print(f"  warm start from {a.init}", flush=True)
        t0 = time.perf_counter()
        chains, cov, hist, ann = gpu_sc.anneal_cover(
            SP, 1, a.cmax, a.S, a.iters, a.epochs, beta=a.beta, gamma=a.gamma,
            ptrack=a.ptrack, seed=a.seed, time_limit=a.time_limit, init_chains=init,
            guide=a.guide, guide_max=a.guide_max)
        dt = time.perf_counter() - t0
        dom = SP.domain_of(chains[0])
        ok = len(dom) == SP.nf
        print(f"  one chain covers {int(cov)}/{SP.nf} facets  ({dt:.0f}s, k={1})"
              f"  -> {'DOMAIN (chain exists)' if ok else 'no chain found in budget'}", flush=True)
        results.append(dict(name=name, facets=facs, cov=int(cov), nf=int(SP.nf),
                            ok=bool(ok), chain=chains[0].tolist(), hist=hist,
                            domain=[int(SP.sub[f]) for f in dom.tolist()]))

    covered = np.zeros(P.nf, bool)
    for r in results:
        if r["ok"]:
            covered[r["domain"]] = True
    print(f"\nunion of the domains that admitted a chain: {int(covered.sum())}/{P.nf}")
    if a.outdir:
        out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(dict(complex=a.complex, params=vars(a),
            results=results, union_coverage=int(covered.sum())), indent=2, default=str))
        print(f"wrote {out}/summary.json")
    if covered.all() and sum(r["ok"] for r in results) >= 2:
        print(f"*** {sum(r['ok'] for r in results)} chained domains cover all {P.nf} facets"
              f"  =>  SC_strict({P.name}) <= {sum(r['ok'] for r in results) - 1}")
        if a.outdir:
            parts = gpu_sc.partition_from_chains(P, [np.array(r["chain"], np.int8) for r in results if r["ok"]])
            if parts is not None:
                gpu_sc.verify_system(P, parts)
                gpu_sc.write_certificate(P, parts, a.outdir, dict(method="fixed-domain chains",
                                          params=vars(a), results=[{k: v for k, v in r.items() if k != "chain"} for r in results]))
                print(f"VERIFIED certificate written to {a.outdir}")
            else:
                print("(domains cover, but partition_from_chains could not split them — "
                      "the cover is still valid as overlapping domains)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
