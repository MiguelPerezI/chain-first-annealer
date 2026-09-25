#!/usr/bin/env python3
"""stall_audit.py — deterministic stall analysis for an annealer snapshot.

Reads a stall snapshot (or any incomplete summary.json with best_chains),
profiles the uncovered facets of the incumbent exactly the way the mobius 149
and rp2 544 audits did (experiments/surface_stall_analysis.md 4.1), and writes
stall_audit.json next to the snapshot plus a short human summary on stdout:

  * per leftover f: defect depth min_j bad_j[f] and the bad-link positions of
    each chain (the seam structure — a depth-2 chain with two far-apart links
    is the seam-stitch target of --pseam);
  * H1 absorption doors: for each (f, chain j) the GF(2) rank of (pi1 - pi2)_*
    on domain_j + {f} (rank 0 = topologically absorbable; h1_obstruction);
  * a machine-readable suggestion block: guide targets (leftovers with live
    doors), seam targets (chain, bad links, gap), reheat advice.

Total H1 tests are capped (default 2048, like --h1-guide); snapshots with
more leftovers x chains are audited on depth/seams only.

Usage: python3 stall_audit.py <snapshot.json> [--complex name] [--h1-cap N]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import gpu_sc
from h1_obstruction import h1_map_obstruction


def bad_links(P, chain, ft):
    """Link indices i (link i spans maps i, i+1) where facet ft is not
    contiguous under the chain."""
    masks = []
    for i in range(chain.shape[0]):
        m = 0
        for q in range(int(P.facet_nv[ft])):
            m |= 1 << int(chain[i, P.facet_verts[ft, q]])
        masks.append(m)
    return [i for i in range(len(masks) - 1)
            if not P.is_simplex[masks[i] | masks[i + 1]]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot", help="stall_snapshot.json (or an incomplete summary.json)")
    ap.add_argument("--complex", default=None,
                    help="complex key for gpu_sc.build (default: params.complex of the snapshot)")
    ap.add_argument("--h1-cap", type=int, default=2048, dest="h1_cap",
                    help="max leftover x chain H1 tests (0 disables the H1 audit)")
    a = ap.parse_args()

    s = json.loads(Path(a.snapshot).read_text())
    name = a.complex or (s.get("params") or {}).get("complex")
    if not name:
        sys.exit("no complex key: pass --complex (snapshot params carry the CLI name)")
    P = gpu_sc.build(name)
    k = len(s["best_chains"])
    chains = [np.array(c, np.int8) for c in s["best_chains"]]
    doms = [set(int(f) for f in P.domain_of(c)) for c in chains]
    covered = np.zeros(P.nf, bool)
    for d in doms:
        covered[list(d)] = True
    L = [int(f) for f in np.flatnonzero(~covered)]

    do_h1 = a.h1_cap > 0 and len(L) * k <= a.h1_cap
    doors, leftovers = [], []
    for f in L:
        entry = {"facet": f, "bad_min": int(min(int(P.chain_bad_counts(c)[f]) for c in chains)),
                 "absorbers": []}
        for j in range(k):
            links = bad_links(P, chains[j], f)
            rank = dim = None
            if do_h1:
                dim, rank = h1_map_obstruction(P, sorted(doms[j]) + [f])
            entry["absorbers"].append({"chain": j, "domain": len(doms[j]),
                                       "bad_links": links, "h1_rank": rank, "h1_dim": dim})
            if rank == 0:
                doors.append({"facet": f, "chain": j, "bad_links": links,
                               "seam_gap": links[-1] - links[0] if len(links) > 1 else None})
        leftovers.append(entry)

    suggestion = {
        "guide_targets": sorted({d["facet"] for d in doors}),
        "seam_targets": [{"facet": d["facet"], "chain": d["chain"],
                          "bad_links": d["bad_links"], "gap": d["seam_gap"]}
                         for d in doors if d["seam_gap"] is not None],
        "depth2_without_door": [
            {"facet": e["facet"],
             "chains": [ab["chain"] for ab in e["absorbers"] if len(ab["bad_links"]) >= 2]}
            for e in leftovers if e["bad_min"] >= 2],
        "reheat": s.get("stall_epochs", 0) >= 10,
        "h1_audited": do_h1,
    }
    audit = {"snapshot": str(a.snapshot), "complex": name, "coverage": int(covered.sum()),
             "num_facets": P.nf, "k": k, "domain_sizes": [len(d) for d in doms],
             "leftovers": leftovers, "doors": doors, "suggestion": suggestion}
    out = Path(a.snapshot).parent / "stall_audit.json"
    out.write_text(json.dumps(audit, indent=2))

    print(f"{name}: {int(covered.sum())}/{P.nf} covered, |L|={len(L)}, "
          f"domains {[len(d) for d in doms]}")
    if do_h1:
        print(f"H1 doors: {len(doors)}/{len(L)} leftovers have a rank-0 absorber")
        for d in doors:
            print(f"  facet {d['facet']} -> chain {d['chain']} (bad links {d['bad_links']}, "
                  f"gap {d['seam_gap']})")
    else:
        print(f"H1 audit skipped ({len(L)}x{k} > cap {a.h1_cap}); depth/seam profile only")
    print(f"depth histogram: {dict(sorted(np.bincount([e['bad_min'] for e in leftovers]).items()))}")
    print(f"stall_audit.json written to {out}")


if __name__ == "__main__":
    main()