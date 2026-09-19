#!/usr/bin/env python3
"""make_certificate_md.py — human-readable certificate (all domains, all
planner maps, all facets) from an audited results directory written by
gpu_sc.write_certificate / star_cover.py / hybrid_cover.py.

usage: make_certificate_md.py <complex> <results dir> <out.md> "<one-paragraph description of how it was obtained>"
"""
import sys, json, re
from pathlib import Path
import numpy as np
import gpu_sc


def parse_planners(path, nv):
    """planners.txt -> list of chains (list of int8 arrays, '@' -> -1)."""
    chains, cur, rows = [], None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith("=== domain"):
            if cur is not None:
                chains.append(cur)
            cur = []
        elif re.match(r"\s*map \d+:", line):
            rows = []
            cur.append(rows)
        elif cur is not None and re.match(r"^\s{4}\S", line) and cur and isinstance(cur[-1], list) and len(cur[-1]) < nv:
            cur[-1].append([-1 if x == "@" else int(x) for x in line.split()])
    if cur is not None:
        chains.append(cur)
    return [[np.array(sum(rows, []), np.int64) for rows in ch] for ch in chains]


def main():
    name, resdir, out, how = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]
    P = gpu_sc.build(name); nv = P.nv
    summ = json.loads((resdir / "summary.json").read_text())
    domains = [sorted(int(f) for f in d) for d in summ["domains"]]
    chains = parse_planners(resdir / "planners.txt", nv)
    assert len(chains) == len(domains), (len(chains), len(domains))
    # re-verify with the plain definition (entries outside the domain are irrelevant; fill with pi1/pi2-consistent values)
    parts = []
    for facs, ch in zip(domains, chains):
        full = []
        for m in ch:
            mm = m.copy(); mm[mm < 0] = P.pi1[mm < 0]
            full.append(mm.astype(np.int8))
        parts.append((facs, full))
    # chain endpoints on the domain must be pi1 / pi2; interior '@' entries are never read on the domain's facets
    gpu_sc.verify_system(P, parts)
    L = [f"# Verified {len(domains)}-domain motion-planner system on {P.name}: SC_strict ≤ {len(domains) - 1}\n",
         f"Source: `{resdir}` (summary.json, cover.txt, planners.txt), re-verified here with the plain Definition 2.3 check "
         f"(`gpu_sc.verify_system`) and audited by `independent_check.py` (shipped in this repository).\n",
         "## How it was obtained\n", how + "\n",
         "## Setup\n",
         f"`K` = {P.name}: {nv} vertices `0..{nv-1}`, {len(P.kfacets)} facets:\n",
         "```\n" + "\n".join(f"  facet {i:2d}: {list(f)}" for i, f in enumerate(P.kfacets)) + "\n```\n",
         f"`K × K` is the staircase (ordered) product on the `{nv} × {nv}` grid; grid vertex `(a, b)` has id `{nv}a + b`; the "
         f"{P.nf} maximal facets have {P.maxfv} vertices. Facet ids enumerate pairs of facets `(i, j)` of `K` (`i` slowest), then the "
         f"staircase paths as bit words with as many 1‑bits as `dim(facet_j)` in increasing numerical order, read from the lowest bit "
         f"(0 = step in the first factor, 1 = step in the second). `π₁(a, b) = a`, `π₂(a, b) = b`.\n",
         "## Summary\n", "| part | facets | chain links |", "|---:|---:|---:|"]
    for i, (facs, ch) in enumerate(zip(domains, chains)):
        L.append(f"| J{i} | {len(facs)} | {len(ch) - 1} |")
    L.append(f"\nTotal {sum(map(len, domains))} = {P.nf} facets, pairwise disjoint.\n")
    for i, (facs, ch) in enumerate(zip(domains, chains)):
        L.append(f"\n## Part J{i} — {len(facs)} facets, {len(ch) - 1} links\n")
        L.append(f"Planner maps on the vertices of J{i} ('@' = grid vertex not in the domain); rows `a = 0..{nv-1}`, columns `b = 0..{nv-1}`. "
                 f"Map 0 is `π₁`, the last map is `π₂`; consecutive maps are 1‑contiguous on every facet of J{i}.\n")
        for mi, m in enumerate(ch):
            rows = [" ".join("@" if m[a * nv + b] < 0 else str(int(m[a * nv + b])) for b in range(nv)) for a in range(nv)]
            L.append(f"map {mi}:\n```\n" + "\n".join("  " + r for r in rows) + "\n```")
        L.append(f"\nFacets of J{i} (id: grid vertices):\n```")
        for f in facs:
            L.append(f"  {f:4d}: " + " ".join(f"({g // nv},{g % nv})" for g in P.facets[f]))
        L.append("```")
    Path(out).write_text("\n".join(L) + "\n")
    print(f"wrote {out}: {len(domains)} domains, {sum(map(len, domains))} facets, chain lengths {[len(c) - 1 for c in chains]}")


if __name__ == "__main__":
    main()
