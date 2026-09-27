"""C3: interpolation-walk feasibility for s3 (pure combinatorics).

If chain maps are restricted to interpolation maps phi^S (grid vertex v
takes pi2(v) iff v in flipped set S), a facet f over (s,t) stays chainable
across single-vertex flips iff:  all s\\t-vertices of f flip BEFORE any
t\\s-vertex of f.  If this precedence system has a linear extension over the
20 off-diagonal grid vertices, the single-vertex walk is a valid chain with
<= 20 links on D.  Test per domain D (constraints from facets in D).
"""
import sys, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_sc

P = gpu_sc.build("s3")
NV, NG, NF = P.nv, P.ngrid, P.nf

# off-diagonal grid vertices: pi1 != pi2
offdiag = [v for v in range(NG) if v // NV != v % NV]
idx = {v: i for i, v in enumerate(offdiag)}
n = len(offdiag)

def precedences(D):
    """edges u -> v : u must flip before v."""
    edges = set()
    for f in D:
        s = {g // NV for g in P.facets[f]}
        t = {g % NV for g in P.facets[f]}
        if s == t:
            continue
        sigma = (s - t).pop()
        tau = (t - s).pop()
        sig_verts = [idx[g] for g in P.facets[f] if g // NV == sigma and g in idx]
        tau_verts = [idx[g] for g in P.facets[f] if g % NV == tau and g in idx]
        for u in sig_verts:
            for w in tau_verts:
                edges.add((u, w))
    return edges

def has_linear_extension(n, edges):
    """Kahn's algorithm."""
    indeg = [0] * n
    out = [[] for _ in range(n)]
    for u, w in edges:
        out[u].append(w)
        indeg[w] += 1
    avail = [i for i in range(n) if indeg[i] == 0]
    seen = 0
    while avail:
        u = avail.pop()
        seen += 1
        for w in out[u]:
            indeg[w] -= 1
            if indeg[w] == 0:
                avail.append(w)
    return seen == n

def topo_order(n, edges):
    indeg = [0] * n
    out = [[] for _ in range(n)]
    for u, w in edges:
        out[u].append(w)
        indeg[w] += 1
    avail = sorted(i for i in range(n) if indeg[i] == 0)
    order = []
    while avail:
        u = avail.pop(0)
        order.append(u)
        for w in out[u]:
            indeg[w] -= 1
            if indeg[w] == 0:
                avail.append(w)
        avail.sort()
    return order

if __name__ == "__main__":
    import json
    print(f"off-diagonal grid vertices: {n}")
    # 1. full constraint set from all facets (hardest case)
    E = precedences(range(NF))
    print(f"all 500 facets: |edges|={len(E)} linear extension: "
          f"{has_linear_extension(n, E)}")
    # 2. per real cover domain
    for pdir in ["results/s3_starcover", "results/s3_k3_v2",
                 "results/s3_k3_run1", "results/s3_k3_c80"]:
        s = json.load(open(f"{pdir}/summary.json"))
        flags = []
        for d in s["domains"]:
            D = [int(x) for x in d]
            E = precedences(D)
            flags.append(has_linear_extension(n, E))
        print(f"{pdir}: per-domain linear extension: {flags}")
    # 3. random subsets of facets of increasing size
    import numpy as np
    rng = np.random.default_rng(5)
    for size in [50, 100, 200, 300, 400, 500]:
        okc = 0
        for _ in range(20):
            D = sorted(int(x) for x in rng.choice(NF, size=size, replace=False))
            if has_linear_extension(n, precedences(D)):
                okc += 1
        print(f"random |D|={size}: {okc}/20 admit linear extension")
    # 4. is there any 2-facet set forcing a cycle? scan pairs
    cyc = 0
    ex = None
    for f1, f2 in itertools.combinations(range(NF), 2):
        if not has_linear_extension(n, precedences([f1, f2])):
            cyc += 1
            if ex is None:
                ex = (f1, f2)
    print(f"facet pairs with precedence cycles: {cyc} (example {ex})")
