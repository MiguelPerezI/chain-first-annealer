"""Validity check: the closures of genuine contiguity domains (recomputed from the annealer/CP-SAT chains with this
package's own staircase) must contain none of the rigid slice triangles, and the domains must cover every facet."""
import os
import json, sys
import relax, h1cegar, slices
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', '')
def domains(kf, chains):
    kf = [tuple(sorted(f)) for f in kf]; nv = 1 + max(v for f in kf for v in f)
    F = relax.staircase(kf, nv); simp = [set(f) for f in kf]
    out = []
    for ch in chains:
        dom = []
        for fi, (verts, s, t) in enumerate(F):
            ok = True
            for i in range(len(ch) - 1):
                U = {ch[i][g] for g in verts} | {ch[i + 1][g] for g in verts}
                if not any(U <= S for S in simp): ok = False; break
            if ok: dom.append(fi)
        out.append(dom)
    return out
for name, rdir in [('mobius', 'mobius_k3_sat'), ('rp2', 'rp2_k6'), ('klein', 'klein_k15_8233_v2')]:
    H, empties, cyc = slices.slice_family(name)
    S = json.load(open(R + rdir + '/summary.json'))
    doms = domains(H.kf, S['best_chains'])
    cov = set().union(*map(set, doms))
    res = []
    for D in doms:
        E = set()
        for f in D: E.update(H.fedges[f])
        res.append(sum(1 for cy in cyc if all(e in E for e in cy)))
    print(f"{name:7s} {rdir:20s}: {len(doms)} domains cover {len(cov)}/{H.nf} facets; rigid slice triangles contained per domain: {res}", flush=True)
