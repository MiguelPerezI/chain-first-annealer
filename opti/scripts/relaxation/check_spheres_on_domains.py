import os
import json, sys, itertools
import relax, spheres
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', '')
N = 5
F, T, sph = spheres.build(N, verbose=False)
kf = spheres.boundary_simplex(N)
simp = [set(f) for f in kf]
def chain_domain(chain):
    dom = []
    for fi, (verts, s, t) in enumerate(F):
        ok = True
        for i in range(len(chain) - 1):
            U = {chain[i][g] for g in verts} | {chain[i + 1][g] for g in verts}
            if not any(U <= S for S in simp): ok = False; break
        if ok: dom.append(fi)
    return dom
def contains_sphere(D):
    Dset = set(D)
    bad = 0
    for sp in sph:
        if all(any(f in Dset for f in T[z]) for z in sp): bad += 1
    return bad
for name in ['s3_k3_v2', 's3_k2_A', 's3_k2_D', 's3_k2_longflat_5', 's3_k2_longcor_5', 's3_k2_v2_point', 's3_k2_v2_track', 's3_k2_B']:
    try:
        S = json.load(open(R + name + '/summary.json'))
    except Exception as e:
        print(name, 'ERR', e); continue
    ch = S.get('best_chains')
    if not ch: print(name, 'no chains'); continue
    res = []
    for c in ch:
        D = chain_domain(c)
        res.append((len(D), contains_sphere(D)))
    print(name, 'domains (size, #rigid spheres contained):', res, flush=True)
# star rectangles
for (v, w) in [(0, 0), (1, 1), (2, 2), (3, 2), (0, 4)]:
    D = [i for i, (vs, s, t) in enumerate(F) if v in s and w in t]
    print('rect', (v, w), len(D), contains_sphere(D))
# negative control: the whole complex and the diagonal blocks
print('all facets', contains_sphere(range(len(F))))
