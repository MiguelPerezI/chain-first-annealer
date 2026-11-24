import itertools, sys, random
import spheres, sphere_core
N = int(sys.argv[1]); maxv = int(sys.argv[2]); seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
F, T, sph = spheres.build(N, verbose=False)
meta = []
for w in spheres.nondecr_maps(N):
    meta.append(('A', w)); meta.append(('B', w))
idx = [i for i, (t, w) in enumerate(meta) if len(set(w)) <= maxv]
cur = list(idx)
assert not sphere_core.sat_with(F, T, [sph[i] for i in cur])
random.seed(seed); random.shuffle(cur)
i = 0
while i < len(cur):
    trial = cur[:i] + cur[i + 1:]
    if not sphere_core.sat_with(F, T, [sph[j] for j in trial]): cur = trial
    else: i += 1
print(f"N={N} (<= {maxv} distinct values), seed {seed}: minimal unsat family has {len(cur)} spheres")
for i in sorted(cur, key=lambda i: (meta[i][0], meta[i][1])): print("  ", meta[i])
