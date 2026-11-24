import itertools, sys
import spheres, sphere_core
N = int(sys.argv[1])
F, T, sph = spheres.build(N, verbose=False)
# rebuild sphere metadata: (type, w)
meta = []
for w in spheres.nondecr_maps(N):
    meta.append(('A', w)); meta.append(('B', w))
assert len(meta) == len(sph)
def fam(pred):
    sub = [sp for sp, (t, w) in zip(sph, meta) if pred(t, w)]
    r = sphere_core.sat_with(F, T, sub)
    return len(sub), ("SAT" if r else "UNSAT")
print("constants only          :", fam(lambda t, w: len(set(w)) == 1))
print("<=2 distinct values     :", fam(lambda t, w: len(set(w)) <= 2))
print("<=3 distinct values     :", fam(lambda t, w: len(set(w)) <= 3))
print("<=4 distinct values     :", fam(lambda t, w: len(set(w)) <= 4))
print("A-type only (all w)     :", fam(lambda t, w: t == 'A'))
print("B-type only (all w)     :", fam(lambda t, w: t == 'B'))
print("A-type, <=2 values      :", fam(lambda t, w: t == 'A' and len(set(w)) <= 2))
print("A-type, <=3 values      :", fam(lambda t, w: t == 'A' and len(set(w)) <= 3))
print("A-type, <=4 values      :", fam(lambda t, w: t == 'A' and len(set(w)) <= 4))
print("all                     :", fam(lambda t, w: True))
