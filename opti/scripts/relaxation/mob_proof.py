import itertools, relax, h1cegar
C = relax.load_complexes(); kf = C['mobius']; nv = 5
H = h1cegar.H1(kf)
E = H.E
tris = [  # (first-coordinates, second-coordinates) of the four hollow triangles
    ((3, 3, 3), (0, 2, 3)),
    ((1, 3, 4), (1, 1, 1)),
    ((3, 3, 3), (1, 3, 4)),
    ((4, 4, 4), (1, 3, 4)),
]
T = {}   # edge -> set of facets containing it
for fi, es in enumerate(H.fedges):
    for e in es: T.setdefault(e, set()).add(fi)
edges_of = []
for (fa, sb) in tris:
    pts = [a * nv + b for a, b in zip(fa, sb)]
    es = [E[(pts[0], pts[1])], E[(pts[1], pts[2])], E[(pts[0], pts[2])]]
    edges_of.append(es)
    print("triangle", list(zip(fa, sb)), "edges:", [(H.edges[e][0] // nv, H.edges[e][0] % nv, H.edges[e][1] // nv, H.edges[e][1] % nv) for e in es],
          "facets per edge:", [len(T[e]) for e in es])
    assert not (T[es[0]] & T[es[1]]) and not (T[es[1]] & T[es[2]]) and not (T[es[0]] & T[es[2]])
ok = True
count = 0
for a in itertools.product(range(3), repeat=4):          # edge of triangle i declared 'absent from D1'  (all facets of it lie in D2)
    for b in itertools.product(range(3), repeat=4):      # edge of triangle i declared 'absent from D2'
        if any(a[i] == b[i] for i in range(4)): continue
        A = [edges_of[i][a[i]] for i in range(4)]; B = [edges_of[i][b[i]] for i in range(4)]
        count += 1
        if not any(T[x] & T[y] for x in A for y in B):
            ok = False; print("COUNTEREXAMPLE", a, b)
print("assignments checked:", count, "-> every one has an edge of A and an edge of B sharing a facet:", ok)
