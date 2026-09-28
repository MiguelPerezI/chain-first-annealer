"""Aut(K) as vertex permutations, with its diagonal action on K x K.

On the hemi-icosahedron this is A5 (order 60, triangle transitive,
rp2_new_mathematics.md section 2). The functions are generic: any complex
built by gpu_sc.build works, and the group found is that complex's own.
"""
import itertools

import numpy as np


def automorphisms(P):
    """All vertex permutations that map the maximal simplices of K onto themselves."""
    K = set(P.kfacets)
    out = []
    for perm in itertools.permutations(range(P.nv)):
        if all(tuple(sorted(perm[v] for v in t)) in K for t in K):
            out.append(perm)
    return out


def act_grid(perm, g, nv):
    """Diagonal action on the grid vertex g = a*nv + b."""
    a, b = divmod(g, nv)
    return perm[a] * nv + perm[b]


_FACET_INDEX = {}


def _facet_index(P):
    key = id(P)
    if key not in _FACET_INDEX:
        _FACET_INDEX[key] = {
            tuple(sorted(int(v) for v in P.facet_verts[f][:P.facet_nv[f]])): f
            for f in range(P.nf)}
    return _FACET_INDEX[key]


def act_facet(P, perm, f):
    """Facet id of the image of facet f under the diagonal action of perm."""
    idx = _facet_index(P)
    verts = [int(v) for v in P.facet_verts[f][:P.facet_nv[f]]]
    return idx[tuple(sorted(act_grid(perm, v, P.nv) for v in verts))]


def preserves_facets(P, perm):
    """True iff the diagonal action of perm maps every staircase facet to a
    staircase facet. On RP2 only the identity does: staircases are monotone in
    the vertex order, so the facet-level action is not available for the split."""
    idx = _facet_index(P)
    for f in range(P.nf):
        verts = [int(v) for v in P.facet_verts[f][:P.facet_nv[f]]]
        if tuple(sorted(act_grid(perm, v, P.nv) for v in verts)) not in idx:
            return False
    return True


def act_pair(perm, pair):
    """Diagonal action on a pair (s, t) of maximal simplices of K."""
    s, t = pair
    return (tuple(sorted(perm[v] for v in s)), tuple(sorted(perm[v] for v in t)))


def pair_orbit_reps(P, G):
    """Orbit representatives of the |K|^2 triangle pairs under the diagonal
    action; the level at which the case split is done."""
    pairs = [(s, t) for s in P.kfacets for t in P.kfacets]
    return orbit_reps(pairs, act_pair, G)


def facet_perm_table(P, G):
    """|G| x nf table: row g, column f -> act_facet(P, G[g], f)."""
    return np.array([[act_facet(P, g, f) for f in range(P.nf)] for g in G], dtype=np.int32)


def _key(item):
    return tuple(item) if isinstance(item, (list, np.ndarray)) else item


def orbit_reps(items, act, G):
    """Representatives of the orbits of `items` under G, with act(g, item).

    Returns (reps, covered). `covered` counts the items that lie in the orbit
    of some representative; it must equal len(items) for the split to be
    exhaustive, and callers assert exactly that.
    """
    items = list(items)
    seen, reps = set(), []
    for it in items:
        if _key(it) in seen:
            continue
        reps.append(it)
        for g in G:
            seen.add(_key(act(g, it)))
    covered = sum(1 for it in items if _key(it) in seen)
    return reps, covered
