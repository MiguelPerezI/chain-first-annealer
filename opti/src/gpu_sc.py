"""
gpu_sc.py — GPU-accelerated versions of the optimized algorithms in
optimized_algorithms.md for the paper

    Ortiz, Lara, González, Borat,
    "A randomized greedy algorithm for piecewise linear motion planning",
    arXiv:2008.13290v2.

Two CUDA kernels (numba), self-contained (only numpy + numba):

  * chain_anneal_kernel  — the "chain-first" formulation (Section 10 of the
    document): a chain of vertex maps Phi = (pi1 = phi_0, ..., phi_c = pi2)
    on the WHOLE vertex grid of L = K x K certifies, for free, the domain
        D(Phi) = { sigma in F_L : phi_i(sigma) U phi_{i+1}(sigma) in K for all i }.
    A k-tuple of chains whose domains cover F_L is a system of k piecewise
    linear motion planners.  Thousands of GPU threads run independent
    simulated-annealing walks on k-tuples of chains, maximizing the weighted
    number of facets covered by at least one chain.  Every accepted move
    costs O(|star(w)|) simplex look-ups.

  * ls_kernel — the optimized LocalSearch (Algorithm 1) of the document,
    batched the way the Abstract_Motion_GPU port batches the paper's walk:
    one CUDA block per test, one thread per independent walk, arbitrary
    start/target maps (warm starts), exact enumeration of admissible moves
    via Ext bitmasks, defect-weighted vertex choice, steepest descent with
    exploration probability r, tabu on immediate reversal, and termination
    as soon as the current map is 1-contiguous to the target.  A CPU twin
    (same code, njit) replays a winning walk deterministically.

Facet ids of K x K follow the Abstract_Motion_GPU staircase order, so the
output directories are auditable with that repo's independent_check.py.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from numbers import Integral
from validation import validate_parameters
from pathlib import Path

import numpy as np
from numba import cuda, njit, int8, int32, uint8, uint32, uint64, float32, float64

# --------------------------------------------------------------------------- #
#  splitmix64: identical CPU / GPU twins (one implementation, two decorators)  #
# --------------------------------------------------------------------------- #
GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MUL1 = np.uint64(0xBF58476D1CE4E5B9)
_MUL2 = np.uint64(0x94D049BB133111EB)
_INV53 = 1.0 / 9007199254740992.0


def _sm_next(state):
    state = state + GOLDEN
    z = state
    z = (z ^ (z >> np.uint64(30))) * _MUL1
    z = (z ^ (z >> np.uint64(27))) * _MUL2
    z = z ^ (z >> np.uint64(31))
    return state, z


def _sm_u01(x):
    return np.float64(x >> np.uint64(11)) * _INV53


sm_next_cpu = njit(inline="always")(_sm_next)
sm_u01_cpu = njit(inline="always")(_sm_u01)
sm_next_dev = cuda.jit(device=True, inline=True)(_sm_next)
sm_u01_dev = cuda.jit(device=True, inline=True)(_sm_u01)


# --------------------------------------------------------------------------- #
#  K and the staircase product L = K x K (facet order == Abstract_Motion_GPU)  #
# --------------------------------------------------------------------------- #
class Product:
    """K given by maximal facets over {0..nv-1}; L = K x K with the staircase
    triangulation; every lookup table the kernels need, as flat numpy arrays.

    Grid vertex (a, b) of K x K has id  a*nv + b;  pi1(id) = a, pi2(id) = b.
    """

    def __init__(self, kfacets, name="K"):
        self.name = name
        self.kfacets = [tuple(sorted(f)) for f in kfacets]
        self.nv = 1 + max(v for f in self.kfacets for v in f)
        nv = self.nv
        # What actually bounds nv: is_simplex is a 2^nv byte table and the
        # maps' vertex masks are int32, so nv <= 30; 24 keeps the table at
        # 16 MB.  The grid-id arrays (facet_verts: a product facet's vertices
        # are grid ids, up to ngrid - 1) are int16, which allows nv <= 181 and
        # is why the old nv <= 11 int8 guard was raised.
        assert nv <= 24, (f"is_simplex table would be 2^{nv} bytes; the "
                          f"streaming tools (make_definition_md.py / "
                          f"stream_starcover.py) cover larger K")
        self.ngrid = nv * nv
        # is_simplex[S] for S a bitmask over V_K: S is a simplex iff it is a
        # subset of some facet.  Enumerate the subsets of each facet (2^|facet|
        # of them) rather than filtering all 2^nv masks — for the dodecahedron
        # that is 36 x 8 instead of 1,048,576 full scans.
        masks = [sum(1 << v for v in f) for f in self.kfacets]
        tab = np.zeros(1 << nv, dtype=np.uint8)
        for m in masks:
            s = m
            while True:
                tab[s] = 1
                if s == 0:
                    break
                s = (s - 1) & m
        self.is_simplex = tab
        # BFS distances on the 1-skeleton
        adj = np.zeros((nv, nv), bool)
        for f in self.kfacets:
            for a in f:
                for b in f:
                    if a != b:
                        adj[a, b] = True
        dk = np.full((nv, nv), 127, np.int8)
        for s in range(nv):
            dist = np.full(nv, -1, np.int32)
            dist[s] = 0
            frontier = [s]
            d = 0
            while frontier:
                d += 1
                nxt = []
                for u in frontier:
                    for v in range(nv):
                        if adj[u, v] and dist[v] < 0:
                            dist[v] = d
                            nxt.append(v)
                frontier = nxt
            assert (dist >= 0).all(), "K must be connected"
            dk[s] = dist
        self.dk = dk.reshape(-1)
        # staircase facets in the GPU repo's order (facet i of K, facet j of K,
        # paths by increasing bitmask with popcount == dim(tau))
        facets = []
        for s0 in self.kfacets:
            for s1 in self.kfacets:
                p, q = len(s0) - 1, len(s1) - 1
                for mask in range(1 << (p + q)):
                    bits = [(mask >> b) & 1 for b in range(p + q)]
                    if sum(bits) != q:
                        continue
                    pi = pj = 0
                    tet = [s0[0] * nv + s1[0]]
                    for step in bits:
                        if step == 0:
                            pi += 1
                        else:
                            pj += 1
                        tet.append(s0[pi] * nv + s1[pj])
                    facets.append(tet)
        self.facets = facets
        self.nf = len(facets)
        self.maxfv = max(len(f) for f in facets)
        # int16: these are grid ids (a*nv + b), not K-vertex ids, so they reach
        # ngrid - 1 = nv^2 - 1 -- 399 for the 20-vertex dodecahedron, past int8
        fv = np.zeros((self.nf, self.maxfv), np.int16)
        fnv = np.zeros(self.nf, np.int8)
        for i, f in enumerate(facets):
            fnv[i] = len(f)
            fv[i, : len(f)] = f
        self.facet_verts, self.facet_nv = fv, fnv
        # incidence CSR: grid vertex -> facets
        counts = np.zeros(self.ngrid, np.int32)
        for f in facets:
            for v in f:
                counts[v] += 1
        off = np.zeros(self.ngrid + 1, np.int32)
        np.cumsum(counts, out=off[1:])
        inc = np.zeros(off[-1], np.int32)
        cur = off[:-1].copy()
        for fi, f in enumerate(facets):
            for v in f:
                inc[cur[v]] = fi
                cur[v] += 1
        self.incid_off, self.incid_fac = off, inc
        self.starmax = int(counts.max())
        self.pi1 = np.array([g // nv for g in range(self.ngrid)], np.int8)
        self.pi2 = np.array([g % nv for g in range(self.ngrid)], np.int8)
        self.maskw = (self.nf + 63) // 64

    # ---- host-side helpers -------------------------------------------------
    def verts_of(self, facet_ids):
        s = set()
        for f in facet_ids:
            s.update(self.facets[f])
        return sorted(s)

    def chain_bad_counts(self, chain):
        """bad[f] = number of links of the chain that are not 1-contiguous on
        facet f (vectorized).  chain: int array [c+1, ngrid]."""
        ch = np.asarray(chain, dtype=np.int64)
        fv = self.facet_verts.astype(np.int64)
        valid = self.facet_nv[:, None] > np.arange(self.maxfv)[None, :]
        img = ch[:, fv]                                   # [c+1, nf, maxfv]
        m = np.zeros(img.shape[:2], np.int64)
        for q in range(self.maxfv):
            m |= np.where(valid[None, :, q], 1 << img[:, :, q], 0)
        link = m[:-1] | m[1:]
        ok = self.is_simplex[link]
        return (ok == 0).sum(0)

    def domain_of(self, chain):
        return np.flatnonzero(self.chain_bad_counts(chain) == 0)


def build(name):
    if name == "circle":
        return Product([[0, 1], [0, 2], [1, 2]], "∂Δ² (S¹)")
    if name == "s2":
        return Product(list(itertools.combinations(range(4), 3)), "∂Δ³ (S²)")
    if name == "s3":
        return Product(list(itertools.combinations(range(5), 4)), "∂Δ⁴ (S³)")
    if name == "s4":
        return Product(list(itertools.combinations(range(6), 5)), "∂Δ⁵ (S⁴)")
    if name == "s5":
        return Product(list(itertools.combinations(range(7), 6)), "∂Δ⁶ (S⁵)")
    if name == "s6":
        return Product(list(itertools.combinations(range(8), 7)), "∂Δ⁷ (S⁶)")
    if name == "s7":
        return Product(list(itertools.combinations(range(9), 8)), "∂Δ⁸ (S⁷)")
    if name == "s8":
        return Product(list(itertools.combinations(range(10), 9)), "∂Δ⁹ (S⁸)")
    if name == "s9":
        return Product(list(itertools.combinations(range(11), 10)), "∂Δ¹⁰ (S⁹)")
    if name == "wedge":
        return Product([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3]], "wedge of two circles")
    if name == "bary":
        # The barycenter triangle: Δ² = {0,1,2} subdivided by its barycenter 3
        # into the three triangles {0,1,3}, {0,2,3}, {1,2,3}.  A cone on ∂Δ²
        # (3 lies in every facet), hence contractible: a single motion planner
        # on all of K x K exists, so the target is one domain, SC_strict = 0.
        return Product([[0, 1, 3], [0, 2, 3], [1, 2, 3]], "barycenter triangle (cone on ∂Δ²)")
    if name == "cube":
        # The cube boundary: 8 vertices 0..7 and the six square faces
        # {0,3,7,4}, {1,2,6,5}, {0,1,5,4}, {2,3,7,6}, {4,5,6,7}, {0,1,2,3},
        # each split along its drawn diagonal (0-7, 1-6, 0-5, 2-7, 4-6, 0-2)
        # into two triangles: 12 triangles, f-vector (8, 18, 12), every edge in
        # exactly two of them, every vertex link a single cycle, chi = 2 and
        # orientable -- a triangulated S^2 (a hexahedron, not ∂Δ³).
        return Product([[0, 3, 7], [0, 7, 4], [1, 2, 6], [1, 6, 5], [0, 1, 5],
                        [0, 5, 4], [2, 3, 7], [2, 7, 6], [4, 5, 6], [4, 6, 7],
                        [0, 1, 2], [0, 2, 3]],
                       "cube boundary (8 vertices, 12 triangles)")
    if name == "dodeca":
        # The dodecahedron boundary: 20 vertices 0..19, 12 pentagonal faces,
        # each pentagon fanned from one of its vertices into 3 triangles ->
        # 36 triangles, f-vector (20, 54, 36), every edge in exactly two
        # triangles, chi = 2 -- a triangulated S^2 like the cube, but with no
        # boundary-simplex structure and, unlike ∂Δ³, no symmetry group small
        # enough to give the product family a cheap cover.
        # (facet list from Documents/c++_scripts/dodeca.hpp, Dodecahedron::tri_)
        return Product([(3, 14, 19), (13, 14, 19), (3, 4, 19), (8, 13, 19),
                        (8, 12, 19), (12, 16, 19), (4, 16, 19), (0, 4, 16),
                        (0, 6, 16), (0, 2, 3), (0, 1, 2), (0, 3, 4),
                        (2, 15, 17), (2, 14, 15), (2, 3, 14), (8, 13, 14),
                        (8, 14, 15), (8, 9, 15), (5, 6, 12), (5, 11, 12),
                        (6, 12, 16), (8, 9, 12), (9, 10, 12), (10, 11, 12),
                        (9, 10, 15), (10, 15, 17), (10, 17, 18), (7, 10, 18),
                        (5, 7, 10), (5, 10, 11), (2, 17, 18), (2, 7, 18),
                        (1, 2, 7), (0, 1, 7), (0, 5, 7), (0, 5, 6)],
                       "dodecahedron boundary (20 vertices, 36 triangles)")
    if name == "torus":
        f = []
        for i in range(7):
            f.append(sorted((i, (i + 1) % 7, (i + 3) % 7)))
            f.append(sorted((i, (i + 2) % 7, (i + 3) % 7)))
        return Product(f, "Császár torus")
    if name == "rp2":
        # The vertex-minimal triangulation of the real projective plane RP²:
        # 6 vertices 0..5, 10 triangles, f-vector (6, 15, 10), chi = 1,
        # every pair of vertices spans an edge (1-skeleton = K6).
        # 6 vertices is provably minimal: a triangulation of RP² with V
        # vertices has F = 2(V-1) faces (Euler + 3F=2E), so V >= 6 to get
        # F >= 10 = C(V,2)-V+1.  The type is unique up to relabelling
        # (it is the hemi-icosahedron: antipodal identification of the
        # icosahedron's 12 vertices, 30 edges, 20 faces -> 6, 15, 10).
        return Product([(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5), (0, 1, 5),
                        (1, 2, 4), (2, 3, 5), (3, 4, 1), (4, 5, 2), (5, 1, 3)],
                       "RP² (6 vertices, 10 triangles, hemi-icosahedron)")
    if name == "klein":
        # The vertex-minimal triangulation of the Klein bottle: 8 vertices 0..7,
        # 16 triangles, f-vector (8, 24, 16), chi = 0, valence vector
        # (7,7,6,6,6,6,5,5).  It is the "242" member of Cervone's
        # classification of the six combinatorially-distinct 8-vertex
        # triangulations of K^2 (Vertex-Minimal Simplicial Immersions of the
        # Klein Bottle in Three Space, Figure 5.10), relabelled
        # A,B,...,H -> 0,1,...,7.  8 vertices is provably minimal: the
        # Heawood-type bound V^2 - 7V + 6*chi >= 0 gives V >= 7, and
        # Franklin's 6-colour theorem for the Klein bottle rules out V = 7
        # (a 7-vertex Klein bottle would be 2-neighbourly, hence its
        # 1-skeleton K7 would embed in K^2, needing 7 colours).
        # independent_check.check_klein re-proves ||K|| ~= K^2 from this
        # list alone: closed surface, chi = 0, no coherent orientation.
        return Product([(0, 1, 2), (1, 2, 6), (0, 2, 4), (0, 4, 6),
                        (0, 3, 6), (0, 1, 5), (1, 3, 5), (0, 5, 7),
                        (0, 3, 7), (1, 3, 7), (2, 3, 5), (2, 3, 6),
                        (2, 4, 5), (4, 5, 7), (1, 4, 7), (1, 4, 6)],
                       "Klein bottle (8 vertices, 16 triangles, Cervone 242)")
    if name == "mobius":
        # The vertex-minimal triangulation of the Möbius band: 5 vertices,
        # 5 triangles {i, i+1, i+2} (mod 5), f-vector (5, 10, 5), chi = 0,
        # every vertex on the boundary, each in exactly 3 triangles.  The
        # 1-skeleton is K5: the 5 "consecutive" edges (i, i+1) lie in 2
        # triangles each (the central circle), the 5 "pentagram" edges
        # (i, i+2) in 1 (together the single boundary 5-cycle 0-2-4-1-3).
        # 5 vertices is provably minimal and the type unique up to
        # relabelling: with b boundary edges, 3F = 2(E-b) + b and chi = 0
        # give E = 3V - b; the boundary cycle is simple (b <= V) and
        # E <= C(V,2), so 2V <= C(V,2), V >= 5.  At V = 5 everything is
        # forced (b = 5, E = 10 = K5, F = 5) and an exhaustive search over
        # the C(10,5) candidate lists finds one combinatorial type (12
        # labelings; Aut = D5, order 10).  independent_check.check_mobius
        # re-proves ||K|| = Möbius band from this list alone: manifold with
        # boundary, one boundary circle, chi = 0, non-orientable.
        return Product([(i, (i + 1) % 5, (i + 2) % 5) for i in range(5)],
                       "Möbius band (5 vertices, 5 triangles, minimal & unique)")
    if name == "s1xw":
        # S^1 x (S^1 v S^1) = the polyhedral product (S^1,*)^K for K the flag
        # path P_3 (the smallest connected flag complex beyond the edge,
        # whose toric complex is the torus).  Aspherical: it is the
        # classifying space of the right-angled Artin group Z x F_2.  The
        # triangulation is the staircase product of this repo's own circle
        # (0,1),(0,2),(1,2) with this repo's own wedge
        # (0,1),(0,2),(0,3),(1,2),(1,3): grid vertex (a,b) gets id a*4+b,
        # so the 12 product vertices are 0..11 and the 3x5 squares split
        # into 30 triangles.  f-vector (12, 42, 30), chi = 0, homology
        # (Z, Z^3, Z^2) by Kunneth (b_1 = 1 + 2, b_2 = 1*2, chi = 0*(-1)).
        # TC = 4 EXACTLY: zero-divisor cup length 3 over Z/2 (g_t g_a g_b
        # survives, e.g. the term tb (x) a), and TC(X x Y) <= TC(X) +
        # TC(Y) - 1 = 2 + 3 - 1 with TC(S^1) = 2, TC(S^1 v S^1) = 3.
        # independent_check.check_s1xw re-proves the staircase structure and
        # the homology from this list alone; the goal for the annealer is
        # 4 domains without subdivision (SC_strict = 3 = TC - 1).
        return Product([(0, 1, 5), (0, 4, 5), (0, 2, 6),
                        (0, 4, 6), (0, 3, 7), (0, 4, 7),
                        (1, 2, 6), (1, 5, 6), (1, 3, 7),
                        (1, 5, 7), (0, 1, 9), (0, 8, 9),
                        (0, 2, 10), (0, 8, 10), (0, 3, 11),
                        (0, 8, 11), (1, 2, 10), (1, 9, 10),
                        (1, 3, 11), (1, 9, 11), (4, 5, 9),
                        (4, 8, 9), (4, 6, 10), (4, 8, 10),
                        (4, 7, 11), (4, 8, 11), (5, 6, 10),
                        (5, 9, 10), (5, 7, 11), (5, 9, 11)],
                       "S¹×(S¹∨S¹) (polyhedral product (S¹,*)^P₃, 12 vertices, 30 triangles)")
    raise ValueError(name)


# --------------------------------------------------------------------------- #
#  Kernel A: joint k-chain simulated annealing ("chain-first" covering)        #
# --------------------------------------------------------------------------- #
def make_anneal_kernel(KCH, C1, NGRID, NF, NV, STARMAX, LMAX=8):
    """Compile the annealer for fixed sizes (numba needs constant local shapes).

    Move = "track move": chain j, vertex w, a block of consecutive interior
    maps i1..i2 (length L, L=1 is the point mutation), new value u for
    phi_i(w), i1 <= i <= i2.  Only the links i1-1..i2 on the facets of
    star(w) change.  Energy (minimized):
        E = sum_f weights[f] * pen(min_j bad_j[f]),  pen(0)=0, pen(b)=1+beta*b
        + gamma * sum_{j,f} bad_j[f]   (per-chain cleanliness: every chain is kept
        contiguous wherever it can be, not only where it is the unique cover)
    Coverage (maximized, unweighted, drives best-state tracking):
        cov = #{ f : min_j bad_j[f] == 0 }.
    """
    CMAX = C1 - 1
    L1 = LMAX + 2

    @cuda.jit
    def anneal_kernel(chains, best, bad, score, seeds,
                      facet_verts, facet_nv, incid_off, incid_fac, is_simplex,
                      weights, iters, T0, T1, beta, gamma, mu, pmove_copy, ptrack):
        s = cuda.grid(1)
        if s >= chains.shape[0]:
            return
        ch = cuda.local.array((KCH, C1, NGRID), int8)
        bd = cuda.local.array((KCH, NF), uint8)
        dl = cuda.local.array(STARMAX, int8)
        mo_ = cuda.local.array(L1, int32)   # old masks of the touched maps on one facet
        mn_ = cuda.local.array(L1, int32)   # new masks

        for j in range(KCH):
            for i in range(C1):
                for g in range(NGRID):
                    ch[j, i, g] = chains[s, j, i, g]
        for j in range(KCH):
            for f in range(NF):
                cnt = 0
                mprev = int32(0)
                for q in range(facet_nv[f]):
                    mprev |= int32(1) << int32(ch[j, 0, facet_verts[f, q]])
                for i in range(1, C1):
                    mcur = int32(0)
                    for q in range(facet_nv[f]):
                        mcur |= int32(1) << int32(ch[j, i, facet_verts[f, q]])
                    if is_simplex[mprev | mcur] == 0:
                        cnt += 1
                    mprev = mcur
                bd[j, f] = uint8(cnt)
        cov = int32(0)
        for f in range(NF):
            b = bd[0, f]
            for j in range(1, KCH):
                if bd[j, f] < b:
                    b = bd[j, f]
            if b == 0:
                cov += 1
        bestcov = cov
        for j in range(KCH):
            for i in range(C1):
                for g in range(NGRID):
                    best[s, j, i, g] = ch[j, i, g]

        state = seeds[s]
        state, x = sm_next_dev(state)
        lnr = math.log(T1 / T0)
        for it in range(iters):
            T = T0 * math.exp(lnr * (float64(it) / float64(iters)))
            state, x = sm_next_dev(state)
            j = int32(x % uint64(KCH))
            state, x = sm_next_dev(state)
            w = int32(x % uint64(NGRID))
            # block length
            state, x = sm_next_dev(state)
            L = int32(1)
            if sm_u01_dev(x) < ptrack:
                state, x = sm_next_dev(state)
                L = int32(2) + int32(x % uint64(LMAX - 1))
            if L > CMAX - 1:
                L = CMAX - 1
            state, x = sm_next_dev(state)
            i1 = int32(1) + int32(x % uint64(CMAX - L))      # i1..i2 inside 1..CMAX-1
            i2 = i1 + L - 1
            old = ch[j, i1, w]
            state, x = sm_next_dev(state)
            pm = sm_u01_dev(x)
            state, x = sm_next_dev(state)
            if pm < pmove_copy:
                if (x & uint64(1)) == uint64(0):
                    u = ch[j, i1 - 1, w]
                else:
                    u = ch[j, i2 + 1, w]
                if u == old:
                    state, x = sm_next_dev(state)
                    kk = int32(x % uint64(NV - 1))
                    u = kk if kk < old else kk + 1
            else:
                kk = int32(x % uint64(NV - 1))
                u = kk if kk < old else kk + 1

            # ---- delta over the star of w: links i1-1 .. i2 ----
            dE = float64(0.0)
            dcov = int32(0)
            idx = 0
            for ii in range(incid_off[w], incid_off[w + 1]):
                f = incid_fac[ii]
                ob = int32(0)
                nb = int32(0)
                if L == 1:
                    # fast path (point mutation): three masks in registers
                    mprev = int32(0)
                    mnext = int32(0)
                    mcur = int32(0)
                    for q in range(facet_nv[f]):
                        v = facet_verts[f, q]
                        mprev |= int32(1) << int32(ch[j, i1 - 1, v])
                        mnext |= int32(1) << int32(ch[j, i1 + 1, v])
                        if v != w:
                            mcur |= int32(1) << int32(ch[j, i1, v])
                    oc = mcur | (int32(1) << int32(old))
                    nc = mcur | (int32(1) << int32(u))
                    if is_simplex[mprev | oc] == 0:
                        ob += 1
                    if is_simplex[oc | mnext] == 0:
                        ob += 1
                    if is_simplex[mprev | nc] == 0:
                        nb += 1
                    if is_simplex[nc | mnext] == 0:
                        nb += 1
                else:
                    # masks of maps i1-1 .. i2+1 on facet f, old and new
                    for l in range(L + 2):
                        i = i1 - 1 + l
                        m = int32(0)
                        for q in range(facet_nv[f]):
                            v = facet_verts[f, q]
                            if v != w:
                                m |= int32(1) << int32(ch[j, i, v])
                        mo_[l] = m | (int32(1) << int32(ch[j, i, w]))
                        if l == 0 or l == L + 1:
                            mn_[l] = mo_[l]
                        else:
                            mn_[l] = m | (int32(1) << int32(u))
                    for l in range(L + 1):
                        if is_simplex[mo_[l] | mo_[l + 1]] == 0:
                            ob += 1
                        if is_simplex[mn_[l] | mn_[l + 1]] == 0:
                            nb += 1
                d = nb - ob
                dl[idx] = int8(d)
                idx += 1
                if d != 0:
                    dE += gamma * float64(d)      # per-chain cleanliness (all chains)
                    bo = int32(bd[j, f])
                    bn = bo + d
                    mo = bo
                    mn = bn
                    mult = int32(1) if bo == 0 else int32(0)   # chains covering f, before
                    for jj in range(KCH):
                        if jj != j:
                            bb = int32(bd[jj, f])
                            if bb < mo:
                                mo = bb
                            if bb < mn:
                                mn = bb
                            if bb == 0:
                                mult += 1
                    po = float64(0.0) if mo == 0 else 1.0 + beta * float64(mo)
                    pn = float64(0.0) if mn == 0 else 1.0 + beta * float64(mn)
                    dE += weights[f] * (pn - po)
                    # redundancy penalty: a facet covered by m > 1 chains wastes
                    # m - 1 of them (they could be elsewhere); mu charges for it
                    if mu > float64(0.0):
                        mult_n = mult
                        if bo == 0 and bn != 0:
                            mult_n -= 1
                        elif bo != 0 and bn == 0:
                            mult_n += 1
                        dE += mu * (float64(mult_n - 1 if mult_n > 1 else 0)
                                    - float64(mult - 1 if mult > 1 else 0))
                    if mo == 0 and mn != 0:
                        dcov -= 1
                    elif mo != 0 and mn == 0:
                        dcov += 1
            accept = dE <= 0.0
            if not accept:
                state, x = sm_next_dev(state)
                if sm_u01_dev(x) < math.exp(-dE / T):
                    accept = True
            if accept:
                for i in range(i1, i2 + 1):
                    ch[j, i, w] = u
                idx = 0
                for ii in range(incid_off[w], incid_off[w + 1]):
                    f = incid_fac[ii]
                    d = dl[idx]
                    idx += 1
                    if d != 0:
                        bd[j, f] = uint8(int32(bd[j, f]) + int32(d))
                cov += dcov
                if cov > bestcov:
                    bestcov = cov
                    for jj in range(KCH):
                        for i2_ in range(C1):
                            for g in range(NGRID):
                                best[s, jj, i2_, g] = ch[jj, i2_, g]
        for j in range(KCH):
            for i in range(C1):
                for g in range(NGRID):
                    chains[s, j, i, g] = ch[j, i, g]
            for f in range(NF):
                bad[s, j, f] = bd[j, f]
        score[s, 0] = float32(cov)
        score[s, 1] = float32(bestcov)

    return anneal_kernel


class ChainAnnealer:
    """Population of S independent k-chain annealers on the GPU.

    chains: int8 [S, k, c+1, ngrid] with chains[:, :, 0] = pi1, chains[:, :, c] = pi2.
    """

    def __init__(self, P: Product, k: int, cmax: int, S: int, seed: int = 0,
                 weights=None, threads=64):
        validate_parameters(k=k, cmax=cmax, S=S, seed=seed, threads=threads)
        self.P, self.k, self.cmax, self.S = P, k, cmax, S
        self.C1 = cmax + 1
        self.threads = threads
        self.rng = np.random.default_rng(seed)
        self.kernel = make_anneal_kernel(k, self.C1, P.ngrid, P.nf, P.nv, P.starmax)
        self.weights = np.ones(P.nf, np.float32) if weights is None else np.asarray(weights, np.float32)
        self.d_fv = cuda.to_device(P.facet_verts)
        self.d_fnv = cuda.to_device(P.facet_nv)
        self.d_off = cuda.to_device(P.incid_off)
        self.d_inc = cuda.to_device(P.incid_fac)
        self.d_iss = cuda.to_device(P.is_simplex)
        self.d_w = cuda.to_device(self.weights)
        self.chains = self.random_init(S)
        self.best = self.chains.copy()
        self.bestscore = np.full(S, -1.0, np.float32)
        self.kernel_time = 0.0
        self.moves = 0

    # -- initial chains --------------------------------------------------------
    def random_init(self, S):
        P = self.P
        ch = np.zeros((S, self.k, self.C1, P.ngrid), np.int8)
        for s in range(S):
            for j in range(self.k):
                ch[s, j] = self.sweep_chain()
        return ch

    def sweep_chain(self):
        """Chain that switches grid vertices from pi1 to pi2 one at a time in a
        random order (each link differs at one vertex); padded with copies of
        pi2 up to cmax links.  Random extra noise on a few interior entries."""
        P = self.P
        order = self.rng.permutation(P.ngrid)
        ch = np.tile(P.pi1, (self.C1, 1)).astype(np.int8)
        cur = P.pi1.copy()
        nsw = min(P.ngrid, self.cmax)
        # spread the ngrid switches over the available links
        pos = np.sort(self.rng.choice(np.arange(1, self.C1), size=nsw, replace=False)) \
            if self.cmax >= nsw else np.arange(1, self.C1)
        pi = 0
        for i in range(1, self.C1):
            while pi < nsw and pos[pi] == i:
                cur[order[pi]] = P.pi2[order[pi]]
                pi += 1
            ch[i] = cur
        # force the switches that did not fit
        for q in range(pi, P.ngrid):
            cur[order[q]] = P.pi2[order[q]]
        ch[-1] = P.pi2
        return ch

    # -- one epoch ---------------------------------------------------------------
    def set_weights(self, w):
        self.weights = np.asarray(w, np.float32)
        self.d_w = cuda.to_device(self.weights)

    def epoch(self, iters, T0, T1, beta=0.05, pmove_copy=0.5, ptrack=0.3, gamma=0.0, mu=0.0):
        validate_parameters(iters=iters, T0=T0, T1=T1, beta=beta,
                            gamma=gamma, pcopy=pmove_copy, ptrack=ptrack, mu=mu)
        S = self.S
        d_ch = cuda.to_device(self.chains)
        d_best = cuda.to_device(self.chains)
        d_bad = cuda.device_array((S, self.k, self.P.nf), np.uint8)
        d_score = cuda.device_array((S, 2), np.float32)
        seeds = self.rng.integers(0, 2**63 - 1, size=S, dtype=np.int64).astype(np.uint64)
        d_seeds = cuda.to_device(seeds)
        blocks = (S + self.threads - 1) // self.threads
        t0 = time.perf_counter()
        self.kernel[blocks, self.threads](
            d_ch, d_best, d_bad, d_score, d_seeds,
            self.d_fv, self.d_fnv, self.d_off, self.d_inc, self.d_iss, self.d_w,
            np.int32(iters), np.float64(T0), np.float64(T1), np.float64(beta), np.float64(gamma),
            np.float64(mu), np.float64(pmove_copy), np.float64(ptrack))
        cuda.synchronize()
        self.kernel_time += time.perf_counter() - t0
        self.moves += S * iters
        self.chains = d_ch.copy_to_host()
        score = d_score.copy_to_host()
        best = d_best.copy_to_host()
        improved = score[:, 1] > self.bestscore
        self.best[improved] = best[improved]
        self.bestscore[improved] = score[improved, 1]
        return score

    # -- population step: clone the elite over the weakest --------------------------
    def tournament(self, frac=0.5, noise=0, crossover=0.0):
        S = self.S
        order = np.argsort(-self.bestscore)
        n = int(S * frac)
        elite = order[:max(1, S - n)]
        weak = order[S - n:]
        src = elite[self.rng.integers(0, len(elite), size=len(weak))]
        self.chains[weak] = self.best[src]
        self.bestscore[weak] = self.bestscore[src]
        self.best[weak] = self.best[src]
        for s in weak[:noise]:
            # random re-initialisation of a few slots for diversity
            for j in range(self.k):
                self.chains[s, j] = self.sweep_chain()
            self.bestscore[s] = -1.0
        # -- chain-level crossover: recombine two elite incumbents -------------------
        # The missing move class on the winding family (klein_anneal_log.md §3-4):
        # point/track mutations can only extend or nudge domains, never hand a band
        # of facets from one chain to another.  A child built from two different
        # basin states carries a partition neither parent could reach by extension.
        # crossover = 0 keeps the RNG stream bit-identical (no draws below).
        nc = min(int(len(weak) * crossover), max(0, len(weak) - noise))
        for s in weak[noise:noise + nc]:
            ia = int(self.rng.integers(0, len(elite)))
            ib = int(self.rng.integers(0, len(elite)))
            if ib == ia and len(elite) > 1:
                ib = (ib + 1) % len(elite)
            pa, pb = elite[ia], elite[ib]
            from_a = self.rng.integers(0, 2, size=self.k).astype(bool)
            child = np.empty_like(self.best[pa])
            child[from_a] = self.best[pa][from_a]
            child[~from_a] = self.best[pb][~from_a]
            self.chains[s] = child
            self.best[s] = child
            self.bestscore[s] = -1.0

    def top(self):
        s = int(np.argmax(self.bestscore))
        return self.best[s], float(self.bestscore[s])

    def coverage_of(self, chains_k):
        """domains D_j (facet index arrays) and the covered set, on the host."""
        doms = [self.P.domain_of(chains_k[j]) for j in range(self.k)]
        covered = np.zeros(self.P.nf, bool)
        for d in doms:
            covered[d] = True
        return doms, covered


# --------------------------------------------------------------------------- #
#  Kernel B: optimized LocalSearch (Algorithm 1 of the document), batched      #
# --------------------------------------------------------------------------- #
def make_ls_kernel(NV, NGRID, MASKW):
    """Compile the batched optimized LocalSearch.  Returns (kernel, cpu_walk).
    The walk body is written once; the CUDA device version and the CPU twin
    are the same source bound to the matching splitmix64 twins."""

    def make_walk(rnd, u01):
        def walk(jm, verts, nvj, start, target, wseed, M, rr, early_exit, early_flag,
                 facet_verts, facet_nv, incid_off, incid_fac, is_simplex, dk,
                 img, defect, badbits, record, hist):
            for g in range(NGRID):
                img[g] = start[g]
                defect[g] = 0
            for ww in range(MASKW):
                badbits[ww] = uint64(0)
            sumdef = int32(0)
            for q in range(nvj):
                v = verts[q]
                dd = int32(dk[int32(img[v]) * NV + int32(target[v])])
                defect[v] = dd
                sumdef += dd
            nbad = int32(0)
            nf = facet_nv.shape[0]
            for f in range(nf):
                if ((jm[f >> 6] >> uint64(f & 63)) & uint64(1)) == uint64(0):
                    continue
                S = int32(0)
                for q in range(facet_nv[f]):
                    v = facet_verts[f, q]
                    S |= int32(1) << int32(img[v])
                    S |= int32(1) << int32(target[v])
                if is_simplex[S] == 0:
                    badbits[f >> 6] = badbits[f >> 6] | (uint64(1) << uint64(f & 63))
                    nbad += 1
            n_hist = 0
            if record != 0:
                for g in range(NGRID):
                    hist[0, g] = img[g]
                n_hist = 1
            if nbad == 0:
                return 1, 0, n_hist

            state = uint64(wseed)
            state, x = rnd(state)
            tabu_w = int32(-1)
            tabu_u = int32(-1)
            i = int32(0)
            while i < M:
                if early_exit != 0 and (i & 63) == 0 and early_flag[0] != 0:
                    break
                i += 1
                # ---- vertex choice, weight 4*defect + 1 ----
                total = uint64(int32(4) * sumdef + nvj)
                state, x = rnd(state)
                r0 = int32(x % total)
                w = int32(verts[0])
                acc = int32(0)
                for q in range(nvj):
                    v = int32(verts[q])
                    acc += int32(4) * int32(defect[v]) + int32(1)
                    if r0 < acc:
                        w = v
                        break
                # ---- admissible values: AND over star(w) ∩ J of Ext(img(sigma)) ----
                A = int32((1 << NV) - 1)
                for ii in range(incid_off[w], incid_off[w + 1]):
                    f = incid_fac[ii]
                    if ((jm[f >> 6] >> uint64(f & 63)) & uint64(1)) == uint64(0):
                        continue
                    S = int32(0)
                    for q in range(facet_nv[f]):
                        S |= int32(1) << int32(img[facet_verts[f, q]])
                    for u in range(NV):
                        if ((A >> u) & 1) != 0 and is_simplex[S | (int32(1) << u)] == 0:
                            A &= ~(int32(1) << u)
                    if A == 0:
                        break
                old = int32(img[w])
                A &= ~(int32(1) << old)
                if tabu_w == w and (A & ~(int32(1) << tabu_u)) != 0:
                    A &= ~(int32(1) << tabu_u)
                if A == 0:
                    continue
                # ---- step choice ----
                state, x = rnd(state)
                p = u01(x)
                tw = int32(target[w])
                cnt = int32(0)
                u = int32(0)
                if p < rr:
                    for uu in range(NV):
                        if ((A >> uu) & 1) != 0:
                            cnt += 1
                    state, x = rnd(state)
                    pick = int32(x % uint64(cnt))
                    for uu in range(NV):
                        if ((A >> uu) & 1) != 0:
                            if pick == 0:
                                u = uu
                                break
                            pick -= 1
                else:
                    bestd = int32(127)
                    for uu in range(NV):
                        if ((A >> uu) & 1) != 0:
                            dd = int32(dk[uu * NV + tw])
                            if dd < bestd:
                                bestd = dd
                                cnt = 1
                            elif dd == bestd:
                                cnt += 1
                    if bestd >= int32(defect[w]):
                        continue
                    state, x = rnd(state)
                    pick = int32(x % uint64(cnt))
                    for uu in range(NV):
                        if ((A >> uu) & 1) != 0 and int32(dk[uu * NV + tw]) == bestd:
                            if pick == 0:
                                u = uu
                                break
                            pick -= 1
                # ---- apply ----
                tabu_w = w
                tabu_u = old
                img[w] = int8(u)
                nd = int32(dk[u * NV + tw])
                sumdef += nd - int32(defect[w])
                defect[w] = nd
                for ii in range(incid_off[w], incid_off[w + 1]):
                    f = incid_fac[ii]
                    if ((jm[f >> 6] >> uint64(f & 63)) & uint64(1)) == uint64(0):
                        continue
                    S = int32(0)
                    for q in range(facet_nv[f]):
                        v = facet_verts[f, q]
                        S |= int32(1) << int32(img[v])
                        S |= int32(1) << int32(target[v])
                    wasbad = (badbits[f >> 6] >> uint64(f & 63)) & uint64(1)
                    if is_simplex[S] == 0:
                        if wasbad == uint64(0):
                            badbits[f >> 6] = badbits[f >> 6] | (uint64(1) << uint64(f & 63))
                            nbad += 1
                    else:
                        if wasbad != uint64(0):
                            badbits[f >> 6] = badbits[f >> 6] & ~(uint64(1) << uint64(f & 63))
                            nbad -= 1
                if record != 0:
                    for g in range(NGRID):
                        hist[n_hist, g] = img[g]
                    n_hist += 1
                if nbad == 0:
                    return 1, i, n_hist
            return 0, i, n_hist
        return walk

    walk_dev = cuda.jit(device=True)(make_walk(sm_next_dev, sm_u01_dev))
    walk_cpu = njit(make_walk(sm_next_cpu, sm_u01_cpu))

    @cuda.jit
    def ls_kernel(jmask, verts, nverts, start, target, seeds,
                  facet_verts, facet_nv, incid_off, incid_fac, is_simplex, dk,
                  M, rr, early_exit, out_found, out_comb):
        t = cuda.blockIdx.x
        w = cuda.threadIdx.x
        s_flag = cuda.shared.array(1, int32)
        s_res = cuda.shared.array(2, uint32)
        if w == 0:
            s_flag[0] = 0
            s_res[0] = uint32(0)
            s_res[1] = uint32(0xFFFFFFFF)
        cuda.syncthreads()
        img = cuda.local.array(NGRID, int8)
        defect = cuda.local.array(NGRID, int8)
        badbits = cuda.local.array(MASKW, uint64)
        hist = cuda.local.array((1, 1), int8)
        wseed = seeds[t] + GOLDEN * uint64(w + 1)
        found, steps, nh = walk_dev(jmask[t], verts[t], nverts[t], start[t], target[t],
                                    wseed, M, rr, early_exit, s_flag,
                                    facet_verts, facet_nv, incid_off, incid_fac,
                                    is_simplex, dk, img, defect, badbits, int32(0), hist)
        if found != 0:
            comb = uint32(w) * uint32(1048576) + uint32(steps)
            cuda.atomic.min(s_res, 1, comb)
            cuda.atomic.max(s_res, 0, uint32(1))
            if early_exit != 0:
                s_flag[0] = 1
        cuda.syncthreads()
        if w == 0:
            out_found[t] = int32(s_res[0])
            out_comb[t] = s_res[1]

    return ls_kernel, walk_cpu


class LocalSearchGPU:
    """Batched optimized LocalSearch tests: one block per test, `walks` threads."""

    def __init__(self, P: Product, walks=128):
        self.P = P
        self.walks = walks
        self.kernel, self.walk_cpu = make_ls_kernel(P.nv, P.ngrid, P.maskw)
        self.d_fv = cuda.to_device(P.facet_verts)
        self.d_fnv = cuda.to_device(P.facet_nv)
        self.d_off = cuda.to_device(P.incid_off)
        self.d_inc = cuda.to_device(P.incid_fac)
        self.d_iss = cuda.to_device(P.is_simplex)
        self.d_dk = cuda.to_device(P.dk)
        self.kernel_time = 0.0

    def pack(self, tests):
        P = self.P
        T = len(tests)
        jm = np.zeros((T, P.maskw), np.uint64)
        verts = np.zeros((T, P.ngrid), np.int8)
        nv = np.zeros(T, np.int32)
        start = np.zeros((T, P.ngrid), np.int8)
        target = np.zeros((T, P.ngrid), np.int8)
        seeds = np.zeros(T, np.uint64)
        for i, t in enumerate(tests):
            for f in t["facets"]:
                jm[i, f >> 6] |= np.uint64(1) << np.uint64(f & 63)
            vs = P.verts_of(t["facets"])
            nv[i] = len(vs)
            verts[i, : len(vs)] = vs
            start[i] = t.get("start", P.pi1)
            target[i] = t.get("target", P.pi2)
            seeds[i] = np.uint64(t["seed"])
        return jm, verts, nv, start, target, seeds

    def run(self, tests, M, r, early_exit=1, walks=None):
        """tests: list of dict(facets=iterable, seed=int, start=, target=)
        -> list of (found, winner_walk, steps)."""
        walks = self.walks if walks is None else walks
        jm, verts, nv, start, target, seeds = self.pack(tests)
        T = len(tests)
        d = [cuda.to_device(a) for a in (jm, verts, nv, start, target, seeds)]
        out_found = cuda.device_array(T, np.int32)
        out_comb = cuda.device_array(T, np.uint32)
        t0 = time.perf_counter()
        self.kernel[T, walks](*d, self.d_fv, self.d_fnv, self.d_off, self.d_inc,
                              self.d_iss, self.d_dk, np.int32(M), np.float64(r),
                              np.int32(early_exit), out_found, out_comb)
        cuda.synchronize()
        self.kernel_time += time.perf_counter() - t0
        fh, ch = out_found.copy_to_host(), out_comb.copy_to_host()
        out = []
        for i in range(T):
            c = int(ch[i])
            out.append((int(fh[i]), -1, -1) if c == 0xFFFFFFFF else (int(fh[i]), c >> 20, c & 0xFFFFF))
        return out

    def replay(self, test, walk_id, M, r):
        """Deterministic CPU replay of one walk -> list of accepted maps (+ target)."""
        P = self.P
        jm, verts, nv, start, target, seeds = self.pack([test])
        img = np.zeros(P.ngrid, np.int8)
        defect = np.zeros(P.ngrid, np.int8)
        badbits = np.zeros(P.maskw, np.uint64)
        hist = np.zeros((M + 2, P.ngrid), np.int8)
        flag = np.zeros(1, np.int32)
        wseed = np.uint64((int(seeds[0]) + int(GOLDEN) * (walk_id + 1)) & ((1 << 64) - 1))
        found, steps, nh = self.walk_cpu(jm[0], verts[0], np.int32(nv[0]), start[0], target[0],
                                         wseed, np.int32(M), np.float64(r), np.int32(0), flag,
                                         P.facet_verts, P.facet_nv, P.incid_off, P.incid_fac,
                                         P.is_simplex, P.dk, img, defect, badbits, np.int32(1), hist)
        assert found, "replay of a winning walk must succeed"
        chain = [hist[i].copy() for i in range(nh)]
        if not np.array_equal(chain[-1], target[0]):
            chain.append(target[0].copy())
        return chain


# --------------------------------------------------------------------------- #
#  Certificates: partition, Reduce (Algorithm 2), independent verification     #
# --------------------------------------------------------------------------- #
def maps_contiguous_on(P: Product, A, B, facet_ids):
    for f in facet_ids:
        S = 0
        for q in range(P.facet_nv[f]):
            v = int(P.facet_verts[f, q])
            S |= 1 << int(A[v])
            S |= 1 << int(B[v])
        if not P.is_simplex[S]:
            return False
    return True


def reduce_chain(P: Product, chain, facet_ids):
    """Algorithm 2 with the document's optimizations: O(1) adjacency
    pre-filter on the differing vertices, checks restricted to facets touching
    a differing vertex, forward + backward passes to a fixpoint."""
    verts = P.verts_of(facet_ids)
    fset = set(int(f) for f in facet_ids)
    star = {v: [f for f in range(P.incid_off[v], P.incid_off[v + 1])] for v in verts}
    star = {v: [int(P.incid_fac[i]) for i in idx if int(P.incid_fac[i]) in fset] for v, idx in star.items()}
    dk = P.dk.reshape(P.nv, P.nv)

    def cont(a, b):
        diff = [v for v in verts if a[v] != b[v]]
        if not diff:
            return True
        for v in diff:
            if dk[a[v], b[v]] > 1:
                return False
        seen = set()
        for v in diff:
            for f in star[v]:
                if f in seen:
                    continue
                seen.add(f)
                S = 0
                for q in range(P.facet_nv[f]):
                    u = int(P.facet_verts[f, q])
                    S |= (1 << int(a[u])) | (1 << int(b[u]))
                if not P.is_simplex[S]:
                    return False
        return True

    def one_pass(ch):
        c = len(ch) - 1
        out = [ch[0]]
        j = 0
        while j < c:
            i = c
            while i > j + 1 and not cont(ch[j], ch[i]):
                i -= 1
            out.append(ch[i])
            j = i
        return out

    best = list(chain)
    while True:
        fwd = one_pass(best)
        bwd = one_pass(best[::-1])[::-1]
        cand = fwd if len(fwd) <= len(bwd) else bwd
        if len(cand) >= len(best):
            return best
        best = cand


def verify_system(P: Product, parts):
    """parts: list of (facet_ids, chain).  Plain Definition 2.3 check:
    partition of F_L, chain from pi1 to pi2 on the part's vertices, every
    map simplicial and consecutive maps 1-contiguous on every facet."""
    if not parts:
        raise ValueError("empty cover")
    covered = set()
    for facets, chain in parts:
        facets = list(facets)
        if not facets or any(isinstance(f, (bool, np.bool_)) or not isinstance(f, Integral)
                             or not 0 <= f < P.nf for f in facets):
            raise ValueError("empty part or invalid facet id")
        if len(set(facets)) != len(facets) or covered.intersection(facets):
            raise ValueError("duplicate facets or overlapping parts")
        covered.update(facets)
        if len(chain) == 0:
            raise ValueError("empty planner chain")
        verts = P.verts_of(facets)
        for m in chain:
            if len(m) != P.ngrid or any(
                    isinstance(m[v], (bool, np.bool_)) or not isinstance(m[v], Integral)
                    or not 0 <= m[v] < P.nv for v in verts):
                raise ValueError("invalid planner vertex map")
            if not maps_contiguous_on(P, m, m, facets):
                raise ValueError("map not simplicial")
        if any(chain[0][v] != P.pi1[v] for v in verts):
            raise ValueError("chain does not start at pi1")
        if any(chain[-1][v] != P.pi2[v] for v in verts):
            raise ValueError("chain does not end at pi2")
        for a, b in zip(chain, chain[1:]):
            if not maps_contiguous_on(P, a, b, facets):
                raise ValueError("link not 1-contiguous")
    if covered != set(range(P.nf)):
        raise ValueError("not a covering")
    return True


def partition_from_chains(P: Product, chains_k):
    """Disjoint parts from the domains of k chains (a facet goes to the first
    chain covering it).  Returns list of (facet_ids, reduced_chain) sorted by
    size, or None if the domains do not cover F_L."""
    covered = np.zeros(P.nf, bool)
    parts = []
    for j in range(len(chains_k)):
        dom = P.domain_of(chains_k[j])
        mine = [int(f) for f in dom if not covered[f]]
        covered[mine] = True
        if mine:
            parts.append((mine, [np.array(m, np.int8) for m in chains_k[j]]))
    if not covered.all():
        return None
    out = []
    for facets, chain in parts:
        out.append((facets, reduce_chain(P, chain, facets)))
    out.sort(key=lambda t: (-len(t[0]), t[0]))
    return out


def write_certificate(P: Product, parts, outdir, meta):
    """cover.txt / planners.txt / summary.json in the Abstract_Motion_GPU format
    (auditable with its independent_check.py)."""
    verify_system(P, parts)
    reserved = {"complex", "n_domains", "sc_strict_bound", "n_verified",
                "all_domains_verified", "domain_sizes", "chain_lengths", "domains",
                "certification", "provenance"}
    if reserved.intersection(meta):
        raise ValueError("metadata cannot override certificate fields")
    from provenance import environment_record
    provenance = environment_record(meta.get("params", {}).get("init"))
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    nv = P.nv
    with open(outdir / "cover.txt", "w") as fh:
        fh.write(f"KxK: {P.nf} maximal facets; cover with {len(parts)} domains "
                 f"(SC_strict <= {len(parts) - 1})\n\n")
        for di, (facets, _) in enumerate(parts):
            fh.write(f"--- domain {di}  ({len(facets)} facets) ---\n")
            for f in sorted(facets):
                vs = " ".join(f"({v // nv},{v % nv})" for v in P.facets[f])
                fh.write(f"  facet {f:3d}: {vs}\n")
    with open(outdir / "planners.txt", "w") as fh:
        fh.write("Planner chains (chain-first GPU annealer, then Algorithm 2 reduce)\n\n")
        for di, (facets, chain) in enumerate(parts):
            fh.write(f"=== domain {di}: {len(facets)} facets ===\n")
            fh.write(f"  reduced chain: {len(chain)} maps\n")
            vset = set(P.verts_of(facets))
            for mi, m in enumerate(chain):
                fh.write(f"  map {mi}:\n")
                for a in range(nv):
                    row = " ".join(str(int(m[a * nv + b])) if a * nv + b in vset else "@"
                                   for b in range(nv))
                    fh.write(f"    {row}\n")
            fh.write("\n")
    summary = dict(complex=dict(name=P.name, nv=nv, ngrid=P.ngrid, num_facets=P.nf,
                                facets=[list(f) for f in P.kfacets]),
                   n_domains=len(parts), sc_strict_bound=len(parts) - 1,
                   n_verified=len(parts), all_domains_verified=True,
                   domain_sizes=[len(f) for f, _ in parts],
                   chain_lengths=[len(c) - 1 for _, c in parts],
                   domains=[sorted(f) for f, _ in parts],
                   certification={"internal": "passed", "independent_audit": "not_run"},
                   provenance=provenance, **meta)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


# --------------------------------------------------------------------------- #
#  Driver: chain-first covering search                                         #
# --------------------------------------------------------------------------- #
def anneal_cover(P: Product, k: int, cmax=40, S=4096, iters=20000, epochs=50,
                 T0=0.6, T1=0.03, beta=0.05, pcopy=0.5, seed=0, time_limit=None,
                 log=print, stop_at_full=True, tournament_every=2, tournament_frac=0.25,
                 init_chains=None, ptrack=0.05, guide=0.0, guide_max=4.0, gamma=0.0,
                 mu=0.0, crossover=0.0, h1_guide=0, on_epoch=None, stop_when=None):
    """guide > 0: adaptive weights (guided local search) — after each epoch the
    facets left uncovered by the best state get weight += guide (capped at
    guide_max), all weights decay toward 1; the reported coverage is always
    unweighted.  A facet left uncovered for many epochs saturates the cap:
    w* = 1 + guide/0.3, so guide = 1 already reaches the old default cap of 4
    (1 + 3.33) — escalating the increment alone does nothing, the cap has to
    move with it (exposed as --guide-max).

    crossover > 0: that fraction of tournament-replacement slots are filled by
    chain-uniform recombination of two elite incumbents instead of clones.
    h1_guide > 0: every N epochs, the uncovered facets of the incumbent are
    tested with the GF(2) H1 obstruction (h1_obstruction.h1_map_obstruction)
    against each of the k chain domains; facets provably unabsorbable for ALL
    chains from the current state (klein_anneal_log.md §3) get weight 0, so
    the guide stops paying for closed doors.  Weight 0 still leaves a facet in
    the (unweighted) coverage count and in the top() ranking — the objective
    stays the true cover."""
    validate_parameters(k=k, cmax=cmax, S=S, iters=iters, epochs=epochs,
                        T0=T0, T1=T1, beta=beta, gamma=gamma, pcopy=pcopy,
                        ptrack=ptrack, guide=guide, guide_max=guide_max, mu=mu,
                        tournament_every=tournament_every, crossover=crossover,
                        h1_guide=h1_guide,
                        tournament_frac=tournament_frac, seed=seed, time_limit=time_limit)
    if h1_guide:
        # lazy import: h1_obstruction imports gpu_sc only in its __main__
        from h1_obstruction import h1_map_obstruction
    ann = ChainAnnealer(P, k, cmax, S, seed=seed)
    if init_chains is not None:
        n = min(len(init_chains), S) // 2
        ann.chains[:n] = init_chains[:n]
    t_start = time.perf_counter()
    hist = []
    full = float(P.nf)
    best_chains, best_cov = None, -1.0
    h1_dead = None   # (P.nf,) bool: facets obstructed for all chains of the incumbent
    for ep in range(1, epochs + 1):
        score = ann.epoch(iters, T0, T1, beta, pcopy, ptrack, gamma, mu)
        chains, cov = ann.top()
        covered = None
        recompute_h1 = h1_guide and (ep - 1) % h1_guide == 0
        if recompute_h1:
            doms, covered = ann.coverage_of(chains)
            U = [int(f) for f in np.nonzero(~covered)[0]]
            if len(U) * k <= 2048:   # ~0.03 s/test; masks matter near plateaus, not in the early climb
                dsets = [set(int(f) for f in d) for d in doms]
                dead = np.zeros(P.nf, bool)
                for f in U:
                    if all(h1_map_obstruction(P, sorted(d | {f}))[1] for d in dsets):
                        dead[f] = True
                h1_dead = dead
                if dead.any():
                    log(f"  h1-guide: {int(dead.sum())}/{len(U)} uncovered facets are H1-obstructed "
                        f"for all {k} chains (weight 0)")
        if guide > 0:
            if covered is None:
                _, covered = ann.coverage_of(chains)
            w = 1.0 + (ann.weights - 1.0) * 0.7
            w[~covered] += guide
            w = np.minimum(w, guide_max)
            if h1_dead is not None:
                w[h1_dead] = 0.0   # the decay would otherwise heal the mask (0 -> 0.3 -> ...)
            ann.set_weights(w)
        elif recompute_h1:
            w = np.ones(P.nf, np.float32)
            if h1_dead is not None:
                w[h1_dead] = 0.0
            ann.set_weights(w)
        el = time.perf_counter() - t_start
        srt = np.sort(ann.bestscore)[::-1]
        log(f"  epoch {ep:3d}: best {cov:6.1f}/{P.nf}  top10 {srt[:10].astype(int).tolist()}  "
            f"median {np.median(srt):.0f}  now-median {np.median(score[:,0]):.0f}  "
            f"{el:7.1f}s  {ann.moves/ann.kernel_time/1e6:8.3f} Mmoves/s")
        hist.append(dict(epoch=ep, best=cov, median=float(np.median(srt)), elapsed=el))
        if cov > best_cov:
            best_cov, best_chains = cov, chains.copy()
        # Both hooks are None by default, so an unflagged run does exactly what
        # it did before: nothing is written and no signal handling is involved.
        # `on_epoch` is the webapp's live feed (JSONL telemetry + a consistent
        # snapshot of the incumbent); `stop_when` is how a stop lands on an
        # epoch boundary and still reaches the end-of-run writer below, so the
        # run keeps its partial summary instead of vanishing (see --graceful-stop).
        if on_epoch is not None:
            on_epoch(ep, cov, best_chains, hist, ann, el)
        if stop_when is not None and stop_when():
            log(f"  stop requested after epoch {ep} — writing the partial summary")
            break
        if stop_at_full and cov >= full - 1e-6:
            log(f"  full coverage reached at epoch {ep}")
            break
        if time_limit is not None and el > time_limit:
            log("  time limit reached")
            break
        if ep % tournament_every == 0:
            ann.tournament(frac=tournament_frac, noise=max(1, S // 32), crossover=crossover)
    return best_chains, best_cov, hist, ann


def init_from_summary(P: Product, path: Path, k: int, cmax: int, S: int):
    """Population seeded from a previous run: take the k chains with the
    largest domains, reduce each on its domain (Algorithm 2), pad with copies
    of pi2 to cmax links.  Half the population gets the seed, half stays random."""
    s = json.loads(path.read_text())
    if "best_chains" not in s:
        sys.exit(f"{path}: not an annealer summary (no 'best_chains'); a star_cover.py "
                 f"certificate records domains but no map tables, so it cannot seed the "
                 f"population -- pass the summary.json of an earlier gpu_sc.py run instead.")
    chains = [np.array(c, np.int8) for c in s["best_chains"]]
    doms = [P.domain_of(c) for c in chains]
    order = sorted(range(len(chains)), key=lambda j: -len(doms[j]))[:k]
    out = np.zeros((k, cmax + 1, P.ngrid), np.int8)
    for slot, j in enumerate(order):
        red = reduce_chain(P, [m for m in chains[j]], doms[j].tolist()) if len(doms[j]) else list(chains[j])
        if len(red) > cmax + 1:
            raise ValueError("reduced chain longer than cmax")
        pad = [np.array(m, np.int8) for m in red] + [P.pi2.copy()] * (cmax + 1 - len(red))
        out[slot] = np.array(pad)
    init = np.zeros((S, k, cmax + 1, P.ngrid), np.int8)
    init[: S // 2] = out
    return init


# ---- live monitoring: opt-in telemetry and a survivable stop ---------------- #
_STOP = {"requested": False, "reason": None}


def _request_stop(signum, _frame):
    """Signal handler: only records the request.  The main thread is inside the
    CUDA call for the whole epoch, and CPython runs this between bytecodes, so
    the flag is seen at the next epoch boundary -- which is where the loop
    checks it, exactly like the time limit.  Nothing is interrupted mid-move."""
    import signal as _signal
    _STOP["requested"] = True
    _STOP["reason"] = _signal.Signals(signum).name


def _atomic_write_json(path, obj):
    """Write JSON via a sibling .tmp + replace: a reader polling the file either
    sees the old version or the new one, never a half-written document.  The
    .tmp suffix also keeps the file out of `find_state`'s *.json scan."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def make_monitors(a, P):
    """(on_epoch, stop_when) for `anneal_cover`, or (None, None) when unused.

    With none of --jsonl / --checkpoint-every / --graceful-stop / --stop-file
    given, both are None and the run behaves exactly as it always has.
    """
    if a.checkpoint_every and not a.outdir:
        raise SystemExit("--checkpoint-every needs --outdir")
    if a.jsonl:                                     # its directory may not exist yet
        Path(a.jsonl).parent.mkdir(parents=True, exist_ok=True)
    jsonl_fh = open(a.jsonl, "a", buffering=1) if a.jsonl else None      # line-buffered
    live_path = Path(a.outdir) / "live_state.json" if (a.checkpoint_every and a.outdir) else None

    def on_epoch(ep, cov, best_chains, hist, ann, el):
        row = dict(epoch=ep, best=cov, median=float(hist[-1]["median"]), elapsed=el,
                   moves_per_s=(ann.moves / ann.kernel_time if ann.kernel_time else 0.0))
        if jsonl_fh:
            jsonl_fh.write(json.dumps(row) + "\n")
        if live_path is not None and ep % a.checkpoint_every == 0 and best_chains is not None:
            _atomic_write_json(live_path, dict(
                complex=dict(name=P.name, nv=P.nv, num_facets=P.nf),
                status="running", epoch=ep, coverage=int(cov),
                best_chains=best_chains.tolist(), history=hist[-200:], params=vars(a)))

    if a.graceful_stop:
        import signal as _signal
        for s in (_signal.SIGINT, _signal.SIGTERM):
            _signal.signal(s, _request_stop)

    def stop_when():
        if a.stop_file and Path(a.stop_file).exists():
            _STOP["requested"] = True
            _STOP["reason"] = "stop-file"
        return _STOP["requested"]

    if not a.graceful_stop and not a.stop_file:
        stop_when = None
    return on_epoch if (jsonl_fh or live_path) else None, stop_when


def cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--complex", default="s2", choices=["circle", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "wedge", "bary", "cube", "dodeca", "rp2", "klein", "torus", "mobius", "s1xw"])
    ap.add_argument("--k", type=int, default=3, help="number of chains / domains")
    ap.add_argument("--cmax", type=int, default=40, help="links per chain")
    ap.add_argument("--S", type=int, default=4096, help="population (GPU threads)")
    ap.add_argument("--iters", type=int, default=20000, help="moves per thread per epoch")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--T0", type=float, default=0.6)
    ap.add_argument("--T1", type=float, default=0.03)
    ap.add_argument("--beta", type=float, default=0.05, help="penalty slope per bad link of the best chain on a facet")
    ap.add_argument("--gamma", type=float, default=0.0, help="per-chain cleanliness: cost per bad link of every chain")
    ap.add_argument("--mu", type=float, default=0.0, dest="mu",
                    help="redundancy penalty: cost for a facet covered by m > 1 chains "
                         "(m - 1 wasted chains); 0 = off.  Pushes the k domains apart "
                         "instead of letting them overlap on easy facets")
    ap.add_argument("--pcopy", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ptrack", type=float, default=0.05, help="probability of a multi-map track move")
    ap.add_argument("--guide", type=float, default=0.0, help="adaptive weight increment for uncovered facets")
    ap.add_argument("--guide-max", type=float, default=4.0, dest="guide_max",
                    help="cap on those adaptive weights; a persistently uncovered facet saturates at "
                         "1 + guide/0.3, so the cap must be raised for --guide above ~0.9 to bite")
    ap.add_argument("--h1-guide", type=int, default=0, dest="h1_guide", metavar="N",
                    help="every N epochs, mask to weight 0 the uncovered facets that the GF(2) H1 "
                         "obstruction provably forbids for ALL chains of the incumbent "
                         "(klein_anneal_log.md section 3); 0 = off")
    ap.add_argument("--tournament-every", type=int, default=2)
    ap.add_argument("--tournament-frac", type=float, default=0.25)
    ap.add_argument("--crossover", type=float, default=0.0,
                    help="fraction of tournament-replacement slots filled by chain-level crossover "
                         "of two elite incumbents instead of clones — a restructuring move class "
                         "(klein_anneal_log.md sections 3-4); 0 = off")
    ap.add_argument("--time-limit", type=float, default=None, help="seconds")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--require-cover", action="store_true", help="exit 2 for a completed search without a full cover")
    ap.add_argument("--no-stop", action="store_true", help="keep annealing after full coverage")
    ap.add_argument("--init", default=None, help="summary.json of a previous run to seed the population")
    ap.add_argument("--jsonl", default=None, metavar="PATH",
                    help="append one JSON object per epoch to PATH (epoch, best, median, "
                         "now-median, elapsed, moves/s, top10, stretch/collapse counters).  "
                         "For live monitoring; the search is unaffected")
    ap.add_argument("--checkpoint-every", type=int, default=0, dest="checkpoint_every", metavar="N",
                    help="every N epochs write <outdir>/live_state.json (atomically) holding the "
                         "incumbent's best_chains with the same keys as the partial summary, so a "
                         "monitor can show the run's current map and a later run can --init from it; "
                         "0 = off (nothing is written during the run, as before)")
    ap.add_argument("--graceful-stop", action="store_true", dest="graceful_stop",
                    help="catch SIGINT/SIGTERM and stop at the next epoch boundary, keeping the "
                         "partial summary.json (without it a signal kills the run and saves nothing, "
                         "which is the long-standing behaviour)")
    ap.add_argument("--stop-file", default=None, metavar="PATH", dest="stop_file",
                    help="also stop when PATH appears (a stop channel that survives a lost pid)")
    a = ap.parse_args()
    try:
        validate_parameters(**{key: getattr(a, key) for key in (
            "k", "cmax", "S", "iters", "epochs", "T0", "T1", "beta", "gamma",
            "mu", "pcopy", "ptrack", "guide", "tournament_every", "tournament_frac",
            "seed", "time_limit", "guide_max", "crossover", "h1_guide")})
    except ValueError as exc:
        ap.error(str(exc))

    P = build(a.complex)
    print(f"=== gpu_sc chain-first annealer ===  K={P.name}  |F_L|={P.nf}  grid={P.ngrid}  "
          f"k={a.k} cmax={a.cmax} S={a.S} iters={a.iters}  T0={a.T0} T1={a.T1}  "
          f"beta={a.beta} gamma={a.gamma} mu={a.mu} ptrack={a.ptrack}  "
          f"guide={a.guide}/(max {a.guide_max})  seed={a.seed}"
          + (f"  crossover={a.crossover} h1-guide={a.h1_guide}" if a.crossover or a.h1_guide else ""))
    init = None
    if a.init:
        init = init_from_summary(P, Path(a.init), a.k, a.cmax, a.S)
        print(f"  seeded population from {a.init} (reduced chains of the {a.k} largest domains, padded to cmax)")
    on_epoch, stop_when = make_monitors(a, P)
    t0 = time.perf_counter()
    chains, cov, hist, ann = anneal_cover(P, a.k, a.cmax, a.S, a.iters, a.epochs, a.T0, a.T1,
                                          a.beta, a.pcopy, a.seed, a.time_limit,
                                          stop_at_full=not a.no_stop, init_chains=init,
                                          on_epoch=on_epoch, stop_when=stop_when,
                                          tournament_every=a.tournament_every,
                                          tournament_frac=a.tournament_frac,
                                          ptrack=a.ptrack, guide=a.guide, guide_max=a.guide_max,
                                          gamma=a.gamma, mu=a.mu,
                                          crossover=a.crossover, h1_guide=a.h1_guide)
    dt = time.perf_counter() - t0
    doms = [P.domain_of(chains[j]) for j in range(a.k)]
    covered = np.zeros(P.nf, bool)
    for d in doms:
        covered[d] = True
    print(f"best coverage {int(covered.sum())}/{P.nf}  domain sizes {[len(d) for d in doms]}  "
          f"time {dt:.1f}s  kernel {ann.kernel_time:.1f}s  moves {ann.moves:.3g}")
    meta = dict(params=vars(a), elapsed_s=dt, kernel_time_s=ann.kernel_time, moves=ann.moves,
                history=hist, best_chains=chains.tolist(), coverage=int(covered.sum()),
                chain_domain_sizes=[int(len(d)) for d in doms])
    if covered.all():
        parts = partition_from_chains(P, chains)
        verify_system(P, parts)
        print(f"VERIFIED system of {len(parts)} motion planners: sizes {[len(f) for f, _ in parts]}, "
              f"reduced chain lengths {[len(c) - 1 for _, c in parts]}  =>  SC_strict({P.name}) <= {len(parts) - 1}")
        if a.outdir:
            write_certificate(P, parts, a.outdir, meta)
            print(f"wrote {a.outdir}/cover.txt planners.txt summary.json")
    else:
        print(f"NOT a covering: {int((~covered).sum())} facets uncovered")
        if a.outdir:
            if any((Path(a.outdir) / f).exists() for f in ("cover.txt", "planners.txt")):
                raise ValueError("incomplete run cannot overwrite an existing certificate; use a new outdir")
            Path(a.outdir).mkdir(parents=True, exist_ok=True)
            (Path(a.outdir) / "summary.json").write_text(json.dumps(
                dict(complex=dict(name=P.name, nv=P.nv, num_facets=P.nf), status="incomplete", sc_strict_bound=None, all_domains_verified=False, **meta), indent=2, default=str))
            print(f"wrote {a.outdir}/summary.json (partial)")
        return 2 if a.require_cover else 0
    return 0


if __name__ == "__main__":
    sys.exit(cli())
