#!/usr/bin/env python3
"""test_stretch_cpu.py — correctness tests for the stretch move (annealer_stretch.py).

Run on a CUDA-less machine with the numba CUDA simulator; the env var must be
set BEFORE numba is imported, so run this file directly in a fresh process:

    NUMBA_ENABLE_CUDASIM=1 ~/whisper_env/bin/python src/test_stretch_cpu.py
    # or simply (the module sets it before importing numba):
    ~/whisper_env/bin/python src/test_stretch_cpu.py

Tests:
  1. test_stretch_delta_reference — a numpy mirror of the kernel's incremental
     stretch delta vs a from-scratch recompute (P.chain_bad_counts) on the real
     mobius_seed_666 plateau state, for both leftover facets x L' in 1..8 x
     both clone directions.  Also checks the left-clone guarantee: after a
     left clone the links i1-1..i2-1 of the target facet are contiguous.
  2. test_kernel_incremental_state — the compiled kernel (CUDASIM), warm-started
     at the plateau with stretch on: for every population slot, the kernel's
     reported coverage must equal the host recompute from the returned chains.
     (This is the check that catches bad-link/cov bookkeeping corruption; it
     failed before the boundary-map fix in make_anneal_kernel.)
     test_kernel_with_collapse (below) additionally turns pcollapse on at the
     149/150 state: same bookkeeping assertions, plus the collapse counters
     must fire.
  4. test_collapse_delta_reference — the numpy mirror of the collapse delta
     (a block of maps, all of ft's columns = ONE fresh K-vertex value) vs
     from-scratch recompute on the 149/150 state, including the known
     covering move (chain 2, map 1, v=0) whose coverage must hold at the
     −7 trade the H1 obstruction predicts.
  5. test_pstretch0_bit_identity — annealer_stretch at pstretch=0 must return
     exactly gpu_sc's best_chains at the same seed (regression contract).
     pcollapse defaults to 0 there and draws nothing extra.
  6. test_seam_delta_reference — the numpy mirror of the seam stitch (TWO
     small collapse blocks at the first two bad links of the depth-2 chain,
     one proposal) vs from-scratch recompute on the 149/150 state.
  7. test_bookkeeping_with_seam / _with_pself — kernel runs with pseam / pself
     on: coverage bookkeeping must match the host recompute and the seam
     counters must fire.
  8. test_epoch_temperatures / test_stall_hook — the reheat schedule and the
     stall hook (snapshot + deterministic audit + claude-advice log) without
     touching the GPU.
"""
import os
os.environ.setdefault("NUMBA_ENABLE_CUDASIM", "1")   # before numba import (fresh process!)

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_sc
import annealer_stretch as st

SUM666 = Path(__file__).resolve().parents[1] / "results" / "mobius_seed_666" / "summary.json"
SUM149 = Path(__file__).resolve().parents[1] / "results" / "mobius_k3_stretch_from_149" / "summary.json"


def build_state(name="mobius", summary=SUM666, k=3, cmax=10, S=4, seed=1):
    """Product + a ChainAnnealer whose first S//2 slots carry the plateau state."""
    P = gpu_sc.build(name)
    A = st.ChainAnnealer(P, k, cmax, S, seed=seed)
    init = st.init_from_summary(P, summary, k, cmax, S)
    A.chains[: S // 2] = init[: S // 2]
    return P, A, init


def stretch_mirror(P, chain, ft, i1, i2, direction):
    """Numpy reference of one stretch proposal.  Returns (new_chain, affected)
    where affected is the dedup union of the stars of ft's grid vertices."""
    ch = chain.copy()
    src = i1 - 1 if direction == "left" else i2 + 1
    for w in P.facet_verts[ft][: int(P.facet_nv[ft])]:
        ch[i1:i2 + 1, w] = ch[src, w]
    affected = set()
    for q in range(int(P.facet_nv[ft])):
        w = int(P.facet_verts[ft, q])
        affected.update(int(f) for f in
                        P.incid_fac[P.incid_off[w]:P.incid_off[w + 1]])
    return ch, affected


def collapse_mirror(P, chain, ft, i1, i2, v):
    """Numpy reference of one collapse proposal: the block maps i1..i2 take the
    single fresh K-vertex value v on EVERY column of ft.  Returns (new_chain,
    affected) with affected the dedup union of the stars of ft's columns."""
    ch = chain.copy()
    for q in range(int(P.facet_nv[ft])):
        w = int(P.facet_verts[ft, q])
        ch[i1:i2 + 1, w] = v
    affected = set()
    for q in range(int(P.facet_nv[ft])):
        w = int(P.facet_verts[ft, q])
        affected.update(int(f) for f in
                        P.incid_fac[P.incid_off[w]:P.incid_off[w + 1]])
    return ch, affected


def seam_mirror(P, chain, ft, a1, a2, b1, b2, vA, vB):
    """Numpy reference of one seam-stitch proposal: block A = maps a1..a2 and
    block B = maps b1..b2 (separated by untouched maps) each carry ONE fresh
    K-vertex value (vA / vB) on EVERY column of ft — the depth-2 repair of
    the mobius 149 audit.  Returns (new_chain, affected)."""
    ch = chain.copy()
    for q in range(int(P.facet_nv[ft])):
        w = int(P.facet_verts[ft, q])
        ch[a1:a2 + 1, w] = vA
        ch[b1:b2 + 1, w] = vB
    affected = set()
    for q in range(int(P.facet_nv[ft])):
        w = int(P.facet_verts[ft, q])
        affected.update(int(f) for f in
                        P.incid_fac[P.incid_off[w]:P.incid_off[w + 1]])
    return ch, affected


def bad_links_of(P, chain, ft):
    """Link indices i (link i spans maps i, i+1) where ft is not contiguous."""
    masks = []
    for i in range(chain.shape[0]):
        m = 0
        for q in range(int(P.facet_nv[ft])):
            m |= 1 << int(chain[i, P.facet_verts[ft, q]])
        masks.append(m)
    return [i for i in range(len(masks) - 1)
            if not P.is_simplex[masks[i] | masks[i + 1]]]


def energy(P, chains_k, weights, beta, gamma):
    """The kernel's energy, computed from scratch on the host."""
    bads = [P.chain_bad_counts(c) for c in chains_k]
    bads = np.stack(bads)                       # [k, nf]
    minbad = bads.min(axis=0)
    pen = np.where(minbad == 0, 0.0, 1.0 + beta * minbad)
    return float((weights * pen).sum() + gamma * float(bads.sum()))


class TestStretchDelta(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        cls.P = gpu_sc.build("mobius")
        s = json.loads(SUM666.read_text())
        cls.chains = [np.array(c, np.int8) for c in s["best_chains"]]
        cls.doms = [sorted(int(f) for f in cls.P.domain_of(c)) for c in cls.chains]
        covered = np.zeros(cls.P.nf, bool)
        for d in cls.doms:
            covered[d] = True
        cls.leftovers = [int(f) for f in np.flatnonzero(~covered)]
        assert len(cls.leftovers) == 2, cls.leftovers   # the 148/150 plateau

    def _first_bad_link(self, chain, ft):
        P = self.P
        bads = P.chain_bad_counts(chain)
        if bads[ft] == 0:
            return -1
        masks = []
        for i in range(chain.shape[0]):
            m = 0
            for q in range(int(P.facet_nv[ft])):
                m |= 1 << int(chain[i, P.facet_verts[ft, q]])
            masks.append(m)
        for i in range(len(masks) - 1):
            if not P.is_simplex[masks[i] | masks[i + 1]]:
                return i
        return -1

    def test_left_clone_guarantee(self):
        """After a LEFT clone from i1 = max(i*, 1): links i1-1 .. i2-1 of the
        target facet are contiguous by construction (the scientific point of
        the move — a compound edit the track move cannot make)."""
        P = self.P
        CMAX = self.chains[0].shape[0] - 1
        for ft in self.leftovers:
            # argmin chain on ft
            bads = [P.chain_bad_counts(c)[ft] for c in self.chains]
            j = int(np.argmin(bads))
            istar = self._first_bad_link(self.chains[j], ft)
            self.assertGreaterEqual(istar, 0)
            i1 = max(istar, 1)
            for Ls in range(1, 9):
                i2 = min(i1 + Ls - 1, CMAX - 1)
                new_ch, _ = stretch_mirror(P, self.chains[j], ft, i1, i2, "left")
                bads_new = P.chain_bad_counts(new_ch)
                # every link of the block except the far boundary is good:
                self.assertLessEqual(bads_new[ft], 1,
                                     f"ft={ft} Ls={Ls}: bad={bads_new[ft]}")

    def test_delta_vs_full_recompute(self):
        """A numpy mirror of the kernel's incremental delta (window masks are
        replaced by chain_bad_counts; the per-facet energy bookkeeping is
        copied line-for-line) == the from-scratch recompute, for every
        (leftover, L', direction)."""
        P = self.P
        CMAX = self.chains[0].shape[0] - 1
        beta, gamma = 5.0, 0.5
        weights = np.ones(P.nf, np.float32)
        for ft in self.leftovers:
            weights[ft] = 11.0                      # guide weight on the leftover
            bads_all = [P.chain_bad_counts(c) for c in self.chains]
            j = int(np.argmin([b[ft] for b in bads_all]))
            istar = self._first_bad_link(self.chains[j], ft)
            i1 = max(istar, 1)
            for Ls in range(1, 9):
                i2 = min(i1 + Ls - 1, CMAX - 1)
                for direction in ("left", "right"):
                    new_ch, affected = stretch_mirror(P, self.chains[j], ft, i1, i2, direction)
                    bad_old = bads_all[j]
                    bad_new = P.chain_bad_counts(new_ch)
                    others = np.stack([b for jj, b in enumerate(bads_all) if jj != j])
                    min_others = others.min(axis=0)

                    # per-facet delta set: exactly the affected facets actually change
                    changed = {int(g): int(bad_new[g]) - int(bad_old[g])
                               for g in affected if bad_new[g] != bad_old[g]}
                    recomputed = {int(g): int(bad_new[g]) - int(bad_old[g])
                                  for g in np.flatnonzero(bad_new != bad_old)}
                    self.assertEqual(changed, recomputed,
                                     f"ft={ft} Ls={Ls} dir={direction}: affected-set mismatch")

                    # kernel-style incremental dE over the affected facets
                    dE = 0.0
                    for g, d in changed.items():
                        dE += gamma * float(d)
                        bo = int(bad_old[g]); bn = bo + d
                        mo = min(bo, int(min_others[g])); mn = min(bn, int(min_others[g]))
                        po = 0.0 if mo == 0 else 1.0 + beta * mo
                        pn = 0.0 if mn == 0 else 1.0 + beta * mn
                        dE += float(weights[g]) * (pn - po)

                    # full-recompute energy difference (all chains, from scratch)
                    chains0 = [self.chains[j]] + [c for jj, c in enumerate(self.chains) if jj != j]
                    chains1 = [new_ch] + [c for jj, c in enumerate(self.chains) if jj != j]
                    e0 = energy(P, chains0, weights, beta, gamma)
                    e1 = energy(P, chains1, weights, beta, gamma)
                    self.assertAlmostEqual(dE, e1 - e0, places=6,
                                           msg=f"ft={ft} Ls={Ls} dir={direction}")


class TestKernelIncremental(unittest.TestCase):
    def test_kernel_cov_matches_host_recompute(self):
        """Warm-started at the plateau, stretch on: the kernel's per-slot
        coverage must equal the host recompute from the returned chains."""
        P, A, init = build_state()
        _, covered = A.coverage_of(init[0])
        unc = np.flatnonzero(~covered)
        self.assertEqual(len(unc), 2)
        w = np.ones(P.nf, np.float32); w[unc] = 11.0
        A.set_weights(w)
        A.set_targets(unc)
        score = A.epoch(iters=400, T0=0.5, T1=0.05, beta=5.0,
                        pmove_copy=0.5, ptrack=0.3, gamma=0.5, mu=0.0, pstretch=0.8)
        self.assertGreater(float(score[:, 2].sum()), 0, "no stretch move fired")
        for s in range(A.S):
            doms, cov_host = A.coverage_of(A.chains[s])
            self.assertEqual(int(cov_host.sum()), int(score[s, 0]),
                             f"slot {s}: kernel cov != host recompute")
            bads = [P.chain_bad_counts(A.chains[s, j]) for j in range(A.k)]
            bads = np.stack(bads)
            self.assertEqual(int((bads.min(axis=0) == 0).sum()), int(score[s, 0]),
                             f"slot {s}: bad-array vs coverage mismatch")


class TestCollapseDelta(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        cls.P = gpu_sc.build("mobius")
        s = json.loads(SUM149.read_text())
        cls.chains = [np.array(c, np.int8) for c in s["best_chains"]]
        cls.doms = [sorted(int(f) for f in cls.P.domain_of(c)) for c in cls.chains]
        covered = np.zeros(cls.P.nf, bool)
        for d in cls.doms:
            covered[d] = True
        cls.leftovers = [int(f) for f in np.flatnonzero(~covered)]
        assert cls.leftovers == [90], cls.leftovers      # the 149/150 state

    def test_known_covering_collapse(self):
        """The move the 149-audit found: chain 2, map 1, v=0 covers facet 90 —
        and the H1 obstruction shows as its price: total coverage is the −7
        trade (142), not 150."""
        P, chains = self.P, self.chains
        new, _ = collapse_mirror(P, chains[2], 90, 1, 1, 0)
        self.assertEqual(int(P.chain_bad_counts(new)[90]), 0)
        cov = np.zeros(P.nf, bool)
        for c in (chains[0], chains[1], new):
            cov[list(P.domain_of(c))] = True
        self.assertEqual(int(cov.sum()), 142)

    def test_collapse_delta_vs_full_recompute(self):
        """Same contract as TestStretchDelta.test_delta_vs_full_recompute, for
        the collapse value source: affected set == actually-changed facets and
        the kernel-style incremental dE == the from-scratch energy difference,
        over the leftover facet, maps 1..5, v in 0..nv-1, L' in {1,2,4,8}."""
        P, chains = self.P, self.chains
        CMAX = chains[0].shape[0] - 1
        beta, gamma = 5.0, 0.5
        ft = 90
        weights = np.ones(P.nf, np.float32)
        weights[ft] = 11.0
        bads_all = [P.chain_bad_counts(c) for c in chains]
        for j in range(3):
            bad_old = bads_all[j]
            others = np.stack([b for jj, b in enumerate(bads_all) if jj != j])
            min_others = others.min(axis=0)
            for m in (1, 2, 3, 5):
                for v in range(P.nv):
                    for Ls in (1, 2, 4, 8):
                        i1, i2 = m, min(m + Ls - 1, CMAX - 1)
                        new_ch, affected = collapse_mirror(P, chains[j], ft, i1, i2, v)
                        bad_new = P.chain_bad_counts(new_ch)
                        changed = {int(g): int(bad_new[g]) - int(bad_old[g])
                                   for g in affected if bad_new[g] != bad_old[g]}
                        recomputed = {int(g): int(bad_new[g]) - int(bad_old[g])
                                      for g in np.flatnonzero(bad_new != bad_old)}
                        self.assertEqual(changed, recomputed,
                                         f"j={j} m={m} v={v} L'={Ls}: affected-set mismatch")
                        dE = 0.0
                        for g, d in changed.items():
                            dE += gamma * float(d)
                            bo = int(bad_old[g]); bn = bo + d
                            mo = min(bo, int(min_others[g])); mn = min(bn, int(min_others[g]))
                            po = 0.0 if mo == 0 else 1.0 + beta * mo
                            pn = 0.0 if mn == 0 else 1.0 + beta * mn
                            dE += float(weights[g]) * (pn - po)
                        chains0 = list(chains)
                        chains1 = list(chains); chains1[j] = new_ch
                        e0 = energy(P, chains0, weights, beta, gamma)
                        e1 = energy(P, chains1, weights, beta, gamma)
                        self.assertAlmostEqual(dE, e1 - e0, places=6,
                                               msg=f"j={j} m={m} v={v} L'={Ls}")


class TestCollapseKernel(unittest.TestCase):
    def test_bookkeeping_with_collapse(self):
        """Warm start from the 149 state with BOTH compound moves on: kernel
        coverage of every slot must equal the host recompute, bad arrays must
        match, and the collapse move must actually fire."""
        P, A, init = build_state(summary=SUM149, cmax=40, S=4, seed=11)
        _, covered = A.coverage_of(init[0])
        unc = np.flatnonzero(~covered)
        self.assertEqual(unc.tolist(), [90])
        w = np.ones(P.nf, np.float32); w[unc] = 11.0
        A.set_weights(w)
        A.set_targets(unc)
        score = A.epoch(iters=400, T0=0.5, T1=0.05, beta=5.0,
                        pmove_copy=0.5, ptrack=0.3, gamma=0.5, mu=0.0,
                        pstretch=0.6, pcollapse=0.5)
        self.assertGreater(float(score[:, 4].sum()), 0, "no collapse move fired")
        for s in range(A.S):
            doms, cov_host = A.coverage_of(A.chains[s])
            self.assertEqual(int(cov_host.sum()), int(score[s, 0]),
                             f"slot {s}: kernel cov != host recompute")
            bads = np.stack([P.chain_bad_counts(A.chains[s, j]) for j in range(A.k)])
            self.assertEqual(int((bads.min(axis=0) == 0).sum()), int(score[s, 0]),
                             f"slot {s}: bad-array vs coverage mismatch")


class TestSeamDelta(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        cls.P = gpu_sc.build("mobius")
        s = json.loads(SUM149.read_text())
        cls.chains = [np.array(c, np.int8) for c in s["best_chains"]]
        cls.doms = [sorted(int(f) for f in cls.P.domain_of(c)) for c in cls.chains]
        covered = np.zeros(cls.P.nf, bool)
        for d in cls.doms:
            covered[d] = True
        cls.leftovers = [int(f) for f in np.flatnonzero(~covered)]
        assert cls.leftovers == [90], cls.leftovers      # the 149/150 state

    def test_seam_delta_vs_full_recompute(self):
        """Kernel-style incremental seam delta == the from-scratch energy
        difference, for every depth-2 chain on the leftover, block lengths
        up to 3 and all fresh-value pairs — the contract the kernel's seam
        branch must keep (window masks a1-1..b2+1, two interior bands)."""
        P, chains, ft = self.P, self.chains, 90
        CMAX = chains[0].shape[0] - 1
        beta, gamma = 5.0, 0.5
        weights = np.ones(P.nf, np.float32)
        weights[ft] = 11.0
        bads_all = [P.chain_bad_counts(c) for c in chains]
        tested = 0
        for j in range(len(chains)):
            links = bad_links_of(P, chains[j], ft)
            if len(links) < 2:
                continue        # the seam move targets depth-2 chains only
            a1 = max(links[0], 1)
            b1 = links[1]
            self.assertGreaterEqual(b1 - a1, 1)
            bad_old = bads_all[j]
            others = np.stack([b for jj, b in enumerate(bads_all) if jj != j])
            min_others = others.min(axis=0)
            for LA in range(1, min(3, b1 - a1) + 1):
                a2 = a1 + LA - 1
                for LB in (1, 2, 3):
                    b2 = min(b1 + LB - 1, CMAX - 1)
                    for vA in range(P.nv):
                        for vB in range(P.nv):
                            new_ch, affected = seam_mirror(P, chains[j], ft,
                                                            a1, a2, b1, b2, vA, vB)
                            bad_new = P.chain_bad_counts(new_ch)
                            changed = {int(g): int(bad_new[g]) - int(bad_old[g])
                                       for g in affected if bad_new[g] != bad_old[g]}
                            recomputed = {int(g): int(bad_new[g]) - int(bad_old[g])
                                          for g in np.flatnonzero(bad_new != bad_old)}
                            self.assertEqual(changed, recomputed,
                                             f"j={j} a=({a1},{a2}) b=({b1},{b2}) "
                                             f"v=({vA},{vB}): affected-set mismatch")
                            dE = 0.0
                            for g, d in changed.items():
                                dE += gamma * float(d)
                                bo = int(bad_old[g]); bn = bo + d
                                mo = min(bo, int(min_others[g])); mn = min(bn, int(min_others[g]))
                                po = 0.0 if mo == 0 else 1.0 + beta * mo
                                pn = 0.0 if mn == 0 else 1.0 + beta * mn
                                dE += float(weights[g]) * (pn - po)
                            chains0 = list(chains)
                            chains1 = list(chains); chains1[j] = new_ch
                            e0 = energy(P, chains0, weights, beta, gamma)
                            e1 = energy(P, chains1, weights, beta, gamma)
                            self.assertAlmostEqual(dE, e1 - e0, places=6,
                                                   msg=f"j={j} a=({a1},{a2}) b=({b1},{b2}) v=({vA},{vB})")
                            tested += 1
        self.assertGreater(tested, 0, "no depth-2 chain on the leftover — seam untested")


class TestSeamKernel(unittest.TestCase):
    def test_bookkeeping_with_seam(self):
        """Warm start from the 149 state with stretch, collapse AND seam on:
        kernel coverage of every slot must equal the host recompute, bad
        arrays must match, and the seam counters must fire."""
        P, A, init = build_state(summary=SUM149, cmax=40, S=4, seed=3)
        _, covered = A.coverage_of(init[0])
        unc = np.flatnonzero(~covered)
        self.assertEqual(unc.tolist(), [90])
        w = np.ones(P.nf, np.float32); w[unc] = 11.0
        A.set_weights(w)
        A.set_targets(unc)
        score = A.epoch(iters=400, T0=0.5, T1=0.05, beta=5.0,
                        pmove_copy=0.5, ptrack=0.3, gamma=0.5, mu=0.0,
                        pstretch=0.6, pcollapse=0.3, pseam=1.0)
        self.assertGreater(float(score[:, 6].sum()), 0, "no seam proposal fired")
        for s in range(A.S):
            doms, cov_host = A.coverage_of(A.chains[s])
            self.assertEqual(int(cov_host.sum()), int(score[s, 0]),
                             f"slot {s}: kernel cov != host recompute")
            bads = np.stack([P.chain_bad_counts(A.chains[s, j]) for j in range(A.k)])
            self.assertEqual(int((bads.min(axis=0) == 0).sum()), int(score[s, 0]),
                             f"slot {s}: bad-array vs coverage mismatch")


class TestPselfKernel(unittest.TestCase):
    def test_bookkeeping_with_pself(self):
        """All compound proposals target the slot's OWN uncovered facets
        (pself=1): bookkeeping contract unchanged — kernel coverage of every
        slot must equal the host recompute."""
        P, A, init = build_state()
        _, covered = A.coverage_of(init[0])
        unc = np.flatnonzero(~covered)
        w = np.ones(P.nf, np.float32); w[unc] = 11.0
        A.set_weights(w)
        A.set_targets(unc)
        score = A.epoch(iters=400, T0=0.5, T1=0.05, beta=5.0,
                        pmove_copy=0.5, ptrack=0.3, gamma=0.5, mu=0.0,
                        pstretch=0.8, pself=1.0)
        ncompound = float(score[:, 2].sum() + score[:, 4].sum() + score[:, 6].sum())
        self.assertGreater(ncompound, 0, "no compound proposal fired")
        for s in range(A.S):
            doms, cov_host = A.coverage_of(A.chains[s])
            self.assertEqual(int(cov_host.sum()), int(score[s, 0]),
                             f"slot {s}: kernel cov != host recompute")
            bads = np.stack([P.chain_bad_counts(A.chains[s, j]) for j in range(A.k)])
            self.assertEqual(int((bads.min(axis=0) == 0).sum()), int(score[s, 0]),
                             f"slot {s}: bad-array vs coverage mismatch")


class TestReheat(unittest.TestCase):
    def test_epoch_temperatures(self):
        """The reheat schedule: constant reheat_T once the best coverage has
        stalled for a positive multiple of reheat_every epochs; disabled at
        reheat_every <= 0 or reheat_T <= 0."""
        f = st._epoch_temperatures
        self.assertEqual(f(0, 0.6, 0.02, 10, 0.35), (0.6, 0.02, False))
        self.assertEqual(f(3, 0.6, 0.02, 10, 0.35), (0.6, 0.02, False))
        self.assertEqual(f(9, 0.6, 0.02, 10, 0.35), (0.6, 0.02, False))
        self.assertEqual(f(10, 0.6, 0.02, 10, 0.35), (0.35, 0.35, True))
        self.assertEqual(f(20, 0.6, 0.02, 10, 0.35), (0.35, 0.35, True))
        self.assertEqual(f(10, 0.6, 0.02, 0, 0.35), (0.6, 0.02, False))
        self.assertEqual(f(10, 0.6, 0.02, 10, 0.0), (0.6, 0.02, False))


class TestStallHook(unittest.TestCase):
    def test_hook_snapshot_audit_and_advice(self):
        """The stall hook writes the snapshot, the deterministic audit runs
        against it (leftovers + H1 doors + suggestion block), and the claude
        stub's output lands in claude_advice.log — without raising."""
        import tempfile
        import json
        P, A, init = build_state()
        with tempfile.TemporaryDirectory() as tmp:
            old = os.environ.get("STRETCH_CLAUDE")
            os.environ["STRETCH_CLAUDE"] = "/bin/echo"
            try:
                lines = []
                st._run_stall_hook(P, A, init[0], [{"epoch": 1}], tmp, 7, 8,
                                   dict(complex="mobius"), lines.append)
            finally:
                if old is None:
                    os.environ.pop("STRETCH_CLAUDE", None)
                else:
                    os.environ["STRETCH_CLAUDE"] = old
            self.assertTrue((Path(tmp) / "stall_snapshot.json").exists(),
                            "no snapshot written")
            audit = json.loads((Path(tmp) / "stall_audit.json").read_text())
            self.assertEqual(audit["complex"], "mobius")
            self.assertEqual(len(audit["leftovers"]), 2)   # the 148/150 plateau
            self.assertIn("guide_targets", audit["suggestion"])
            self.assertTrue(all(k in audit["suggestion"]
                                for k in ("seam_targets", "reheat", "h1_audited")))
            advice = (Path(tmp) / "claude_advice.log").read_text()
            self.assertIn("stalled 8 epochs", advice)


    def test_pstretch0_equals_gpu_sc(self):
        """pstretch = 0 draws no extra RNG values: the walk is bit-identical to
        gpu_sc.py's at the same seed (the regression contract)."""
        P = gpu_sc.build("mobius")
        kw = dict(k=3, cmax=10, S=4, iters=120, epochs=2, T0=1.2, T1=0.05,
                  beta=1.0, pcopy=0.5, seed=7, guide=1.0, guide_max=4.0,
                  gamma=0.5, ptrack=0.3)
        r0 = gpu_sc.anneal_cover(P, **kw)
        r1 = st.anneal_cover(P, pstretch=0.0, **kw)
        c0 = np.array(r0[0]); c1 = np.array(r1[0])
        self.assertEqual(c0.shape, c1.shape)
        self.assertTrue((c0 == c1).all(),
                        f"pstretch=0 diverged from gpu_sc: best cov {r0[1]} vs {r1[1]}")


if __name__ == "__main__":
    unittest.main(verbosity=2)