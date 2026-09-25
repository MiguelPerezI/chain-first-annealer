#!/usr/bin/env python3
"""plateau_barrier_audit.py — exhaustive one-move neighborhood audit of a stalled state.

Enumerates EVERY move of the annealer's move set (gpu_sc.make_anneal_kernel:
a "track move" = chain j, grid vertex w, a block i1..i2 of consecutive
interior maps with length L in 1..LMAX=8, all set to a new value u) applied to
the incumbent best_chains of a stalled summary.json, and reports:

  * best coverage reachable by ONE move  (== C* certifies strict local
    optimality in the unweighted coverage objective);
  * among "repair moves" — moves that newly cover at least one currently
    uncovered facet — the best coverage reachable.  Every repair therefore
    costs at least  C* - that  covered facets: the certified depth of the
    valley the Metropolis rule must accept to leave the plateau.

The audit is exhaustive over the kernel's move set (all chains, all grid
vertices, all blocks of length 1..8 inside the interior maps, all target
values), so its verdicts are proofs about the recorded state — but only
one-move proofs: they do not rule out escaping by a coordinated sequence of
individually costly moves.

Usage: python3 plateau_barrier_audit.py [complex] [summary.json]
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import gpu_sc

name = sys.argv[1] if len(sys.argv) > 1 else "rp2"
summary_path = sys.argv[2] if len(sys.argv) > 2 else "results/rp2_k5_hybrid/summary.json"
LMAX = 8
# kernel energy weights of the recorded runs (beta/gamma from params; mu absent
# in these runs; facet weights taken at their floor w=1 -- the guide raises
# weights on uncovered facets, which can only LOWER the uphill deltas below,
# so with w=1 the certificate is the conservative one for repairs).
BETA, GAMMA = 5.0, 0.5

P = gpu_sc.build(name)
s = json.loads(Path(summary_path).read_text())
raw = [np.array(c, np.int8) for c in s["best_chains"]]
chains = np.array(s["best_chains"], np.int64)          # [k, C1, ngrid]
k, C1, ngrid = chains.shape
CMAX = C1 - 1

def pen(b):
    return 0.0 if b == 0 else 1.0 + BETA * b

bads = np.array([P.chain_bad_counts(c) for c in raw], np.int64)   # [k, nf]
minbad = bads.min(0)
covered = minbad == 0
Cstar = int(covered.sum())
Lset = set(np.where(~covered)[0].tolist())
print(f"{name}: {summary_path}")
print(f"  k={k} CMAX={CMAX} grid={ngrid}  C*={Cstar}/{P.nf}  |L|={len(Lset)}")

fv = P.facet_verts.astype(np.int64)
fnv = P.facet_nv.astype(np.int64)
istab = P.is_simplex.astype(bool)
nv = P.nv

best_after = Cstar          # best coverage reachable by one move
best_repair_after = -1      # best coverage among repair moves
best_repair_move = None
count_repair = 0
count_improve = 0
nmoves = 0
min_dE = np.inf             # cheapest energy of any non-noop move
min_dE_move = None
improve_dE = []              # energy deltas of coverage-improving moves

for j in range(k):
    bd_other = np.array([bads[i].copy() for i in range(k) if i != j]).min(0)  # [nf]
    ch = chains[j]                                          # [C1, ngrid]
    for w in range(ngrid):
        star = P.incid_fac[P.incid_off[w]:P.incid_off[w + 1]].astype(np.int64)
        if len(star) == 0:
            continue
        # per-facet, per-map image masks WITHOUT the moving vertex w
        mwo = np.zeros((len(star), C1), np.int64)
        for t, f in enumerate(star):
            verts = fv[f, :fnv[f]]
            verts = verts[verts != w]
            imgs = ch[:, verts]                            # [C1, nfv-1]
            mwo[t] = np.bitwise_or.reduce(1 << imgs, axis=1)
        val = ch[:, w]                                      # [C1] current values
        # full bad count per star facet for chain j
        b0 = bads[j][star]                                  # [nstar]
        oth = bd_other[star]
        covered0 = (b0 == 0) | (oth == 0)
        inL = np.array([f in Lset for f in star])
        for L in range(1, LMAX + 1):
            if L > CMAX - 1:
                break
            for i1 in range(1, CMAX - L + 1):
                i2 = i1 + L - 1
                # maps i1-1 .. i2+1 involved
                lo, hi = i1 - 1, i2 + 1
                m = mwo[:, lo:hi + 1] | (1 << val[None, lo:hi + 1])   # old masks [nstar, L+2]
                old_bad_links = np.zeros(len(star), np.int64)
                for l in range(L + 1):
                    old_bad_links += ~istab[m[:, l] | m[:, l + 1]]
                for u in range(nv):
                    if all(val[i1:i2 + 1] == u):
                        continue                              # exact no-op
                    newval = val[lo:hi + 1].copy()
                    newval[1:i2 - i1 + 2] = u                 # positions i1..i2
                    nm = mwo[:, lo:hi + 1] | (1 << newval[None, :])
                    new_bad_links = np.zeros(len(star), np.int64)
                    for l in range(L + 1):
                        new_bad_links += ~istab[nm[:, l] | nm[:, l + 1]]
                    b1 = b0 - old_bad_links + new_bad_links
                    covered1 = (b1 == 0) | (oth == 0)
                    dcov = int(covered1.sum() - covered0.sum())
                    # kernel energy delta (w=1): weighted coverage term
                    # + gamma * per-chain cleanliness, over the star only
                    m0 = np.minimum(b0, oth)
                    m1 = np.minimum(b1, oth)
                    pens = np.where(m1 != m0,
                                    np.array([pen(int(x)) for x in m1]) -
                                    np.array([pen(int(x)) for x in m0]), 0.0)
                    dE = float(pens.sum() + GAMMA * (new_bad_links - old_bad_links).sum())
                    nmoves += 1
                    if dE < min_dE:
                        min_dE = dE
                        min_dE_move = (j, w, i1, i2, int(u), dcov, dE)
                    after = Cstar + dcov
                    if after > best_after:
                        best_after = after
                    repairs = inL & covered1 & ~covered0
                    if repairs.any():
                        count_repair += 1
                        if after > best_repair_after:
                            best_repair_after = after
                            best_repair_move = (j, w, i1, i2, int(u),
                                                int(repairs.sum()))
                    if dcov > 0:
                        count_improve += 1
                        improve_dE.append(dE)

print(f"  moves enumerated: {nmoves}")
print(f"  best coverage reachable by ONE move: {best_after} "
      f"({'STRICT LOCAL OPTIMUM' if best_after == Cstar else 'improvable!'})")
print(f"  improving moves: {count_improve}")
if improve_dE:
    d = np.array(improve_dE)
    print(f"  energy of improving moves: min {d.min():+.2f} max {d.max():+.2f}; "
          f"energy-downhill (dE<0): {(d < 0).sum()}")
print(f"  cheapest non-noop move energy: dE = {min_dE:+.2f} at "
      f"(chain,grid,i1,i2,u,dcov,dE) = {min_dE_move}")
if min_dE > 0:
    import math
    for T in (0.6, 0.1, 0.03):
        log_esc = min_dE / T - math.log(nmoves)
        print(f"    T={T}: expected one-move escape time >= exp({min_dE:.1f}/{T})/"
              f"{nmoves} ~ e^{log_esc:.0f} moves "
              f"({'astronomical' if log_esc > 40 else 'feasible'})")
print(f"  repair moves (newly cover >=1 leftover): {count_repair}")
if best_repair_after >= 0:
    print(f"  best coverage among repair moves: {best_repair_after} "
          f"-> net one-move repair outcome {best_repair_after - Cstar:+d} facets")
    print(f"  witness move (chain, grid vtx, i1, i2, u, #repaired): {best_repair_move}")