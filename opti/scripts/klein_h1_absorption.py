#!/usr/bin/env python3
"""klein_h1_absorption.py — why the k=8 plateau cannot be fixed by extension.

For the incumbent state of an annealer run (a summary.json with best_chains),
take the k chain domains D_j and the uncovered facets L, and ask for every
(f, j) in L x domains whether D_j u {f} passes the H1 obstruction test
(h1_obstruction.h1_map_obstruction): if rank (pi1 - pi2)_* != 0 on some cycle
of D_j u {f}, then NO contiguity chain covers that set, whatever the budget --
chain j can provably never absorb facet f.  A pass is only a necessary
condition (the absorption may still be impossible for local reasons); a fail
is a proof.

Usage:
  python3 scripts/klein_h1_absorption.py [complex] [summary.json]
  # defaults: klein  results/klein_k8_r1/summary.json
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import gpu_sc
from h1_obstruction import h1_map_obstruction

name = sys.argv[1] if len(sys.argv) > 1 else "klein"
summary = sys.argv[2] if len(sys.argv) > 2 else "results/klein_k8_r1/summary.json"

P = gpu_sc.build(name)
s = json.loads(Path(summary).read_text())
chains = [np.array(c, np.int8) for c in s["best_chains"]]
doms = [sorted(int(f) for f in P.domain_of(c)) for c in chains]
k = len(doms)

covered = np.zeros(P.nf, bool)
for d in doms:
    covered[d] = True
L = [f for f in range(P.nf) if not covered[f]]
print(f"{P.name}: state {summary}")
print(f"  |F_L| = {P.nf}, k = {k}, covered = {int(covered.sum())}, uncovered L = {len(L)}")
print(f"  domain sizes {[len(d) for d in doms]}")

# reference: the whole complex is obstructed (no single chain covers K x K)
h1all, rkall = h1_map_obstruction(P, list(range(P.nf)))
print(f"  all of K x K: dim H1 = {h1all}, rank (pi1-pi2)_* = {rkall} -> "
      f"{'OBSTRUCTED' if rkall else 'passes'} (expected: obstructed)")

# sanity: each D_j carries a chain, so each must pass
print("-- each plateau domain alone (carries its chain; must pass) --")
for j, d in enumerate(doms):
    h1, rk = h1_map_obstruction(P, d)
    print(f"  D_{j}: {len(d)} facets, dim H1 = {h1}, rank = {rk} -> "
          f"{'OBSTRUCTED (BUG?)' if rk else 'passes'}")

# the leftover complex by itself
h1L, rkL = h1_map_obstruction(P, L)
print(f"-- L alone: {len(L)} facets, dim H1 = {h1L}, rank = {rkL} -> "
      f"{'OBSTRUCTED: no fresh chain could take even just the leftovers' if rkL else 'passes'}")

# can one chain take its domain plus all of L?
print("-- D_j u L (chain j absorbs everything left) --")
for j, d in enumerate(doms):
    h1, rk = h1_map_obstruction(P, sorted(set(d) | set(L)))
    print(f"  D_{j} u L: dim H1 = {h1}, rank = {rk} -> "
          f"{'OBSTRUCTED' if rk else 'passes'}")

# the absorption matrix
print(f"-- absorption matrix: for each uncovered f, chains whose D_j u {{f}} passes --")
absorbable = {}
for n, f in enumerate(L):
    ok = []
    for j, d in enumerate(doms):
        _, rk = h1_map_obstruction(P, sorted(set(d) | {f}))
        if rk == 0:
            ok.append(j)
    absorbable[f] = ok
    print(f"  f={f:4d}  H1-pass chains: {ok if ok else 'NONE -- obstructed for all k chains'}"
          f"  ({n + 1}/{len(L)})", flush=True)

free = [f for f in L if absorbable[f]]
dead = [f for f in L if not absorbable[f]]
print(f"-- summary: {len(free)}/{len(L)} leftover facets have at least one H1-pass "
      f"absorber; {len(dead)}/{len(L)} are obstructed for ALL {k} chains")
if dead:
    print(f"  provably unabsorbable (restructuring required, extension cannot work): {dead}")