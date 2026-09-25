#!/usr/bin/env python3
"""rp2_targeted_push.py — drive the RP² k=5 search past the rectangle plateau.

Strategy informed by the stall analysis (experiments/surface_stall_analysis.md,
annealer_formal_properties.md):

Phase 1: Standard anneal with strong guide — let the annealer find whatever
         shape it converges to (rectangle plateau at ~540 or beyond).
Phase 2: Ridge walk — continue from the Phase 1 incumbent with elevated guide
         and lowered γ (the cleanliness penalty that blocks restructuring
         moves). The analysis proved 254 repair moves become downhill at
         guide weights; lowering γ opens more.
Phase 3: Targeted rewrite — at each new plateau, identify H₁-absorbable
         (facet, chain) pairs and use SubProduct single-chain annealing to
         rewrite the absorbing chain. This is the move class the standard
         annealer's point/track mutations cannot produce efficiently.
Phase 4: If still incomplete, output the state for CP-SAT cold attack.

Usage:
  python3 scripts/rp2_targeted_push.py [--time-limit 3600] [--seed 42]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import gpu_sc
from h1_obstruction import h1_map_obstruction
from hybrid_cover import SubProduct


def anneal_phase(P, k, outdir, seed, epochs, guide, gamma, T0, T1,
                 init=None, time_limit=600, cmax=40, S=4096,
                 beta=5.0, mu=0.5, ptrack=0.3):
    """Run the standard chain-first annealer and return (coverage, summary_path)."""
    import subprocess
    cmd = [sys.executable, "-u", "src/gpu_sc.py",
           "--complex", "rp2", "--k", str(k),
           "--cmax", str(cmax), "--S", str(S),
           "--iters", "5000", "--epochs", str(epochs),
           "--T0", str(T0), "--T1", str(T1),
           "--beta", str(beta), "--gamma", str(gamma),
           "--mu", str(mu), "--ptrack", str(ptrack),
           "--guide", str(guide), "--seed", str(seed),
           "--time-limit", str(time_limit),
           "--outdir", outdir]
    if init:
        cmd += ["--init", init]
    print(f"  → {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, capture_output=False)
    sp = Path(outdir) / "summary.json"
    if sp.exists():
        s = json.loads(sp.read_text())
        return s["coverage"], str(sp)
    return None, None


def h1_absorption_scan(P, chains):
    """Return dict: uncovered facet → list of chain indices that can absorb it."""
    doms = [set(int(f) for f in P.domain_of(c)) for c in chains]
    covered = set()
    for d in doms:
        covered |= d
    L = sorted(set(range(P.nf)) - covered)
    absorb = {}
    for f in L:
        ok = []
        for j, d in enumerate(doms):
            _, rk = h1_map_obstruction(P, sorted(d | {f}))
            if rk == 0:
                ok.append(j)
        absorb[f] = ok
    return absorb, L, doms


def single_chain_rewrite(P, chain_j, target_facets, seed, epochs=100,
                         cmax=40, time_limit=120):
    """Try to rewrite one chain to cover target_facets using SubProduct annealing.

    Returns the new chain (np.int8 array) if successful, else None.
    """
    sub = SubProduct(P, target_facets)
    print(f"    SubProduct: {sub.nf} facets, grid {sub.ngrid}", flush=True)

    ann = gpu_sc.ChainAnnealer(sub, k=1, cmax=cmax, S=2048, seed=seed)
    t0 = time.time()
    for ep in range(epochs):
        ann.epoch(iters=5000, T0=0.6, T1=0.03, beta=5.0, gamma=0.1)
        best_chains, score = ann.top()       # best_chains: [1, C1, ngrid]
        doms, covered = ann.coverage_of(best_chains)
        cov = int(covered.sum())
        if cov == sub.nf:
            # return the chain mapped back to full-product grid coordinates
            ch = best_chains[0].copy()       # [C1, ngrid] on the SubProduct grid
            print(f"    SubProduct COVERED at epoch {ep+1} ({time.time()-t0:.1f}s)", flush=True)
            return ch
        if time.time() - t0 > time_limit:
            break
    print(f"    SubProduct stalled at {cov}/{sub.nf} ({time.time()-t0:.1f}s)", flush=True)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=3600, help="total wall time budget (s)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="results/rp2_k5_push")
    ap.add_argument("--init", default=None, help="warm start from a summary.json")
    a = ap.parse_args()

    P = gpu_sc.build("rp2")
    k = 5
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print(f"=== RP² k=5 targeted push ===")
    print(f"  |F| = {P.nf}, grid = {P.ngrid}, nv = {P.nv}")
    print(f"  time budget: {a.time_limit}s, seed: {a.seed}")

    # --- Phase 1: standard anneal with strong guide ---
    print(f"\n--- Phase 1: standard anneal (guide=3.0, γ=0.5) ---")
    cov1, sp1 = anneal_phase(
        P, k, str(outdir / "phase1"), a.seed,
        epochs=100, guide=3.0, gamma=0.5, T0=0.6, T1=0.03,
        init=a.init, time_limit=min(600, a.time_limit * 0.3))
    if cov1 is None:
        print("Phase 1 failed to produce a summary"); return
    print(f"  Phase 1 result: {cov1}/{P.nf}")
    if cov1 == P.nf:
        print("  COVERED in Phase 1!"); return

    # --- Phase 2: ridge walk with lowered γ ---
    print(f"\n--- Phase 2: ridge walk (guide=5.0, γ=0.1) ---")
    cov2, sp2 = anneal_phase(
        P, k, str(outdir / "phase2"), a.seed + 1,
        epochs=200, guide=5.0, gamma=0.1, T0=0.6, T1=0.03,
        init=sp1, time_limit=min(900, a.time_limit * 0.4))
    if cov2 is None:
        cov2, sp2 = cov1, sp1
    print(f"  Phase 2 result: {cov2}/{P.nf}")
    if cov2 == P.nf:
        print("  COVERED in Phase 2!"); return

    # --- Phase 3: targeted single-chain rewrite ---
    print(f"\n--- Phase 3: targeted rewrite ---")
    s = json.loads(Path(sp2).read_text())
    chains = [np.array(c, np.int8) for c in s["best_chains"]]

    for attempt in range(10):
        if time.time() - t_start > a.time_limit * 0.9:
            print("  time budget exhausted"); break

        absorb, L, doms = h1_absorption_scan(P, chains)
        n_open = sum(1 for f in L if absorb[f])
        n_dead = len(L) - n_open
        print(f"  attempt {attempt+1}: {len(L)} uncovered, {n_open} H1-open, {n_dead} H1-dead")

        if not L:
            print("  COVERED!"); break
        if n_open == 0:
            print("  all remaining facets H1-dead — restructuring needed beyond this script")
            break

        # pick the chain with the most absorbable facets
        chain_absorb_count = [0] * k
        for f in L:
            for j in absorb[f]:
                chain_absorb_count[j] += 1
        best_j = max(range(k), key=lambda j: chain_absorb_count[j])
        target = sorted(set(doms[best_j]) | {f for f in L if best_j in absorb[f]})
        print(f"  rewriting chain {best_j} ({len(doms[best_j])} facets) to absorb "
              f"{chain_absorb_count[best_j]} H1-open leftovers → target {len(target)} facets")

        new_ch = single_chain_rewrite(
            P, chains[best_j], target, a.seed + 100 + attempt,
            time_limit=min(180, (a.time_limit - (time.time() - t_start)) * 0.5))
        if new_ch is not None:
            chains[best_j] = new_ch
            # update doms and check coverage
            new_doms = [set(int(f) for f in P.domain_of(c)) for c in chains]
            new_cov = len(set().union(*new_doms))
            print(f"  → coverage improved to {new_cov}/{P.nf}")
            if new_cov == P.nf:
                print("  COVERED in Phase 3!")
                # build parts for write_certificate via verify_system
                parts = gpu_sc.verify_system(P, chains)
                if parts is not None:
                    gpu_sc.write_certificate(P, parts, str(outdir / "final"), {
                        "method": "rp2_targeted_push phases 1-3",
                        "seed": a.seed, "elapsed_s": time.time() - t_start})
                    print(f"  Certificate written to {outdir / 'final'}")
                else:
                    print("  verify_system failed — domains do not actually cover (bug?)")
                break
        else:
            print("  SubProduct rewrite failed — trying next chain")

    elapsed = time.time() - t_start
    # save final state regardless
    cov_final = len(set().union(*[set(int(f) for f in P.domain_of(c)) for c in chains]))
    print(f"\n=== Final: {cov_final}/{P.nf} in {elapsed:.0f}s ===")
    if cov_final < P.nf:
        print(f"  {P.nf - cov_final} facets remain — state saved for CP-SAT cold attack")
        # save summary for CP-SAT warm start
        summary = {
            "complex": {"name": P.name, "nv": P.nv, "num_facets": P.nf},
            "params": {"complex": "rp2", "k": k, "cmax": 40, "seed": a.seed},
            "coverage": cov_final,
            "chain_domain_sizes": [len(list(P.domain_of(c))) for c in chains],
            "best_chains": [c.tolist() for c in chains],
            "status": "incomplete",
            "method": "rp2_targeted_push",
            "elapsed_s": elapsed
        }
        (outdir / "final_state.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
