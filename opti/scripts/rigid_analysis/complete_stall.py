#!/usr/bin/env python3
"""complete_stall.py -- exact single-chain completion of a stalled annealer state.

For a stalled summary.json (best_chains saved), compute the k domains and the leftover set L, then ask the exact
solver (domain_chain_sat.decide) whether D_j + L admits a chain of <= L links, for each j.  SAT means the state
was one single-chain move away from a complete k-cover (which the annealer did not find); the completed chain is
saved.  UNSAT at every j and every L tried means no single-domain completion with chains that short exists.

usage: complete_stall.py <complex> <summary.json> [--L 8 12] [--budget CONFLICTS] [--only j]
"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import rigid, relax, domain_chain_sat as dcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('complex'); ap.add_argument('summary')
    ap.add_argument('--L', type=int, nargs='+', default=[8, 12])
    ap.add_argument('--budget', type=int, default=0, help='conflict budget per solve (0 = unlimited)')
    ap.add_argument('--only', type=int, default=None)
    ap.add_argument('--drop', type=int, default=None, help='instead of the leftovers, try to absorb domain DROP into each other domain')
    a = ap.parse_args()
    C = relax.load_complexes(); K = rigid.Complex(C[a.complex]); P = rigid.Product(K)
    s = json.load(open(a.summary))
    doms = rigid.domains_from_chains(P, s['best_chains'])
    covered = set().union(*map(set, doms)); L = sorted(set(range(P.nf)) - covered)
    tag = os.path.basename(os.path.dirname(a.summary))
    if a.drop is not None:
        L = sorted(doms[a.drop]); print(f"{tag}: absorbing domain {a.drop} ({len(L)} facets) into each other domain")
    else:
        print(f"{tag}: {len(doms)} domains {[len(d) for d in doms]}, leftovers {len(L)}")
    order = [a.only] if a.only is not None else sorted(range(len(doms)), key=lambda j: len(doms[j]))
    for j in order:
        if a.drop is not None and j == a.drop: continue
        for Ltry in a.L:
            print(f"  domain {j} (|D|={len(doms[j])}) + {len(L)} facets, L={Ltry}:", flush=True)
            t0 = time.time()
            r, chain = dcs.decide(P, sorted(set(doms[j]) | set(L)), Ltry, return_chain=True, budget=a.budget)
            if r is None:
                print(f"    UNKNOWN (budget exhausted after {time.time()-t0:.0f}s)", flush=True); continue
            if r:
                out = f"{HERE}/certificates/{tag}_absorb{'_dropdom%d' % a.drop if a.drop is not None else ''}_domain{j}_L{Ltry}.json"
                json.dump({'domain_index': j, 'facets': sorted(set(doms[j]) | set(L)), 'chain': chain}, open(out, 'w'))
                print(f"    -> COMPLETION FOUND, chain saved to {out}", flush=True)
                return 0
    print("  no single-domain completion found within the tried chain lengths")
    return 1


if __name__ == '__main__':
    sys.exit(main())
