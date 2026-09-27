"""A2: witness round-trips + padding argument (reduced chains from planners.txt).

Uses make_certificate_md.parse_planners ('@' -> -1, fill holes with pi1 as the
certificate tool does; holes are grid vertices outside the domain's facets).

Checks per chain:
  1. sat-cover semantics == ref (definition) == gpu_sc.domain_of on the filled chain;
  2. the recorded domain is a subset of that (holes may only add facets);
  3. padding c links -> L links (repeat pi2 at tail) never shrinks the domain
     -- the monotonicity behind INFEASIBLE(L) => INFEASIBLE(L' < L).
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import gpu_sc
from a1_equiv import sat_cover, ref_cover
from make_certificate_md import parse_planners

def fill(m, P, first=False, last=False):
    """Fill '@' holes so the chain is a genuine pi1 -> ... -> pi2 chain:
    first map fills with pi1, last map with pi2, interior arbitrary (pi1).
    Only then do sat/ref/gsc evaluate the same object."""
    mm = m.copy()
    if last:
        mm[mm < 0] = P.pi2[mm < 0]
    else:
        mm[mm < 0] = P.pi1[mm < 0]
    return mm.astype(np.int8)

def pad_tail(chain, L):
    c = len(chain) - 1
    assert c <= L, f"chain has {c} links > L={L}"
    out = list(chain)
    while len(out) < L + 1:
        out.append(out[-1].copy())
    return out

def roundtrip(name, pdir, L_target=16):
    P = gpu_sc.build(name)
    s = json.load(open(f"{pdir}/summary.json"))
    print(f"\n== {pdir} ==")
    chains = parse_planners(f"{pdir}/planners.txt", P.nv)
    recorded = [sorted(int(f) for f in d) for d in s["domains"]]
    if len(chains) != len(recorded):
        print(f"  FAIL chain/domain count {len(chains)} != {len(recorded)}")
        return False
    ok = True
    union = set()
    for j, ch in enumerate(chains):
        raw = [np.asarray(m, np.int64) for m in ch]
        arr = [fill(m, P, first=(i == 0), last=(i == len(raw) - 1))
               for i, m in enumerate(raw)]
        c = len(arr) - 1
        # endpoints on the recorded domain's vertices must be pi1/pi2 (no holes there)
        hole_on_dom = 0
        ep_bad = 0
        for f in recorded[j]:
            for g in P.facets[f]:
                if any(m[g] < 0 for m in raw):
                    hole_on_dom += 1
                if arr[0][g] != P.pi1[g] or arr[-1][g] != P.pi2[g]:
                    ep_bad += 1
        d_sat = set(sat_cover(P, arr, c))
        d_ref = set(ref_cover(P, arr))
        d_gsc = set(int(f) for f in P.domain_of(np.stack(arr)))
        same = d_sat == d_ref == d_gsc
        covers_rec = set(recorded[j]) <= d_sat
        union |= set(recorded[j])
        print(f"  chain {j}: {c} links, sat domain {len(d_sat)}, "
              f"recorded {len(recorded[j])}; sat==ref==gsc: {'OK' if same else 'MISMATCH'}, "
              f"recorded ⊆ sat: {'OK' if covers_rec else 'FAIL'}, "
              f"holes on recorded facets: {hole_on_dom}, endpoint errors: {ep_bad}")
        ok &= same and covers_rec and hole_on_dom == 0 and ep_bad == 0
        padded = pad_tail(arr, L_target)
        d_pad = set(sat_cover(P, padded, L_target))
        grew = d_pad >= d_sat
        print(f"    pad to L={L_target}: sat domain {len(d_pad)} "
              f"{'superset OK' if grew else f'SHRANK (lost {len(d_sat - d_pad)})'}")
        ok &= grew
    cover = len(union) == P.nf
    print(f"  recorded union: {len(union)}/{P.nf} {'COVER' if cover else 'NOT A COVER'}")
    return ok and cover

if __name__ == "__main__":
    ok = True
    for d in ["results/s3_k3_v2", "results/s3_k3_run1", "results/s3_k3_c80",
              "results/s3_k3_20260918T164956Z", "results/s3_starcover"]:
        try:
            ok &= roundtrip("s3", d)
        except FileNotFoundError as e:
            print(f"\n== {d} == SKIP ({e})")
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
