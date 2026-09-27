# Cluster runbook — CP-SAT decisions for RP² k=5 and Klein k=9

*Opened 2026-09-27. Purpose: settle by `paper-overleaf/paper/scs_sat.py` (CP-SAT)
whether the annealer's hard stalls are dynamics or combinatorics —
RP² k=5 stuck at 544/600 (from a 6-cover) and Klein k=9 stuck at 1494/1536.
Everything here was validated locally first: encoding audit `s3_sat_validation_log.md`,
ladder methodology `s3_sat_validation_log.md` §B, and live smoke runs whose
vars/cons counts and peak RSS appear below.*

## 1. The two decision questions

Convention (paper `sc_sc_strict_and_tc.tex`): `SC_strict(K)` = (min number of
planner domains covering K×K) **− 1**. So solver `k` = number of domains, and
the theorem value is `k_min − 1`.

| question | solver run | SAT means | UNSAT (at L=40) means |
|---|---|---|---|
| does a 5-domain cover of RP²×RP² exist? | `--complex rp2 --k 5` | min domains ≤ 5 → `SC_strict⁰(RP²) ≤ 4` | min domains ≥ 6; with the verified 6-cover `results/rp2_k6` → **`SC_strict⁰(RP²) = 5`** |
| does a 9-domain cover of K²×K² exist? | `--complex klein --k 9` | min domains ≤ 9 → `SC_strict⁰(K²) ≤ 8` | min domains ≥ 10; with the verified 10-cover `results/klein_k10` → **`SC_strict⁰(K²) = 9`** |

Either answer is a theorem (first exact `SC_strict⁰` value for RP²; the
Klein bound sharpens from the current `≤ 9`). The known brackets:
`2 ≤ SC_strict⁰(RP²) ≤ 5` (`rp2_literature_check.md`), `TC(K²)=4 ≤ SC_strict⁰(K²) ≤ 9`.

**Scope of an UNSAT rung.** `INFEASIBLE(L)` rules out chains of ≤ L links
(padding: repeat π₂ at the tail, domain never shrinks — `s3_sat_validation_log.md` §A2).
The annealer stores chains with `cmax=40` links; L=40 is therefore the
"whole annealer search space" rung. Every chain ever observed here has only
3–10 **active** links (below), and the compression measurements
(`s3_compression_lemma.md`) found min chain length ≤ 4 on every chainable
domain tested — so an UNSAT at L=40 is unconditionally meaningful and, up
to the named compression conjecture, the final answer.

## 2. Ground truth (recomputed 2026-09-27 from `best_chains`)

| complex | NV | NG | NF | NT | best stall | covered | known cover |
|---|---|---|---|---|---|---|---|
| RP² (hemi-icosahedron) | 6 | 36 | 600 | 10 | `results/rp2_k5_from_k6_stretch_collapse7232` | **544/600** | `results/rp2_k6` 600/600 |
| Klein (Cervone 242) | 8 | 64 | 1536 | 16 | `results/klein_k9_v3` (also v4) | **1494/1536** | `results/klein_k10` 1536/1536 |

* (`results/rp2_k5_hybrid` = 540/600; `results/klein_k9` = 1491/1536 — the
  older, slightly weaker stalls. Earlier notes claiming "780/800" for RP²
  were wrong: NF(RP²×RP²)=600.)
* Stored chains are 41 maps (40 links) but collapse to 3–10 active links:
  RP² stretch chains use 10/4/5/3/4, Klein v3 chains 4–8. Every domain is
  reproduced exactly by the collapsed chain.

## 3. Instrument sizing (validated)

Model size is exactly
`vars ≈ k·NF·NT·(L−2) + k·NG·(L−1) + k·NF + O(NV)`,
`cons ≈ 2·|verts|·k·NF·NT·(L−2) + k·NF·(L−2) + 2·|verts|·k·NF + NF`,
with `|verts|=5` grid vertices per staircase facet on both complexes.
Validated by live builds: rp2 k=5 L=8 → 184 266 vars / 1 848 604 cons
(≈2.2 GB peak RSS, workers=4); klein k=9 L=4 → 457 928 / 4 591 112 (≈3.7 GB).

Cold ladder sizes:

| L | rp2 k=5 vars | cons | klein k=9 vars | cons | est. RAM |
|---|---|---|---|---|---|
| 4 | 64 k | 0.64 M | 458 k | 4.6 M | 4 GB |
| 5 | 94 k | 0.94 M | 680 k | 6.8 M | 5 GB |
| 6 | 124 k | 1.24 M | 901 k | 9.0 M | 7 GB |
| 8 | 184 k | 1.85 M | 1.34 M | 13.5 M | 11 GB |
| 12 | 305 k | 3.06 M | 2.23 M | 22.4 M | 18 GB |
| 16 | 426 k | 4.27 M | 3.12 M | 31.3 M | 25 GB |
| 24 | 667 k | 6.70 M | 4.89 M | 49.1 M | 40 GB |
| 32 | 909 k | 9.12 M | 6.67 M | 66.9 M | 55 GB |
| 40 | 1.15 M | 11.5 M | 8.44 M | 84.7 M | 70 GB |

RAM ≈ 0.6–0.8 GB per M cons (measured at two points; extrapolated).
Hardware: any 16-core node runs all RP² rungs (max ~10 GB); Klein L ≤ 16 fits
32 GB; Klein L ≥ 24 wants 64 GB. No GPU — pure CPU CP-SAT
(`ortools` + `numpy` + `numba` needed to import `gpu_sc`; locally:
`/home/mike666/default_env/bin/python`). Solver workers: `--workers 16` on
big nodes, `--workers 8` on the L ≥ 24 giants (memory scales with workers).

## 4. Track 0 — controls (run first; gate for trusting UNSAT)

The encoding is already validated on S¹/S²/S³/Möbius (`logs/controls/`,
`scripts/sat_validation/`). These two controls validate the *pipeline at
these complexes' scale* — an UNSAT answer is only reportable after they pass.

```bash
PY=/home/mike666/default_env/bin/python   # cluster: any python with ortools
cd /home/mike666/chain-first-annealer/opti
# known 6-cover at k=6 L=8 must go SAT (its chains have <=4 active links, padded)
setsid nohup $PY paper-overleaf/paper/scs_sat.py --complex rp2 --k 6 --L 8 \
  --timeout 21600 --workers 16 --witness --outdir results/cp_sat_rp2_k5 \
  < /dev/null > logs/cp_sat_decisions/rp2_k6_cold_L8_control.log 2>&1 & disown
# known 10-cover at k=10 L=8 must go SAT (chains <=7 active links)
setsid nohup $PY paper-overleaf/paper/scs_sat.py --complex klein --k 10 --L 8 \
  --timeout 21600 --workers 16 --witness --outdir results/cp_sat_klein_k9 \
  < /dev/null > logs/cp_sat_decisions/klein_k10_cold_L8_control.log 2>&1 & disown
```

If a control cannot reproduce SAT within its budget, do not report the
corresponding UNSAT — escalate the control first (bigger L makes SAT easier:
any ≤ L-link cover is representable at L by padding).

## 5. Track A — RP² k=5, cold L-ladder (the decisive run)

Independent jobs per rung (padding makes rungs monotone — run them all in
parallel on the cluster). Cheap rungs first to bank UNSAT certificates;
SAT at any rung ends the campaign.

```bash
for L in 4 5 6 8 12 16; do
  setsid nohup $PY paper-overleaf/paper/scs_sat.py --complex rp2 --k 5 \
    --L $L --timeout 86400 --workers 16 --witness --outdir results/cp_sat_rp2_k5 \
    < /dev/null > logs/cp_sat_decisions/rp2_k5_cold_L$L.log 2>&1 & disown
done
# giants on 64GB nodes, after the first wave:
for L in 24 32 40; do
  setsid nohup $PY paper-overleaf/paper/scs_sat.py --complex rp2 --k 5 \
    --L $L --timeout 259200 --workers 8 --witness --outdir results/cp_sat_rp2_k5 \
    < /dev/null > logs/cp_sat_decisions/rp2_k5_cold_L$L.log 2>&1 & disown
done
```

## 6. Track A′ — Klein k=9, cold L-ladder (the decisive run)

Same structure, 2–2.5× the RP² size. Do **not** skip straight to L=40
(84.7 M cons / ~70 GB): the L ≤ 8 rungs are where UNSAT certificates are
cheap, and L=12/16 already cover every chain length ever observed for Klein.

```bash
for L in 4 5 6 8; do
  setsid nohup $PY paper-overleaf/paper/scs_sat.py --complex klein --k 9 \
    --L $L --timeout 86400 --workers 16 --witness --outdir results/cp_sat_klein_k9 \
    < /dev/null > logs/cp_sat_decisions/klein_k9_cold_L$L.log 2>&1 & disown
done
for L in 12 16; do   # 32GB nodes fine
  setsid nohup $PY paper-overleaf/paper/scs_sat.py --complex klein --k 9 \
    --L $L --timeout 172800 --workers 16 --witness --outdir results/cp_sat_klein_k9 \
    < /dev/null > logs/cp_sat_decisions/klein_k9_cold_L$L.log 2>&1 & disown
done
for L in 24 32 40; do   # 64GB nodes
  setsid nohup $PY paper-overleaf/paper/scs_sat.py --complex klein --k 9 \
    --L $L --timeout 259200 --workers 8 --witness --outdir results/cp_sat_klein_k9 \
    < /dev/null > logs/cp_sat_decisions/klein_k9_cold_L$L.log 2>&1 & disown
done
```

## 7. Track B — hybrid residual attacks (SAT-hunting; cheap)

Hybrid freezes the biggest warm domains and re-solves the rest against the
residual facets. **SAT here is an unconditional theorem** (the frozen chains
are verified planners); **UNSAT is conditional** on the chosen freezing and
L — never report it as non-existence. Annealer chains must first be
collapsed (`scs_sat.py` rejects warm chains longer than `--L`; stored chains
are 40 links with only 3–10 active):

```bash
$PY scripts/sat_validation/reduce_warm.py rp2   results/rp2_k5_from_k6_stretch_collapse7232/summary.json results/warm_reduced/rp2_k5_stretch_red.json   # max active 10
$PY scripts/sat_validation/reduce_warm.py klein results/klein_k9_v3/summary.json                        results/warm_reduced/klein_k9_v3_red.json          # max active 8
```

Residual sizes for `freeze top-n / free (k−n)` (top by domain size):

| warm (k) | free 1 | free 2 | free 3 | free 4 |
|---|---|---|---|---|
| rp2_k5_stretch (k=5, 544/600) | 146 (frozen 4) | 227 (frozen 3) | 323 (frozen 2) | 449 (frozen 1) |
| klein_k9_v3 (k=9, 1494/1536) | 121 (frozen 8) | 232 (frozen 7) | 305 (frozen 6) | 446 (frozen 5) |

Recommended sweep (start free=1 — smallest models, ~150 k–100 k vars):

```bash
# RP²: free 1 and 2 at L=10 (max active link count of the warm chains)
setsid nohup $PY paper-overleaf/paper/scs_sat.py --complex rp2 --k 5 --L 10 \
  --timeout 3600 --workers 8 --witness --outdir results/cp_sat_rp2_k5 \
  --warm results/warm_reduced/rp2_k5_stretch_red.json --free 1 \
  < /dev/null > logs/cp_sat_decisions/rp2_k5_hybrid_free1_L10.log 2>&1 & disown
setsid nohup $PY paper-overleaf/paper/scs_sat.py --complex rp2 --k 5 --L 10 \
  --timeout 7200 --workers 8 --witness --outdir results/cp_sat_rp2_k5 \
  --warm results/warm_reduced/rp2_k5_stretch_red.json --free 2 \
  < /dev/null > logs/cp_sat_decisions/rp2_k5_hybrid_free2_L10.log 2>&1 & disown
# Klein: free 1, 2, 3 at L=8
for FREE in 1 2 3; do
  setsid nohup $PY paper-overleaf/paper/scs_sat.py --complex klein --k 9 --L 8 \
    --timeout 7200 --workers 8 --witness --outdir results/cp_sat_klein_k9 \
    --warm results/warm_reduced/klein_k9_v3_red.json --free $FREE \
    < /dev/null > logs/cp_sat_decisions/klein_k9_hybrid_free${FREE}_L8.log 2>&1 & disown
done
```

Escalation on all-UNSAT: free 4 with L=12/16 for the freed domains (frozen
chains are padded to any L ≥ their length automatically), and the second
warm file (`results/rp2_k5_hybrid`, `results/klein_k9_v4`).

Recorded hybrid verdicts (2026-09-27, local, all conditional on the freezing):

| run | frozen/free | residual | L | verdict | time |
|---|---|---|---|---|---|
| rp2_k5_stretch | 4 / 1 | 146 | 10 | INFEASIBLE (presolve) | 4 s |
| rp2_k5_hybrid (older warm) | 3 / 2 | 216 | 8 | INFEASIBLE (prior, `annealer_formal_properties.md` §2) | — |
| klein_k9_v3 | 8 / 1 | 121 | 8 | INFEASIBLE (presolve) | 14 s |

## 8. Job management

* Launch pattern: `setsid nohup CMD < /dev/null > LOG 2>&1 & disown` — a bare
  `&` dies with the launching shell's process group.
* One job per (complex, k, L, mode). No inter-job dependencies.
* Logs grow large (`log_search_progress=True`); keep them on scratch if
  quota-bound. Progress: `grep -E "model built|solution #|k=.*: " LOG`.
* Status lines end `k=K L=L: OPTIMAL|FEASIBLE|INFEASIBLE|UNKNOWN in Xs`.
  **UNKNOWN proves nothing** — re-launch the same rung with 2× `--timeout`
  (and/or `--workers`); there is no checkpointing.
* UNSAT wall-time grows steeply with L (S³ ladder: 12 s at L=4 → 1 988 s at
  L=48). Budgets above assume RP² ≲ S³ difficulty and Klein ≈ 2× RP².

## 9. Witness verification (mandatory before claiming SAT)

`solve_cold`/`solve_hybrid` write `scs_witness_k{K}_L{L}.json` =
`{"chains": [k][L+1][NG], "domains": [k][lists of facets]}`. Check:

```python
import json, sys
sys.path.insert(0, "src"); import gpu_sc
P = gpu_sc.build("rp2")  # or "klein"
w = json.load(open("results/cp_sat_rp2_k5/scs_witness_k5_L10.json"))
tri = [set(t) for t in P.kfacets]
FAC = [sorted(int(v) for v in P.facet_verts[i][:P.facet_nv[i]]) for i in range(P.nf)]
PI1 = [v // P.nv for v in range(P.ngrid)]; PI2 = [v % P.nv for v in range(P.ngrid)]
for d, (ch, dom) in enumerate(zip(w["chains"], w["domains"])):
    assert ch[0] == PI1 and ch[-1] == PI2
    for f in dom:
        for i in range(len(ch) - 1):
            u = {ch[i][v] for v in FAC[f]} | {ch[i+1][v] for v in FAC[f]}
            assert any(u <= t for t in tri), (d, f, i)
covered = set(f for dom in w["domains"] for f in dom)
assert len(covered) == P.nf, f"covers only {len(covered)}/{P.nf}"
print("witness OK:", len(w["chains"]), "domains, all links contiguous, cover complete")
```

(All links contiguous on each claimed facet *and* union = all facets.
SAT domains may overlap — that is a cover, not a partition.)

## 10. What each outcome buys

| outcome | publishable statement |
|---|---|
| rp2 k=5 SAT (any L) | `SC_strict⁰(RP²) ≤ 4` — annealer stall is dynamics |
| rp2 k=5 INFEASIBLE through L=40 | `SC_strict⁰(RP²) = 5` (first exact value for a closed non-orientable surface; the stall is combinatorics) |
| klein k=9 SAT (any L) | `SC_strict⁰(K²) ≤ 8` — annealer stall is dynamics |
| klein k=9 INFEASIBLE through L=40 | `SC_strict⁰(K²) = 9` (annealer's 1494 stall is combinatorics) |
| klein k=9 INFEASIBLE only through L=16 | same claim scoped "chains of ≤ 16 links" (= all observed chains + compression margin 4×) |
| hybrid SAT | the corresponding ≤ claim, unconditional (verify witness §9) |
| hybrid INFEASIBLE | only "this freezing cannot be extended" — dynamics evidence, not a bound |

**Lower rungs accumulate**: UNSAT at L=8 after UNSAT at L=6 means "no cover
with chains ≤ 8 links" — if the campaign stops early, the scoped theorem
still stands and is recorded in the ladder table (format of
`s3_sat_validation_log.md` §B). Append final results there or in a new
`experiments/rp2_klein_cp_sat_results.md`.

## Appendix — solver interface recap

```
scs_sat.py --complex {rp2,klein,...} --k K --L L --timeout S --workers N
           --witness --outdir DIR [--warm summary.json --free M]
```
* `--L 1` is rejected (encoding needs ≥ 2 links).
* `--warm` with `--free 0` runs **cold** (warm file ignored) — that is not a
  hybrid run.
* Frozen warm chains longer than `--L` raise `ValueError` (the old silent
  truncation could fabricate witnesses; feasibility verdicts of the old
  hybrid runs are unaffected — frozen chains enter no constraint).
* Hybrid `k` must equal the warm file's domain count (frozen = `k − free`).
