# Guided two-chain test on the spheres — plan and prior evidence

*Written 2026-09-19, before the test. Question: do the k=2 plateaus on the odd
spheres mean the two-domain cover does not exist (Conjecture C1), or did the
searches simply run out of budget?*

> **Outcome (2026-09-19, same day).** The plan was executed on ∂Δ⁴ and the
> budget hypothesis is dead: with ~25× the search budget (69 epochs, 5.65 × 10⁹
> moves, corrected energy) the population still sits at exactly 460/500 with
> zero spread in every single epoch. Two corrections to this plan were needed
> first — the `--guide` escalation below was a no-op against a hard-coded
> weight cap, and the pass-1 command line contradicts the corrected energy of
> `optimized_algorithms.md` §14.3 — and the campaign went on to characterise
> what 460 is (the exhaustive 2-rectangle optimum) and to close two structured
> families that could have contained a cover. Full log, including the runs that
> were deliberately *not* made: `s3_k2_push_log.md`. The plan below is left
> as written, as the pre-registration.

## The question in context

- Corollary 5.3 of the paper (all boundary-of-simplex spheres): ρ(∂Δ^{n+1}) = 3,
  so SCst(∂Δ^{n+1}) ≤ 2 for every n ≥ 1, and SCst = 2 = TC(S^n) exactly for even
  n ≥ 2, sharp at subdivision level 0. Even spheres are **done**.
- Odd spheres: TC(S^odd) = 1, so SCst ∈ {1, 2}. A **two-domain** cover would give
  SCst = 1, and by the corollary's lower-bound part it would have to use
  **non-product domains** (ρ = 3 means two rectangles never cover; the circle is
  the precedent: SCst(∂Δ²) = 1 via non-product domains while ρ − 1 = 2).
- C1 claims SCst(∂Δ⁴) = 2, i.e. two domains do **not** suffice at level 0.
  C2 generalizes: with k < ρ chains, searches always converge to (sub)rectangle
  covers of ρ − 1 rectangles — the plateaus 84/96, 460/500, 2380/2520.

## Prior evidence on record (all in `results/`, `logs/`)

| run | epochs asked / run | energy | guide | outcome |
|---|---|---|---|---|
| `s3_k2_run1` | 17, stopped early | point | none | 460/500 |
| throughput probe | 3 (18 s) | point | none | 460/500 (295 + 184) |
| `s3_k2_A` | 200 / **3** (300 s TL, ~130 s/epoch) | point | ptrack 0.05, guide 1.0 | 460/500 (284 + 265) |
| `s3_k2_B` | 200 / **3** | point + track | adaptive weights | 460/500 (203 + 296) |
| `s3_k2_C` | 200 / **2** | point | guide 1.0 | 460/500 (284 + 216) |
| `s3_k2_D` | 200 / **3** | point | guide 1.0 | 460/500 (275 + 212) |
| `push_s4` (∂Δ⁵, k=2) | — / 1 (716 s) | point | — | 2380/2520 (1746 + 1750) |

Twelve two-chain searches total (old and corrected energy, all seeds): **every
one pinned at exactly 460/500**, which is precisely the best cover by ρ − 1 = 2
rectangles. The pinning is total: top10 all 460, median 460, from epoch 1.

**The cube contrast (why this is not automatically conclusive):**
the recorded cube completion was *not* an epochs story — cold k3 → 662/864
(300 s); `cube_k3_v2` ran **60 full epochs** (814 s) and stayed pinned at
820/864; what closed the last 44 was warm start + the coverage guide
(`cube_k3_guided`: 864/864 in 22 epochs). But note the plateau *shapes* differ:

- cube plateau: population still exploring — top10 spread 810–820, median
  climbing 468 → 785. A guide had something to push.
- sphere k=2 plateaus: zero spread at exactly the rectangle optimum from
  epoch 1, guide already on. Signature of a structural attractor (C2): when
  k < ρ, the only good basins are rectangle covers, and any single-facet move
  out of a rectangle sheds many covered facets at once.

## What has actually never been tried

1. The k=2 runs never ran their epochs — the 300 s time limit cut every one at
   2–3 epochs. "Enough epochs" is genuinely an untested variable.
2. The cube's escalation ladder was never applied to k=2: cold → continue from
   its **own** best state → continue + stronger guide. The pushes so far seeded
   from the *k=3 certificate*, not from chained k=2 continuations.
3. Escalating guide weight (1.0 → 3.0 → 10.0) on the pinned 460 state.

## The test

Run on the GPU box (same env as `run_s3.sh`: numba-CUDA python, then audit with
`independent_check.py`). Three passes; a pass 1 population that stays frozen at
exactly 460 with zero spread for a full hour is already strong evidence for
C1/C2 — pass 2 exists to rule out the "one basin away" cube scenario.

```bash
# pass 1 — a real budget for once (was: 300 s)
$PY -u gpu_sc.py --complex s3 --k 2 --cmax 40 --S 4096 --iters 20000 \
    --epochs 200 --time-limit 3600 --ptrack 0.05 --guide 1.0 \
    --seed 5 --outdir results/s3_k2_long_5 2>&1 | tee logs/s3_k2_long_5.log

# pass 2 — cube-style escalation: warm-start from pass 1's OWN best state,
# stronger guide
$PY -u gpu_sc.py --complex s3 --k 2 --cmax 40 --S 4096 --iters 20000 \
    --epochs 200 --time-limit 3600 --ptrack 0.05 --guide 3.0 \
    --init results/s3_k2_long_5/summary.json --seed 6 \
    --outdir results/s3_k2_esc_6 2>&1 | tee logs/s3_k2_esc_6.log

# pass 3 — if pass 2 moved at all: same again with guide 10 and cmax 80
# (the wider cmax lets a domain abandon rectangle structure wholesale)
$PY -u gpu_sc.py --complex s3 --k 2 --cmax 80 --S 4096 --iters 20000 \
    --epochs 200 --time-limit 3600 --ptrack 0.05 --guide 10.0 --T1 0.1 \
    --init results/s3_k2_esc_6/summary.json --seed 7 \
    --outdir results/s3_k2_esc_7 2>&1 | tee logs/s3_k2_esc_7.log

# any cover.txt that appears: audit it immediately
python3 independent_check.py results/s3_k2_* | tail -5
```

Repeat pass 1 with 2–3 seeds (5, 11, 23) — one seed per hour of GPU.

## Decision criteria

- **Still exactly 460/500, zero spread, after 3600 s + escalated-guide
  continuations** → the plateau is structural, not budgetary. C1 stands on
  much firmer ground; record the runs and cite them as the strengthened
  evidence in the conjecture's parenthetical (12 → 12 + N runs, real budgets).
- **Any movement past 460, or a non-product domain larger than the 329 maximum
  on record** → the attractor is escapable. Keep escalating; every 10 facets
  past 460 is new information about the landscape.
- **A full 500/500 two-domain cover** → SCst(∂Δ⁴) = 1 exactly (TC(S³) = 1 is the
  matching lower bound), beating Corollary 5.3's 2 via non-product domains —
  the circle phenomenon one dimension up, and C1 is false. Audit, then repeat
  the same ladder on ∂Δ⁵ (plateau 2380/2520) to see how far the family goes.

## Cost

~1 h GPU per pass-1 seed, ~1 h per escalation pass; ≲ 6 h GPU for the whole
ladder on ∂Δ⁴. The ∂Δ⁵ rerun is ~1.2 h/epoch-scale — only if ∂Δ⁴ moves.