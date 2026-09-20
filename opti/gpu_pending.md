# Would a bigger GPU help? — pending question, with the evidence

*2026-09-20. Written at the author's request during the 100-try dodecahedron
run, which was stopped at 70 tries later that day; the figures below are the
final ones for those 70 (the run's own write-up is in `dodeca_anneal_log.md`).*

## Short answer

A bigger GPU makes this search **run faster**. It does not make it **reach
more**. The two are different claims and only the first is safe.

## The evidence that the plateau is not throughput-limited

All from the dodecahedron campaign, `dodeca_anneal_log.md`:

1. **Per-move productivity has collapsed by ~50×.** The cold run gained +644
   facets in 2.17 × 10⁹ moves — 0.30 facets per million moves. The most recent
   hour-long rung gained **+13** from 2.23 × 10⁹ moves — **0.006 facets per
   million**. The same number of moves now buys 50× less. More moves is exactly
   what a faster GPU sells, and moves are what stopped paying.
2. **73% of restart tries gain literally nothing.** Distribution over 70
   independent warm-started tries of 3 epochs each (fresh seed every time):
   `{0: 51, 1: 12, 2: 2, 3: 4, 4: 1}` — mean +0.46, best-ever exit **+4**.
3. **The domains do not move.** 1425 + 1232 + 433 → 1425 + 1235 + 433 across
   hours of search. The chains are not being restructured; they are being
   nudged.
4. **Time does not help either.** A 60-minute rung is 3600× one try. It gained
   +13. If search volume were the constraint, that run should have dwarfed a
   3-epoch try; it did not.

## The sampling argument — the strongest form

Each try runs S = 4096 independent annealers. 55 tries is therefore **~225,000
separate annealer runs**, each with a fresh seed and a warm start, and the best
of them improved the total by +4. That is a large sample of what this search
can reach from this state, and the reachable set looks exhausted near 2,600 of
7,776.

A faster GPU would draw *more* samples from the same set. It would find the
best point in that set sooner; it would not enlarge the set.

## Where a bigger GPU genuinely would help

1. **Wall-clock compression, roughly linearly.** A current datacentre card is
   plausibly 20–50× a T4 on this kernel. The ~10 GPU-hours spent getting from 0
   to 2,600 would take ~20–30 minutes. It would then sit at the same plateau.
2. **A much larger population.** S = 4096 is bounded by memory; a bigger card
   could run S = 100,000. Population diversity is a real lever — it is why the
   warm start's transient exists at all — and it is the one variant of this
   search that the campaign has *not* tested. If anything about this plateau is
   a sampling artefact rather than a structural one, a much larger S is where
   it would show up.
3. **`∂Δ⁴` k = 2 (`s3_k2_push_log.md`).** That result's open question is
   evidence, not reachability: more seeds at a real budget would strengthen
   "the plateau is structural". Cheap and worth doing.
4. **Not `∂Δ⁸` (S⁷)** — `s7_chain_first_wall.md`. That wall is a per-thread
   local-memory limit (764 KB against a 512 KB ceiling). It fails identically
   on an H100, or on anything else.

## The honest caveat

Saturation is a strong *inference* from the distribution, the domain
stagnation and the time-independence — not a proof. Nothing here rules out a
rare exit that a much larger population would eventually find.

But what would move this problem is a **different formulation**, not more
silicon: a move attribution that does not touch the whole star of a grid vertex
(listed as not-tried in the anneal log), more chains competing, or
initialisations built from collapsing maps. A bigger GPU makes the present
formulation finish its thinking faster; it does not give it new thoughts.

## If you do rent one

The cheapest experiment that answers the question directly: **one hour on the
biggest card available, with S as large as fits, on the dodecahedron from
`results/dodeca_tries/rung100/summary.json`.** If the per-try distribution
sharpens but the ceiling holds near 2,600, the saturation reading is confirmed
and further hardware is pointless. If a single try breaks well past 2,600 — say
+50 or more — then the plateau was a sampling artefact after all, and that
would be worth knowing before spending anything else.
