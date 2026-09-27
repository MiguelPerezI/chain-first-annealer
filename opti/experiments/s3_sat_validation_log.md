# Validation of `scs_sat.py` on ∂Δ⁴, the L-ladder, and the compression question

*Opened 2026-09-27. Follows the C1 line of `s3_k2_push_log.md` and answers the
open scope gap in `s3_compression_lemma.md`. Everything below runs
`paper-overleaf/paper/scs_sat.py` (CP-SAT encoding) against `src/gpu_sc.py`
(the annealer's own definition). Raw logs: `logs/controls/`, `logs/sat_ladder/`;
the checkers are kept in `scripts/sat_validation/` (`a1_equiv.py`,
`a2_witness.py`, `c2_growth.py`, `c3_precedence.py`) and re-run from the
repo root with the default_env interpreter.*

## The question, in one paragraph

The earlier CP-SAT runs (k = 2 at L = 4, 5, 8, 16 — all INFEASIBLE) would give
SC_strict⁰(∂Δ⁴) = 2 > 1 = TC(S³), the second exact level-0 gap after the
Möbius band. Before trusting an UNSAT certificate from a hand-built model, the
model itself has to be validated: does its notion of "chain covers facet"
match the paper's Definition 2.3 and `gpu_sc.chain_bad_counts` exactly, does
it accept the covers we already know, and does it reject what must be
rejected?  And the UNSAT itself only rules out chains of bounded length —
the *compression* question (can any chain be assumed short?) has to be scoped
honestly.  This log records the validation battery (A), the widened L-ladder
(B), and the compression experiments (C, detailed in
`s3_compression_lemma.md`).

## A1 — predicate equivalence (the encoding audit)

The model's constraints were reimplemented as a pure Python predicate
`sat_cover` (endpoint constraints `m[1] ∈ sig`, `m[L-1] ∈ tau`, middle links
`∃ maximal facet T ⊇ both images`) and compared on random chains against
two independent notions: the paper's definition (`ref_cover`: every link's
union of vertex images lies in some maximal facet of K) and the repo's
`gpu_sc.domain_of`.  Result: **1100/1100 chains agree on all three**
(500 s3, 300 circle, 300 klein; L ∈ {2..7}).

Two facts make the endpoint constraints *exact*, not merely sound: `sig(f)`
is the full π₁-image of f (a maximal facet `s₀`), so `π₁ ∪ img(m[1])` spans a
simplex iff `img(m[1]) ⊆ s₀`; same for `tau`/`s₁` at the tail.  Verified
structurally on all 500 facets: `sig`/`tau` are always kfacets, every facet
has 7 grid vertices, the 25 (s,t) staircase pairs contribute exactly 20
facets each (= 25·C(6,3) = 500), and off-diagonal pairs satisfy
|s ∩ t| = 3, |s \ t| = |t \ s| = 1 (diagonal pairs: s = t).

## A2 — witness round-trips and the padding argument

All five recorded s3 3-covers (`s3_k3_v2`, `s3_k3_run1`, `s3_k3_c80`,
`s3_k3_20260918T164956Z`, `s3_starcover` — 15 reduced chains in
`planners.txt`) were run through `sat_cover`:

* sat == ref == domain_of on every chain;
* every recorded domain is contained in the recomputed one (the recorded
  domain is the disjoint partition piece; the chain's true D(Φ) may be
  larger — e.g. 314 ⊇ 309);
* padding each chain to 16 links by repeating π₂ at the tail never shrinks
  a domain.  This is the monotonicity behind "INFEASIBLE(L) ⇒ INFEASIBLE(L′<L)"
  and validates the L-ladder reading.

One artifact found and understood along the way: `planners.txt` marks grid
vertices outside the domain with `@` (make_definition_md.py:208).  Filling
those holes with π₁ values in *every* map — as the certificate tool does,
safely for its purpose — makes the last map no longer π₂, and the three
predicates then disagree (sat's `tau` constraint correctly flags it).  With
endpoint-consistent filling the disagreement vanishes.  The model was never
wrong; the checker was.

## A3 — solver controls (both directions)

| run | expected | result |
|---|---|---|
| `s3 --k 3 --L 4` | SAT | SAT, 500/500, witness saved |
| `circle --k 1 --L 4` | INFEASIBLE | INFEASIBLE 0.0 s |
| `circle --k 2 --L 4` | SAT | SAT 18/18 |
| `mobius --k 2 --L 6` | INFEASIBLE | INFEASIBLE 11 s |
| `mobius --k 3 --L 6` | SAT | SAT 150/150 |

5/5.  Each SAT witness re-verified independently: π₁/π₂ endpoints exact,
every link contiguous on the claimed domain (`gpu_sc.maps_contiguous_on`),
claimed ⊆ recomputed domain, union = all facets.  (`independent_check.py`
needs a `cover.txt` results directory and rejects SAT witnesses' overlapping
domains — SAT covers may overlap, partitions need not — so the per-chain
check was used instead.)

## A4 — code fixes to `scs_sat.py`

1. `assert L >= 2` in both solvers (the L=1 encoding collides the endpoint
   constraints on `m[1,*]` and is wrong).
2. Naming: `TRIS`/docstring/print called the maximal facets of K
   "triangles"; for s3 they are the 5 tetrahedra.  Semantics were always
   right ("one maximal facet"), only the labels were misleading.
3. Hybrid mode truncated warm chains longer than L (`src_i = min(i, chain_len)`),
   silently breaking `m[d,L] = π₂`; now raises with the required `--L`.

Regression: `circle --k 2 --L 4` still SAT after the edits, `--L 1` now
rejected.

## B — the ladder: k = 2 at L = 24, 32, 40, 48

| L | vars | cons | result | time |
|---|---|---|---|---|
| 4 | 11 155 | 156 501 | INFEASIBLE | 12 s |
| 5 | 16 205 | 227 501 | INFEASIBLE | 33 s |
| 8 | 31 355 | 440 501 | INFEASIBLE | 77 s |
| 16 | 71 755 | 1 008 501 | INFEASIBLE | 239 s |
| 24 | 112 155 | 1 576 501 | INFEASIBLE | 799 s |
| 32 | 152 555 | 2 144 501 | INFEASIBLE | 983 s |
| 40 | 192 955 | 2 712 501 | INFEASIBLE | 1 622 s |
| 48 | 233 355 | 3 280 501 | INFEASIBLE | 1 988 s |

Model size follows `vars = 5000(L−2) + 50(L−1) + 1005` exactly (y-vars =
k·NF·(L−2)·NT).  Padding (A2) makes the family monotone, so the reading is:
**no 2-cover of ∂Δ⁴ × ∂Δ⁴ with chains of ≤ 48 links exists.**

## C — compression (summary; full detail in `s3_compression_lemma.md`)

The UNSAT statement is about bounded link counts.  To upgrade it to
"SC_strict⁰(∂Δ⁴) = 2" one needs: every contiguity domain admits short chains.

* **Proved (Interpolation Lemma):** if φ ~ ψ on D, every map on the Hamming
  box between them is contiguous to both.  Single-vertex changes are free
  *within* a contiguous pair.
* **Falsified (proof route):** the naive π₁→π₂ single-vertex walk fails —
  81 820 facet pairs force flip-order cycles (scripts/sat_validation/c3_precedence.py).  Compression,
  where it exists, uses non-interpolation intermediates (constant maps).
* **Measured (exact min-L via the same SAT constraints at k=1):** every
  chainable domain ever found — cover domains, staircase rectangles, and all
  six k = 2 plateau domains (the largest chainable sets in the repo) — needs
  **≤ 4 links**; grown 320-facet domains reject all 131 attempted additions;
  random sets of 400+ facets are not chainable at all.  Margins below the
  UNSAT ceiling of 48: 12×.

## The statement

SC_strict⁰(∂Δ⁴) = 2 > 1 = TC(S³) — **proven for chains of ≤ 48 links**
(union of the eight rungs above), validated end to end (A1–A3).  Every cover
ever observed in this repo uses ≤ 9 links; every domain ever measured needs
≤ 4.  The unconditional form differs only by the compression claim
(every chainable D has dist_D(π₁,π₂) ≤ 48), which is unfalsified and, on the
evidence in `s3_compression_lemma.md`, overwhelmingly likely.  The same
battery (A1–A4) validates the instrument for the pending rp2 k = 5 and
klein k = 9 decisions.
