# Verification of §3 ("How the greedy dies") — experiments of 2026-09-22

The old greedy (the OLGB21 pipeline, faithful GPU port in
`~/Documents/Abstract_Motion_GPU`) was executed on **S² = ∂Δ³ (96 facets)** at
the cluster's own settings — `M=40000, r=0.1`, batched-retry growth with
**unbounded retry passes** (the C++ law that ran 120-hour jobs on the 96-core
cluster), 8 replicas, two independent runs — plus instrumented replays of the
exact LocalSearch walk, a faithful k=1 miniature of the paper's annealer, and
the original C++ step benchmark.  S² confirmed everything the article claims,
so no S³ rerun was needed (the existing `s3.log` corroborates the same
pattern at 500 facets).

## Verdict table

| Claim in main.tex | Experiment | Verdict |
|---|---|---|
| "growth froze after the first few dozen facets" (§3) | 16 replicas across 2 runs froze at 30–54 facets | **Confirmed** |
| "dozens of consecutive retry passes flipped nothing at full computational cost" (§3) | 65 consecutive all-fail passes, ~26.6 s each; 68.1% of tests / 81.7% of wall time / 1.35×10¹¹ steps bought nothing | **Confirmed, now quantified** |
| "about 70 µs per step … ~3 seconds per candidate walk" (§3) | Compiled and ran the original C++ `bench_step`: 68.3 µs whole-domain `contiguous()` rescan + 2×4.3 µs `d()` (Dijkstra) ≈ 77 µs/step → 3.07 s per failed M=40000 walk | **Confirmed** (bench_step's final estimate line has a variable-reuse bug — prints 0.5 s; its two raw measurements are the evidence) |
| "flat energy … no gradient over the number of defects" (§3) | Instrumented replays: all 42 candidates of a frozen 54-facet front score identically "fail" while plateauing at d = 1…10; 18/42 sit at d=1 (128/128 replays, one vertex flip from π₂); 9.8% of 37M rejections break contiguity on exactly 1 facet, 13.9% on 2; hundreds of d-improving proposals per walk blocked on a single facet; baseline contrast: at |J|=10 the same machinery succeeds 32/32 | **Confirmed and measured** |
| "chain drift … the likely microscopic mechanism of the macroscopic freeze" (§3) | The greedy cannot drift: every walk rebuilds from π₁ and every accepted step is a fully contiguous link. Measured freeze mechanism is different (below) | **Needs re-attribution** |
| (new) the actual freeze mechanism | Self-cornering: every frozen front is a near-rectangle (31–47 of a maximal 54-facet D(v,w) inside, 5–14 outside); 0/146 pending candidates across three fronts (54, 52, 36 facets) attachable by the walk (up to 3.3×10⁸ steps/candidate) **or** by the k=1 annealer miniature at the paper's parameters (β=5, γ=0.5, p_track=0.3, up to 128 chains × 20 epochs = 5.1×10⁷ proposals/candidate) — all runs stall exactly one defect short. Miniature is sound: it solves a known-feasible 45-facet certified domain in one epoch (verified by `check_chain`) | **New finding**: fronts are genuine local optima; first k=1 evidence for Conjecture C2 |
| (context) "the search domains are near-rectangles" (§4, stated for the annealer) | The **greedy's** frozen domains are near-rectangles too | Extends the §4 remark to the greedy |

## Proposed edits to main.tex

### 1. Abstract (the stall-tracing sentence)

```latex
% OLD
% We trace this stall to the objective function---the energy landscape is
% flat exactly where the search needs a gradient, causing chain entries
% to drift to noise unless actively tracked.
% NEW
We trace this stall to two defects of the search itself: an all-or-nothing
score that is flat exactly where the search needs a gradient, and a
sequential growth rule that corners itself at near-product local optima
whose remaining facets no retry can attach.
```

### 2. Introduction (the "three ingredients" sentence)

```latex
% OLD
% a diagnosis of the two ways the greedy search dies, flat scores and
% drifting chains (Section~\ref{sec:fail});
% NEW
a diagnosis of the two ways the greedy search dies, flat scores and
self-cornering growth (Section~\ref{sec:fail});
```

### 3. §3 opening line

```latex
% OLD
% Three autopsy findings, each verified experimentally.
% NEW
Three autopsy findings, each verified experimentally; the first two are
now measured on the $2$-sphere model itself, by running the faithful GPU
port of the pipeline under the cluster's own retry law and instrumenting
the frozen front it reaches.
```

### 4. §3 "Flat energy" — replace the closing hedge by the measurement

```latex
% OLD (last two sentences of the paragraph)
% That is a documented failure mode of our own scoring function; that the
% same flatness is what stalls the pipeline of \cite{OLGB21} is a reading
% of the two behaviours, not a causal link we have isolated.
% NEW
That is a documented failure mode of our own scoring function, and on the
pipeline's own walk the same flatness is now measured directly.  At a
frozen front of $54$ facets, exact replays of \textsc{LocalSearch} (same
pseudo-random stream, same acceptance rule) score all $42$ pending
candidates identically ``fail'' while their true distance to completion
spans an order of magnitude: $18$ of them plateau at $d=1$---one vertex
flip from the target, where $d$ is the walk's internal score, the sum of
graph distances from the current map to $\pi_2$ over the domain's
vertices---and stay there for the remaining $86\%$ of the step budget on
$128$ independent replays; the rest plateau at $d=5$ to $10$.  The
step-level predicate behaves the same way: of $37$ million rejected
proposals, $9.8\%$ break $1$-contiguity on exactly one facet and $13.9\%$
on two, and each walk sees hundreds of proposals that would strictly
improve its score yet are blocked by contiguity on a single facet.  A
score that counted defecting facets would see all of this; the binary
score reports only ``fail''.
```

(For reference: the first sentences of the paragraph — the "one bad link ≈
forty bad links" statement and the β=0.05 anecdote — stay as they are. If
desired, "forty" can be grounded as "ten", the measured range.)

### 5. §3 — replace the "Chain drift" paragraph by "Self-cornering" + "Drifting chains"

```latex
% OLD: the whole \textbf{Chain drift.} paragraph
% NEW:
\textbf{Self-cornering.}  The freeze has a mechanism sharper than
flatness: growth is sequential and irrevocable---one facet at a time,
never undone---so each accepted choice constrains the next, and the walk's
own reachability shapes the domain.  In the instrumented run all eight
replicas froze at $36$--$54$ facets, and every frozen domain is a
\emph{near-rectangle}: it contains $31$--$47$ of the $54$ facets of some
maximal rectangle $D(v,w)$ of Theorem~\ref{thm:prod}, plus $5$--$14$
facets outside it.  At three such fronts ($54$, $52$ and $36$ facets) we
tested every pending candidate---$146$ tests in all---with the original
walk (up to $3\times10^{8}$ steps per candidate across the pipeline's
own retries, plus fresh replays) and with a single-chain miniature of our
annealer at the parameters of Section~\ref{sec:annealer} ($\beta=5$,
$\gamma=0.5$, $p_{\mathrm{track}}=0.3$, up to $128$ chains and
$5\times10^{7}$ proposals per candidate).  Nothing attaches: every
candidate resists both searches, and both stall \emph{exactly one defect
short}---the walk at $d=1$ from $\pi_2$, the annealer with a single
defecting link remaining.  The miniature is sound; on a known feasible
$45$-facet domain it finds a verified chain in one epoch.  The front is
thus a genuine local optimum of the growth dynamics, not a flat patch a
gradient would cross; the wedge-of-two-circles reproduction of the first
phase showed the same signature at the refinement level, where exhaustive
pairwise-merge probing certified the greedy's partition a stable local
optimum one domain above the true one.  Flatness is what turns this
corner into the observed cost profile: a failed test carries no
information, so the retry law cannot distinguish a doomed candidate from
an unlucky one and retests the whole front forever.  In the instrumented
run, $65$ consecutive retry passes retested all $407$ pending
candidates---about $2\times10^{9}$ walk steps per pass---with zero
successes: $68\%$ of all tests and $82\%$ of the wall time bought
nothing.

\textbf{Drifting chains.}  When several chains compete to cover the same
facets and the score takes the minimum over chains, entries of a chain
that are momentarily irrelevant receive no signal at all and drift to
noise; when a facet should migrate from one chain to another, the
receiving chain has been randomized exactly where it matters.  This is a
disease of the \emph{annealer's} energy, and the cleanliness term
$\gamma$ of the next section is its suppression ($\gamma=0$ fails again,
$\gamma=0.1$ is measurably too weak).  It is not a mechanism of the
greedy, whose walks rebuild each chain from $\pi_1$ at every attempt and
accept only steps that keep every link contiguous---there is nothing to
drift.  An earlier reading of ours attributed the cluster freeze to
drift; the instrumented replays above trace it to self-cornering instead.
```

### 6. §3 "Quadratic accounting" — insert the measured breakdown

```latex
% OLD
% ... from scratch:
% about $70\,\mu$s per step in the C++/MPI implementation, so one failed
% growth attempt at budget $M=40000$ costs about $3$ seconds per
% candidate, sequentially per core.
% NEW
... from scratch.  Measured on the original C++ sources, the rescan
costs $68\,\mu$s at $|J|=96$ and the two distance evaluations
$4.3\,\mu$s each, about $77\,\mu$s per step, so one failed growth
attempt at budget $M=40000$ costs about $3$ seconds per candidate,
sequentially per core.
```

### 7. §3 closing sentence

```latex
% OLD
% The repair of the two diseases is the energy of the next section, and
% it is what made the corrected search worth running.
% NEW
The repair of the two diseases is the energy of the next section:
grading defects replaces the flat score, and annealing whole systems
replaces sequential cornering.  It is what made the corrected search
worth running.
```

### 8. Conjecture C2 (§8) — add the k=1 evidence

```latex
% OLD
% ... the observed plateaus $84/96$, $460/500$ and $2380/2520$ are the
% general case.
% NEW
... the observed plateaus $84/96$, $460/500$ and $2380/2520$ are the
general case.  The single-chain growth runs on $\partial\Delta^3$ add
the $k=1$ case: all sixteen replicas froze at near-rectangle fronts of
$30$--$54$ facets, each within a few facets of a maximal $D(v,w)$
(Section~\ref{sec:fail}).
```

## Artifacts (Abstract_Motion_GPU/)

- `s2_stall_witness.log` + `results/s2_stall_20260922/` — the main
  faithful run (35-min box; covering froze, singleton-completed, all 43
  planner chains verified; summary.json inside).
- `probe_stall.log`, `probe_stall_snapshot_s2.json` — second run with
  exact per-job/per-pass bookkeeping (pass-success decay
  37→6→3→0→0…), frozen-front snapshot with growth order.
- `probe_micro.py` + `probe_micro_mainrun.json`, `probe_micro_mainrun_j36.json`,
  `probe_micro_probe.json` — instrumented walk replays (per-candidate
  min_d ladder, rejection histograms, baseline contrast).
- `probe_anneal.py` + `probe_anneal_mainrun.json`, `probe_anneal_mainrun_j36.json`,
  `probe_anneal_probe.json` — the k=1 annealer miniature (all chains
  verified with `absmotion.verify.check_chain`).
- `analyze_stall.py` — pass/freeze statistics from any verbose run log.
- `snapshot_from_checkpoint.py`, `snapshot_mainrun_s2.json` — frozen-front
  extraction from the main run's checkpoint.
- `~/Documents/Abstract_Motion_Final/bench_step` (recompiled) — the C++
  step-cost measurements.

## Open items (user decision)

1. Where the §3 measurements should be citable: the paper's repository
   paragraph (§6) covers the annealer repo only; the stall-verification
   artifacts live in the phase-1 GPU-port repo.  A one-line pointer in §3
   ("logs, replay scripts and raw statistics in the phase-1 repository")
   would make the new numbers auditable like the rest.
2. Whether to keep the rhetorical "forty bad links" or ground it in the
   measured range ("ten").
3. Whether to also cite the S³ corroboration (`s3.log`: same pattern at
   500 facets, pass caps truncating it) in §3 or leave S² as the sole
   exhibit.
