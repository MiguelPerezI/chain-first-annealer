# S⁶ chain-first campaign for the laboratory host

This campaign searches for a three-domain cover of
`S6 = boundary(Delta^7)`, whose product triangulation has 59,136 maximal
facets. It uses only the chain-first annealer in `gpu_sc.py`. It does not use
the closed-form star cover or any other certificate as an initialization.

The existing scaling probe took 81.8 seconds for 6,400 moves on its host.
The schedule therefore starts with one unguided epoch and keeps a fixed,
auditable move budget. If that epoch is incomplete, four guided rungs warm
start from the best previous chain system. The rungs increase `guide` and
`guide_max` together, because increasing the increment after the weights have
hit their cap would have no effect. They also increase `cmax` and the move
budget gradually.

## Schedule

The committed schedule is [s6_ladder.json](s6_ladder.json). With population
`S=64`, its upper limit is 17 epochs and 710,400 moves:

| rung | mode | cmax | iters/epoch | epochs | guide / cap | seed |
|---:|---|---:|---:|---:|---:|---:|
| 0 | cold, unguided | 16 | 100 | 1 | 0 / 1 | 601 |
| 1 | guided warm restart | 16 | 250 | 4 | 1 / 4 | 602 |
| 2 | guided warm restart | 24 | 500 | 4 | 2 / 8 | 603 |
| 3 | guided warm restart | 32 | 1,000 | 4 | 4 / 16 | 604 |
| 4 | guided warm restart | 40 | 1,000 | 4 | 8 / 32 | 605 |

A warm restart loads the three chains with greatest domains from the best
previous `summary.json`, reduces and pads them, and places them in half of the
new population. It does not restore the previous population, facet weights,
RNG state, or an interrupted epoch. The other half starts from random sweep
chains, so every rung is still a chain-first annealer run.

## First laboratory result and continuation

The first complete ladder ran on an NVIDIA RTX PRO 6000 Blackwell Server
Edition on 23 September 2026. Its unweighted coverage progressed as follows:

| rung | coverage | uncovered | elapsed |
|---:|---:|---:|---:|
| 0 | 24,711 / 59,136 | 34,425 | 61 s |
| 1 | 50,019 / 59,136 | 9,117 | 510 s |
| 2 | 56,775 / 59,136 | 2,361 | 1,084 s |
| 3 | 58,484 / 59,136 | 652 | 2,228 s |
| 4 | 59,072 / 59,136 | 64 | 2,289 s |

The campaign is incomplete and records no bound. Its hashed incumbent is
`results/s6_k3_chain_first_ladder_run4/rung04/summary.json`.
The 64 uncovered facets lie in 13 of the 64 ordered pairs of base facets,
with at most 12 missing staircase facets in any one pair. Across the full
product, 13,963 facets are covered by exactly one chain, 36,429 by two, and
8,680 by all three. Thus the residual is localized amid substantial overlap;
this supports concentrating the next search but does not prove that the same
three chains can be completed.

The continuation schedule in
[s6_ladder_continuation.json](s6_ladder_continuation.json) starts from that
incumbent. It uses `S=256`, four times the previous population, while its first
rung keeps the same total 256,000 moves as rung 4 by reducing iterations per
thread. Later rungs increase `cmax`, guide, cap and move budget. The complete
continuation is bounded at 16 epochs and 2,560,000 moves:

| rung | cmax | S | iters/epoch | epochs | guide / cap | seed |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 40 | 256 | 250 | 4 | 12 / 48 | 606 |
| 1 | 48 | 256 | 500 | 4 | 16 / 64 | 607 |
| 2 | 56 | 256 | 750 | 4 | 24 / 96 | 608 |
| 3 | 64 | 256 | 1,000 | 4 | 32 / 128 | 609 |

Run it from the repository root with a fresh output directory:

```bash
.venv/bin/python opti/scripts/run_s6_ladder.py \
  --execute \
  --python "$PWD/.venv/bin/python" \
  --plan opti/experiments/s6_ladder_continuation.json \
  --initial-summary opti/results/s6_k3_chain_first_ladder_run4/rung04/summary.json \
  --outdir opti/results/s6_k3_chain_first_continuation_01
```

The runner verifies that the input is the hashed best result of an incomplete
chain-first campaign, then copies both its summary and campaign manifest into
the new directory. Subsequent warm starts are selected from the copied input
and the best new rung. Resume at a clean rung boundary with the same `--plan`
and `--outdir`, `--resume`, and no `--initial-summary`.

### Continuation result

The continuation completed on the same laboratory host:

| rung | coverage | uncovered | elapsed | outcome |
|---:|---:|---:|---:|---|
| 0 | 59,124 / 59,136 | 12 | 629 s | incomplete |
| 1 | 59,133 / 59,136 | 3 | 1,238 s | incomplete |
| 2, epoch 1 | 59,135 / 59,136 | 1 | 464 s | continued |
| 2, epoch 2 | 59,136 / 59,136 | 0 | 960 s total for rung | verified cover |

The final certificate is
`results/s6_k3_chain_first_continuation_01/rung02`. Its disjoint domain sizes
are 44,909, 11,451 and 2,776 facets, and the reduced planner chains have 10, 9
and 7 links. The independent checker rebuilt all 59,136 staircase facets,
verified their partition, and checked every map and every contiguity link on
each domain. Its output is retained in `rung02/audit.log`; the campaign
manifest binds the checker and all certificate artifacts by SHA-256.

The initial and continuation sequences used 1,862,400 moves and approximately
2.5 hours of accumulated rung wall time. This is a sequence of warm restarts,
not a single stationary annealing run. It establishes the explicit upper bound
`SC_strict(boundary(Delta^7)) <= 2`; it does not establish optimality by itself.

## Laboratory execution

From the repository root, first inspect the plan. This command does not import
CUDA, create the output directory, or launch a search:

```bash
python3 opti/scripts/run_s6_ladder.py \
  --outdir opti/results/s6_k3_chain_first_ladder
```

Create a CUDA 12 environment with the committed compatible constraints. The
NumPy `<2.5` pin is required because `numba-cuda 0.30.4` still refers to the
`np.row_stack` alias removed by NumPy 2.5:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-cuda.txt
```

Launch the full bounded ladder with:

```bash
.venv/bin/python opti/scripts/run_s6_ladder.py \
  --execute \
  --python "$PWD/.venv/bin/python" \
  --outdir opti/results/s6_k3_chain_first_ladder
```

To run only the one-epoch pilot, add `--stop-after 1`. It exits 3 at the clean
rung boundary. Continue with the same repository revision, plan, Python path,
and unmodified artifacts:

```bash
.venv/bin/python opti/scripts/run_s6_ladder.py \
  --execute --resume \
  --python "$PWD/.venv/bin/python" \
  --outdir opti/results/s6_k3_chain_first_ladder
```

The process exit codes are:

- `0`: full coverage and the independent audit passed;
- `2`: all scheduled rungs completed without full coverage;
- `3`: paused at a requested rung boundary;
- `1`: invalid inputs, inconsistent output, failed audit, or another failure;
- `130`: interrupted by the operator.

Do not reuse an output directory after an interrupted rung. The manifest keeps
that partial directory for diagnosis and refuses to treat it as resumable.
Start a new campaign directory instead.

## Evidence and interpretation

`campaign.json` records the exact plan, commands, host sessions, GPU inventory,
Python packages, Git revision, source hashes, input hashes, elapsed wall time,
coverage and artifact hashes for every completed rung. It selects subsequent
warm starts by maximum unweighted coverage, not by the most recent rung.
Before creating the output directory, execution compiles and runs a one-element
CUDA kernel. Missing CUDA support and incompatible compiler dependencies fail
during this preflight rather than consuming a rung directory.

An incomplete rung must return exit 2, contain no certificate files, and state
`sc_strict_bound: null`. A reported full coverage is accepted only after
`independent_check.py` verifies the exported domains and all planner chains in
a separate process with assertions enabled. An audit failure records campaign
status `failed` and never publishes a bound.

The committed CPU tests exercise escalation, best-incumbent selection,
pause/resume, plan and artifact tampering, bad exit/status contracts, stale
certificate rejection, assertion enforcement, corrupt-certificate rejection,
and the rule that only an audited full cover grants a bound. These tests do not
launch the S⁶ search; the GPU artifacts above were produced on the laboratory
host and independently re-audited after being pulled into this repository.
