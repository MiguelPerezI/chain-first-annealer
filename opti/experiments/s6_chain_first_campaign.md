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

## Laboratory execution

From the repository root, first inspect the plan. This command does not import
CUDA, create the output directory, or launch a search:

```bash
python3 opti/scripts/run_s6_ladder.py \
  --outdir opti/results/s6_k3_chain_first_ladder
```

Use the laboratory Python environment containing the versions in
`requirements.txt`. Launch the full bounded ladder with:

```bash
python3 opti/scripts/run_s6_ladder.py \
  --execute \
  --python /absolute/path/to/venv/bin/python \
  --outdir opti/results/s6_k3_chain_first_ladder
```

To run only the one-epoch pilot, add `--stop-after 1`. It exits 3 at the clean
rung boundary. Continue with the same repository revision, plan, Python path,
and unmodified artifacts:

```bash
python3 opti/scripts/run_s6_ladder.py \
  --execute --resume \
  --python /absolute/path/to/venv/bin/python \
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

An incomplete rung must return exit 2, contain no certificate files, and state
`sc_strict_bound: null`. A reported full coverage is accepted only after
`independent_check.py` verifies the exported domains and all planner chains in
a separate process with assertions enabled. An audit failure records campaign
status `failed` and never publishes a bound.

The committed CPU tests exercise escalation, best-incumbent selection,
pause/resume, plan and artifact tampering, bad exit/status contracts, stale
certificate rejection, assertion enforcement, corrupt-certificate rejection,
and the rule that only an audited full cover grants a bound. No S⁶ GPU search
was run while preparing this campaign.
