# CP-SAT on the Ibero Kubernetes cluster

Jobs for the RP² k=5 decision in
`opti/experiments/cluster_runbook_rp2_k5_klein_k9.md`. Each Indexed-Job pod runs
one rung with `scripts/sat_rung.sh` (solve, time and RSS, witness conversion,
independent audit, one log row) and writes to a PVC.

| file | role |
|---|---|
| `run_rung.sh` | pod entrypoint: installs git/time if missing, clones the repo at `REPO_REF`, builds the ortools venv, runs the rung chosen by `JOB_COMPLETION_INDEX` |
| `rp2_k5_jobs.yaml` | three Jobs: `rp2-controls` (k=6 L=8), `rp2-k5-ladder` (L=4..16, 14 CPU / 32 GiB), `rp2-k5-giants` (L=24..40, 8 CPU / 64 GiB) |

## Before applying

Fill the placeholders in `rp2_k5_jobs.yaml`:

- `[NAMESPACE]`: the namespace assigned by the lab.
- `[PVC_NAME]`: a PersistentVolumeClaim for results (a few MB per rung plus
  solver logs; logs grow with `log_search_progress`).
- `[GIT_TOKEN_SECRET]`: a Secret holding a read-only GitHub token for
  `MiguelPerezI/chain-first-annealer` under key `token`. The repository is
  private. Create it without writing the token to any file:

```bash
kubectl -n [NAMESPACE] create secret generic [GIT_TOKEN_SECRET] --from-literal=token="$(read -rs t; echo "$t")"
```

The image is `python:3.14-slim`, which runs as root and installs `git` and
`time` with apt at start, and the ortools venv with pip. Both need outbound
access to Debian mirrors, github.com and pypi.org. If the cluster forbids root
or egress, build an image with `git`, `time` and `opti/requirements-sat.txt`
preinstalled and replace `image:`.

The CPU request (14) matches the current per-workspace ceiling. Raise it if
the lab grants a larger batch quota; CP-SAT does not scale linearly past
16-32 workers, and RAM grows with workers.

## Run

```bash
kubectl -n [NAMESPACE] create configmap sat-rung --from-file=run_rung.sh=opti/cluster/run_rung.sh
# 1. controls (runbook §4): must end SAT before trusting any UNSAT
kubectl apply -f <(yq 'select(.metadata.name == "rp2-controls")' opti/cluster/rp2_k5_jobs.yaml)
kubectl -n [NAMESPACE] logs -f -l job-name=rp2-controls
# 2. ladder, 3. giants: same pattern with rp2-k5-ladder, then rp2-k5-giants
```

Without `yq`, split the YAML at the `---` lines and apply each file.

Progress: `kubectl -n [NAMESPACE] get pods -l job-name=rp2-k5-ladder`. Each
pod prints its log row on exit. Results land in
`/results/cp_sat_rp2_k5/rp2_k5_L{L}/` with `row.md`, `revision.txt`,
`nproc.txt` and `run/` (solver log, GNU time, witness and audited certificate
when SAT).

## Reading the outcome

Follow runbook §10. An UNKNOWN row proves nothing: relaunch that rung with a
longer `TIMEOUT`. An INFEASIBLE row at L rules out 5-domain covers with chains
of at most L links. Append every row to
`opti/experiments/rp2_klein_cp_sat_results.md`.
