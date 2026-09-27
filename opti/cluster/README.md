# CP-SAT on the Ibero AI Lab (Kubeflow Pipelines)

Runs the decision ladders of `opti/experiments/cluster_runbook_rp2_k5_klein_k9.md`
as a KFP v2 pipeline on the lab (Kubeflow 26.03, KFP backend 2.16, SDK 2.17 in
the notebook image).

```
control: known cover (default RP² k=6 L=8) must come back SAT and audited
  └─ if certificate == "audited"
       └─ ParallelFor L in ladder (parallelism N): one rung per L
```

Each rung runs `opti/scripts/sat_rung.sh` (solve, peak RSS, witness conversion,
independent audit, one log row) and returns:

- `status` and `certificate` as output parameters (visible in the KFP UI);
- a `rung` artifact (tar.gz of the rung directory) in the pipeline artifact store;
- the same directory on the workspace PVC under `~/cp_sat_results/<complex>_k<k>_L<L>/`.

| file | role |
|---|---|
| `kfp/pipeline.py` | pipeline definition; compiles to `kfp/compiled/cp_sat_ladder*.yaml` |
| `kfp/rung_entry.sh` | container-component entrypoint for one rung |
| `kfp/lanzar.py` | submits a run from the notebook with `kfp.Client()` |
| `kfp/setup_workspace.sh` | one-time setup of repo and CP-SAT venv on the workspace PVC |

## Design, following the lab's working pattern

Mirrors `gemelo-digital-cdmx/pipelines/gemelo_tgcn_lab.py`, which ran end to
end on the lab:

- **Container components** with the notebook image
  (`cr.saptiva.com/apps/jupyter-saptiva-ds-cpu:v1.11.0-saptiva2`); nothing is
  installed at run time.
- **Repo and venv on the workspace PVC**, mounted at `/home/jovyan` in every
  pod. No project image to publish and no GitHub token inside the pipeline.
- **Non-root** (`runAsUser: 1000`), per-task CPU/RAM requests equal to limits,
  pod timeout, pod labels `app=cp-sat, paso=control|ladder`, caching off.
- **No GNU time in the image**: `sat_rung.sh` falls back to
  `scripts/rss_run.py` for peak RSS.

## One-time setup (notebook terminal)

```bash
git clone https://github.com/MiguelPerezI/chain-first-annealer.git ~/chain-first-annealer
bash ~/chain-first-annealer/opti/cluster/kfp/setup_workspace.sh
```

It creates `~/venv-sat` with `opti/requirements-sat.txt`, runs the rung smoke
test and installs `kfp-kubernetes` for compiling.

## Compile and launch (notebook terminal)

```bash
cd ~/chain-first-annealer
PVC=<workspace PVC of this notebook>   # Notebooks UI > this notebook > Volumes (e.g. <notebook>-workspace)
SAT_PVC=$PVC python opti/cluster/kfp/pipeline.py
python opti/cluster/kfp/lanzar.py                 # RP² control + k=5 ladder L=4..16
```

Giants (runbook §5, L = 24 32 40) need more RAM per pod than the default:

```bash
SAT_PVC=$PVC SAT_MEM=60Gi SAT_VARIANT=giants python opti/cluster/kfp/pipeline.py
python opti/cluster/kfp/lanzar.py --yaml opti/cluster/kfp/compiled/cp_sat_ladder_giants.yaml \
    --ladder 24,32,40 --workers 4 --solver-timeout-s 259200
```

Klein k=9 (runbook §6), same pipeline:

```bash
python opti/cluster/kfp/lanzar.py --complex klein --control-k 10 --k 9 --ladder 4,5,6,8
```

## Sizing against the namespace quota

The lab namespace quota (`kf-resource-quota`) caps around 17 CPU / 66 GiB in
total, shared with the running notebook and with each pod's Istio sidecar
(up to 200m / 256Mi). Defaults: 8 CPU / 28 GiB per rung, 6 CP-SAT workers,
one rung at a time. RAM grows with workers: RP² k=6 L=8 (2.2 M constraints)
passed 6 GB with 8 workers on a laptop. Raise `SAT_PARALLELISM` only if
`parallelism x (SAT_CPU, SAT_MEM)` plus the notebook fits the quota; stop the
notebook's kernels while a ladder runs.

## Reading the outcome

Runbook §10. An UNKNOWN rung proves nothing (rerun with a longer
`--solver-timeout-s`). INFEASIBLE at L rules out covers with chains of at most
L links. A skipped ladder means the control was not audited: fix that first.
Append each row (the `row.md` in the rung directory) to
`opti/experiments/rp2_klein_cp_sat_results.md`.
