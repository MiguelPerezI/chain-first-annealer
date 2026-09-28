"""KFP v2 pipeline for the CP-SAT ladders of the cluster runbook.

    control (known cover must be SAT and audited)
        └─ dsl.If(certificate == "audited")
             └─ dsl.ParallelFor(L in ladder, parallelism=N)  ->  one rung per L

Default parameters are runbook §4-5 (RP², control k=6 L=8, ladder k=5).
The same pipeline runs Klein k=9 (runbook §6) by changing the parameters.

Follows the Ibero lab pattern of gemelo-digital-cdmx/pipelines/gemelo_tgcn_lab.py:
container components with the notebook image, repo and venv on the workspace
PVC (no image to publish, no token inside the pipeline), non-root security
context, per-task CPU/RAM/timeout, no caching.

Settings fixed in the IR at compile time (KFP cannot template them):
    SAT_PVC          workspace PVC of the notebook (required)
    SAT_IMAGE        default cr.saptiva.com/apps/jupyter-saptiva-ds-cpu:v1.11.0-saptiva2
    SAT_CPU          default 8     (CPU request = limit per rung)
    SAT_MEM          default 28Gi  (RAM grows with CP-SAT workers)
    SAT_PARALLELISM  default 1     (rungs at once; namespace quota ~17 CPU / 66Gi)
    SAT_TIMEOUT_S    default 90000 (pod timeout; solver --timeout is a parameter)
    SAT_VARIANT      name suffix of the compiled YAML (e.g. "giants")
    SAT_ENCODING     default factored (y = original scs_sat model)

Usage:
    SAT_PVC=<workspace-pvc> python opti/cluster/kfp/pipeline.py
    SAT_PVC=<workspace-pvc> SAT_MEM=60Gi SAT_VARIANT=giants python opti/cluster/kfp/pipeline.py
"""
import os
import pathlib

from kfp import compiler, dsl, kubernetes

PVC = os.environ.get("SAT_PVC", "")
IMAGE = os.environ.get("SAT_IMAGE", "cr.saptiva.com/apps/jupyter-saptiva-ds-cpu:v1.11.0-saptiva2")
CPU = os.environ.get("SAT_CPU", "8")
MEM = os.environ.get("SAT_MEM", "28Gi")
PARALLELISM = int(os.environ.get("SAT_PARALLELISM", "1"))
TIMEOUT_S = int(os.environ.get("SAT_TIMEOUT_S", "90000"))
VARIANT = os.environ.get("SAT_VARIANT", "")
# middle-link encoding passed to scs_sat.py through sat_rung.sh; "factored"
# used about half the RAM of "y" on RP2 k=5 L=4 (rp2_klein_cp_sat_results.md)
ENCODING = os.environ.get("SAT_ENCODING", "factored")

HOME = "/home/jovyan"
REPO = f"{HOME}/chain-first-annealer"
UID_JOVYAN = 1000


@dsl.container_component
def cp_sat_rung(
    complex: str,
    k: int,
    L: int,
    solver_timeout_s: int,
    workers: int,
    out_root: str,
    warm_file: str,
    free: int,
    status: dsl.OutputPath(str),
    certificate: dsl.OutputPath(str),
    rung: dsl.Output[dsl.Artifact],
):
    """One rung: scs_sat.py, time and RSS, witness conversion, independent audit.
    warm_file/free select a hybrid case (e.g. orbit case A1); "" / 0 is cold."""
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["bash", f"{REPO}/opti/cluster/kfp/rung_entry.sh"],
        args=[complex, k, L, solver_timeout_s, workers, out_root,
              status, certificate, rung.path, warm_file, free],
    )


def _harden(task, step: str):
    kubernetes.mount_pvc(task, pvc_name=PVC, mount_path=HOME)
    kubernetes.set_security_context(task, run_as_user=UID_JOVYAN, run_as_non_root=True)
    kubernetes.set_timeout(task, seconds=TIMEOUT_S)
    kubernetes.add_pod_label(task, "app", "cp-sat")
    kubernetes.add_pod_label(task, "paso", step)
    task.set_env_variable("HOME", HOME)
    task.set_env_variable("PYTHONUNBUFFERED", "1")
    task.set_env_variable("SCS_ENCODING", ENCODING)
    task.set_cpu_request(CPU).set_cpu_limit(CPU)
    task.set_memory_request(MEM).set_memory_limit(MEM)
    # every rung is a fresh solver run; a cached result would hide a rerun
    task.set_caching_options(False)
    return task


@dsl.pipeline(
    name="cp-sat-ladder" + (f"-{VARIANT}" if VARIANT else ""),
    description="CP-SAT cover decision: audited control, then an L-ladder "
                "(cluster_runbook_rp2_k5_klein_k9.md)",
)
def cp_sat_ladder(
    complex: str = "rp2",
    control_k: int = 6,
    control_L: int = 8,
    control_timeout_s: int = 21600,
    k: int = 5,
    ladder: list = [4, 5, 6, 8, 12, 16],
    solver_timeout_s: int = 86400,
    workers: int = 6,
    out_root: str = f"{HOME}/cp_sat_results",
    warm_file: str = "",
    free: int = 0,
):
    # the control is always cold: it validates the encoding, not the case
    control = cp_sat_rung(complex=complex, k=control_k, L=control_L,
                          solver_timeout_s=control_timeout_s, workers=workers,
                          out_root=out_root, warm_file="", free=0)
    control.set_display_name("control (known cover)")
    _harden(control, "control")

    # runbook §4: no UNSAT is reportable unless the control reproduces SAT
    with dsl.If(control.outputs["certificate"] == "audited", name="control-audited"):
        with dsl.ParallelFor(items=ladder, parallelism=PARALLELISM, name="ladder") as L:
            rung = cp_sat_rung(complex=complex, k=k, L=L,
                               solver_timeout_s=solver_timeout_s, workers=workers,
                               out_root=out_root, warm_file=warm_file, free=free)
            rung.set_display_name("rung")
            _harden(rung, "ladder")


if __name__ == "__main__":
    if not PVC:
        raise SystemExit("SAT_PVC is required: the notebook's workspace PVC name")
    name = "cp_sat_ladder" + (f"_{VARIANT}" if VARIANT else "") + ".yaml"
    out = pathlib.Path(__file__).resolve().parent / "compiled" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    compiler.Compiler().compile(cp_sat_ladder, str(out))
    print(f"compiled {out}")
