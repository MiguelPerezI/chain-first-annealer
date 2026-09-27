"""Submit the CP-SAT ladder from the lab notebook with the KFP SDK.

kfp.Client() with no arguments works inside the notebook: it takes the
endpoint and the ServiceAccount token that Kubeflow mounts at
/var/run/secrets/kubeflow/pipelines/token (PodDefault access-ml-pipeline).

Examples:
    python opti/cluster/kfp/lanzar.py                          # RP²: control k=6 L=8, ladder k=5 L=4..16
    python opti/cluster/kfp/lanzar.py --ladder 24,32,40 \
        --yaml opti/cluster/kfp/compiled/cp_sat_ladder_giants.yaml --workers 4
    python opti/cluster/kfp/lanzar.py --complex klein --control-k 10 --k 9 --ladder 4,5,6,8
    python opti/cluster/kfp/lanzar.py --wait
"""
import argparse
import pathlib

SA_NAMESPACE = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
DEFAULT_YAML = pathlib.Path(__file__).resolve().parent / "compiled" / "cp_sat_ladder.yaml"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yaml", default=str(DEFAULT_YAML))
    ap.add_argument("--complex", default="rp2")
    ap.add_argument("--control-k", type=int, default=6)
    ap.add_argument("--control-L", type=int, default=8)
    ap.add_argument("--control-timeout-s", type=int, default=21600)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--ladder", default="4,5,6,8,12,16", help="comma-separated L values")
    ap.add_argument("--solver-timeout-s", type=int, default=86400)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-root", default="/home/jovyan/cp_sat_results")
    ap.add_argument("--experiment", default="cp-sat-ladders")
    ap.add_argument("--namespace", help="default: the notebook's own namespace")
    ap.add_argument("--wait", action="store_true")
    a = ap.parse_args()

    import kfp
    ns = a.namespace or open(SA_NAMESPACE).read().strip()
    args = {
        "complex": a.complex, "control_k": a.control_k, "control_L": a.control_L,
        "control_timeout_s": a.control_timeout_s, "k": a.k,
        "ladder": [int(x) for x in a.ladder.split(",")],
        "solver_timeout_s": a.solver_timeout_s, "workers": a.workers,
        "out_root": a.out_root,
    }
    client = kfp.Client()
    run = client.create_run_from_pipeline_package(
        a.yaml, arguments=args, experiment_name=a.experiment, namespace=ns,
        run_name=f"{a.complex}-k{a.k}-L{a.ladder.replace(',', '-')}",
        enable_caching=False)
    print(f"run {run.run_id} in {ns}: {args}")
    if a.wait:
        res = client.wait_for_run_completion(run.run_id, timeout=a.solver_timeout_s * 10)
        print("state:", getattr(res, "state", res))


if __name__ == "__main__":
    main()
