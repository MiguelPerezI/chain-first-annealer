#!/usr/bin/env python3
"""Bounded S6 campaign: chain-first annealing only; no search unless --execute.

The incumbent seeds half of each new population. These are warm restarts,
not restoration of a population, RNG, weights, or an in-progress epoch.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

OPTI = Path(__file__).resolve().parents[1]
ROOT = OPTI.parent
SRC = OPTI / "src"
sys.path.insert(0, str(SRC))
from validation import validate_parameters

NF = 59136  # 8**2 * binomial(12, 6), for S6 = boundary of Delta^7
COMMON = {"S", "beta", "gamma", "ptrack", "pcopy", "T0", "T1",
          "tournament_every", "tournament_frac"}
RUNG = {"cmax", "iters", "epochs", "guide", "guide_max", "seed"}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    if set(plan) != {"schema_version", "complex", "k", "start", "common", "rungs"}:
        raise ValueError("unexpected plan fields")
    if plan["schema_version"] != 2 or plan["complex"] != "s6" or type(plan["k"]) is not int or plan["k"] != 3:
        raise ValueError("this campaign requires s6 with k=3")
    if plan["start"] not in {"cold", "guided_continuation"}:
        raise ValueError("start must be cold or guided_continuation")
    if set(plan["common"]) != COMMON or not plan["rungs"]:
        raise ValueError("missing common parameters or rungs")
    previous = 2
    for i, rung in enumerate(plan["rungs"]):
        if set(rung) != RUNG:
            raise ValueError("unexpected rung fields")
        validate_parameters(k=3, **plan["common"], **rung)
        if plan["common"]["S"] < 2 or rung["cmax"] < previous:
            raise ValueError("warm starts need S >= 2 and nondecreasing cmax")
        if i == 0 and plan["start"] == "cold" and (rung["epochs"] != 1 or rung["guide"] != 0):
            raise ValueError("the first rung must be one unguided epoch")
        if (i or plan["start"] == "guided_continuation") and (
                rung["epochs"] < 2 or rung["guide"] <= 0 or rung["guide_max"] <= 1):
            raise ValueError("guided rungs need >=2 epochs, guide>0 and guide_max>1")
        previous = rung["cmax"]
    return plan


def parameters(plan, i):
    return dict(complex="s6", k=3, **plan["common"], **plan["rungs"][i])


def validate_chains(chains, cmax):
    if len(chains) != 3:
        raise ValueError("expected three annealer chains")
    for chain in chains:
        if len(chain) != cmax + 1:
            raise ValueError("unexpected chain length")
        if any(len(m) != 64 or any(type(v) is not int or not 0 <= v < 8 for v in m) for m in chain):
            raise ValueError("invalid map tables")
        if chain[0] != [v // 8 for v in range(64)] or chain[-1] != [v % 8 for v in range(64)]:
            raise ValueError("invalid chain endpoints")


def validate_partial_summary(path):
    path = Path(path).resolve()
    summary = json.loads(path.read_text())
    params = summary.get("params", {})
    if (summary.get("complex", {}).get("num_facets") != NF
            or summary.get("complex", {}).get("nv") != 8
            or params.get("complex") != "s6" or params.get("k") != 3):
        raise ValueError("initial summary is not an S6 k=3 annealer result")
    cov = summary.get("coverage")
    if type(cov) is not int or not 0 <= cov < NF:
        raise ValueError("initial summary must have incomplete unweighted coverage")
    if (summary.get("status") != "incomplete" or summary.get("sc_strict_bound") is not None
            or summary.get("all_domains_verified") is not False):
        raise ValueError("initial summary must be explicitly incomplete with no bound")
    cmax = params.get("cmax")
    if type(cmax) is not int:
        raise ValueError("initial summary has no valid cmax")
    validate_chains(summary.get("best_chains"), cmax)
    if any((path.parent / f).exists() for f in ("cover.txt", "planners.txt")):
        raise ValueError("incomplete initial result has stale certificate files")
    return summary


def validate_initial_summary(path):
    path = Path(path).resolve()
    summary = validate_partial_summary(path)
    campaign_path = path.parent.parent / "campaign.json"
    campaign = json.loads(campaign_path.read_text())
    relative = str(path.relative_to(campaign_path.parent))
    records = [r for r in campaign.get("rungs", []) if r.get("directory") == path.parent.name]
    if (campaign.get("status") != "incomplete" or campaign.get("sc_strict_bound") is not None
            or campaign.get("best_summary") != relative
            or campaign.get("best_coverage") != summary["coverage"]
            or campaign.get("method") != "chain-first annealer only; warm restarts from this campaign"
            or len(records) != 1
            or records[0].get("artifact_sha256", {}).get("summary.json") != digest(path)):
        raise ValueError("initial summary is not the hashed incumbent of an incomplete chain-first campaign")
    return summary, campaign_path


def command(python, params, out, init):
    args = [str(python), "-u", str(SRC / "gpu_sc.py")]
    # gpu_sc has underscore spellings for tournament options only through
    # argparse dest names; all those CLI option names use hyphens.
    for key, value in params.items():
        args.extend(["--" + key.replace("_", "-"), str(value)])
    args += ["--require-cover", "--outdir", str(out)]
    if init is not None:
        args += ["--init", str(init)]
    return args


def checked_summary(out, params, rc):
    """Check execution contract; only the independent auditor grants success."""
    if rc not in (0, 2):
        raise ValueError(f"annealer process failed with exit {rc}")
    summary = json.loads((out / "summary.json").read_text())
    if summary["complex"]["num_facets"] != NF or summary["complex"]["nv"] != 8:
        raise ValueError("unexpected complex")
    if any(summary["params"].get(k) != v for k, v in params.items()):
        raise ValueError("summary parameters differ from scheduled rung")
    cov = summary["coverage"]
    if type(cov) is not int or not 0 <= cov <= NF:
        raise ValueError("invalid unweighted coverage")
    validate_chains(summary["best_chains"], params["cmax"])
    if cov < NF:
        if rc != 2 or summary.get("status") != "incomplete" or summary.get("sc_strict_bound") is not None or summary.get("all_domains_verified") is not False:
            raise ValueError("partial run must be explicitly incomplete with no bound")
        if any((out / f).exists() for f in ("cover.txt", "planners.txt")):
            raise ValueError("partial run has stale certificate files")
    else:
        nd = summary.get("n_domains")
        if rc != 0 or type(nd) is not int or not 1 <= nd <= 3 or summary.get("all_domains_verified") is not True or summary.get("sc_strict_bound") != nd - 1:
            raise ValueError("full coverage lacks consistent certificate metadata")
        if not all((out / f).is_file() for f in ("cover.txt", "planners.txt")):
            raise ValueError("full coverage lacks certificate files")
    return summary


def provenance():
    # Hash all annealer dependencies and orchestration/auditor code. Unrelated
    # manuscript edits may change HEAD between boundary resumes.
    paths = list(SRC.glob("*.py")) + [Path(__file__).resolve()]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def git_info():
    def run(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    return dict(commit=run("rev-parse", "HEAD"), dirty=bool(run("status", "--porcelain")))


def runtime_info(python):
    code = """import json, platform, importlib.metadata as m
import numpy as np
from numba import cuda

versions = {}
for p in ['numpy', 'numba', 'llvmlite', 'numba-cuda']:
    try: versions[p] = m.version(p)
    except m.PackageNotFoundError: versions[p] = None

if not cuda.is_available():
    raise RuntimeError('CUDA is not available to the selected Python environment')

@cuda.jit
def preflight_kernel(values):
    i = cuda.grid(1)
    if i < values.size:
        values[i] += 1

values = cuda.to_device(np.zeros(1, np.int32))
preflight_kernel[1, 1](values)
cuda.synchronize()
if values.copy_to_host().tolist() != [1]:
    raise RuntimeError('CUDA preflight kernel returned an invalid result')

device = cuda.get_current_device()
print(json.dumps(dict(python=platform.python_version(), packages=versions,
                      cuda_preflight='passed', compute_capability=list(device.compute_capability))))
"""
    info = json.loads(subprocess.check_output([str(python), "-c", code], text=True, timeout=30))
    info.update(started_at=now(), host=platform.node(), platform=platform.platform(),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total",
                                 "--format=csv"], capture_output=True, text=True, timeout=10)
        info["nvidia_smi"] = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        info["nvidia_smi"] = dict(error=str(exc))
    return info


def run_logged(args, log):
    env = dict(os.environ)
    env.pop("PYTHONOPTIMIZE", None)  # independent_check uses assertions
    with log.open("w") as fh:
        return subprocess.run(args, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode


def campaign(plan, out, python, resume=False, stop_after=None, runner=run_logged,
             initial_summary=None):
    state_path = out / "campaign.json"
    sources = provenance()
    incumbent = None
    best = -1
    if resume:
        state = json.loads(state_path.read_text())
        if state.get("schema_version") != 2:
            raise ValueError("unsupported campaign schema for resume")
        if state["plan"] != plan or state["source_sha256"] != sources or state["python"] != str(python):
            raise ValueError("resume requires the same plan, source hashes and Python path")
        if state["status"] not in {"paused", "incomplete", "verified"}:
            raise ValueError("resume only at a completed rung boundary; preserve interrupted artifacts")
        initial = state.get("initial")
        if initial:
            incumbent = out / initial["summary_copy"]
            campaign_copy = out / initial["campaign_copy"]
            if (digest(incumbent) != initial["summary_sha256"]
                    or digest(campaign_copy) != initial["campaign_sha256"]):
                raise ValueError("modified initial continuation artifacts")
            seed = validate_partial_summary(incumbent)
            best = seed["coverage"]
        for i, record in enumerate(state["rungs"]):
            rd = out / record["directory"]
            for name, sha in record["artifact_sha256"].items():
                if digest(rd / name) != sha:
                    raise ValueError(f"modified rung artifact: {rd / name}")
            summary = checked_summary(rd, parameters(plan, i), record["returncode"])
            if summary["coverage"] > best:
                best, incumbent = summary["coverage"], rd / "summary.json"
        if state["status"] == "verified":
            return 0
    else:
        if plan["start"] == "guided_continuation":
            if initial_summary is None:
                raise ValueError("guided continuation requires --initial-summary")
            seed, source_campaign = validate_initial_summary(initial_summary)
            if plan["rungs"][0]["cmax"] < seed["params"]["cmax"]:
                raise ValueError("first continuation cmax cannot be shorter than the incumbent")
        elif initial_summary is not None:
            raise ValueError("a cold campaign cannot use --initial-summary")
        environment = runtime_info(python)
        repository = git_info()
        out.mkdir(parents=True, exist_ok=False)
        initial = None
        if initial_summary is not None:
            summary_copy = out / "initial_summary.json"
            campaign_copy = out / "initial_campaign.json"
            shutil.copy2(Path(initial_summary).resolve(), summary_copy)
            shutil.copy2(source_campaign, campaign_copy)
            incumbent, best = summary_copy, seed["coverage"]
            initial = dict(source_path=str(Path(initial_summary).resolve()),
                           source_campaign=str(source_campaign),
                           summary_copy=summary_copy.name,
                           campaign_copy=campaign_copy.name,
                           summary_sha256=digest(summary_copy),
                           campaign_sha256=digest(campaign_copy),
                           coverage=best)
        state = dict(schema_version=2, created_at=now(), status="running", plan=plan,
                     python=str(python), host=platform.node(), platform=platform.platform(),
                     git=repository, source_sha256=sources, rungs=[],
                     best_coverage=None, sc_strict_bound=None,
                     sessions=[environment], initial=initial,
                     method="chain-first annealer only; warm restarts from this campaign")
    if resume:
        state["sessions"].append(runtime_info(python))
    state["status"] = "running"
    save(state_path, state)
    try:
        for i in range(len(state["rungs"]), len(plan["rungs"])):
            if stop_after is not None and i >= stop_after:
                state["status"] = "paused"
                save(state_path, state)
                return 3
            if provenance() != sources:
                raise ValueError("source code changed during campaign")
            rd = out / f"rung{i:02d}"
            rd.mkdir()  # never reuse outputs of an interrupted process
            params = parameters(plan, i)
            args = command(python, params, rd, incumbent)
            active = dict(directory=rd.name, started_at=now(), command=args,
                          session=len(state["sessions"]) - 1,
                          init=str(incumbent.relative_to(out)) if incumbent else None,
                          init_sha256=digest(incumbent) if incumbent else None)
            state["active_rung"] = active
            save(state_path, state)
            print(f"Rung {i}: best={best}/{NF}; log={rd / 'run.log'}", flush=True)
            start = time.monotonic()
            rc = runner(args, rd / "run.log")
            active.update(returncode=rc, elapsed_s=time.monotonic() - start)
            if provenance() != sources:
                raise ValueError("source code changed while running the rung")
            summary = checked_summary(rd, params, rc)
            files = ["summary.json", "run.log"]
            cov = summary["coverage"]
            if cov == NF:
                audit_cmd = [str(python), "-B", str(SRC / "independent_check.py"), str(rd)]
                audit_rc = runner(audit_cmd, rd / "audit.log")
                active["audit"] = dict(command=audit_cmd, returncode=audit_rc,
                                       checker_sha256=digest(SRC / "independent_check.py"))
                if audit_rc != 0:
                    raise ValueError(f"independent audit failed with exit {audit_rc}")
                if provenance() != sources:
                    raise ValueError("source code changed while auditing the certificate")
                files += ["cover.txt", "planners.txt", "audit.log"]
            if cov > best:
                best, incumbent = cov, rd / "summary.json"
            active.update(finished_at=now(), coverage=cov,
                          artifact_sha256={f: digest(rd / f) for f in files})
            state["rungs"].append(active)
            state.pop("active_rung")
            state.update(best_coverage=best, best_summary=str(incumbent.relative_to(out)))
            if cov == NF:
                state.update(status="verified", sc_strict_bound=summary["sc_strict_bound"])
                save(state_path, state)
                return 0
            save(state_path, state)
        state["status"] = "incomplete"
        save(state_path, state)
        return 2
    except (Exception, KeyboardInterrupt) as exc:
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                     error=f"{type(exc).__name__}: {exc}", sc_strict_bound=None)
        save(state_path, state)
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, default=OPTI / "experiments/s6_ladder.json")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--python", type=Path, default=Path(sys.executable))
    ap.add_argument("--execute", action="store_true", help="launch on the laboratory GPU host")
    ap.add_argument("--resume", action="store_true", help="resume completed rung boundaries only")
    ap.add_argument("--initial-summary", type=Path,
                    help="hashed incumbent summary from an incomplete chain-first campaign")
    ap.add_argument("--stop-after", type=int, help="pause after this total number of completed rungs")
    a = ap.parse_args()
    try:
        plan = load_plan(a.plan)
        if a.resume and a.initial_summary is not None:
            raise ValueError("--resume uses the copied incumbent; omit --initial-summary")
        if not a.resume and plan["start"] == "guided_continuation" and a.initial_summary is None:
            raise ValueError("guided continuation requires --initial-summary")
        if not a.resume and plan["start"] == "cold" and a.initial_summary is not None:
            raise ValueError("a cold campaign cannot use --initial-summary")
        if a.stop_after is not None and not 1 <= a.stop_after <= len(plan["rungs"]):
            raise ValueError("stop-after must be between 1 and the rung count")
        out, python = a.outdir.resolve(), a.python.absolute()
        if not a.execute:
            print(json.dumps(dict(mode="plan_only_no_GPU", plan=plan,
                                  facets=NF, max_epochs=sum(r["epochs"] for r in plan["rungs"]),
                                  max_moves=sum(plan["common"]["S"] * r["iters"] * r["epochs"] for r in plan["rungs"]),
                                  initial_summary=str(a.initial_summary.resolve()) if a.initial_summary else None,
                                  outdir=str(out)), indent=2))
            return 0
        return campaign(plan, out, python, a.resume, a.stop_after,
                        initial_summary=a.initial_summary)
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(f"S6 campaign failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
