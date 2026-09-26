#!/usr/bin/env python3
"""server.py — local webapp for the K x K annealer campaign.

What it does
------------
* **Browses** every state in `results/` and draws K x K the way
  `scripts/render_kxk.py` draws it — but interactively, and at every zoom level:
  red = uncovered, every covered facet in the colour of the domain that owns it,
  one colour per domain (overlaps are marked, never hidden).
* **Launches** a run on the GPU host over ssh: warm-started from a chosen
  previous state (`--init`, always a *file* — `init_from_summary` dies on a
  directory) or cold, with the composed command shown for confirmation first.
* **Follows** a running job live — epoch curve, log tail, and the incumbent's
  K x K map, from `--jsonl` telemetry and the `--checkpoint-every`
  `live_state.json` the run writes.  Also attaches to runs it did not start, by
  tailing their log (and their checkpoints, if the run was given the flag).

Design notes
------------
* **No dependencies**: stdlib `http.server`, one HTML page, Canvas 2D.  The
  data layer is `scripts/kxk_state.py` (matplotlib-free, so this process can
  import it while the box is busy annealing) and the export button shells out
  to `scripts/render_kxk.py` so the page and the paper figures can never
  disagree.
* **Nothing is recomputed per request**: coverage costs 0.04 s on Klein but
  ~9 s on S^6, so payloads are cached by `(realpath, mtime_ns, size, inode)`
  and served with an ETag; a cold S^6 request returns 202 and computes in a
  worker thread.
* **Safety**: the launch form has no free-text that reaches a shell — numbers
  go through `src/validation.validate_parameters`, `--complex` through a
  whitelist, paths are built from a validated run name, and everything is
  `shlex.quote`d into an argv list (never `shell=True`).  Launch and stop each
  require a token minted by the preceding *plan* call, so what runs is exactly
  the command that was displayed.

Run:  bash webapp/serve.sh --port 8770     (then open http://127.0.0.1:8770)
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import hmac
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                   # opti/
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import gpu_sc          # noqa: E402
import kxk_state as K  # noqa: E402
import live            # noqa: E402

# rho(K): the rectangle-cover number, i.e. the fewest product domains that can
# cover K x K at all.  Only values recorded in this repo are listed; anything
# else is reported as unknown rather than guessed.  Sources: README.md (torus
# 11, cube 8, wedge 4), experiments/surface_stall_analysis.md (rp2 6, Mobius 4,
# klein 12), gpu_sc.py (bary: a cone, one domain).
RHO = {"bary": 1, "wedge": 4, "cube": 8, "torus": 11, "rp2": 6, "mobius": 4, "klein": 12}

RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")
TOKEN_TTL = 120.0
_SECRET = os.urandom(32)


# --------------------------------------------------------------------------- #
#  state cache                                                                 #
# --------------------------------------------------------------------------- #
class StateCache:
    """Payloads keyed by the state file's identity, so a live checkpoint that is
    rewritten every epoch is re-read, and an unchanged file never is."""

    def __init__(self, ttl_runs=5.0):
        self._lock = threading.Lock()
        self._products = {}
        self._payload = {}          # key -> (body bytes, etag)
        self._errors = {}
        self._runs = (0.0, None)
        self._ttl_runs = ttl_runs

    def product(self, name):
        with self._lock:
            if name not in self._products:
                self._products[name] = gpu_sc.build(name)
            return self._products[name]

    @staticmethod
    def stat_key(path):
        st = os.stat(path)
        return (os.path.realpath(path), st.st_mtime_ns, st.st_size, st.st_ino)

    def payload(self, state_path, block=False):
        """(body bytes, etag) for a state, or (None, eta) while computing."""
        key = self.stat_key(state_path)
        with self._lock:
            hit = self._payload.get(key)
            if hit:
                return hit, None
            if key in self._errors:
                raise RuntimeError(self._errors[key])
        t0 = time.time()
        state, chains = K.load_state(state_path)
        name = K.infer_complex(state, state_path)
        if not name:
            raise RuntimeError("cannot tell which complex %s belongs to" % state_path)
        P = self.product(name)
        rec = state.get("complex")
        if isinstance(rec, dict) and rec.get("num_facets") not in (None, P.nf):
            raise RuntimeError("%s records %s facets but %s builds %d"
                               % (state_path, rec.get("num_facets"), name, P.nf))
        A = K.analyse(P, chains)
        S = K.state_stats(P, chains, A)
        pay = K.state_payload(P, chains, A, S, complex_name=name, state_path=str(state_path))
        pay["schema"] = 1
        pay["rho"] = RHO.get(name)
        pay["h1_report"] = str(K.find_h1_report(state_path, ROOT / "results", S["nf"], S["cov"]) or "")
        pay["compute_s"] = round(time.time() - t0, 2)
        body = json.dumps(pay).encode()
        etag = '"%d-%d"' % (key[1], key[2])
        with self._lock:
            self._payload[key] = (body, etag)
            while len(self._payload) > 8:                # keep memory bounded
                self._payload.pop(next(iter(self._payload)))
        return (body, etag), None

    def runs(self):
        """The results index, behind a short TTL (the scan touches 160 dirs)."""
        with self._lock:
            ts, cached = self._runs
            if cached is not None and time.time() - ts < self._ttl_runs:
                return cached
        runs = K.list_runs(ROOT / "results")
        for r in runs:
            r["rho"] = RHO.get(r["complex"])
            r["k_below_rho"] = bool(r["rho"] and r["k"] and r["k"] < r["rho"])
        with self._lock:
            self._runs = (time.time(), runs)
        return runs


CACHE = StateCache()


# --------------------------------------------------------------------------- #
#  the runner: config, composition, jobs over ssh                              #
# --------------------------------------------------------------------------- #
def load_config():
    cfg = {"host": "", "identity": "~/.ssh/gcp_key", "remote_repo": "~/Documents/opti",
           "remote_python": "~/Documents/Abstract_Motion_GPU/venv/bin/python",
           "ssh_options": ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                           "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
                           "-o", "StrictHostKeyChecking=accept-new"]}
    for p in (Path(os.environ.get("OPTIMAP_CONFIG", "")), HERE / "runner.json"):
        if p and p.is_file():
            cfg.update(json.loads(p.read_text()))
            break
    for key, env in (("host", "OPTIMAP_HOST"), ("identity", "OPTIMAP_KEY"),
                     ("remote_repo", "OPTIMAP_REPO"), ("remote_python", "OPTIMAP_PYTHON")):
        if os.environ.get(env):
            cfg[key] = os.environ[env]
    return cfg


CFG = load_config()


def is_local():
    """`"host": "local"` in runner.json (or OPTIMAP_HOST=local) runs the annealer
    on THIS machine through numba's CUDA simulator instead of over ssh."""
    return str(CFG.get("host", "")).strip() == "local"


def ssh_argv(remote_cmd, cfg=None):
    cfg = cfg or CFG
    return ["ssh", "-i", os.path.expanduser(cfg["identity"]), *cfg["ssh_options"],
            cfg["host"], "--", remote_cmd]


def rq(p):
    """Quote a path for the *remote* shell, letting a leading ~ expand there.

    `shlex.quote("~/x")` would stop the remote shell from expanding the tilde
    (it would look for a directory literally named "~"), and expanding it
    locally would be wrong too -- the remote home is a different machine's.
    """
    p = str(p)
    if p == "~":
        return '"$HOME"'
    if p.startswith("~/"):
        return '"$HOME"/%s' % shlex.quote(p[2:])
    return shlex.quote(p)


def next_outdir(run, complex_name, k, continuation):
    """Where a run writes: a fresh timestamped dir, or <run>/phase<N>/ when
    continuing (the results/rp2_k5_push_warm/phase1,phase2 convention)."""
    results = ROOT / "results"
    if continuation:
        base = results / continuation
        n = 1 + max([int(m.group(1)) for p in base.glob("phase*") if (m := re.match(r"phase(\d+)$", p.name))] or [0])
        return base / ("phase%d" % n), "%s_phase%d" % (continuation, n)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    name = "%s_k%d_%s" % (complex_name, k, stamp)
    return results / name, name


def compose_run(req):
    """Validate a launch request and build (remote command, display, outdir, log).

    Everything user-supplied is either a number checked by the CLI's own
    validator, a whitelisted complex name, or a run directory name matching
    RUN_NAME_RE -- nothing is interpolated into a shell unquoted.
    """
    params = req.get("params") or {}
    complex_name = req.get("complex")
    if complex_name not in K.COMPLEXES:
        raise ValueError("unknown complex %r (choose one of %s)"
                         % (complex_name, ", ".join(K.COMPLEXES)))
    from validation import validate_parameters
    checked = dict(k=int(params.get("k", 5)), cmax=int(params.get("cmax", 40)),
                   S=int(params.get("S", 4096)), iters=int(params.get("iters", 5000)),
                   epochs=int(params.get("epochs", 100)), seed=int(params.get("seed", 0)),
                   tournament_every=2, h1_guide=0)
    floats = {key: float(params.get(key, default)) for key, default in (
        ("T0", 0.9), ("T1", 0.03), ("beta", 5.0), ("gamma", 0.5), ("mu", 0.0),
        ("guide", 3.0), ("guide_max", 11.0), ("pcopy", 0.5), ("ptrack", 0.3),
        ("crossover", 0.0), ("tournament_frac", 0.25),
        ("pstretch", 0.0), ("pcollapse", 0.0))}
    if floats["pstretch"] == 0:
        floats["pcollapse"] = 0.0
    time_limit = params.get("time_limit")
    # `validate_parameters` is the CLI's own checker, so it takes exactly the
    # keys the CLI passes; pstretch/pcollapse ride the stretch branch and are
    # checked here (both are probabilities).
    VALIDATED = ("T0", "T1", "beta", "gamma", "mu", "guide", "guide_max",
                 "pcopy", "ptrack", "crossover", "tournament_frac")
    validate_parameters(**checked, **{k: floats[k] for k in VALIDATED},
                        time_limit=(None if time_limit in (None, "", 0) else float(time_limit)))
    for key in ("pstretch", "pcollapse"):
        if not 0.0 <= floats[key] <= 1.0:
            raise ValueError("%s must be in [0, 1]" % key)
    complex_name = dict(complex=complex_name, **checked, **floats)

    warm = req.get("warm_from")
    init_path = None
    if warm not in (None, "", "cold"):
        s = str(warm)
        if not s.endswith(".json"):
            raise ValueError("--init takes the state FILE, not a directory "
                             "(init_from_summary dies on a directory): got %r" % s)
        init_abs = (ROOT / s).resolve() if not s.startswith("/") else Path(s)
        if not str(init_abs).startswith(str((ROOT / "results").resolve())):
            raise ValueError("the warm-start state must live under results/")
        if not init_abs.is_file():
            raise ValueError("no such state file: %s" % s)
        init_path = init_abs.relative_to(ROOT)

    cont = req.get("continue_run")
    if cont is not None and not RUN_NAME_RE.match(str(cont)):
        raise ValueError("bad run name for a continuation: %r" % cont)
    outdir, run_name = next_outdir(None, complex_name["complex"], checked["k"], cont)

    flags = ["--complex", complex_name["complex"], "--k", "%d" % checked["k"],
             "--cmax", "%d" % checked["cmax"], "--S", "%d" % checked["S"],
             "--iters", "%d" % checked["iters"], "--epochs", "%d" % checked["epochs"],
             "--seed", "%d" % checked["seed"], "--T0", "%g" % floats["T0"],
             "--T1", "%g" % floats["T1"], "--beta", "%g" % floats["beta"],
             "--gamma", "%g" % floats["gamma"], "--mu", "%g" % floats["mu"],
             "--guide", "%g" % floats["guide"], "--guide-max", "%g" % floats["guide_max"],
             "--pcopy", "%g" % floats["pcopy"], "--ptrack", "%g" % floats["ptrack"],
             "--tournament-frac", "%g" % floats["tournament_frac"]]
    if floats["crossover"]:
        flags += ["--crossover", "%g" % floats["crossover"]]
    if floats["pstretch"]:
        flags += ["--pstretch", "%g" % floats["pstretch"]]
    if floats["pcollapse"]:
        flags += ["--pcollapse", "%g" % floats["pcollapse"]]
    if time_limit not in (None, "", 0):
        flags += ["--time-limit", "%g" % float(time_limit)]
    if init_path is not None:
        flags += ["--init", str(init_path)]
    # Everything the remote shell touches must be a path *in the remote repo*:
    # the local absolute path would not exist there.  The local mirror keeps
    # the absolute one for reading.
    rem_out = str(outdir.relative_to(ROOT))
    flags += ["--jsonl", "%s/epochs.jsonl" % rem_out,
              "--checkpoint-every", "%d" % int(req.get("checkpoint_every") or 1),
              "--graceful-stop"]

    rel = shlex.quote(rem_out)
    log = ROOT / "logs" / ("%s.log" % run_name)
    rlog = str(log.relative_to(ROOT))
    if is_local():
        # Simulation mode: the SAME kernels on this machine, through numba's
        # CUDA simulator (NUMBA_ENABLE_CUDASIM=1), which is how
        # src/test_stretch_cpu.py exercises them.  ~1000x slower than a GPU, so
        # it is for exercising the loop (launch -> live -> stop) on a small
        # complex with a tiny population, not for searching.
        argv = [CFG.get("local_python") or sys.executable, "-u",
                str(ROOT / "src/annealer_stretch.py")] + flags + ["--outdir", str(outdir)]
        return dict(remote=" ".join(shlex.quote(a) for a in argv), argv=argv,
                    display="NUMBA_ENABLE_CUDASIM=1 " + " ".join(shlex.quote(a) for a in argv),
                    local=True, run=run_name, outdir=str(outdir.relative_to(ROOT)), log=rlog,
                    init=str(init_path) if init_path else None, params=complex_name)

    remote = ("cd %s && mkdir -p %s %s && PYTHONUNBUFFERED=1 nohup %s -u src/annealer_stretch.py %s "
              "--outdir %s > %s 2>&1 < /dev/null & pid=$!; echo $pid > %s/pid; sleep 1; "
              "if kill -0 $pid 2>/dev/null; then echo LAUNCHED $pid; else echo FAILED; tail -20 %s; fi"
              % (rq(CFG["remote_repo"]), rel, rq(str(Path(rlog).parent)),
                 rq(CFG["remote_python"]), " ".join(flags), rel,
                 rq(rlog), rel, rq(rlog)))
    return dict(remote=remote, argv=ssh_argv(remote), display=" ".join(shlex.quote(a) for a in ssh_argv(remote)),
                local=False,
                run=run_name, outdir=str(outdir.relative_to(ROOT)), log=rlog,
                init=str(init_path) if init_path else None, params=complex_name)


def mint(body):
    """A token binding a confirmation to this exact request body."""
    payload = base64.urlsafe_b64encode(json.dumps(body, sort_keys=True).encode()).decode()
    mac = hmac.new(_SECRET, payload.encode(), hashlib.sha256).hexdigest()[:32]
    return "%s.%d.%s" % (payload, int(time.time()), mac)


def verify(token, body, ttl=TOKEN_TTL):
    try:
        payload, ts, mac = token.split(".")
    except Exception:
        return False, "malformed token"
    if not hmac.compare_digest(mac, hmac.new(_SECRET, payload.encode(), hashlib.sha256).hexdigest()[:32]):
        return False, "token does not match this request (did the form change?)"
    if time.time() - int(ts) > ttl:
        return False, "confirmation expired after %ds -- press the button again" % ttl
    if payload != base64.urlsafe_b64encode(json.dumps(body, sort_keys=True).encode()).decode():
        return False, "token was issued for a different command"
    return True, ""


class Jobs:
    """The job registry and the threads that keep a run observable."""

    def __init__(self):
        self.path = HERE / "jobs.json"
        self.live_dir = HERE / "live"
        self.live_dir.mkdir(exist_ok=True)
        self.lock = threading.Lock()
        self.procs = {}                       # jid -> Popen, for local simulations
        self.jobs = json.loads(self.path.read_text()) if self.path.is_file() else {}

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.jobs, indent=2, default=str))
        tmp.replace(self.path)

    def add(self, job):
        with self.lock:
            self.jobs[job["id"]] = job
            self.save()
        threading.Thread(target=self._watch, args=(job["id"],), daemon=True).start()
        return job

    def get(self, jid):
        return self.jobs.get(jid)

    def dir(self, jid):
        d = self.live_dir / jid
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _ssh(self, cmd, timeout=25):
        return subprocess.run(ssh_argv(cmd), capture_output=True, text=True, timeout=timeout)

    @staticmethod
    def _pid_alive(pid):
        """True only for a live process: a zombie (exited, not yet reaped) still
        answers `kill -0`, which would make a crashed run look like it is going."""
        try:
            os.kill(int(pid), 0)
        except Exception:
            return False
        try:
            with open("/proc/%s/stat" % pid) as fh:
                return fh.read().split(") ", 1)[1].split()[0] != "Z"
        except Exception:
            return True

    def _poll_local(self, job):
        """Liveness and the live checkpoint for a locally simulated run."""
        pid = job.get("pid")
        proc = self.procs.get(job.get("id"))
        if proc is not None and proc.poll() is not None:
            alive = False
            job["exit_code"] = proc.returncode
        else:
            alive = self._pid_alive(pid) if pid else False
        job["ssh_ok"] = True                 # nothing to reach
        job["remote_state"] = alive
        live = ROOT / job["outdir"] / "live_state.json"
        if live.is_file():
            data = live.read_text()
            # mirror it beside the job, so /api/jobs/<id> exposes the same
            # `live_state` field for a local run as for a remote one
            mirror = self.dir(job["id"]) / "live_state.json"
            tmp = mirror.with_suffix(".tmp")
            tmp.write_text(data)
            tmp.replace(mirror)
            try:
                job["live_coverage"] = json.loads(data).get("coverage")
            except Exception:
                pass
        if not alive and job.get("status") in ("running", "launching"):
            done = (ROOT / job["outdir"] / "summary.json").is_file()
            job["status"] = "finished" if done else "failed"

    def _watch(self, jid):
        """Mirror the remote log, pull the live checkpoint, track liveness."""
        job = self.jobs[jid]
        d = self.dir(jid)
        if job.get("local"):
            # a local simulation writes its log here directly: no mirror needed
            job["log_local"] = str(ROOT / job["log"])
        else:
            log_local = d / "run.log"
            # the log mirror: tail -F survives the file not existing yet and
            # keeps following it as epochs are appended
            tail = subprocess.Popen(ssh_argv("tail -n +1 -F %s" % rq(job["log"])),
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            job["log_local"] = str(log_local)

            def pump():
                with log_local.open("a") as fh:
                    for line in tail.stdout:
                        fh.write(line)
                        fh.flush()
            threading.Thread(target=pump, daemon=True).start()

        while True:
            time.sleep(5)
            if job.get("local"):
                self._poll_local(job)
                with self.lock:
                    self.save()
                continue
            alive = self._ssh("kill -0 $(cat %s/pid) 2>/dev/null && echo UP || echo DOWN"
                              % rq(job["outdir"]))
            ssh_ok = alive.returncode == 0
            job["ssh_ok"] = ssh_ok
            if ssh_ok:
                job["remote_state"] = alive.stdout.strip().endswith("UP")
            # pull the live checkpoint (cheap: a few KB) so /api/map can serve it
            got = self._ssh("cat %s/live_state.json 2>/dev/null" % rq(job["outdir"]))
            if got.stdout.startswith("{"):
                tmp = d / "live_state.json.tmp"
                tmp.write_text(got.stdout)
                tmp.replace(d / "live_state.json")
                try:
                    job["live_coverage"] = json.loads(got.stdout).get("coverage")
                except Exception:
                    pass
            if not ssh_ok and job.get("status") == "running":
                job["status"] = "unreachable"
            if ssh_ok and job.get("remote_state") is False and job.get("status") == "running":
                # process gone: finished if the run wrote its own result
                done = self._ssh("ls %s/summary.json %s/live_state.json 2>/dev/null"
                                 % (rq(job["outdir"]), rq(job["outdir"])))
                job["status"] = "finished" if done.stdout.strip() else "failed"
            with self.lock:
                self.save()


JOBS = Jobs()


# --------------------------------------------------------------------------- #
#  HTTP                                                                        #
# --------------------------------------------------------------------------- #
STATIC = {".html": "text/html; charset=utf-8", ".js": "application/javascript",
          ".css": "text/css", ".png": "image/png", ".pdf": "application/pdf",
          ".svg": "image/svg+xml", ".json": "application/json"}


class Handler(BaseHTTPRequestHandler):
    server_version = "opti-kxk/1.0"

    def log_message(self, fmt, *args):                   # quieter than the default
        if "?" in self.path or self.path.startswith("/api/"):
            sys.stderr.write("  %s\n" % (fmt % args))

    # -- helpers ---------------------------------------------------------- #
    def _send(self, code, body=b"", ctype="application/json", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code=200, extra=None):
        self._send(code, json.dumps(obj).encode(), "application/json", extra)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def _guard(self):
        """POSTs must carry our header: a browser cannot send it cross-origin
        without a preflight this server refuses."""
        if self.headers.get("X-OptiMap") != "1":
            self._json({"ok": False, "error": "missing X-OptiMap header"}, 403)
            return False
        return True

    def _query(self):
        return {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).items()}

    def _state_path(self, raw):
        p = (ROOT / raw).resolve() if not str(raw).startswith("/") else Path(raw).resolve()
        if not str(p).startswith(str((ROOT / "results").resolve())) and \
           not str(p).startswith(str((HERE / "live").resolve())):
            raise ValueError("state path must live under results/ (got %s)" % raw)
        return p

    # -- GET --------------------------------------------------------------- #
    def do_GET(self):
        q = self._query()
        path = urllib.parse.urlparse(self.path).path
        try:
            if path == "/api/config":
                return self._json({
                    "repo": str(ROOT), "python": sys.executable,
                    "results": str(ROOT / "results"), "figures": str(ROOT / "figures"),
                    "complexes": K.COMPLEXES, "rho": RHO,
                    "runner": {"configured": bool(CFG["host"]), "local": is_local(),
                               "host": CFG["host"],
                               "remote_repo": CFG["remote_repo"],
                               "remote_python": CFG["remote_python"],
                               "identity": CFG["identity"]}})
            if path == "/api/runs":
                return self._json({"runs": CACHE.runs()})
            if path == "/api/map":
                state = self._state_path(q.get("state", ""))
                if not state.is_file():
                    return self._json({"ok": False, "error": "no such state: %s" % state}, 404)
                (body, etag), pending = CACHE.payload(state)
                if pending:
                    return self._json({"status": "computing"}, 202)
                if self.headers.get("If-None-Match") == etag:
                    return self._send(304, b"", extra={"ETag": etag})
                return self._send(200, body, "application/json", {"ETag": etag})
            if path == "/api/jobs":
                return self._json({"jobs": list(JOBS.jobs.values())})
            if path.startswith("/api/jobs/"):
                jid = path.split("/")[3]
                job = JOBS.get(jid)
                if not job:
                    return self._json({"ok": False, "error": "no such job"}, 404)
                tail = live.tail_last(job.get("log_local", ""), 200)
                mirror = JOBS.dir(jid) / "live_state.json"
                return self._json({**job,
                                   "log": [{"kind": live.describe_line(l), "text": l} for l in tail],
                                   "history": [r for r in live.tail_jsonl(
                                       ROOT / job["outdir"] / "epochs.jsonl")][-500:]
                                   if (ROOT / job["outdir"] / "epochs.jsonl").is_file() else
                                   live.series_from_log(job.get("log_local", "")),
                                   "live_state": str(mirror.relative_to(ROOT)) if mirror.is_file() else None})
            if path == "/api/export":
                return self._export(q)
            if path in ("/", "/index.html"):
                return self._static(HERE / "static/index.html")
            if path.startswith("/static/"):
                return self._static(HERE / "static" / path[len("/static/"):])
            if path.startswith("/files/"):                       # results/ and exports, read-only
                rel = path[len("/files/"):]
                target = (ROOT / rel).resolve()
                if not str(target).startswith(str(ROOT)):
                    return self._json({"ok": False, "error": "outside the repo"}, 403)
                return self._static(target)
            return self._json({"ok": False, "error": "no route %s" % path}, 404)
        except Exception as exc:
            return self._json({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}, 500)

    def _static(self, p):
        p = Path(p)
        if not p.is_file():
            return self._json({"ok": False, "error": "not found: %s" % p}, 404)
        body = p.read_bytes()
        ctype = STATIC.get(p.suffix, "application/octet-stream")
        if p.suffix in (".json", ".js", ".css", ".html") and len(body) > 4096:
            body = gzip.compress(body)
            return self._send(200, body, ctype, {"Content-Encoding": "gzip"})
        return self._send(200, body, ctype)

    def _export(self, q):
        state = self._state_path(q["state"])
        fmt = q.get("fmt", "both")
        run = state.parent.name if state.name in ("summary.json", "final_state.json") else state.stem
        out = HERE / "exports" / run
        (HERE / "exports").mkdir(exist_ok=True)
        cmd = [sys.executable, str(ROOT / "scripts/render_kxk.py"), "--state", str(state), "-o", str(out)]
        if q.get("by_domain") == "1":
            cmd.append("--by-domain")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        files = [str(p) for p in (out.with_suffix(".png"), out.with_suffix(".pdf")) if p.is_file()]
        return self._json({"ok": proc.returncode == 0, "command": " ".join(shlex.quote(c) for c in cmd),
                           "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:],
                           "files": [str(Path(f).relative_to(ROOT)) for f in files],
                           "urls": ["/files/%s" % Path(f).relative_to(ROOT) for f in files]})

    # -- POST -------------------------------------------------------------- #
    def do_POST(self):
        if not self._guard():
            return
        path = urllib.parse.urlparse(self.path).path
        try:
            body = self._body()
            if path == "/api/plan":
                plan = compose_run(body)
                return self._json({"ok": True, "command": plan["display"], "run": plan["run"],
                                   "outdir": plan["outdir"], "log": plan["log"], "init": plan["init"],
                                   "warnings": self._warnings(body, plan),
                                   "token": mint(body)})
            if path == "/api/launch":
                ok, why = verify(body.pop("token", ""), body)
                if not ok:
                    return self._json({"ok": False, "error": why}, 409)
                if not CFG["host"]:
                    return self._json({"ok": False, "error":
                                       "no GPU host configured: copy webapp/runner.example.json to "
                                       "webapp/runner.json and set host (or export OPTIMAP_HOST)"}, 400)
                plan = compose_run(body)
                jid = "j%d" % int(time.time())
                job = dict(id=jid, run=plan["run"], outdir=plan["outdir"], log=plan["log"],
                           command=plan["display"], init=plan["init"], host=CFG["host"],
                           local=bool(plan.get("local")),
                           started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           status="launching", params=plan["params"], live_coverage=None)
                JOBS.add(job)
                if plan.get("local"):
                    log = ROOT / plan["log"]
                    log.parent.mkdir(parents=True, exist_ok=True)
                    (ROOT / plan["outdir"]).mkdir(parents=True, exist_ok=True)
                    env = dict(os.environ, NUMBA_ENABLE_CUDASIM="1", PYTHONUNBUFFERED="1")
                    with log.open("wb") as fh:
                        proc = subprocess.Popen(plan["argv"], cwd=str(ROOT), env=env,
                                                stdout=fh, stderr=subprocess.STDOUT,
                                                stdin=subprocess.DEVNULL, start_new_session=True)
                    JOBS.procs[jid] = proc
                    job["pid"] = str(proc.pid)
                    time.sleep(1.5)                 # an immediate crash must be visible
                    if proc.poll() is not None:
                        job["status"] = "failed"
                        job["exit_code"] = proc.returncode
                        job["error"] = "".join(live.tail_last(log, 12))
                        with JOBS.lock:
                            JOBS.save()
                        return self._json({"ok": False, "job": job, "output": job["error"]}, 502)
                    job["status"] = "running"
                    with JOBS.lock:
                        JOBS.save()
                    return self._json({"ok": True, "job": job, "output": "local simulation pid %s" % proc.pid})
                proc = subprocess.run(plan["argv"], capture_output=True, text=True, timeout=60)
                out = (proc.stdout or "") + (proc.stderr or "")
                if "LAUNCHED" not in out:
                    job["status"] = "failed"
                    job["error"] = out.strip()[-500:]
                    with JOBS.lock:
                        JOBS.save()
                    return self._json({"ok": False, "job": job, "output": out}, 502)
                job["status"] = "running"
                job["pid"] = out.split("LAUNCHED")[-1].strip().split()[0]
                with JOBS.lock:
                    JOBS.save()
                return self._json({"ok": True, "job": job})
            if path == "/api/jobs/stop":
                jid = body.get("id")
                job = JOBS.get(jid)
                if not job:
                    return self._json({"ok": False, "error": "no such job"}, 404)
                if job.get("local"):
                    try:
                        os.kill(int(job["pid"]), signal.SIGTERM)
                        out = "STOPPED (SIGTERM to local pid %s)" % job["pid"]
                    except Exception as exc:
                        out = "GONE (%s)" % exc
                    proc = type("p", (), {"stdout": out})()
                else:
                    proc = subprocess.run(ssh_argv(
                        "kill -TERM $(cat %s/pid) 2>/dev/null && echo STOPPED || echo GONE"
                        % rq(job["outdir"])), capture_output=True, text=True, timeout=30)
                job["status"] = "stopping"
                with JOBS.lock:
                    JOBS.save()
                return self._json({"ok": True, "output": proc.stdout.strip(), "job": job})
            return self._json({"ok": False, "error": "no route %s" % path}, 404)
        except Exception as exc:
            return self._json({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}, 400)

    def _warnings(self, body, plan):
        """Pre-flight notes, computed from the state -- never guessed."""
        out = []
        warm = body.get("warm_from")
        if warm not in (None, "", "cold"):
            try:
                state, chains = K.load_state(ROOT / str(warm))
            except Exception as exc:
                return [{"level": "error", "msg": "cannot read the warm-start state: %s" % exc}]
            p = plan["params"]
            name = K.infer_complex(state, ROOT / str(warm))
            rec = state.get("complex") if isinstance(state.get("complex"), dict) else {}
            if rec.get("num_facets") and rec["num_facets"] != CACHE.product(p["complex"]).nf:
                out.append({"level": "error", "msg":
                            "%s holds %d facets but %s builds %d -- init_from_summary does NOT check "
                            "the complex, this would silently seed garbage"
                            % (warm, rec["num_facets"], p["complex"], CACHE.product(p["complex"]).nf)})
            k_state = len(state.get("best_chains") or [])
            sizes = sorted(state.get("chain_domain_sizes") or [], reverse=True)
            if k_state and p["k"] != k_state:
                if p["k"] < k_state:
                    out.append({"level": "warn", "msg":
                                "k=%d < the state's %d domains: init_from_summary keeps the %d largest "
                                "domains and drops %s, so the seed starts weaker than the state"
                                % (p["k"], k_state, p["k"], sizes[p["k"]:])})
                else:
                    out.append({"level": "warn", "msg":
                                "k=%d > the state's %d domains: %d slots start from random chains"
                                % (p["k"], k_state, p["k"] - k_state)})
            links = max([len(c) - 1 for c in chains] or [0])
            if p["cmax"] < links:
                out.append({"level": "error", "msg":
                            "cmax=%d is shorter than the seed's longest reduced chain (%d links): "
                            "init_from_summary raises ValueError" % (p["cmax"], links)})
        rho = RHO.get(plan["params"]["complex"])
        if rho and plan["params"]["k"] < rho:
            out.append({"level": "warn", "msg":
                        "k=%d < rho(%s)=%d: no product cover that small exists, so this run can only "
                        "plateau (every recorded one did)" % (plan["params"]["k"],
                                                              plan["params"]["complex"], rho)})
        elif rho is None:
            out.append({"level": "info", "msg":
                        "rho(%s) is not recorded in this repo -- unknown, so no cover-existence "
                        "warning is made" % plan["params"]["complex"]})
        if not CFG["host"]:
            out.append({"level": "info", "msg": "no GPU host configured: the command above is what "
                                                "would run; launching is disabled"})
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--allow-remote", action="store_true",
                    help="allow binding a non-loopback address (anyone who can reach the port "
                         "can start GPU jobs on your host)")
    a = ap.parse_args()
    if a.host not in ("127.0.0.1", "localhost", "::1") and not a.allow_remote:
        raise SystemExit("refusing to bind %s without --allow-remote" % a.host)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print("K x K webapp  ->  http://%s:%d" % (a.host, a.port))
    print("  repo    %s" % ROOT)
    print("  runner  %s" % ("%s (%s)" % (CFG["host"], CFG["remote_repo"]) if CFG["host"]
                            else "NOT configured -- browsing and export work, launching is off"))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
