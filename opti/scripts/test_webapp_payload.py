#!/usr/bin/env python3
"""test_webapp_payload.py — the webapp's data contract, without a GPU or matplotlib.

The local webapp draws K x K from a compact JSON payload built by
`scripts/kxk_state.py`.  If that payload is wrong the picture is wrong in a way
nobody notices by eye, so this pins it down against ground truth:

  1. every facet's grid vertices, rebuilt from the block's two K-facets plus the
     staircase masks, equal `Product.facets` -- the tables the GPU kernels use;
  2. the packed `covered` / `owner` / `memb` arrays decode back to exactly what
     `analyse()` computed;
  3. two recorded states keep their documented coverage and hole structure
     (RP^2 k=5 at 540/600 with its 60 private facets of R_5; Klein k=5 at
     1186/1536);
  4. the live log parser reads the annealer's real epoch lines, including the
     optional stretch/collapse/leftovers suffixes, and round-trips through the
     JSONL telemetry format.

Run:  python -B opti/scripts/test_webapp_payload.py     (also in CI)
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # opti/
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "webapp"))

import gpu_sc          # noqa: E402
import kxk_state as K  # noqa: E402
import live            # noqa: E402

FAILURES = []


def check(cond, what):
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILURES.append(what)


def test_facet_geometry():
    """Rebuilt facet vertices must equal Product's own tables, for every block."""
    print("facet geometry (masks -> grid vertices vs Product.facets)")
    for name in ("rp2", "klein", "torus", "mobius", "cube"):
        P = gpu_sc.build(name)
        n = len(P.kfacets)
        bad = 0
        for b in range(n * n):
            i, j = divmod(b, n)
            for p, f in enumerate(K.block_facets(P, i, j)):
                if K.facet_grid_vertices(P, i, j, p) != list(P.facets[f]):
                    bad += 1
        check(bad == 0, "%s: %d/%d facets rebuilt exactly" % (name, P.nf - bad, P.nf))
    # a higher-simplex complex too: S^3 blocks hold 20 product facets, not 6
    P = gpu_sc.build("s3")
    n = len(P.kfacets)
    bad = sum(1 for b in range(n * n) for p, f in enumerate(K.block_facets(P, *divmod(b, n)))
              if K.facet_grid_vertices(P, *divmod(b, n), p) != list(P.facets[f]))
    check(bad == 0, "s3: %d/%d facets rebuilt exactly (binom(6,3)=20 per block)"
          % (P.nf - bad, P.nf))


def test_block_index():
    print("block indexing")
    for name in ("rp2", "klein", "s6"):
        P = gpu_sc.build(name)
        blk, offs = K.block_index(P)
        n = len(P.kfacets)
        check(len(blk) == P.nf and int(blk.max()) == n * n - 1 and offs[-1] == P.nf,
              "%s: %d facets over %d blocks, contiguous" % (name, P.nf, n * n))
        # blocks are contiguous runs
        check(bool((np.diff(blk) >= 0).all()), "%s: facet ids increase with block id" % name)


def decode(payload, nf, k):
    cov = np.unpackbits(np.frombuffer(base64.b64decode(payload["covered"]), np.uint8))[:nf].astype(bool)
    own = np.frombuffer(base64.b64decode(payload["owner"]), np.uint8).astype(int) - 1
    memb = np.unpackbits(np.frombuffer(base64.b64decode(payload["memb"]), np.uint8))
    memb = memb.reshape(nf, -1)[:, :k].astype(bool)
    return cov, own, memb


def test_payload_roundtrip():
    print("payload round-trip")
    cases = [("rp2", ROOT / "results/rp2_k5_push_warm/final_state.json", 540),
             ("klein", ROOT / "results/klein_cold666_seed_999/summary.json", 1186)]
    for name, path, expect in cases:
        if not path.is_file():
            print("  skip %s (no %s)" % (name, path))
            continue
        P = gpu_sc.build(name)
        state, chains = K.load_state(path)
        A = K.analyse(P, chains)
        S = K.state_stats(P, chains, A)
        pay = K.state_payload(P, chains, A, S, complex_name=name, state_path=str(path))
        cov, own, memb = decode(pay, P.nf, len(chains))
        check(S["cov"] == expect, "%s: coverage %d (documented %d)" % (name, S["cov"], expect))
        check(int(cov.sum()) == expect, "%s: packed covered bits decode to %d" % (name, int(cov.sum())))
        check(bool((cov == A["covered"]).all()), "%s: covered bits == analyse()" % name)
        check(bool((own == A["owner"]).all()), "%s: owner bytes == analyse() (first owner)" % name)
        check(bool((memb == K.domain_membership(P, A)).all()), "%s: membership bits exact" % name)
        # the payload's own claim about the hole must match the decoded truth
        check(S["nhole"] == P.nf - int(cov.sum()),
              "%s: nhole %d consistent" % (name, S["nhole"]))
        # sizes in the payload are the domain sizes the run recorded
        check(list(pay["stats"]["sizes"]) == [len(d) for d in A["dom"]],
              "%s: domain sizes %s" % (name, pay["stats"]["sizes"]))
        # the masks shipped for this complex regenerate exactly its block sizes
        p0, q0 = len(P.kfacets[0]) - 1, len(P.kfacets[0]) - 1
        check(len(pay["masks"]["%d,%d" % (p0, q0)]) == pay["stats"]["per_block"],
              "%s: masks per block == %d" % (name, pay["stats"]["per_block"]))


def test_rp2_hole_structure():
    """RP^2 k=5: the 60 uncovered facets are the private cells of R_5."""
    print("RP^2 k=5 hole structure")
    path = ROOT / "results/rp2_k5_push_warm/final_state.json"
    if not path.is_file():
        print("  skip (no state)")
        return
    P = gpu_sc.build("rp2")
    state, chains = K.load_state(path)
    A = K.analyse(P, chains)
    S = K.state_stats(P, chains, A)
    K5 = P.kfacets
    dead = S["dead"]
    check(len(dead) == 60, "60 uncovered facets")
    check(all(len(set(K5[(f // 6) // 10]) & set(K5[(f // 6) % 10])) == 1 for f in dead),
          "every uncovered facet sits in a block with |s cap t| = 1")
    check(all(set(K5[(f // 6) // 10]) & set(K5[(f // 6) % 10]) == {5} for f in dead),
          "and that shared vertex is 5 (the dropped rectangle R_5)")
    check(S["cells"] == 10 and S["full"] == 10 and S["part"] == 0,
          "the hole is 10 whole blocks, no partial ones")
    R5 = A["rect_of_vertex"][5]
    check(int((~A["covered"][R5]).sum()) == 60, "R_5 itself holds exactly those 60")


def test_live_parser():
    print("live progress parser")
    real = "  epoch   1: best  242.0/1536  top10 [242, 236, 235, 234]  median 197  " \
           "now-median 195     35.2s     0.584 Mmoves/s"
    d = live.parse_epoch_line(real)
    check(d and d["epoch"] == 1 and d["best"] == 242.0 and d["nf"] == 1536
          and d["top10"] == [242, 236, 235, 234] and d["median"] == 197
          and abs(d["elapsed"] - 35.2) < 1e-9, "plain epoch line parses")
    stretch = real + ("  stretch prop 4826355 acc 578237 (12%)  collapse prop 2068954 "
                      "acc 82412 (4%)  leftovers 60")
    d = live.parse_epoch_line(stretch)
    check(d and d["stretch"]["acc"] == 578237 and d["collapse"]["pct"] == 4
          and d["leftovers"] == 60, "stretch/collapse/leftovers suffixes parse")
    check(live.parse_epoch_line("best coverage 1186/1536  domain sizes [258, 292]") is None,
          "non-epoch lines are not mistaken for epochs")
    check(live.describe_line(stretch) == "epoch"
          and live.describe_line("  time limit reached") == "done"
          and live.describe_line("NOT a covering: 350 facets uncovered") == "done",
          "line kinds classified")
    # round-trip through the JSONL the run writes under --jsonl
    rows = [{"epoch": i, "best": 500.0 + i, "median": 400.0, "elapsed": 10.0 * i}
            for i in range(1, 4)]
    tmp = Path("/tmp/_kxk_test_jsonl.jsonl")
    tmp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    got = live.tail_jsonl(tmp)
    check([r["best"] for r in got] == [501.0, 502.0, 503.0], "jsonl telemetry round-trips")
    check(live.tail_jsonl(tmp, since=2) == rows[2:], "jsonl tail can skip delivered records")
    tmp.unlink()
    # a real log, if one is around, must yield a monotone epoch series
    log = ROOT / "logs/klein_cold2.log"
    if log.is_file():
        ser = live.series_from_log(log)
        check(len(ser) > 3 and [r["epoch"] for r in ser] == sorted(r["epoch"] for r in ser),
              "real log %s -> %d epochs, ordered" % (log.name, len(ser)))
        check(all(0 <= r["best"] <= r["nf"] for r in ser), "every best is within [0, nf]")


def test_client_geometry():
    """The browser rebuilds each facet's grid vertices from the block's two
    K-facets plus the staircase masks (no vertex tables are shipped).  That
    rule must agree with Python's, facet for facet -- this is what makes the
    tooltips honest.  Skipped when node is unavailable."""
    print("client geometry (static/geometry.js) vs kxk_state")
    import shutil as _sh
    import subprocess as _sp
    node = _sh.which("node")
    if not node:
        print("  skip (no node)")
        return
    for name, rel in (("rp2", "results/rp2_k5_push_warm/final_state.json"),
                      ("klein", "results/klein_cold666_seed_999/summary.json")):
        path = ROOT / rel
        if not path.is_file():
            print("  skip %s (no state)" % name)
            continue
        P = gpu_sc.build(name)
        state, chains = K.load_state(path)
        A = K.analyse(P, chains)
        S = K.state_stats(P, chains, A)
        pay = K.state_payload(P, chains, A, S, complex_name=name, state_path=str(path))
        tmp = Path("/tmp/_kxk_payload.json")
        tmp.write_text(json.dumps(pay))
        js = (
            "const g=require(%r); const p=require(%r);"
            "let bad=0, n=0, labelBad=0;"
            "for(let i=0;i<p.triangles.length;i++)for(let j=0;j<p.triangles.length;j++){"
            "  const b=i*p.triangles.length+j;"
            "  for(let q=0;q<p.block_counts[b];q++){"
            "    const f=p.block_offsets[b]+q;"
            "    process.stdout.write(JSON.stringify([f,g.facetGridVertices(p,i,j,q),"
            "      g.pathLabel(p,i,j,q), p.sub_col[q], p.sub_row[q]])+'\\n');"
            "  }}"
        ) % (str(ROOT / "webapp/static/geometry.js"), str(tmp))
        out = _sp.run([node, "-e", js], capture_output=True, text=True, timeout=120)
        if out.returncode != 0:
            check(False, "%s: node failed: %s" % (name, out.stderr[-300:]))
            continue
        bad = label_bad = 0
        seen = 0
        for line in out.stdout.splitlines():
            f, verts, label, sub_col, sub_row = json.loads(line)
            i, j = divmod(int(K.block_index(P)[0][f]), len(P.kfacets))
            q = int(f) - int(K.block_index(P)[1][i * len(P.kfacets) + j])
            # python returns grid ids (a*nv + b, matching Product.facets); the
            # client returns the (a, b) pairs it prints in tooltips
            py = [(g // P.nv, g % P.nv) for g in K.facet_grid_vertices(P, i, j, q)]
            if [list(v) for v in verts] != [list(v) for v in py]:
                bad += 1
            if label != K.PATH_LABEL.get(q, label):
                label_bad += 1
            if sub_col != K.SUB_COL.get(q) or sub_row != K.SUB_ROW.get(q):
                label_bad += 1
            seen += 1
        check(seen == P.nf, "%s: client walked all %d facets" % (name, seen))
        check(bad == 0, "%s: client facet vertices == python for all %d" % (name, seen))
        check(label_bad == 0, "%s: client path labels / sub-cell layout agree" % name)
        tmp.unlink()


def test_telemetry_hooks():
    """The live-monitoring hooks: default-off, and what they write is readable
    by the same code that reads a finished run."""
    print("annealer telemetry hooks")
    sys.path.insert(0, str(ROOT / "src"))
    try:
        import annealer_stretch as A
    except Exception as exc:                       # no numba / no CUDA host
        print("  skip (cannot import annealer_stretch: %s)" % exc)
        return
    P = gpu_sc.build("rp2")
    args = type("a", (), dict(jsonl=None, checkpoint_every=0, outdir=None,
                              graceful_stop=False, stop_file=None, complex="rp2",
                              k=5, seed=1, init=None))

    # 1. nothing asked for -> nothing wired up (the default is today's behaviour)
    on_epoch, stop_when = A.make_monitors(args, P)
    check(on_epoch is None and stop_when is None,
          "no flags -> (None, None): the run behaves exactly as before")

    # 2. checkpointing writes a live_state.json a reader can use as a state
    tmp = Path("/tmp/_kxk_live_test")
    if tmp.exists():
        for f in tmp.iterdir():
            f.unlink()
    tmp.mkdir(parents=True, exist_ok=True)
    args2 = type("a", (), dict(jsonl=str(tmp / "epochs.jsonl"), checkpoint_every=2,
                               outdir=str(tmp), graceful_stop=False, stop_file=None,
                               complex="rp2", k=5, seed=1, init=None))
    on_epoch, _ = A.make_monitors(args2, P)
    state, chains = K.load_state(ROOT / "results/rp2_k5_push_warm/final_state.json")
    best = np.array(chains)                     # a plausible incumbent, k=5
    hist = [dict(epoch=1, best=480.0, median=400.0, elapsed=10.0)]
    on_epoch(1, 480.0, best, hist, type("ann", (), dict(moves=1e8, kernel_time=10.0)), 10.0)
    check(not (tmp / "live_state.json").exists(), "epoch 1 of every-2: nothing written yet")
    on_epoch(2, 540.0, best, hist + [dict(epoch=2, best=540.0, median=500.0, elapsed=20.0)],
             type("ann", (), dict(moves=2e8, kernel_time=20.0)), 20.0)
    live = tmp / "live_state.json"
    check(live.exists() and not list(tmp.glob("*.tmp")), "epoch 2: live_state.json written, no .tmp left")
    # the run's own reader must accept it as a state, and the map must decode
    path, st, _ = K.find_state(tmp)
    check(path.name == "live_state.json" and st["status"] == "running"
          and st["coverage"] == 540, "find_state reads the live checkpoint (coverage 540)")
    A_ = K.analyse(P, [[np.array(m, np.int64) for m in ch] for ch in st["best_chains"]])
    S_ = K.state_stats(P, [[np.array(m, np.int64) for m in ch] for ch in st["best_chains"]], A_)
    check(S_["cov"] == 540, "the checkpoint's chains cover 540/600 exactly as recorded")
    check((tmp / "epochs.jsonl").exists() and len(live.read_text()) > 100,
          "jsonl telemetry + checkpoint both written")
    # a warm start can be seeded from the live file (init_from_summary reads best_chains)
    seed = A.init_from_summary(P, live, 5, 40, 16)
    check(seed.shape == (16, 5, 41, P.ngrid), "init_from_summary accepts the live checkpoint")

    # 3. graceful stop: the handler only sets a flag, and stop_when sees it
    import signal as _sig
    args3 = type("a", (), dict(jsonl=None, checkpoint_every=0, outdir=None,
                               graceful_stop=True, stop_file=None, complex="rp2",
                               k=5, seed=1, init=None))
    _, stop3 = A.make_monitors(args3, P)
    check(stop3() is False, "graceful-stop: not requested yet")
    _sig.raise_signal(_sig.SIGTERM)
    check(stop3() is True, "SIGTERM sets the stop flag (checked at the epoch boundary)")

    # 4. stop-file is an independent channel.  _STOP is process-global and once
    #    set stays set (a run only ever stops once), so clear it between cases.
    A._STOP["requested"] = False
    flag = tmp / "STOP"
    args4 = type("a", (), dict(jsonl=None, checkpoint_every=0, outdir=None,
                               graceful_stop=False, stop_file=str(flag), complex="rp2",
                               k=5, seed=1, init=None))
    _, stop4 = A.make_monitors(args4, P)
    check(stop4() is False, "stop-file: absent -> keep going")
    flag.write_text("")
    check(stop4() is True, "stop-file: appears -> stop")


if __name__ == "__main__":
    print("== webapp payload contract ==")
    test_facet_geometry()
    test_block_index()
    test_payload_roundtrip()
    test_rp2_hole_structure()
    test_live_parser()
    test_telemetry_hooks()
    test_client_geometry()
    print("== %s ==" % ("all checks passed" if not FAILURES
                        else "%d FAILURES: %s" % (len(FAILURES), FAILURES)))
    sys.exit(1 if FAILURES else 0)
