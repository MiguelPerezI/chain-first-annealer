#!/usr/bin/env python3
"""live.py — reading a running anneal's progress without touching the GPU.

The annealer reports itself on stdout, one line per epoch (see
`src/gpu_sc.py:1243` and `src/annealer_stretch.py:1548`):

    epoch   1: best  242.0/1536  top10 [242, 236, ...]  median 197  now-median 195     35.2s     0.584 Mmoves/s

`annealer_stretch.py` appends `stretch prop/acc` and (with `--pcollapse`)
`collapse prop/acc`, and `leftovers [...]` under `--verbose`.  Everything the
webapp shows live is derived from those lines plus, optionally, the JSONL the
run writes under `--jsonl`:

  * tail_last(path)          -> the last N lines of a log (the text panel)
  * parse_epoch_line(line)   -> dict, or None for a non-epoch line
  * series_from_log(path)    -> [{epoch, best, median, elapsed, ...}] for the curve
  * tail_jsonl(path)         -> the same, straight from the run's own writer

Nothing here imports matplotlib, numba or gpu_sc: the server may call it while
the box is busy annealing.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# the exact epoch line, tolerant of the optional telemetry suffixes
EPOCH_RE = re.compile(
    r"^\s*epoch\s+(?P<epoch>\d+):\s+best\s+(?P<best>[\d.]+)/(?P<nf>\d+)\s+"
    r"top10\s+\[(?P<top10>[^\]]*)\]\s+median\s+(?P<median>[\d.]+)\s+"
    r"now-median\s+(?P<now_median>[\d.]+)\s+(?P<elapsed>[\d.]+)s\s+"
    r"(?P<mmoves>[\d.]+)\s*Mmoves/s")
STRETCH_RE = re.compile(r"stretch prop\s+(\d+) acc\s+(\d+) \((\d+)%\)")
COLLAPSE_RE = re.compile(r"collapse prop\s+(\d+) acc\s+(\d+) \((\d+)%\)")
LEFTOVERS_RE = re.compile(r"leftovers\s+(\[[^\]]*\]|\d+)")
RESULT_RE = re.compile(
    r"^best coverage\s+(?P<cov>\d+)/(?P<nf>\d+)\s+domain sizes\s+\[(?P<sizes>[^\]]*)\]\s+"
    r"time\s+(?P<time>[\d.]+)s\s+kernel\s+(?P<kernel>[\d.]+)s\s+moves\s+(?P<moves>[\d.e+]+)")
HEADER_RE = re.compile(r"^===\s*(?P<which>\w+)\s+chain-first annealer\s*===\s*(?P<rest>.*)$")
DONE_RE = re.compile(r"^(?P<kind>VERIFIED system of|NOT a covering:|wrote |full coverage reached)")


def parse_epoch_line(line):
    """dict for an epoch progress line, else None."""
    m = EPOCH_RE.match(line)
    if not m:
        return None
    d = {k: (float(v) if k in ("best", "median", "now_median", "elapsed", "mmoves") else int(v))
         for k, v in m.groupdict().items() if k != "top10"}
    d["top10"] = [int(x) for x in re.findall(r"\d+", m.group("top10"))]
    s = STRETCH_RE.search(line)
    if s:
        d["stretch"] = {"prop": int(s.group(1)), "acc": int(s.group(2)), "pct": int(s.group(3))}
    c = COLLAPSE_RE.search(line)
    if c:
        d["collapse"] = {"prop": int(c.group(1)), "acc": int(c.group(2)), "pct": int(c.group(3))}
    lo = LEFTOVERS_RE.search(line)
    if lo:
        body = lo.group(1)
        d["leftovers"] = (len(json.loads(body)) if body.startswith("[") else int(body))
    return d


def describe_line(line):
    """A coarse kind for a log line, for colouring the tail panel."""
    if parse_epoch_line(line):
        return "epoch"
    if RESULT_RE.match(line):
        return "result"
    if HEADER_RE.match(line):
        return "header"
    if DONE_RE.match(line):
        return "done"
    if line.strip().startswith("seeded population"):
        return "init"
    if "time limit reached" in line or "full coverage reached" in line:
        return "done"
    return "other"


def tail_last(path, n=200):
    """The last n lines of a text file (missing file -> empty list)."""
    p = Path(path)
    if not p.is_file():
        return []
    size = p.stat().st_size
    with p.open("rb") as fh:                     # read a bounded window from the end
        fh.seek(max(0, size - 512 * max(n, 1)))
        data = fh.read().decode("utf-8", errors="replace")
    return data.splitlines()[-n:]


def series_from_log(path):
    """Every epoch line of a log, parsed, in order."""
    out = []
    for line in tail_last(path, n=100000):
        d = parse_epoch_line(line)
        if d:
            out.append(d)
    return out


def tail_jsonl(path, since=0):
    """Parse a run's `--jsonl` telemetry file (one JSON object per line).

    `since` skips records already delivered, so a poller can ask for the tail
    repeatedly without re-reading the whole file.
    """
    p = Path(path)
    if not p.is_file():
        return []
    out = []
    with p.open() as fh:
        for i, line in enumerate(fh):
            if i < since:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def progress_snapshot(log_path=None, jsonl_path=None, state=None, tail=80):
    """One poll's worth of progress for the UI: curve, tail, headline numbers."""
    rows = tail_jsonl(jsonl_path) if jsonl_path else []
    if not rows and log_path:
        rows = series_from_log(log_path)
    snap = {
        "history": [{"epoch": r.get("epoch"), "best": r.get("best"),
                     "median": r.get("median"), "elapsed": r.get("elapsed")} for r in rows],
        "last": rows[-1] if rows else None,
        "tail": [{"kind": describe_line(l), "text": l} for l in tail_last(log_path, tail)]
        if log_path else [],
    }
    if state:                                    # the run's own recorded numbers
        snap["state"] = {"coverage": state.get("coverage"),
                         "sizes": state.get("chain_domain_sizes"),
                         "status": state.get("status"),
                         "epoch": state.get("epoch"),
                         "params": {k: v for k, v in (state.get("params") or {}).items()
                                    if k in ("complex", "k", "seed", "init", "epochs",
                                             "time_limit", "beta", "gamma", "mu", "guide",
                                             "ptrain", "ptrack", "pstretch", "pcollapse")}}
    return snap
