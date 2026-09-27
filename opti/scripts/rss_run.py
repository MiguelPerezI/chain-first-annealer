#!/usr/bin/env python3
"""Run a command and record its peak RSS in GNU `time -v` format.

Fallback for images without /usr/bin/time (the Kubeflow notebook image runs as
non-root, so it cannot be installed there). Usage: rss_run.py OUT CMD [ARGS...]
"""
import resource
import subprocess
import sys

out, cmd = sys.argv[1], sys.argv[2:]
rc = subprocess.call(cmd)
kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
with open(out, "w") as fh:
    fh.write("\tmeasured by rss_run.py\n")
    fh.write(f"\tMaximum resident set size (kbytes): {kb}\n")
    fh.write(f"\tExit status: {rc}\n")
sys.exit(rc)
