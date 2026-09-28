"""Opt-in flags of scs_sat.py: --fix-domain0 and --tables leave defaults untouched."""
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

OPTI = Path(__file__).resolve().parents[1]
SCS = OPTI / "paper-overleaf" / "paper" / "scs_sat.py"
BASE = ["--complex", "mobius", "--k", "3", "--L", "4", "--timeout", "1", "--workers", "1"]


def run(args):
    return subprocess.run([sys.executable, str(SCS), *args], capture_output=True, text=True).stdout


def size(out):
    m = re.search(r"model built: (\d+) vars, (\d+) constraints", out)
    assert m, out[-500:]
    return int(m.group(1)), int(m.group(2))


class FixDomain0(unittest.TestCase):
    def test_default_model_size_unchanged(self):
        # measured on main before any flag existed (pins the default model)
        self.assertEqual(size(run(BASE)), (5180, 50552))

    def test_fix_domain0_adds_only_unit_constraints(self):
        with tempfile.TemporaryDirectory() as d:
            fx = Path(d, "fx.json")
            fx.write_text(json.dumps([0, 1, 2]))
            nv, nc = size(run(BASE))
            nv2, nc2 = size(run(BASE + ["--fix-domain0", str(fx)]))
            self.assertEqual((nv2, nc2), (nv, nc + 3))


if __name__ == "__main__":
    unittest.main()
