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


def status(out):
    m = re.search(r"k=\d+ L=\d+: (\w+) in", out)
    assert m, out[-500:]
    return m.group(1)


class FactoredEncoding(unittest.TestCase):
    MOB = ["--complex", "mobius", "--L", "6", "--timeout", "120", "--workers", "4"]

    def test_same_verdicts_as_y_on_moebius(self):
        for k, want in (("3", "OPTIMAL"), ("2", "INFEASIBLE")):
            y = run(self.MOB + ["--k", k])
            fa = run(self.MOB + ["--k", k, "--encoding", "factored"])
            self.assertEqual(status(y), want)
            self.assertEqual(status(fa), want)
            # no size check here: with 5 triangles the one-hot costs more than
            # it saves; the RAM advantage is measured on RP2 (hybrid test below)

    def test_hybrid_free_domains_use_factored(self):
        warm = str(OPTI / "results" / "rp2_cases" / "A1_star0.json")
        args = ["--complex", "rp2", "--k", "5", "--L", "4", "--timeout", "2",
                "--workers", "2", "--warm", warm, "--free", "4"]
        self.assertLess(size(run(args + ["--encoding", "factored"]))[1], size(run(args))[1])


if __name__ == "__main__":
    unittest.main()
