import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

OPTI = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OPTI / "src"))
sys.path.insert(0, str(OPTI / "paper-overleaf" / "paper"))
import gpu_sc  # noqa: E402
import scs_sat  # noqa: E402


class Cases(unittest.TestCase):
    def test_a1_case_is_the_star_rectangle(self):
        with tempfile.TemporaryDirectory() as d:
            subprocess.run([sys.executable, str(OPTI / "scripts/orbits/make_cases.py"),
                            "--out", d], check=True, capture_output=True)
            cases = json.loads(Path(d, "cases.json").read_text())
            a1 = [c for c in cases if c["family"] == "A1"]
            self.assertEqual(len(a1), 1)
            self.assertEqual(len(a1[0]["domain0_facets"]), 150)
            self.assertEqual(a1[0]["residual"], 450)
            NV, NG, NF, FAC, TRIS, _, _, _ = scs_sat.build_complex("rp2")
            warm = Path(a1[0]["warm_file"])
            warm = warm if warm.is_absolute() else OPTI / warm
            chains, doms, _, _ = scs_sat.load_warm(str(warm), NV, NG, NF, FAC, TRIS)
            self.assertEqual(sorted(doms[0]), sorted(a1[0]["domain0_facets"]))
            self.assertEqual(len(chains[0]), 3)   # pi1, constant, pi2

    def test_a2_case_pins_facets_instead_of_a_warm_chain(self):
        # A frozen chain always covers its whole domain, so a warm file cannot
        # express "star(0) x star(0) minus some cells": load_warm recomputes the
        # full 150. A2 must be a cold case that pins its facets (--fix-domain0).
        with tempfile.TemporaryDirectory() as d:
            subprocess.run([sys.executable, str(OPTI / "scripts/orbits/make_cases.py"),
                            "--out", d], check=True, capture_output=True)
            a1 = json.loads(Path(d, "cases.json").read_text())[0]
            cells = a1["domain0_facets"][:2]
            orbits = Path(d, "a2_orbits.json")
            orbits.write_text(json.dumps({"x": cells}))
            subprocess.run([sys.executable, str(OPTI / "scripts/orbits/make_cases.py"),
                            "--out", d, "--a2-orbits", str(orbits)],
                           check=True, capture_output=True)
            a2 = [c for c in json.loads(Path(d, "cases.json").read_text())
                  if c["family"] == "A2"]
            self.assertEqual(len(a2), 1)
            self.assertNotIn("warm_file", a2[0])
            want = sorted(set(a1["domain0_facets"]) - set(cells))
            self.assertEqual(a2[0]["domain0_facets"], want)
            self.assertEqual(a2[0]["residual"], 600 - 148)
            pin = Path(a2[0]["fix_domain0_file"])
            pin = pin if pin.is_absolute() else OPTI / pin
            self.assertEqual(json.loads(pin.read_text()), want)


if __name__ == "__main__":
    unittest.main()
