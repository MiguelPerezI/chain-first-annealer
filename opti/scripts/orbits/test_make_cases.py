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
            chains, doms, _, _ = scs_sat.load_warm(a1[0]["warm_file"], NV, NG, NF, FAC, TRIS)
            self.assertEqual(sorted(doms[0]), sorted(a1[0]["domain0_facets"]))
            self.assertEqual(len(chains[0]), 3)   # pi1, constant, pi2


if __name__ == "__main__":
    unittest.main()
