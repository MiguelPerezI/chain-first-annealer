"""Converter from scs_sat.py witnesses to audited certificates."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

OPTI = Path(__file__).resolve().parents[1]
CONV = OPTI / "scripts" / "sat_witness_to_certificate.py"
WARM = OPTI / "results" / "mobius_k5" / "summary.json"


def run_conv(args):
    return subprocess.run([sys.executable, str(CONV), *args],
                          capture_output=True, text=True)


def scs_order():
    sys.path.insert(0, str(OPTI / "paper-overleaf" / "paper"))
    import scs_sat
    NV, NG, NF, FAC, TRIS, _, _, _ = scs_sat.build_complex("mobius")
    return scs_sat.load_warm(str(WARM), NV, NG, NF, FAC, TRIS)


def truncated(chain, L):
    # what solve_hybrid puts in the witness for a frozen chain
    return [chain[min(i, len(chain) - 1)] for i in range(L + 1)]


class LoadWarm(unittest.TestCase):
    def test_domains_are_paired_with_their_own_chains(self):
        # complete summaries store domains sorted by size, not in chain order
        import numpy as np
        sys.path.insert(0, str(OPTI / "src"))
        import gpu_sc
        P = gpu_sc.build("mobius")
        chains, domains, _, _ = scs_order()
        for c, d in zip(chains, domains):
            own = set(int(f) for f in P.domain_of(np.array(c, np.int8)))
            self.assertLessEqual(set(d), own)


class Converter(unittest.TestCase):
    def test_cold_witness_certifies(self):
        s = json.loads(WARM.read_text())
        with tempfile.TemporaryDirectory() as d:
            w = Path(d) / "w.json"
            w.write_text(json.dumps({"chains": s["best_chains"],
                                     "domains": s["domains"]}))
            r = run_conv([str(w), f"{d}/cert", "--complex", "mobius",
                          "--k", "5", "--L", "40"])
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertTrue((Path(d) / "cert" / "cover.txt").is_file())

    def test_hybrid_restores_truncated_frozen_chains(self):
        chains, _, _, order = scs_order()
        L = 4
        witness = [truncated(chains[i], L) for i in order[:4]] + [chains[order[4]]]
        with tempfile.TemporaryDirectory() as d:
            w = Path(d) / "w.json"
            w.write_text(json.dumps({"chains": witness, "domains": []}))
            r = run_conv([str(w), f"{d}/cert", "--complex", "mobius", "--k", "5",
                          "--L", str(L), "--warm", str(WARM), "--free", "1"])
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_truncated_without_warm_is_rejected(self):
        chains, _, _, order = scs_order()
        witness = [truncated(chains[i], 4) for i in order]
        with tempfile.TemporaryDirectory() as d:
            w = Path(d) / "w.json"
            w.write_text(json.dumps({"chains": witness, "domains": []}))
            r = run_conv([str(w), f"{d}/cert", "--complex", "mobius",
                          "--k", "5", "--L", "4"])
            self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
            self.assertRegex(r.stderr, r"certificate rejected|do not cover")


if __name__ == "__main__":
    unittest.main()
