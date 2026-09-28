"""A5 on the hemi-icosahedron and its diagonal action on pairs of triangles.

Finding (2026-09-27): A5 does NOT act on the staircase facets of RP2 x RP2.
Only the identity preserves the staircase facet set, because a staircase is
monotone in the vertex order. The usable symmetry is the diagonal action on
pairs of triangles (s, t): 100 pairs, 3 orbits (|s and t| = 0, 1, 2).
"""
import sys
import unittest
from pathlib import Path

OPTI = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OPTI / "src"))
sys.path.insert(0, str(OPTI / "scripts"))
import gpu_sc  # noqa: E402
from orbits import rp2_aut as A  # noqa: E402


class Aut(unittest.TestCase):
    def setUp(self):
        self.P = gpu_sc.build("rp2")
        self.G = A.automorphisms(self.P)

    def test_group_of_order_60(self):
        self.assertEqual(len(self.G), 60)
        ids = set(self.G)
        for g in self.G:
            for h in self.G:
                self.assertIn(tuple(g[h[i]] for i in range(6)), ids)

    def test_maps_triangles_to_triangles_transitively(self):
        K = set(self.P.kfacets)
        for g in self.G:
            for t in K:
                self.assertIn(tuple(sorted(g[v] for v in t)), K)
        orbit = {tuple(sorted(g[v] for v in self.P.kfacets[0])) for g in self.G}
        self.assertEqual(orbit, K)

    def test_staircase_facets_are_not_preserved(self):
        # documents the finding: the case split cannot be done facet by facet
        good = [g for g in self.G if A.preserves_facets(self.P, g)]
        self.assertEqual(good, [tuple(range(6))])

    def test_pair_orbits_are_exhaustive(self):
        reps, covered = A.pair_orbit_reps(self.P, self.G)
        self.assertEqual(covered, 100)
        self.assertEqual(len(reps), 3)
        sizes = sorted(len(set(r[0]) & set(r[1])) for r in reps)
        self.assertEqual(sizes, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
