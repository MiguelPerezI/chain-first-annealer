# Verified 3-domain motion-planner system on barycenter triangle (cone on ∂Δ²): SC_strict ≤ 2

Source: `results/bary_k3` (summary.json, cover.txt, planners.txt), re-verified here with the plain Definition 2.3 check (`gpu_sc.verify_system`) and audited by `independent_check.py` (shipped in this repository).

## How it was obtained

Chain-first GPU annealer (gpu_sc.py, beta 5 / gamma 0.5 / track moves); see optimized_algorithms.md and the barycenter run log.

## Setup

`K` = barycenter triangle (cone on ∂Δ²): 4 vertices `0..3`, 3 facets:

```
  facet  0: [0, 1, 3]
  facet  1: [0, 2, 3]
  facet  2: [1, 2, 3]
```

`K × K` is the staircase (ordered) product on the `4 × 4` grid; grid vertex `(a, b)` has id `4a + b`; the 54 maximal facets have 5 vertices. Facet ids enumerate pairs of facets `(i, j)` of `K` (`i` slowest), then the staircase paths as bit words with as many 1‑bits as `dim(facet_j)` in increasing numerical order, read from the lowest bit (0 = step in the first factor, 1 = step in the second). `π₁(a, b) = a`, `π₂(a, b) = b`.

## Summary

| part | facets | chain links |
|---:|---:|---:|
| J0 | 25 | 3 |
| J1 | 20 | 3 |
| J2 | 9 | 3 |

Total 54 = 54 facets, pairwise disjoint.


## Part J0 — 25 facets, 3 links

Planner maps on the vertices of J0 ('@' = grid vertex not in the domain); rows `a = 0..3`, columns `b = 0..3`. Map 0 is `π₁`, the last map is `π₂`; consecutive maps are 1‑contiguous on every facet of J0.

map 0:
```
  0 0 0 0
  1 1 1 1
  2 2 2 2
  3 3 3 3
```
map 1:
```
  0 1 1 3
  1 1 1 3
  2 1 2 3
  3 3 3 3
```
map 2:
```
  0 3 2 3
  3 1 2 3
  0 1 2 3
  0 3 3 3
```
map 3:
```
  0 1 2 3
  0 1 2 3
  0 1 2 3
  0 1 2 3
```

Facets of J0 (id: grid vertices):
```
     0: (0,0) (0,1) (0,3) (1,3) (3,3)
     1: (0,0) (0,1) (1,1) (1,3) (3,3)
     2: (0,0) (1,0) (1,1) (1,3) (3,3)
     3: (0,0) (0,1) (1,1) (3,1) (3,3)
     4: (0,0) (1,0) (1,1) (3,1) (3,3)
     5: (0,0) (1,0) (3,0) (3,1) (3,3)
    11: (0,0) (1,0) (3,0) (3,2) (3,3)
    13: (0,1) (0,2) (1,2) (1,3) (3,3)
    14: (0,1) (1,1) (1,2) (1,3) (3,3)
    15: (0,1) (0,2) (1,2) (3,2) (3,3)
    16: (0,1) (1,1) (1,2) (3,2) (3,3)
    17: (0,1) (1,1) (3,1) (3,2) (3,3)
    23: (0,0) (2,0) (3,0) (3,1) (3,3)
    26: (0,0) (2,0) (2,2) (2,3) (3,3)
    28: (0,0) (2,0) (2,2) (3,2) (3,3)
    29: (0,0) (2,0) (3,0) (3,2) (3,3)
    36: (1,0) (1,1) (1,3) (2,3) (3,3)
    37: (1,0) (1,1) (2,1) (2,3) (3,3)
    39: (1,0) (1,1) (2,1) (3,1) (3,3)
    48: (1,1) (1,2) (1,3) (2,3) (3,3)
    49: (1,1) (1,2) (2,2) (2,3) (3,3)
    50: (1,1) (2,1) (2,2) (2,3) (3,3)
    51: (1,1) (1,2) (2,2) (3,2) (3,3)
    52: (1,1) (2,1) (2,2) (3,2) (3,3)
    53: (1,1) (2,1) (3,1) (3,2) (3,3)
```

## Part J1 — 20 facets, 3 links

Planner maps on the vertices of J1 ('@' = grid vertex not in the domain); rows `a = 0..3`, columns `b = 0..3`. Map 0 is `π₁`, the last map is `π₂`; consecutive maps are 1‑contiguous on every facet of J1.

map 0:
```
  0 0 0 0
  1 @ 1 1
  2 2 2 2
  3 3 3 3
```
map 1:
```
  3 0 0 0
  3 @ 1 3
  2 3 2 3
  3 3 3 3
```
map 2:
```
  3 1 0 0
  3 @ 3 3
  3 0 3 3
  0 3 3 3
```
map 3:
```
  0 1 2 3
  0 @ 2 3
  0 1 2 3
  0 1 2 3
```

Facets of J1 (id: grid vertices):
```
     6: (0,0) (0,2) (0,3) (1,3) (3,3)
     7: (0,0) (0,2) (1,2) (1,3) (3,3)
     8: (0,0) (1,0) (1,2) (1,3) (3,3)
     9: (0,0) (0,2) (1,2) (3,2) (3,3)
    10: (0,0) (1,0) (1,2) (3,2) (3,3)
    18: (0,0) (0,1) (0,3) (2,3) (3,3)
    19: (0,0) (0,1) (2,1) (2,3) (3,3)
    20: (0,0) (2,0) (2,1) (2,3) (3,3)
    21: (0,0) (0,1) (2,1) (3,1) (3,3)
    22: (0,0) (2,0) (2,1) (3,1) (3,3)
    24: (0,0) (0,2) (0,3) (2,3) (3,3)
    25: (0,0) (0,2) (2,2) (2,3) (3,3)
    27: (0,0) (0,2) (2,2) (3,2) (3,3)
    38: (1,0) (2,0) (2,1) (2,3) (3,3)
    40: (1,0) (2,0) (2,1) (3,1) (3,3)
    41: (1,0) (2,0) (3,0) (3,1) (3,3)
    42: (1,0) (1,2) (1,3) (2,3) (3,3)
    44: (1,0) (2,0) (2,2) (2,3) (3,3)
    46: (1,0) (2,0) (2,2) (3,2) (3,3)
    47: (1,0) (2,0) (3,0) (3,2) (3,3)
```

## Part J2 — 9 facets, 3 links

Planner maps on the vertices of J2 ('@' = grid vertex not in the domain); rows `a = 0..3`, columns `b = 0..3`. Map 0 is `π₁`, the last map is `π₂`; consecutive maps are 1‑contiguous on every facet of J2.

map 0:
```
  @ 0 0 0
  1 @ 1 1
  @ 2 2 2
  @ 3 3 3
```
map 1:
```
  @ 3 3 3
  3 @ 1 1
  @ 3 3 3
  @ 3 3 3
```
map 2:
```
  @ 1 2 1
  3 @ 2 1
  @ 3 2 3
  @ 1 2 3
```
map 3:
```
  @ 1 2 3
  0 @ 2 3
  @ 1 2 3
  @ 1 2 3
```

Facets of J2 (id: grid vertices):
```
    12: (0,1) (0,2) (0,3) (1,3) (3,3)
    30: (0,1) (0,2) (0,3) (2,3) (3,3)
    31: (0,1) (0,2) (2,2) (2,3) (3,3)
    32: (0,1) (2,1) (2,2) (2,3) (3,3)
    33: (0,1) (0,2) (2,2) (3,2) (3,3)
    34: (0,1) (2,1) (2,2) (3,2) (3,3)
    35: (0,1) (2,1) (3,1) (3,2) (3,3)
    43: (1,0) (1,2) (2,2) (2,3) (3,3)
    45: (1,0) (1,2) (2,2) (3,2) (3,3)
```
