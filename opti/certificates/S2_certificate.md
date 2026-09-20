# Verified 3-domain motion-planner system on ∂Δ³ (S²): SC_strict ≤ 2

Source: `results/s2_k3_v2` (summary.json, cover.txt, planners.txt), re-verified here with the plain Definition 2.3 check (`gpu_sc.verify_system`) and audited by `Abstract_Motion_GPU/independent_check.py`.

## How it was obtained

Found by the chain-first GPU annealer (`gpu_sc.py`, Kernel A of optimized_algorithms.md §12) with the corrected energy of §14 — `python gpu_sc.py --complex s2 --k 3 --cmax 20 --S 4096 --iters 20000 --beta 5 --gamma 0.5 --ptrack 0.3 --seed 51` — which reached full coverage of the 96 facets in its first epoch (21.9 s wall, 8.2 × 10⁷ moves on a Tesla T4, run `s2_k3_v2` of `push_s3_v2.sh`). The three chains' domains (54, 52, 48 facets, overlapping) were made disjoint by assigning each facet to the first chain covering it (`partition_from_chains`), the chains were shortened with Algorithm 2 (`reduce_chain`) to 7, 3 and 4 links, and the system was verified with the plain Definition 2.3 check and audited by `independent_check.py` (exit 0). The same result (3 domains, SC_strict(∂Δ³) ≤ 2 = TC(S²)) also follows without search from the product construction of §16: `star(0)×star(0)`, `star(1)×star(1)`, `star(2)×star(2)` with two-link chains (`python star_cover.py s2 <outdir> 0,0 1,1 2,2`).

## Setup

`K` = ∂Δ³ (S²): 4 vertices `0..3`, 4 facets:

```
  facet  0: [0, 1, 2]
  facet  1: [0, 1, 3]
  facet  2: [0, 2, 3]
  facet  3: [1, 2, 3]
```

`K × K` is the staircase (ordered) product on the `4 × 4` grid; grid vertex `(a, b)` has id `4a + b`; the 96 maximal facets have 5 vertices. Facet ids enumerate pairs of facets `(i, j)` of `K` (`i` slowest), then the staircase paths as bit words with as many 1‑bits as `dim(facet_j)` in increasing numerical order, read from the lowest bit (0 = step in the first factor, 1 = step in the second). `π₁(a, b) = a`, `π₂(a, b) = b`.

## Summary

| part | facets | chain links |
|---:|---:|---:|
| J0 | 54 | 7 |
| J1 | 29 | 3 |
| J2 | 13 | 4 |

Total 96 = 96 facets, pairwise disjoint.


## Part J0 — 54 facets, 7 links

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
  0 2 0 0
  2 2 2 1
  2 2 2 2
  3 2 2 2
```
map 2:
```
  0 1 2 0
  1 1 0 2
  2 0 0 2
  2 2 2 2
```
map 3:
```
  2 2 0 0
  1 1 2 2
  2 1 2 2
  0 1 2 2
```
map 4:
```
  2 2 0 0
  2 2 3 2
  2 2 3 2
  1 1 2 2
```
map 5:
```
  2 0 3 0
  2 0 3 2
  2 0 3 3
  0 2 2 0
```
map 6:
```
  0 2 2 3
  0 2 2 3
  2 2 2 3
  0 2 2 3
```
map 7:
```
  0 1 2 3
  0 1 2 3
  0 1 2 3
  0 1 2 3
```

Facets of J0 (id: grid vertices):
```
     0: (0,0) (0,1) (0,2) (1,2) (2,2)
     1: (0,0) (0,1) (1,1) (1,2) (2,2)
     2: (0,0) (1,0) (1,1) (1,2) (2,2)
     3: (0,0) (0,1) (1,1) (2,1) (2,2)
     4: (0,0) (1,0) (1,1) (2,1) (2,2)
     5: (0,0) (1,0) (2,0) (2,1) (2,2)
    12: (0,0) (0,2) (0,3) (1,3) (2,3)
    13: (0,0) (0,2) (1,2) (1,3) (2,3)
    14: (0,0) (1,0) (1,2) (1,3) (2,3)
    15: (0,0) (0,2) (1,2) (2,2) (2,3)
    16: (0,0) (1,0) (1,2) (2,2) (2,3)
    17: (0,0) (1,0) (2,0) (2,2) (2,3)
    18: (0,1) (0,2) (0,3) (1,3) (2,3)
    19: (0,1) (0,2) (1,2) (1,3) (2,3)
    20: (0,1) (1,1) (1,2) (1,3) (2,3)
    21: (0,1) (0,2) (1,2) (2,2) (2,3)
    22: (0,1) (1,1) (1,2) (2,2) (2,3)
    23: (0,1) (1,1) (2,1) (2,2) (2,3)
    48: (0,0) (0,1) (0,2) (2,2) (3,2)
    49: (0,0) (0,1) (2,1) (2,2) (3,2)
    50: (0,0) (2,0) (2,1) (2,2) (3,2)
    51: (0,0) (0,1) (2,1) (3,1) (3,2)
    52: (0,0) (2,0) (2,1) (3,1) (3,2)
    53: (0,0) (2,0) (3,0) (3,1) (3,2)
    60: (0,0) (0,2) (0,3) (2,3) (3,3)
    61: (0,0) (0,2) (2,2) (2,3) (3,3)
    62: (0,0) (2,0) (2,2) (2,3) (3,3)
    63: (0,0) (0,2) (2,2) (3,2) (3,3)
    64: (0,0) (2,0) (2,2) (3,2) (3,3)
    65: (0,0) (2,0) (3,0) (3,2) (3,3)
    66: (0,1) (0,2) (0,3) (2,3) (3,3)
    67: (0,1) (0,2) (2,2) (2,3) (3,3)
    68: (0,1) (2,1) (2,2) (2,3) (3,3)
    69: (0,1) (0,2) (2,2) (3,2) (3,3)
    70: (0,1) (2,1) (2,2) (3,2) (3,3)
    71: (0,1) (2,1) (3,1) (3,2) (3,3)
    72: (1,0) (1,1) (1,2) (2,2) (3,2)
    73: (1,0) (1,1) (2,1) (2,2) (3,2)
    74: (1,0) (2,0) (2,1) (2,2) (3,2)
    75: (1,0) (1,1) (2,1) (3,1) (3,2)
    76: (1,0) (2,0) (2,1) (3,1) (3,2)
    77: (1,0) (2,0) (3,0) (3,1) (3,2)
    84: (1,0) (1,2) (1,3) (2,3) (3,3)
    85: (1,0) (1,2) (2,2) (2,3) (3,3)
    86: (1,0) (2,0) (2,2) (2,3) (3,3)
    87: (1,0) (1,2) (2,2) (3,2) (3,3)
    88: (1,0) (2,0) (2,2) (3,2) (3,3)
    89: (1,0) (2,0) (3,0) (3,2) (3,3)
    90: (1,1) (1,2) (1,3) (2,3) (3,3)
    91: (1,1) (1,2) (2,2) (2,3) (3,3)
    92: (1,1) (2,1) (2,2) (2,3) (3,3)
    93: (1,1) (1,2) (2,2) (3,2) (3,3)
    94: (1,1) (2,1) (2,2) (3,2) (3,3)
    95: (1,1) (2,1) (3,1) (3,2) (3,3)
```

## Part J1 — 29 facets, 3 links

Planner maps on the vertices of J1 ('@' = grid vertex not in the domain); rows `a = 0..3`, columns `b = 0..3`. Map 0 is `π₁`, the last map is `π₂`; consecutive maps are 1‑contiguous on every facet of J1.

map 0:
```
  0 0 0 0
  1 1 1 1
  2 2 @ 2
  3 3 3 3
```
map 1:
```
  3 0 3 0
  3 1 1 1
  3 3 @ 3
  3 3 3 3
```
map 2:
```
  0 1 0 0
  0 1 0 3
  0 1 @ 3
  0 1 0 3
```
map 3:
```
  0 1 2 3
  0 1 2 3
  0 1 @ 3
  0 1 2 3
```

Facets of J1 (id: grid vertices):
```
    24: (0,0) (0,1) (0,2) (1,2) (3,2)
    25: (0,0) (0,1) (1,1) (1,2) (3,2)
    26: (0,0) (1,0) (1,1) (1,2) (3,2)
    27: (0,0) (0,1) (1,1) (3,1) (3,2)
    28: (0,0) (1,0) (1,1) (3,1) (3,2)
    29: (0,0) (1,0) (3,0) (3,1) (3,2)
    30: (0,0) (0,1) (0,3) (1,3) (3,3)
    31: (0,0) (0,1) (1,1) (1,3) (3,3)
    32: (0,0) (1,0) (1,1) (1,3) (3,3)
    33: (0,0) (0,1) (1,1) (3,1) (3,3)
    34: (0,0) (1,0) (1,1) (3,1) (3,3)
    35: (0,0) (1,0) (3,0) (3,1) (3,3)
    37: (0,0) (0,2) (1,2) (1,3) (3,3)
    38: (0,0) (1,0) (1,2) (1,3) (3,3)
    39: (0,0) (0,2) (1,2) (3,2) (3,3)
    40: (0,0) (1,0) (1,2) (3,2) (3,3)
    41: (0,0) (1,0) (3,0) (3,2) (3,3)
    54: (0,0) (0,1) (0,3) (2,3) (3,3)
    55: (0,0) (0,1) (2,1) (2,3) (3,3)
    56: (0,0) (2,0) (2,1) (2,3) (3,3)
    57: (0,0) (0,1) (2,1) (3,1) (3,3)
    58: (0,0) (2,0) (2,1) (3,1) (3,3)
    59: (0,0) (2,0) (3,0) (3,1) (3,3)
    78: (1,0) (1,1) (1,3) (2,3) (3,3)
    79: (1,0) (1,1) (2,1) (2,3) (3,3)
    80: (1,0) (2,0) (2,1) (2,3) (3,3)
    81: (1,0) (1,1) (2,1) (3,1) (3,3)
    82: (1,0) (2,0) (2,1) (3,1) (3,3)
    83: (1,0) (2,0) (3,0) (3,1) (3,3)
```

## Part J2 — 13 facets, 4 links

Planner maps on the vertices of J2 ('@' = grid vertex not in the domain); rows `a = 0..3`, columns `b = 0..3`. Map 0 is `π₁`, the last map is `π₂`; consecutive maps are 1‑contiguous on every facet of J2.

map 0:
```
  0 0 0 0
  1 1 1 1
  2 2 @ 2
  @ 3 3 3
```
map 1:
```
  0 1 0 1
  1 1 1 1
  1 1 @ 0
  @ 1 1 1
```
map 2:
```
  1 1 1 1
  1 1 1 3
  1 1 @ 3
  @ 1 2 1
```
map 3:
```
  0 1 3 3
  1 1 2 3
  0 1 @ 3
  @ 1 2 3
```
map 4:
```
  0 1 2 3
  0 1 2 3
  0 1 @ 3
  @ 1 2 3
```

Facets of J2 (id: grid vertices):
```
     6: (0,0) (0,1) (0,3) (1,3) (2,3)
     7: (0,0) (0,1) (1,1) (1,3) (2,3)
     8: (0,0) (1,0) (1,1) (1,3) (2,3)
     9: (0,0) (0,1) (1,1) (2,1) (2,3)
    10: (0,0) (1,0) (1,1) (2,1) (2,3)
    11: (0,0) (1,0) (2,0) (2,1) (2,3)
    36: (0,0) (0,2) (0,3) (1,3) (3,3)
    42: (0,1) (0,2) (0,3) (1,3) (3,3)
    43: (0,1) (0,2) (1,2) (1,3) (3,3)
    44: (0,1) (1,1) (1,2) (1,3) (3,3)
    45: (0,1) (0,2) (1,2) (3,2) (3,3)
    46: (0,1) (1,1) (1,2) (3,2) (3,3)
    47: (0,1) (1,1) (3,1) (3,2) (3,3)
```
