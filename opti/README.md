# Certificates for the paper

*Explicit covers and rigid obstructions for simplicial complexity: from
spheres to surfaces* (M. Perez, J. Flores); the arXiv source is in
[paper-overleaf/arxiv/](paper-overleaf/arxiv/). This file documents the
first-phase sphere, torus, wedge and cube certificates; the surface descents,
the rigid-subcomplex lower bounds and the exact completions are described in
the paper and in [../README.md](../README.md).

This repository is the companion artifact of the article. Every system
of piecewise-linear motion planners the paper reports is stored here as
data, together with the search code that found it and an auditor that
shares **no code** with the search.

## Reviewed behavior and reproducibility

Read [REVIEW_NOTES.md](REVIEW_NOTES.md) for the exact energy and state space,
corrections to the rectangle interpretation, validation guarantees and commands.
The 460/500 plateau is an observed count, not a proof of rectangular domains or
of the impossibility of two domains. Historical results below retain their
original timings. Code lives in `src/`, campaign scripts in `scripts/`, and
notebooks in `experiments/`.

The [S⁶ chain-first campaign](experiments/s6_chain_first_campaign.md) used a
bounded, resumable guided ladder. Its first sequence reached 59,072/59,136
facets; a continuation from that hashed incumbent reached full coverage and
passed the independent audit. Partial stages record no bound.

## What to read first (the verified covers)

Standalone certificate directories in `results/` contain the files below.
Campaign-managed certificates retain `cover.txt`, `planners.txt` and
`summary.json` in the successful rung, with `audit.log` and the parent
`campaign.json` providing the independent audit and provenance.

| file | contents |
|---|---|
| `cover.txt` | the domains, facet by facet, as vertex lists |
| `planners.txt` | the contiguity chain of simplicial maps on each domain (every map given as its full vertex table) |
| `definition.md` | the exact definition of K and of the product indexing used by the code (facet lists of K, grid id = nv·a+b, staircase facet enumeration, map-table format) — read this to replicate or hand-check the files |
| `summary.json` | domain sizes, chain lengths, search parameters, run time |
| `certificate.md` | a human-readable rendering of all of the above, with provenance |

The main results of the article:

| complex | bound | explicit (product) certificate | annealed (searched) certificate |
|---|---|---|---|
| circle ∂Δ² | SC ≤ 1, sharp | `results/circle_starcover` (product: only ≤ 2) | `results/circle_probe3` (9+9, 2 domains, §14.3) |
| ∂Δ³ (S²) | SC = 2, sharp | `results/s2_starcover` (54+30+12) | `results/s2_k3_v2` (54+29+13, 22 s) |
| ∂Δ⁴ (S³) | SC ≤ 2 | `results/s3_starcover` (320+140+40) | `results/s3_k3_v2` (309+130+61, 153 s) |
| ∂Δ⁵ (S⁴) | SC = 2, sharp | `results/s4_starcover` (1750+630+140, <1 s) | `results/s4_k3_v2` (1634+653+233, one 13-min epoch) |
| ∂Δ⁶ (S⁵) | SC ≤ 2 | `results/s5_starcover` (9072+2772+504, 1.2 s) | `results/s5_k3_v2` (7061+4239+1048, full coverage in epoch 2, ≈34 min) |
| ∂Δ⁷ (S⁶) | SC ≤ 2 | `results/s6_starcover` (45276+12012+1848, 5 s) | `results/s6_k3_chain_first_continuation_01/rung02` (44909+11451+2776, ≈2.5 h sequence) |
| ∂Δ⁸ (S⁷) | SC ≤ 2 | `results/s7_starcover` (219648+51480+6864, 25 s) | — |
| ∂Δ⁹ (S⁸) | SC ≤ 2 | `results/s8_starcover` (1042470+218790+25740, 2:12) | — |
| Császár torus | SC ≤ 10 | `results/torus_starcover_opt` (11 domains, ρ = 11 optimal) | — |
| wedge of two circles (§5.2) | **SC = 2, sharp** | — (product covers reach only ≤ 3: ρ = 4, §16.2) | `results/wedge_k3_v2` (18+16+16, full coverage in epoch 25, 41 s) |
| barycenter triangle (Δ² split by its barycenter) | **SC = 0, sharp** (contractible) | — (one domain covers all of K×K) | `results/bary_k1_c3` (54 facets, 1 domain, 3 maps, 3 s) |
| cube boundary (8 vertices, 12 triangles) | **SC = 2, sharp** | `results/cube_starcover` (products reach only ≤ 7: ρ = 8, §26) | `results/cube_k3_guided` (482+236+146, 22 epochs, 304 s) |

The star covers are computed in closed form (Corollary 3: `D(v) = star(v) × star(v)`, `v = 0, 1, 2`, closed-form
domain sizes) and audited end to end; S⁸ is the last `n` at which the full facet-by-facet certificate is both
writable and auditable on this machine (S⁹: 5.88 M facets, ≈0.8 GB `cover.txt`, ≈16.5 GB for the auditor). The size
wall, the closed forms and `stream_starcover.py` (the streamed verifier for that regime, cross-checked against the
audited certificates) are Section 21 of `optimized_algorithms.md`. The exact rectangle-cover numbers of the
hemi-polyhedra (all RP²), the Klein bottle and the Möbius band — and the search-free bounds
`SC_strict ≤ ρ − 1` they certify, with `star_cover.py` certificates reusable as annealer `--init` seeds — are
Section 27; the Möbius band is the first non-orientable case where the searched cover beats every product cover
(`SC_strict = 2` vs `ρ − 1 = 3`).

A second and *independent* wall sits on the search side rather than the writing side: the chain-first GPU annealer
cannot be launched on ∂Δ⁸ at all (its per-thread kernel state is 764 KB against a 512 KB limit). ∂Δ⁷ is expensive,
but a staged guided campaign with small populations produced the S⁶ certificate above in about 2.5 accumulated
hours. The launch boundary, earlier scaling probe and corrected campaign result are in `s7_chain_first_wall.md`.

For the wedge of two circles of the paper's §5.2 — like the circle, a complex where no product cover reaches the
bound (`ρ = 4`, so products give only `SC ≤ 3`) — the annealer finds 3 domains (18+16+16) with 2/5/5-link chains
in 41 s, giving the sharp `SC = 2`; and `wedge_paper_check.py` decodes the paper's own facet lists and chain tables
out of its HTML and re-verifies that certificate (20+18+12 domains, 8/11/10-link chains) with the same Definition 2.3
check. Section 23.

The **barycenter triangle** (Δ² cut by its barycenter into three triangles) is the opposite extreme: its barycenter
lies in every facet, so ‖K‖ is contractible and — as the paper notes just before defining `SC_strict` — a single
motion planner exists on all of `K × K`. The annealer finds it in 3 s as one domain (all 54 facets) with the 3-map
chain `π₁ ∼ m ∼ π₂`, the sharp `SC = 0`; the middle map it picks is not the constant `c₃` but the map that keeps the
diagonal and collapses everything else to the cone point, and both are among the exactly 16 possible middle maps.
Section 24.

The **cube boundary** (8 vertices, its six square faces cut along their drawn diagonals into 12
triangles) is a second triangulation of S² — the same space as ∂Δ³ as a different complex. It is the
first place in this report where the product family is *far* from optimal: the exact rectangle-cover
number is ρ = 8 (`rect_cover.py`, ≈2 s), so the eight-rectangle cover `results/cube_starcover`
(216+180+108+108+102+78+48+24 = 864 facets, chains of 2/4/2/4/3/3/4/4 links) certifies only
SC ≤ 7, while the sphere's own bound is 2 — and no 3-domain cover can even be made of rectangles
(each covers at most 216 facets, and 864/3 = 288). The search closes the gap: the guided k = 3 run
`results/cube_k3_guided` covers all 864 facets in 304 s and reduces to 3 domains (482+236+146, chains
5/5/4 links), giving the sharp `SC_strict(cube) = 2` at subdivision level 0 — five domains better than
any product cover. Section 26.

A third triangulation of S², the **dodecahedron boundary** (20 vertices, 54 edges, 36 triangles, χ = 2; fanning
each pentagon from one vertex leaves it far from vertex-transitive — vertex degrees run 3 to 8), is
supported by both the annealer (`--complex dodeca`) and the auditor as of 2026-09-19. The chain-first search does
**not** reach a cover there. Five runs — one cold and four warm-started rungs, 4 h 20 m of GPU — left it at **1802 of
its 7,776 product facets (23.2%)**, and the yield per identical hour *fell monotonically* across the rungs
(644 → 564 → 371 → 170 gained), so the restart route is refuted rather than merely unfinished; a 15-minute rung
confirmed the yield is front-loaded without making short rungs the better strategy (extrapolated 4 h of them ≈ 848
against the 1802 the long rungs delivered). The measured cause is
that a chain's domain requires *every* link to be simplicial: the domain floor is the diagonal cells (2.8% here,
against the cube's 8.3% and ∂Δ⁴'s 20%), coverage beyond that comes from walks across K's dual graph, and this K is
by far the sparsest in the repository — 36 triangles out of 1,140 possible vertex triples (3.2%, against the cube's
21%) with dual-graph diameter 7 (the cube's is 4). Density table, the cmax sweep, the escalation and its corrections:
`dodeca_anneal_log.md`.

Narrative certificates with the full math and every planner map:
`S2_certificate.md`, `S3_certificate.md`, `S4_certificate.md`,
`T2_certificate.md`. For S⁵ and above the same document is generated per
result — `results/s5_k3_v2/certificate.md` (annealed; 3 domains
7061 + 4239 + 1048, chains 9 / 5 / 4 links) and
`results/s{5,6,7,8}_starcover/certificate.md` (product; the S⁷/S⁸ ones render
`cover.txt` facet by facet, 27 MB / 144 MB). The complete lab notebook (failure
diagnosis of the
original greedy algorithms, energy fix, plateaus, H₁ obstruction, all
hybrid experiments) is `optimized_algorithms.md`; **§25** of it explains how to
read a chain of maps as an actual motion plan — evaluating the waypoints at a
query (vertices or interior points), joining them into a polyline in ‖K‖, with
worked examples on the circle and the barycenter triangle.

## Verify a certificate yourself (no search code involved)

```
python3 src/independent_check.py results/s4_starcover   # exit 0, prints ALL CHECKS PASSED
```

`independent_check.py` is pure Python (standard library + nothing from
the search): it rebuilds each triangulation from scratch, checks the
facet tables, disjointness and coverage, and asserts that every chain
starts at π₁, ends at π₂, is simplicial on its domain, and that
consecutive maps are 1-contiguous on **every** facet (Definition 2.3 of
arXiv:2008.13290). It exits nonzero on any discrepancy; during
development it caught real bugs.

## Reproduce everything

```
bash reproduce_all.sh        # ~2 minutes, CPU only:
                             # rebuilds + verifies + audits all explicit
                             # certificates (circle, S², S³, S⁴, torus)
bash reproduce_all.sh gpu    # additionally replays the successful chain-first
                             # annealer runs (22 s, 153 s, ~13 min, 41 s wedge,
                             # 3 s + 11 s barycenter triangle)
./scripts/run_wedge.sh k3            # the wedge of two circles alone (41 s, GPU)
python3 src/wedge_paper_check.py # re-verifies the paper's own §5.2 certificate
./scripts/run_bary.sh k1             # the barycenter triangle alone (11 s, GPU;
                             # k1c3 = 3 s with cmax 2, k3 for comparison)
./scripts/run_cube.sh k3g            # the cube boundary alone: the sharp SC = 2 (304 s, GPU;
                             # k3 = 300 s cold, k3w = the continuation, k2 = the
                             # control TC(S²) = 2 says must never cover)
```

Requires a Python with `numpy`, `numba`, `scipy` (override with
`PY=/path/to/python`); the `gpu` mode additionally needs a CUDA GPU.

## Code map

| file | role |
|---|---|
| `gpu_sc.py` | the chain-first GPU annealer (Algorithm 1 of the article; energy with β bad-link penalty, γ chain cleanliness, tracking moves) |
| `star_cover.py` | builds and verifies product-domain certificates star(v)×star(w) (Algorithm 2) |
| `stream_starcover.py` | streamed verifier for the star covers of ∂Δ^{n+1} above the size wall (never materializes K×K); `--cross-check` reproduces the audited S⁵–S⁸ domain sizes |
| `rect_cover.py` | rectangle-cover number ρ(K): ILP (scipy/HiGHS) and independent branch-and-bound |
| `rect2_optimum.py` | exhaustive optimum of the rectangle family over 1, 2, 3 rectangles (`star(v)×star(w)`), and which rectangles the recorded k=2 plateau states actually reach — the check that the sphere plateaus 84/96, 460/500, 2380/2520 *are* the 2-rectangle optima |
| `hybrid_cover.py` | hybrid experiments (fixed rectangles + annealed chains), the "wall" of Section 8 |
| `fixed_domain_test.py` | the constructive counterpart to the cover search: given a facet set, anneal **one** chain on the grid restricted to it (`hybrid_cover.SubProduct`). Full coverage ⇒ that set is a domain with an explicit planner. Used for the ∂Δ⁴ two-domain push — see s3_k2_push_log.md |
| `h1_obstruction.py`, `band_test.py` | the GF(2) H₁ obstruction test for candidate domains |
| `make_certificate_md.py` | generates `certificate.md` files from data |
| `make_definition_md.py` | generates `definition.md` for any `sN`, the wedge, the barycenter triangle or the cube boundary (pure combinatorics: facet list, product indexing, closed-form counts) |
| `wedge_paper_check.py` | reads the wedge data out of the paper's own §5.2 (facet lists and the three chain tables), decodes it into this repo's facet indexing, and re-verifies the paper's certificate with the Definition 2.3 check; standard library only (`python3 src/wedge_paper_check.py`, no numpy/numba/search code) |
| `independent_check.py` | the code-independent auditor |
| `opti_sc.py` | CPU implementation of the optimized greedy (reference) |
| `article.html`, `article.txt` | local copies of Ortiz–Lara–González–Borat (arXiv:2008.13290v2), the pipeline this work diagnoses and replaces |
| `logs/` | run logs of the reported experiments |
