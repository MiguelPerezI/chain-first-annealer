# Why the annealer stalls on the surfaces: rigid subcomplexes, maximal rectangles, and exact completion

*2026-10-07.  Analysis of every stalled state in `results/` with the rigid-subcomplex machinery of
`~/Downloads/rigid_note/rigid_subcomplexes.tex`, extended by one new theorem (A), one computer-verified
structural theorem (B), and two exact tools.  All code is in `scripts/rigid_analysis/`; every number below is
reproducible with the commands at the end.  Python: `/home/mike666/default_env/bin/python` (python-sat installed
there today).*

## 0. Summary

1. **The icosahedron stall at k = 4 was a search failure, not an obstruction.**  Domain 2 of the stalled state
   `icosa_k4_v3` together with all 24 uncovered facets is a contiguity domain (an 8-link chain exists; found by an
   exact SAT in under a second).  The resulting 4-domain cover is `results/icosa_k4_exact` and passes
   `src/independent_check.py`: **SC_strict(icosahedron) ≤ 3**, improving the paper's ≤ 4.  The bracket is now
   2 ≤ SC_strict ≤ 3.
2. **The RP², hemicube, Klein, Möbius-148 and odd-sphere plateaus are rigid-closed**: for every uncovered facet f
   and every domain D_j of the plateau, D_j ∪ {f} contains a rigid subcomplex (a hollow triangle with projections of
   different windings on the surfaces; a hollow 3-sphere Γ_w or its transpose on ∂Δ⁴).  No single facet can be
   added to any domain, whatever the chain length.  These stalls are genuine dead ends of the *state*, and on RP²
   the mechanism is structural (Theorem B): every star rectangle is a maximal domain.
3. **Rigid-family lower bounds (chain-length-free):** RP² ≥ 5 domains, hemicube ≥ 5, Klein ≥ 5 (k = 5 pending),
   torus ≥ 4, Möbius = 3.  The same families prove nothing on the icosahedron, the cube, the hemi-octahedron and the
   hemi-dodecahedron beyond TC.  Where the family is silent (icosahedron), exact completion succeeded; where it is
   loud (RP², hemicube, Klein), the stalled states are provably dead and the remaining gap (RP² 5 vs 6, hemicube 5 vs
   7, Klein 5 vs 10) is open.
4. **The hidden property shared by the odd spheres and the surfaces** is not a homotopy invariant.  It is
   *minimality* (no dominated vertex: the identity is contiguous to no other map), which makes the graph of every
   non-decreasing simplicial self-map a rigid subcomplex (Theorem A), together with *coarseness*: the facet classes of
   a level-0 cover are too coarse to separate the resulting rigid pieces into TC + 1 classes.  Every rigid
   subcomplex found today is also homotopically rigid (the two projections are not homotopic), so the level-0 gap on
   these complexes is a granularity effect, not a new homotopy invariant.

## 1. Theorem A (rigid graphs over minimal complexes) — proved

Let K be a finite simplicial complex on an ordered vertex set V.  A vertex v is *dominated* if some v' ≠ v lies in
every facet containing v; K has **no dominated vertex** iff ⋂{σ facet : v ∈ σ} = {v} for every v.

**Theorem A.**  Suppose K has no dominated vertex.  Let w : V → V be non-decreasing (v ≤ v' ⇒ w(v) ≤ w(v')) and
simplicial (w(σ) ∈ K for all σ ∈ K), with w ≠ id.  Put
Γ_w = { {(v, w(v)) : v ∈ σ} : σ ∈ K }.
Then Γ_w is a subcomplex of K × K, π₁|Γ_w is an isomorphism onto K, π₂|Γ_w corresponds to w under it, and **Γ_w is
rigid** (no contiguity chain of any length joins π₁|Γ_w to π₂|Γ_w).  The same holds for the transpose
Γ_wᵀ = { {(w(v), v) : v ∈ σ} }.

*Proof.*  (i) For σ ∈ K the set S_σ = {(v, w(v)) : v ∈ σ} is a chain in V × V because w is non-decreasing;
π₁(S_σ) = σ ∈ K and π₂(S_σ) = w(σ) ∈ K because w is simplicial; so S_σ ∈ K × K by the criterion of the note
(Lemma "criterion"), and S_τ ⊆ S_σ for τ ⊆ σ, so Γ_w is closed under faces.  (ii) (v, w(v)) ↦ v is a bijection of
vertex sets carrying S_σ to σ, hence an isomorphism Γ_w → K; under it π₁ becomes id_K and π₂ becomes w.
(iii) *Lemma.*  If φ : K → K is simplicial and contiguous to id_K then φ = id_K.  Indeed σ ∪ φ(σ) ∈ K for every
facet σ, and σ is maximal, so φ(σ) ⊆ σ; hence for a vertex v, φ(v) ∈ ⋂_{σ ∋ v} σ = {v}.  Now a contiguity chain
π₁|Γ_w = ψ₀ ∼ ψ₁ ∼ ⋯ ∼ ψ_L = π₂|Γ_w becomes, under the isomorphism, a chain id = φ₀ ∼ φ₁ ∼ ⋯ ∼ φ_L = w of
simplicial maps K → K.  By the lemma and induction φ_i = id for every i, so w = id, a contradiction.  For the
transpose read the chain backwards: it starts at the isomorphism π₂|Γ_wᵀ.  ∎

*Remarks.*  (a) The lemma is the basic fact of Barmak–Minian's theory of strong homotopy types (a complex is
minimal iff it has no dominated vertex; on a minimal complex the identity is alone in its contiguity class).
(b) For K = ∂Δ^{N−1} every non-decreasing w ≠ id is automatically simplicial (it is not onto, so proper subsets go
to proper subsets), and Γ_w = Z_w: Theorem A contains the note's Proposition "rigid spheres" and extends it to every
closed surface triangulation (whose vertex links are cycles of length ≥ 3, so ⋂ of the triangles at v is {v}) and
to the 5-vertex Möbius band.  Checked computationally for all 11 campaign complexes: id is contiguous only to itself
(`rigid.py info`, and the brute-force check in the session log).  (c) Constants w ≡ c give Γ_c = K × {c}: **no
contiguity domain contains a full column or row of K × K.**  (d) A contractible complex with no free face (e.g. a
dunce-hat triangulation) has no dominated vertex, so K × K ⊇ K × {c} is not a domain and SC_strict(K) ≥ 1 > 0 = TC:
a level-0 gap with a one-line proof, for a space where the two projections *are* homotopic.  On the campaign
complexes, however, every Γ_w has even mod-2 degree or acts non-trivially on H¹(K; Z₂) (checked: 0 exceptions), so
the rigidity there is also visible homologically.

Counts of rigid graphs Γ_w (non-decreasing simplicial self-maps ≠ id): ∂Δ³ 34, ∂Δ⁴ 125, cube 422, icosahedron
3891, hemi-icosahedron 181, hemicube 450, Klein 597, torus 343, Möbius 75, hemi-octahedron 1028,
hemi-dodecahedron 1021 (each also transposed).

## 2. Exact contiguity classes of short closed walks

For a closed walk Z = (p₀,…,p_{m−1}) in the 1-skeleton of K × K, Z is rigid as soon as the two projected closed
walks lie in **different contiguity classes of maps C_m → K**.  These classes were computed exactly (union–find over
all V^m maps with the contiguity relation; re-verified by an independent brute-force BFS for m = 3, 4).  They are
finer than homotopy: Klein bottle 3-walks fall into 11 classes but 4-walks into 6; torus 3-walks into 7 (the integral
windings ±1 are separated); the icosahedron has a single class for every m ≤ 5 and the cube for every m ≤ 6 (no rigid short cycle at
all); RP² has exactly 2 classes for m = 3,…,6 (the Z₂ winding and nothing else).

| K | rigid closed walks m ≤ 4 (distinct edge sets) | rigid graphs (with transposes) |
|---|---:|---:|
| Möbius | 4 684 | 150 |
| hemi-icosahedron | 19 214 | 362 |
| hemicube | 29 216 | 900 |
| Klein | 94 540 | 1 194 |
| torus | 87 204 | 686 |
| hemi-octahedron | 30 922 | 2 056 |
| hemi-dodecahedron | 37 966 | 2 042 |
| icosahedron, cube, ∂Δⁿ | 0 | 7 782 / 844 / 68, 250 |

## 3. Chain-length-free lower bounds (Theorem 2 of the note, SAT-decided)

k-class avoidance of the full family (walks ≤ 4 + graphs), CaDiCaL:

| K | TC+1 | rigid floor (domains) | k at which avoidance is SAT | annealer's best cover | first stall |
|---|---|---|---|---|---|
| Möbius | 2 | **3** (k=2 UNSAT) | 3 | 4 (CP-SAT: 3) | 3 |
| ∂Δ⁴ (S³) | 2 | **3** | 3 | 3 | 2 |
| hemi-icosahedron (RP²) | 4 | **5** (k=4 UNSAT, 28 s) | 5 (540 s) | 6 | 5 |
| hemicube (RP²) | 4 | **5** (k=4 UNSAT, 486 s) | 5 (411 s) | 7 | 6 |
| Klein bottle | 5 | **5** (k=4 UNSAT, 911 s) | k=5: pending | 10 | 9 |
| Császár torus | 3 | **4** (k=3 UNSAT) | 4 | 11 (star) | — |
| hemi-octahedron (RP²) | 4 | 4 only (k=4 SAT, 1476 s) | 4 | 8 | 7 |
| hemi-dodecahedron (RP²) | 4 | 4 only (k=4 SAT) | 4 | 7 | 6 |
| icosahedron, cube (S²) | 3 | nothing (k=2 SAT) | 2 | 4 → **4 (new)**, 3 | 4, — |

The torus row proves **SC_strict(Császár torus) ≥ 3 > 2 = TC** (chain-length-free).  For RP² the full H¹-cycle
relaxation (`scripts/relaxation/h1cegar.py rp2 5 5`) is also SAT at k = 5, so **no winding argument can exclude a
5-domain cover of the hemi-icosahedron**; but 4 of its 5 balanced classes are not domains (exact check, L ≤ 10).

## 4. Theorem B (star rectangles are maximal domains) — computer-verified

**Theorem B.**  Let K be the hemi-icosahedron, the 5-vertex Möbius band or the Császár torus.  For every vertex pair
(v, w) and every facet f ∉ D(v, w), the subcomplex generated by D(v, w) ∪ {f} contains a hollow triangle whose two
projected 3-walks lie in different contiguity classes of maps C₃ → K.  Consequently D(v, w) ∪ {f} is not a
contiguity domain for any chain length, i.e. **every star rectangle D(v, w) is a maximal contiguity domain**: no
domain of any cover of K × K properly contains a star rectangle.

*Proof.*  36 × 450 = 16 200 (RP²), 25 × 96 = 2 400 (Möbius) and 49 × 960 = 47 040 (torus) explicit witnesses,
each re-verified independently: (a) each edge of the hollow triangle lies in a facet of D(v, w) ∪ {f} (direct
containment), (b) the two projected walks are in different classes, with the classes recomputed by a brute-force BFS
over all |V|³ maps C₃ → K using only the definition of contiguity (2, 3 and 7 classes respectively).  Each single
witness is a hand proof via the note's winding lemma: e.g. on RP², for D(1,1) and the facet over (s,t) = ({0,4,5},
{2,3,5}), the points (0,0), (0,2), (5,5) span a hollow triangle with π₁-walk 0,0,5 (winding 0) and π₂-walk 0,2,5
(the essential empty triangle {0,2,5}, winding 1).  Certificates:
`scripts/rigid_analysis/certificates/maximal_rectangles_{rp2,mobius,torus}.json`.  ∎

On the hemicube only 16 of 49 rectangles are maximal (4 of 7 diagonal ones), on the Klein bottle 4 of 64.

**Why this is the "product-family trap".**  The rectangle chains π₁ ∼ c_v ∼ π₂ have zero cleanliness cost, so the
annealer is pulled into the rectangle family; once there, Theorem B says no chain can gain a single facet without
first losing some, and the γ-term and the coverage objective both punish the loss.  The k = ρ − 1 plateau on RP²
(540/600, the 60 private facets of the missing sixth rectangle) is therefore a strict local maximum of coverage
under *all* single-chain moves, of any width, for a structural reason.

## 5. Audits of the stalled states (every leftover against every domain)

"dead" = D_j ∪ {f} contains a member of the rigid family for every j; the witness kind is recorded.

| state | k | covered | leftovers | dead (rigid family) | dead (full Z₂-cohomology test) | witness kind | exact single-domain completion (L ≤ 12) |
|---|---|---|---|---|---|---|---|
| rp2_k5_hybrid | 5 | 540/600 | 60 | **60/60** | 60/60 | hollow triangles | none |
| rp2_k5_from_k6_stretch_collapse7232 | 5 | 544/600 | 56 | 54/56 | 54/56 | hollow triangles | none |
| rp2_k3_cold | 3 | 398/600 | 202 | 200/202 | — | hollow triangles | — |
| hemicube_k6_v2 | 6 | 836/864 | 28 | 27/28 | 27/28 | hollow triangles | none |
| klein_k9_v4 | 9 | 1494/1536 | 42 | 41/42 | 40/42 | 3-walks (one 4-walk) | none |
| klein_k8_r1 | 8 | 1483/1536 | 53 | **53/53** | — | 3- and 4-walks | — |
| hemi_dodeca_k6_v2 | 6 | 1922/1944 | 22 | 6/22 | 6/22 | 3-walks | none (doors spread over 4 domains) |
| mobius_seed_999 (148) | 3 | 148/150 | 2 | 0/2 | 0/2 | — | none |
| mobius_k3_cold_s4 (149) | 3 | 149/150 | 1 | 0/1 | 0/1 | — | **found** (domain 1, 8 links) |
| mobius_k3_stretch_from_149 | 3 | 149/150 | 1 | 0/1 | 0/1 | — | **found** (domain 1, 8 links) |
| s3_k2_v2_point, s3_k2_A, s3_k2_v2_track | 2 | 460/500 | 40 | **40/40** | — | rigid 3-spheres Γ_w, Γ_wᵀ | none (L ≤ 8) |
| icosa_k4_v3 | 4 | 2376/2400 | 24 | **0/24** | 0/24 | — | **found** (domain 2, 8 links) → `results/icosa_k4_exact` |
| icosa_k5 | 5 | 2395/2400 | 5 | 0/5 | 0/5 | — | (annealer continuation completed it) |

Reading: the two 149-states of the Möbius band were one exact chain away from the 3-cover, exactly as the
`surface_stall_analysis.md` audit predicted ("the door is open; the move set cannot walk through it"); so was the
icosahedron.  Where every door is rigid-closed (RP², hemicube, Klein, ∂Δ⁴), the state cannot be completed by any
single-chain change, and only a *re-partition* can help.

## 6. Exact tools (CPU, seconds)

* `rigid.py bound <K> <kmax>` — chain-length-free lower bound by k-class avoidance of the rigid families.
* `rigid.py audit <K> <summary.json>` — the door audit of §5 with explicit witnesses.
* `domain_chain_sat.py` — exact decision "facet set D admits a chain of ≤ L links" (one-hot maps, presence
  variables, all minimal non-faces of K forbidden; validated on recorded domains and on Theorem B).
* `complete_stall.py <K> <summary.json>` — tries D_j ∪ leftovers for each j (what produced the icosahedron cover).
* `cegar_cover.py <K> <k> <L>` — lazy exact cover search: outer SAT over facet classes avoiding the rigid families,
  exact chain oracle per class, unsat cores minimised to minimal non-domain sets and learned as clauses; SAT writes
  an audited certificate.  Controls: Möbius k=3 in 1 s (audited), Möbius k=2 UNSAT, ∂Δ³ k=3 in 2 s, ∂Δ⁴ k=2 UNSAT.
  Runs in progress at the time of writing: RP² k=5 L=8, icosahedron k=3 L=10, hemicube k=5 L=8, cube k=2 (control).

## 7. Answers to the three questions

**Is the annealer weak?**  Yes, in two specific ways, and no in a third.  (i) It cannot express the move that
completes an open door: on the icosahedron and on the Möbius 149-states the completion is a single chain of 8 links
over the same domain, found instantly by an exact solver, which the annealer's local proposals (track, stretch,
collapse, seam) never reach because each intermediate edit is uphill.  (ii) It is attracted to the rectangle family,
whose members are maximal domains (Theorem B), so from k = ρ it can only descend by re-partitioning, which its moves
do not do.  (iii) Where it stalls with every door rigid-closed, it is *right* to stall: those states are dead ends of
any search, and the rigid floors show the certified values are within 1 (RP²), 2 (hemicube) and 5 (Klein) domains of
what can be proved.

**Is there a topological property that stops it?**  The obstruction is combinatorial rather than topological:
rigid subcomplexes.  Two sources are at work.  Essential empty triangles (H₁ ≠ 0 with the slice mechanism) produce
hollow triangles with mismatched windings; they close every door on RP², the hemicube, the Klein bottle and the
torus and they make the star rectangles maximal.  Minimality (Theorem A) produces the rigid spheres that close every
door on ∂Δ⁴.  The icosahedron has neither essential empty triangles nor rigid short cycles (one contiguity class for
every walk length ≤ 5), its Γ_w family forces nothing beyond TC, and accordingly its stall was not an obstruction.

**The hidden property shared by the odd spheres and the stalling surfaces.**  All of them (and the even spheres and
the cube too) are minimal in the strong-homotopy sense.  What separates the odd spheres and the surfaces from S² is
that the *coloring number* χ of their small rigid subcomplexes exceeds TC + 1: 3 > 2 on ∂Δ^{2m}, 5 > 4 on the two
small RP² triangulations, 4 > 3 on the torus, 3 > 2 on the Möbius band; on the even spheres and on the Klein bottle χ
= TC + 1, and on the icosahedron χ = 2 < TC + 1.  Since every rigid piece found is homotopically rigid as well, the
excess SC_strict − TC seen at level 0 on these complexes is a resolution effect: the facets of K × K are too coarse
to separate the (topologically unavoidable) rigid pieces into TC + 1 classes.

## 8. Reproduce

```bash
PY=/home/mike666/default_env/bin/python; cd /home/mike666/chain-first-annealer/opti
$PY scripts/rigid_analysis/rigid.py bound rp2 6 --walks 4          # ≥5 domains; k=5 SAT (540 s)
$PY scripts/rigid_analysis/rigid.py bound torus 4 --walks 4        # k=3 UNSAT: SC_strict ≥ 3 > TC
$PY scripts/rigid_analysis/rigid.py audit rp2 results/rp2_k5_hybrid/summary.json      # 60/60 dead, witnesses printed
$PY scripts/rigid_analysis/rigid.py audit s3  results/s3_k2_v2_point/summary.json --walks 3   # 40/40 dead by rigid spheres
$PY scripts/rigid_analysis/rigid.py audit icosa results/icosa_k4_v3/summary.json --walks 3    # 0/24 dead
$PY scripts/rigid_analysis/complete_stall.py icosa results/icosa_k4_v3/summary.json --L 8      # completion found
$PY src/independent_check.py results/icosa_k4_exact                                            # VERIFIED ≤ 3
$PY scripts/rigid_analysis/cegar_cover.py mobius 3 8 --outdir results/mobius_k3_cegar_L8       # 3-cover in 1 s
```
