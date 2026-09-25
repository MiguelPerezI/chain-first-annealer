# Literature check: what is known vs. new in the RP² results

Date: 2026-09-24
Scope: online investigation of the claims in `rp2_new_mathematics.md` §§2, 6–8 (Aut ≅ A₅ theorem; product-family trap; H₁ obstruction; the three publishable claims).

---

## 1. The automorphism theorem Aut(K_RP²) ≅ A₅ — **KNOWN (classical). Cite, do not claim.**

The result is classical and appears in three independent literature streams:

### Regular maps / abstract polytopes
The hemi-icosahedron is the regular map **{3,5}/2** (= {3,5}₅, Conder–Dobcsányi number **N1.2**) on the projective plane, V/E/F = 6/15/10, skeleton K₆, dual = hemi-dodecahedron (Petersen graph skeleton). Full automorphism group: **A₅, order 60**, acting simply transitively on the 60 flags.

- Coxeter & Moser, *Generators and Relations for Discrete Groups* (Springer, 1957/1980), Ch. 8 "Regular maps" — tabulates all regular maps of positive Euler characteristic (the four projective-plane ones: hemi-cube, hemi-octahedron, hemi-dodecahedron, hemi-icosahedron).
- Marston Conder's census of regular maps: <https://www.math.auckland.ac.nz/~conder/hypermaps.html>; "Regular maps and hypermaps of Euler characteristic −1 to −200" — presentation ⟨r,s,t | r², s², t², (rs)³, (st)⁵, (rt)², ststsrstsrtstr⟩ (image of the (3,5,2) triangle group).
- Regular Map database entry N1.2: <https://www.weddslist.com/rmdb/map.php?a=N1.2> (both rotational and full symmetry group listed A₅(60)). Non-orientable ⇒ no orientation-reversal distinction.
- McMullen & Schulte, *Abstract Regular Polytopes*, §6C "Projective regular polytopes."

### Triangulations of manifolds
The hemi-icosahedron is the **unique 6-vertex triangulation of RP²** (Heawood bound gives f₀ ≥ 6; uniqueness is classical). Sym(RP²₆) ≅ **A₅**, acting transitively on vertices, edges, triangles, and even on complete flags:

- Gaifullin, arXiv:2207.08507 — states it in exactly this form.
- Balagopalan, PhD thesis (Maynooth), vertex-minimal triangulations — RP²₆ unique up to relabeling, A₅ via the degree-6 icosahedral representation, 2-neighborly and complementary.
- Hemi-icosahedral Bier sphere, arXiv:2504.21345 — treats these facts as standard background.

### Group theory
The A₅ action on 6 points is the classical identification **A₅ ≅ PSL(2,5) acting on PG(1,5)** (Klein, Burnside). The 10 triangles of the hemi-icosahedron form one A₅-orbit of 3-subsets of a 6-set (the 20 triples split into two orbits of 10; odd permutations of S₆ swap the orbits — the shadow of the outer automorphism of S₆).

**Actions for the paper:** cite the triangulation stream (Gaifullin; Brehm–Kühnel for the bound context) as the primary reference, Conder N1.2 as the regular-map reference of record, Coxeter–Moser for the classical classification. Present the machine verification as a *check of our encoding of K_RP²*, not new mathematics.

**Fix in our proof sketch:** the step "an order-60 subgroup of A₆ consisting entirely of even permutations, with the quotient S₆/A₆ ≅ Z/2 having no order-60 subgroup, is necessarily a copy of A₅" is garbled (subgroups of A₆ are even by definition; the quotient argument does not parse). Clean classical replacement: a **transitive** subgroup of S₆ of order 60 is unique up to conjugation and ≅ A₅ (PSL(2,5) in its degree-6 action, 6T11 in the classification of transitive groups of degree 6). We already verify vertex-transitivity, so invoke that; alternatively exhibit the Coxeter–Moser presentation.

---

## 2. §6 — Product-family trap and A₅ symmetry breaking — **novel instance of a known generic phenomenon.**

### Known (cite as framing)
- Prestwich & Roli, "Symmetry Breaking and Local Search Spaces" (CP-AI-OR 2005) — symmetry constraints create deep new optima and shrink basins of global optima.
- Prügel-Bennett, "Symmetry Breaking in Population-Based Optimization" (IEEE TEVC 2004) — hard landscapes split into widely separated metastable regions; premature symmetry breaking as a trap.
- Alza et al., "Landscape rotation as a perturbation strategy on the QAP" (GECCO 2021) — problem symmetry governs basin structure and escape efficiency.

### Novel (ours)
No literature exists on annealing covers of this complex — the complex, the objective (cleanliness-costed chain covers), and the observed dynamics (re-convergence from a non-rectangular k = 6 warm start into the diagonal rectangle family; 540/600 forced by the far-cell count; 60-facet hole) are ours.

### ⚠️ CORRECTION REQUIRED: the residual symmetry group is D₁₀, not A₄

The write-up says the symmetry breaks "to the stabilizer of vertex 5, which is A₄ of order 12." **Wrong group.** In the degree-6 action of A₅ on the hemi-icosahedron's 6 vertices, orbit–stabilizer forces

> |stabilizer| = 60 / 6 = **10**, ≅ **D₁₀**

(the pentagon symmetries of the vertex link: the 5 rotations about the axis plus 5 involutions swapping the antipodal pair). A₄ is the point stabilizer of the **degree-5** action of A₅; the two standard actions were conflated.

The correction *repairs* the orbit claim: under D₁₀ the 10 private cells of the missing rectangle form a single orbit of size 10 (trivial stabilizer). Under A₄ it is arithmetically impossible — orbit sizes must divide 12, and 10 does not.

Also note: with the fifth domain only *approaching* star(0)×star(0) (141/150 facets), the actual plateau state's symmetry may be a proper subgroup of D₁₀; state the D₁₀ claim for the idealized rectangle family.

---

## 3. §7 — The H₁ obstruction — **machinery classical; the absorption audit is ours.**

### Known (cite as lineage)
- H₁(RP²; Z/2) = Z/2 — textbook.
- The (π₁ − π₂)_* difference-of-projections computation on a product/deleted-product complex is exactly the classical **deleted-product / van Kampen obstruction** setup (van Kampen 1933, Shapiro 1957, Wu 1965):
  - Freedman–Krushkal–Teichner, "Van Kampen's embedding obstruction is incomplete for 2-complexes in R⁴" (MRL 1994) — <https://math.berkeley.edu/~teichner/Papers/vanKampen.pdf>
  - Krushkal, "Embedding obstructions and 4-dimensional thickenings of 2-complexes" (Proc. AMS 2000).
  - Skopenkov, "On the deleted product criterion for embeddability in Rᵐ" (Proc. AMS 1998).
  - Sarkaria, "Kneser colorings of polyhedra" (Illinois J. Math. 33, 1989) — colorability/embeddability side of the same machinery.

### Novel (ours)
Using the obstruction as a **per-facet absorption audit** — "L alone passes (rank 0) but every D_j ∪ {f} is obstructed" as a machine-certifiable mechanism for why the plateau is dead — nothing comparable found in the literature. The k = 5 row (non-winding leftovers, but every single absorption creates the wind) is the crispest statement.

### Caution
Vanishing of van Kampen-type obstructions does **not** imply extendability in general (FKT counterexample). Phrase "L alone passes" strictly as a one-directional certificate — the table already does this.

---

## 4. §8 — The three publishable claims: novelty assessment

### 4.1 ρ(K_RP²) = 6 = |V| — **appears new; position against two neighbor literatures.**

No prior work found on rectangle covers of this complex or anything equivalent. Two literatures are conceptually adjacent and should be cited:

- **Rectangle/biclique cover numbers.** The rectangle covering number of a 0/1 matrix equals the chromatic number of a Lovász–Saks incompatibility graph; fooling sets give lower bounds:
  - Pourmoradnasseri–Theis, "The rectangle covering number of random Boolean matrices" (EJC 24(2), 2017).
  - Theis, "On some lower bounds on the number of bicliques needed to cover a bipartite graph" (arXiv:0708.1174).
  - Mubayi–Vishwanathan, "Bipartite Coverings and the Chromatic Number" (EJC 16(1), 2009).
  - **Key observation:** our fact "the 50 far cells (|s∩t| = 1) each have exactly one covering diagonal rectangle" **is a fooling-set structure**. State the ρ ≥ 6 lower bound via this general principle (not only via the exhaustive audit).
- **Box covers of discrete structures.**
  - Leader–Milićević–Tan, "Decomposing the complete r-graph" (JCTA 154, 2018) — covers by products of stars/boxes; the uniform-cover ≥ 2ⁿ question.
  - White, "New bounds for double covers of the discrete box {0,1,2}ᵈ" — the multiplicity-2 case (recent arXiv).
  - Alon–Bohman–Holzman–Kleitman partition bound (cited within the above).

**Open question to resolve before writing:** does ρ(K) = |V| hold for *any* 2-neighborly K (i.e., does the far-cell argument generalize)? If yes → a stronger general theorem. If no → the RP²-specificity is itself the content, and we should say why the hemi-icosahedron forces it.

### 4.2 SC_strict⁰(RP²) ≤ 5 — **new.** Nothing online; the invariant is our own definition, so the paper must fully define it before stating the bracket 2 ≤ SC_strict⁰ ≤ 5. Upper bound from the verified non-rectangular k = 6 cover (132+126+112+80+78+72) is entirely ours.

### 4.3 The A₅-symmetric trap + barrier audit + escape to 542 — **new empirical work.** Anchor with the §2 framing citations (symmetry-in-local-search literature).

---

## 5. Action items for the paper

1. **Fix the stabilizer:** A₄ → D₁₀ (order 10) in the k = 5 symmetry-breaking narrative (§6 above). This simultaneously makes the 10-cell-hole-as-single-orbit claim arithmetically valid.
2. **Fix the Aut ≅ A₅ proof sketch:** replace the S₆/A₆ quotient sentence with the transitive-degree-6 classification argument (6T11 ≅ PSL(2,5) ≅ A₅), and cite Coxeter–Moser / Conder N1.2 / Gaifullin rather than claiming novelty.
3. **Cite the classical scaffolding:** van Kampen/Shapiro/Wu/Sarkaria for §7; biclique-cover and box-cover literatures for ρ; symmetry-in-search literature for the trap.
4. **Decide the generality of ρ = |V|** before presenting it as hemi-icosahedron-specific.
5. Optionally restate ρ ≥ 6 as a fooling-set bound for elegance and robustness.

## References (URLs)

- Regular Map database N1.2: https://www.weddslist.com/rmdb/map.php?a=N1.2
- Conder's regular map/hypermap lists: https://www.math.auckland.ac.nz/~conder/hypermaps.html
- Conder, "Regular maps and hypermaps of Euler characteristic −1 to −200": https://www.math.auckland.ac.nz/~conder/preprints/regmapshypermaps.pdf
- Coxeter & Moser, *Generators and Relations for Discrete Groups*: https://link.springer.com/book/10.1007/978-3-662-25739-5
- Wikipedia, Hemi-icosahedron: https://en.wikipedia.org/wiki/Hemi-icosahedron
- Gaifullin, arXiv:2207.08507: http://arxiv.org/pdf/2207.08507
- Balagopalan PhD thesis: https://mural.maynoothuniversity.ie/id/eprint/5024/1/SoBPhDThesisFinal.pdf
- Hemi-icosahedral Bier sphere, arXiv:2504.21345: https://doi.org/10.48550/arxiv.2504.21345
- Freedman–Krushkal–Teichner: https://math.berkeley.edu/~teichner/Papers/vanKampen.pdf
- Krushkal, Proc. AMS 2000: https://www.ams.org/journals/proc/2000-128-12/S0002-9939-00-05458-7/S0002-9939-00-05458-7.pdf
- Skopenkov, Proc. AMS 1998: https://doi.org/10.1090/s0002-9939-98-04142-2
- Sarkaria, Kneser colorings of polyhedra: https://doi.org/10.1215/ijm/1255988573
- Sarkaria papers list: https://kssarkaria.org/List.htm
- Leader–Milićević–Tan, JCTA 2018: https://doi.org/10.1016/j.jcta.2017.08.008
- White, double covers of {0,1,2}ᵈ: https://arxiv.org/html/2607.09014
- Theis, biclique cover bounds: https://doi.org/10.48550/arxiv.0708.1174
- Pourmoradnasseri–Theis, EJC 2017: https://www.combinatorics.org/ojs/index.php/eljc/article/download/v24i2p37/pdf/
- Mubayi–Vishwanathan, EJC 2009: https://www.combinatorics.org/ojs/index.php/eljc/article/view/v16i1n34
- Prestwich & Roli 2005: https://www.kiphub.com/paper/61e50ae477d2412cd5f6335d
- Prügel-Bennett 2004: https://doi.org/10.1109/tevc.2003.819419
- Alza et al., QAP landscape rotation (GECCO 2021): https://dl.acm.org/doi/10.1145/3449726.3463139