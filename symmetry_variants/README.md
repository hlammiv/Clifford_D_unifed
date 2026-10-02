# Symmetry-variant selection (2026-10-02)

For an exact approximant V of a diagonal target R_z(θ), every V' = D V D† with D = diag(±ζ^a, ±ζ^b, ±ζ^c), and its transpose, is an equally accurate exact approximant: 324 conjugation classes × 2 = 648 copies. The canonical reducer decomposes them to different words, so we keep the cheapest.

- `variant_search.py`: generate the copies exactly (ring arithmetic on the 6-integer groups), decompose with the fast reducer, and write per-copy counts.
- `analyze_variants.py`: summarise as-is vs best per symmetry subset, plus fits. Output: `summary_full30.txt`.
- Data: 30 angles × 7 f-levels × 648 copies = 136,080 decompositions of Nick's matrices: `full30_f4-8_candidates.csv.gz`, `full30_f10-16_candidates.csv.gz`.

## Result
- **A constant saving of ~16–18 T per rotation at every f**: 24% at f = 4 down to 7.6% at f = 16.
- **The slope is unchanged:**
  - as given: 18.8 + 9.76·log₃(1/ε);
  - best copy: 2.5 + 9.71·log₃(1/ε).
- **Where it comes from:** the best copy removes ~1.4 level-4 gadgets and ~1 R, i.e. the boundary syllables.
- **Which symmetries contribute:**
  - ω-phase (Clifford) conjugations: nothing;
  - ζ-phases: most of the gain;
  - signs (±): the rest;
  - transpose: nothing.
- **All copies of a matrix have identical ε** (checked for all 210 matrices).
