# EP Implementation Plan — Evra-Parzanchevski 2024 for Qutrit Clifford+D

**Reference:** Evra & Parzanchevski, *Arithmeticity, thinness and efficiency of qutrit Clifford+T gates*, arXiv:2401.16120v2 (Nov 2024). Cited below as "EP".

**Supporting:** Parzanchevski & Sarnak, *Ramanujan complexes and Golden Gates in PU(3)*, arXiv:1810.04710v3. Cited as "PS".

**Status:** plan written 2026-05-12. Phase 1 module written and all 9 unit tests pass:
`unified/hrsa/ep_level.py` — ord_π valuation in Z[ξ, 1/3], level function ℓ(g),
exact arithmetic on the canonical gates H, S, T, D. Verified:
- ℓ(I) = 0, ℓ(S) = 0, ℓ(T) = 0, ℓ(D) = 0 for any D-gate.
- ℓ(H) = 6.
- ord_π(3) = 6 (canonical), ord_π(1-ξ) = 1, ord_π is a valuation (additive).
- N(1-ξ) = 3 (consistent with cyclotomic theory).
- distribution of ℓ(H·D·H) over 728 D-gates: {0: 26, 6: 54, 8: 162, 10: 486} —
  matches EP §3 prediction (some D's in C∩HCH⁻¹ stabilize Hv_0, rest don't).

---

## 0. Scope and what's actually in the EP paper

EP gives **two algorithms** at different levels of explicitness:

- **(EXACT)** Theorem 2.8(2) — Bass-Serre descent on the Π-adic Bruhat-Tits tree T. Given γ ∈ Γ (i.e. an exact element of `U(3)` over `Z[ζ_9, 1/3]`), produce a word in `{H} ∪ D` representing γ. This is **fully constructive** in §2: each descent step uses (i) the level function ℓ, (ii) the C-stabilizer = 1944 monomial matrices, (iii) the H-multiplication trick, (iv) a transversal of the 108 v_0-clans.
- **(APPROX)** Theorem 4.14 — for continuous target g ∈ PU(3), an ε-approximation in `{H} ∪ D` exists with length ≤ K·log_ρ(1/ε) for any K > log_3(105) ≈ 4.236, with ρ = √105. The proof is **non-constructive**: it goes through the Ramanujan/spectral-gap machinery (Theorem 4.2, Corollary 4.5). The covering is proved via automorphic representation theory and is NOT turned into an explicit rounding algorithm in the paper.

> **Key tactical implication:** the **constructive content we can implement directly is the exact synthesis algorithm of §2**. To get approximate synthesis, we will need to bolt our own rounding step on top of it (round the continuous target into Γ, then run exact synthesis). The rounding step is the only genuinely novel mathematical work — everything else is implementation of EP's §2.

This is consistent with the memory note `evra_parzanchevski_2024.md`: "The rounding step (lattice reduction in Z[ζ_9] + norm equation in Z[α]) for approximate input — **this step is the MISSING INGREDIENT** for actually running it on a continuous-angle target."

We have a partial answer for the rounding step already: zeta9's `find_roots_exact_v2.py` solves the norm equation, and the zeta9 row-1 LLL enumeration approximates a row of U over Z[ζ_9, 1/3] for given ε. This is **exactly the missing ingredient** in usable form for SU(3) row-1; extending to full SU(3) is what we owe.

---

## 1. Mathematical setup (EP §2.1, §2.2)

For our case (specialized in EP §2 after Prop 2.2):

| symbol | meaning | concrete value |
|---|---|---|
| ξ | ζ_9 = exp(2πi/9), primitive 9th root | |
| σ | ξ + ξ⁻¹ = 2 cos(2π/9) | algebraic integer; minimal poly x³−3x+1 over Q |
| F | Q(σ), the totally real subfield | degree 3 over Q |
| E | Q(ξ), the CM extension | degree 6 over Q |
| O_F | Z[σ] | PID, unit group ⟨-1⟩ × ⟨1-σ⟩ × ⟨σ⟩ |
| O_E | Z[ξ] | PID, unit group ⟨-ξ⟩ × ⟨1+ξ⟩ × ⟨1+ξ²⟩ |
| π | 1 - ξ, prime in O_E | π̄π = 2 - σ in O_F, with (1-ξ)⁶ = 3 · unit; Π = ππ̄, the prime in O_F above 3 |
| Π | the prime in O_F above 3; ramified in E | Π³ ~ 3 in O_F; Π is ramified in E |
| Γ | G(O_F[1/Π]) ⊂ U(3) | the **3-arithmetic** projective unitary group; all our Clifford+D matrices live here |
| ord_Π | the Π-adic valuation, extends to matrices by min entry | foundational ingredient |
| ℓ(g) | 2·N · ord_Π gg* − 2·ord_Π det g, simplifies to **−2·ord_Π g** for unitary g | (EP eq 2.2); the **level map** = distance from v_0 in T |

**Key facts from §2:**

- T is a 4-regular tree (because Π is ramified in E and N_{F/Q}(Π) = 3, so the tree degree is N+1 = 4).
- v_0 = the base vertex, stabilizer K_Π = G(O_{F_Π}).
- **C := Γ ∩ K_Π = G(O_F)** = monomial matrices with entries in ⟨-ξ⟩ (Lemma 2.3). Size: 3! · 18³ = **6·5832 = 34992** as a subgroup of GU(3,Z); modulo center ⟨-ξ⟩ (size 18) this gives C_0 := C/center of size 1944.
- |C_0| = 1944 = 2 · 18³ / 6 → wait, the paper has |C| in GU(3) = 6 · 18³ = 34992, and the central subgroup ⟨-ξ⟩ ∼ Z/18 acts by scalar; |C_0| = 34992/18 = **1944** (confirmed in EP §3.1 explicitly).
- **ℓ(H) = 6**: the Hadamard takes v_0 to a vertex at distance 6 (i.e., "depth 3" in the tree).
- **{H} ∪ D ⊆ C_3 ∪ ... ⊆ Γ**: every D-gate has ℓ = 0 (lives in C_0); H has ℓ = 6.
- **The descent recipe (Theorem 2.8 proof)**: given γ ∈ Γ with ℓ(γ) > 0,
  1. Compute s = ℓ(γ)/2 (the tree distance from v_0).
  2. By Theorem 2.7(2), there exists c_1 ∈ C such that ℓ(γ · c_1 · H · v_0, v_0) < ℓ(γ v_0, v_0) — i.e., **multiplying γ on the right by c_1·H reduces the tree distance by ≥ 2**.
  3. Find c_1: search over the 108 representatives of C/C∩HCH⁻¹ (these are the 108-clan transversal). For each, multiply γ → γ · c_1 · H, recompute ord_Π, and accept if the level dropped.
  4. Repeat. After r ≤ ℓ(γ)/2 steps, γ has level 0, hence is in C, hence is monomial, hence is a D-gate up to S_3 ⋉ ⟨-ξ⟩ stuff.

- **PS Golden Gates §3.3 optimization (out of scope for Phase 1-3 but a Phase 4 optimization):** instead of trying all 108 c_1's, the **p-adic Iwasawa decomposition** of γ uniquely identifies which c_1 reduces distance, in one step rather than 108. This is the constant-factor speedup that gets us closer to "polylog" rather than "108·polylog".

---

## 2. Milestones

### Milestone 1 (Phase 1): Π-adic valuation and the level function
**File:** `unified/hrsa/ep_level.py`

**Input:** a 3×3 matrix M with entries in `Z[ξ, 1/3]` (represented either via ringZ9 coefficients or as a 6-tuple of integers over the Z-basis {1, ξ, ξ², ξ³, ξ⁴, ξ⁵} plus a 3-power denominator).

**Output:**
- `ord_pi(α)` for α ∈ `Z[ξ, 1/3]`: the unique integer k such that α = π^k · u with u a Π-adic unit (u ∈ O_{E,Π}^×).
- `ord_pi_mat(M)`: minimum of `ord_pi(M_ij)` over all entries.
- `level(g)` = ℓ(g) per EP eq 2.2.
- A verification harness that proves ℓ(H) = 6, ℓ(D_gate) = 0, ℓ(I) = 0 for our actual matrices.

**Key data structure:** since `Z[ξ, 1/3]` is dense in Q_Π (the completion), we represent elements as `(int_coefs[6], denom_power_of_3)`. The Π-adic valuation of (a + b ξ + c ξ² + ...) is computed via the **Newton polygon of the rational integer N_{E/Q}(α) = norm**. For our ring, α has ord_π = k iff π^k | α in O_E, which is testable in O_E directly via repeated division by π = 1 - ξ.

**LOC estimate:** ~250 lines. Self-contained, depends only on numpy (and conceptually on the ringZ9chi multiplication, but we can do everything in pure Python via integer coefficients for the first cut).

**Tests:**
- `ord_pi(0)` = +∞ (sentinel).
- `ord_pi(1)` = 0.
- `ord_pi(3)` = 6 (since π⁶ ~ 3).
- `ord_pi(1 - ξ)` = 1.
- `ord_pi(ξ)` = 0 (unit).
- `level(I)` = 0, `level(D_a,b)` = 0 for any sign-extended D-gate.
- `level(H)` = 6.

**Why this is foundational and not Iwasawa:** without ord_Π you cannot compute distances in the tree, identify v_0-stabilizers, or do any descent step. **Iwasawa is a downstream optimization** to choose the right c_1 in one step instead of 108. We do not need it for the Phase 2 brute-force descent.

---

### Milestone 2: The 1944-element C_0 stabilizer and the 108-clan transversal
**File:** `unified/hrsa/ep_clan.py`

**Input:** the abstract structure of C_0 = M_3 ⋉ D ≅ S_3 ⋉ (Z/18Z)³ (Lemma 2.3).

**Output:**
- An enumeration of all 1944 elements as 3×3 monomial matrices over Z[ξ] (each entry is ±ξ^a for some a ∈ {0..8}, in fixed permutation pattern).
- The subgroup C ∩ HCH⁻¹ (size 18² = 324 per Theorem 2.7 proof).
- The 108 coset representatives of C / (C ∩ HCH⁻¹). These are the **108-clan transversal**: when descending by 2 in T, exactly one of these 108 right-multipliers reduces the level.

**Key data structure:** Each C_0 element ↔ (S_3 permutation, (a_0, a_1, a_2) ∈ (Z/18)³). Store as `(perm: tuple[int,int,int], signed_powers: tuple[int,int,int])` where signed_power = (sign, exponent) packed into [0,18). 1944 such tuples; cache as a list. Matrix realization via `numpy.complex128` + ringZ9chi representation.

**LOC estimate:** ~200 lines.

**Tests:**
- |C_0| = 1944. Group closure: for random pairs c, c' in C_0, verify c·c' ∈ C_0 (i.e., is also monomial-with-entries-in-⟨-ξ⟩).
- Verify HCH⁻¹ ∩ C has size 324 by direct enumeration.
- Verify the 108 clan reps generate distinct cosets.

---

### Milestone 3: Brute-force tree descent (the actual exact-synthesis algorithm)
**File:** `unified/hrsa/ep_descent_v2.py` (the existing `ep_descent.py` is greedy nearest-net, will be retired or renamed).

**Input:** γ ∈ Γ (exact matrix with entries in Z[ξ, 1/3], represented either via ringZ9 ints or via numpy + denominator-tracking).

**Output:** a word in `{H} ∪ D` (list of D-gate parameters + H positions) such that the product equals γ exactly.

**Algorithm (EP Theorem 2.8 proof, made constructive):**
```
def synthesize_exact(gamma):
    word = []
    while level(gamma) > 0:
        # Find c_1 in 108-clan such that level(c_1^{-1} gamma) drops
        for c in C_108:
            candidate = c.inv @ gamma  # or: gamma * c * H, depending on convention
            if level(candidate) < level(gamma):
                gamma = candidate
                word.append(('C', c))
                # then multiply by H
                gamma = H.inv @ gamma
                word.append(('H',))
                break
        else:
            raise RuntimeError("no clan element reduces level — paper claims this is impossible")
    # gamma now has level 0, hence is in C_0; decompose it as a D-gate word
    word += decompose_C0(gamma)
    return reversed(word)  # since we accumulated inverses
```

The "decompose_C0" step at the end is independent and easy: a level-0 element is monomial with entries in ⟨-ξ⟩, hence a product of one permutation matrix and one D-gate.

**LOC estimate:** ~300 lines + ~100 for the C_0 → D-word leaf decomposition.

**Tests:**
- Round-trip on every Clifford in our 648-cache: synthesize H · D · H, recompute, recover original γ.
- Round-trip on the bidir BFS f=2, f=3 reified V's from HRSA (these are all valid γ's that we already know how to express in H·D words via HRSA — so we can cross-check word lengths).
- The expected word length should be `≈ 2·log_3(N_{E/Q}(det γ))` or something close; let's check against bidir empirical for a few targets.

---

### Milestone 4: Iwasawa decomposition (the 108→1 optimization)
**File:** `unified/hrsa/ep_iwasawa.py`

**Background:** PS §3.3 navigation algorithm; for our case (Π ramified in E), use the Iwasawa decomposition of GL_3(Q_Π[ξ]) per PS §3.3 final paragraph: "for p ≡ 3 (mod 4), one should use the Iwasawa decomposition of GL_3(Q_p[i]) instead of GL_3(Q_p)" — same situation for us.

**Input:** γ ∈ Γ with ℓ(γ) > 0.

**Output:** a *single* c_1 ∈ C_108 (rather than searching all 108) that reduces ℓ(γ) by 2.

The Iwasawa decomposition writes g = B · K where B is upper-triangular (Borel) and K is in the maximal compact. The Borel part identifies the unique vertex closest to γ·v_0, hence picks c_1.

**LOC estimate:** ~400 lines (this is the hardest single module — needs careful Π-adic precision tracking).

**Tests:** for each γ where Phase 3 found a c_1 by brute force, verify Iwasawa picks the same c_1 (or one in the same coset of C ∩ HCH⁻¹).

**Why this comes after Phase 3, not before:** Phase 3 is unblocked by Phase 1+2 alone, and yields a working exact-synthesis algorithm — just 108× slower per descent step. We can validate the entire EP pipeline before sinking time into Iwasawa.

---

### Milestone 5: Rounding step (continuous target → element of Γ)
**File:** `unified/hrsa/ep_rounding.py`

**Input:** a continuous target U_target ∈ SU(3), and a precision parameter ε.

**Output:** a γ ∈ Γ such that ‖γ - U_target‖_F < ε.

This is the **novel research piece**, not in EP. The strategy:
- Use existing zeta9 norm-equation machinery (`/home/hlamm/Desktop/efficent_gates/unified/zeta9/`) for the row-by-row Diophantine approximation in Z[ξ, 1/3].
- The first row needs ‖r_1 - target_r_1‖ < O(ε); then the second row is constrained by the unitarity condition to a lattice of dim 2 in Z[ξ]; the third row is then determined up to a phase by orthogonality.
- The min ‖.‖_F over γ ∈ Γ is bounded below by Q ε for some constant Q because the lattice spacing in Γ is finite at scale ε; we don't expect to hit ε exactly but should hit O(ε) reliably.

**LOC estimate:** ~600 lines. **This is the highest-risk module.** The lattice-rounding step is genuinely new.

**Tests:** for random U_target ∈ SU(3), verify the rounding produces a γ ∈ Γ within ε and that downstream synthesis (Phase 3) produces a polylog-length word.

---

### Milestone 6: Integration and word-count benchmarks
**File:** `unified/hrsa/ep_compile.py`

Glue: target U_target + ε → round → synthesize → emit word. Add CLI compatible with the existing `hrsa --json` schema. Measure word lengths at ε = 10⁻², 10⁻⁵, 10⁻⁸, 10⁻¹⁰. Compare against the theorem 4.14 bound 4.236·log_√105(1/ε) ≈ 1.86·log_3(1/ε).

**LOC estimate:** ~150 lines.

---

## 3. Re-used vs net-new infrastructure

| component | source | status |
|---|---|---|
| 648-element Clifford cache | `clifford_cache.h` + `decompose.cpp` | re-use as-is |
| 5184 sign-extended Clifford net | `e0_net_dump` → `/tmp/e0_net_5184.txt` | re-use; serves as C_0 ∩ {sign-extended-Cliffords} sanity check |
| ringZ9 / ringZ9chi exact arithmetic over Z[ξ, 1/3] | `cyclotomic_int9.{cpp,h}` | re-use, but **may need Python bindings** for the EP modules; alternative is to write a pure-Python integer-coef ringZ9 class for these modules |
| Galois conjugation, field norm | `ringZ9::GaloisAut, fieldNorm` | re-use |
| ord_Π for ring elements | **net new** — but trivial given Z[ξ] integer-coef representation | Milestone 1 |
| 1944-element C_0 enumeration | **net new** | Milestone 2 |
| 108-clan transversal | **net new** | Milestone 2 |
| Tree-descent loop | **net new** | Milestone 3 |
| Decompose level-0 γ into D-word | partial — `decompose.cpp` already does this for level-0 Cliffords, may need extension to S_3 ⋉ D | Milestone 3 |
| Π-adic Iwasawa | **net new (hardest)** | Milestone 4 |
| Diophantine rounding into Γ | partial — zeta9's `find_roots_exact_v2.py` does norm-equation row-1; **extending to 3 rows is net new** | Milestone 5 |
| Bidir BFS as cross-check oracle | `bidir_bfs.cpp` | re-use as test ground truth for short words |
| HRSA full pipeline as comparison baseline | existing | re-use; EP should match HRSA for short words and beat it asymptotically |
| zeta9 norm method as comparison baseline | existing | re-use; complementary, different gate-count tradeoff |

---

## 4. Mathematical definitions glossary

For implementers reading this without the paper:

- **Π-adic valuation, ord_Π:** the unique additive map O_F → Z (extended to F → Z ∪ {∞}) such that ord_Π(Π) = 1 and ord_Π(unit) = 0. Extended to matrices entrywise via minimum: `ord_Π(M) = min_{i,j} ord_Π(M_{ij})`. For us: a fast computation is "divide entry by 1-ξ in O_E until you can't anymore, count the divisions; floor-divide by 3 to project from ord_π in E down to ord_Π in F."

- **Bruhat-Tits tree T:** an infinite 4-regular tree on which Γ acts by isometries. Vertex v_0 = G(O_{F_Π})/G(O_F). The "level" ℓ(γ) of a group element = 2·d(γv_0, v_0). Edges correspond to π·v_0-lattices vs v_0-lattices of one step.

- **Iwasawa decomposition (over Q_Π):** for g ∈ GL_3(Q_Π), write g = B·K uniquely with B in the Borel subgroup (upper triangular, diagonal entries ∈ Q_Π^×) and K in the maximal compact GL_3(O_{F_Π}). In our 4-regular tree setting, B has only 4 possible "shapes" mod K, corresponding to the 4 edges of the tree at v_0.

- **108-clan:** EP §2 defines, for v ∈ L_T, a v-clan as a set of 9 vertices in the 6-sphere S_6(v) that share a common grandparent at distance 4 (and lie in the 6-sphere). |S_6(v)| = 4·3⁵ = 972 = 108 × 9. The 108 clans partition S_6(v); Theorem 2.7 says C (the stabilizer of v_0) takes Hv_0 to a representative of each of the 108 clans.

- **Bass-Serre normal form (EP §4.1):** Γ = C_0 *_{C_D} C_3 (amalgamated product). Every γ has a unique presentation γ = c_0 c_1 c_2 ... c_r with c_0 ∈ C_0 and (c_{2j-1}, c_{2j}) ∈ T_3' × T_0' (specific transversals).

- **Almost-optimal almost-cover (a.o.a.c., EP Def 4.1):** a sequence of sets X_r with |X_r| → ∞ is a.o.a.c. of a Lie group L if `μ(L \ B(X_r, ε_r)) → 0` for ε_r = polylog(|X_r|)/|X_r|. This is the spectral-gap/Ramanujan property.

---

## 5. Honest assessment of effort

### How long?

I estimate **8–14 weeks of full-time engineering work**, broken as:

| milestone | optimistic | pessimistic |
|---|---|---|
| 1 (ord_Π / level) | 3 days | 10 days |
| 2 (C_0 + 108-clan) | 5 days | 15 days |
| 3 (brute-force descent) | 2 weeks | 5 weeks |
| 4 (Iwasawa optimization) | 2 weeks | 8 weeks |
| 5 (rounding step) | 3 weeks | 10 weeks |
| 6 (integration) | 1 week | 2 weeks |

The wide range reflects that **Phase 5 (rounding) is genuine research**: nothing in EP, PS, or our existing code does this directly. Phase 4 is also nontrivial because Π-adic precision tracking is subtle.

### Where does the math get really hard?

1. **Phase 4 (Iwasawa over a CM-extension at a ramified prime):** PS §3.3 handles this for E = Q[i] at p ≡ 1 mod 4 (split) and p ≡ 3 mod 4 (inert). Our case is **ramified** (Π in O_E above 3 in O_F is ramified), which PS does not directly address. We will need to carefully follow the building-vs-tree correspondence (`B = B̃#` fixed section under involution) and adapt their Iwasawa to our ramified tree. This is doable but bookkeeping-heavy.

2. **Phase 5 (Diophantine rounding for SU(3)):** memory note `evra_parzanchevski_2024.md` already flags this: "approximate-rounding step (the only genuinely new research): LLL/BKZ in the rank-6 Minkowski lattice with linear half-space cuts from the ε-tube." For SU(3), the rank of the relevant lattice is 6 (Z[ξ] has Z-rank 6) times 3 rows times 2 (real/imag per entry) modulo unitarity constraints. We have a partial precedent: zeta9 already does this for **row 1**, and we have `find_roots_exact_v2.py` for the norm equation. Generalizing to full SU(3) is research-grade but tractable.

3. **What's not hard:** Phases 1, 2, 3, 6 are pure implementation work with a clear algorithmic spec.

### Are there blockers that should change our mind?

**Blocker 1:** *No public EP code exists.* This means we either (a) re-derive everything from the paper (current plan), or (b) email Evra & Parzanchevski. Option (b) is the highest-leverage thing we could do in parallel — even a Sage proof-of-concept from them would compress Phase 1-4 by weeks. **Recommendation: do (b) NOW**, regardless of whether we proceed with (a).

**Blocker 2:** *Iwasawa over ramified-prime Q_Π is mathematically subtle.* If we hit a wall on Phase 4, the **fallback is to keep the brute-force 108-clan search** from Phase 3. The asymptotic complexity is the same up to a 108 constant; the result is still O(log(1/ε)) word length. We'd only suffer a 108× compile-time factor, not asymptotic. **This means Phase 4 is OPTIONAL for first-pass viability.**

**Blocker 3:** *The rounding step (Phase 5) might require iterative/heuristic search rather than a closed-form algorithm.* If so, we lose some of EP's polylog compile-time guarantee but still get the polylog gate-count guarantee. **Acceptable.** Gate count is what dominates the physics resource estimate.

**Conclusion:** the plan is **tractable for one engineer in 2 months**. The headline result (qutrit Clifford+D synthesis at ε=10⁻¹⁰ with polylog word length) is achievable with high confidence by Phase 3 + a heuristic Phase 5. Phase 4 is an asymptotic-constant optimization that we should consider optional.

---

## 6. Phase 1 deliverable (this session)

`unified/hrsa/ep_level.py` — pure-Python module with the following API:

```python
# Cyclotomic integer in Z[ξ, 1/3]:
class Z9Frac:
    coefs: tuple[int, int, int, int, int, int]   # over basis (1, ξ, ξ², ξ³, ξ⁴, ξ⁵)
    denom_pow3: int                                # the denominator is 3^denom_pow3
    def __mul__, __add__, __neg__, __eq__         # basic arithmetic
    def conjugate() -> "Z9Frac"                    # ξ → ξ⁻¹ (complex conj)
    def field_norm() -> Fraction                  # N_{E/Q} → Q
    def ord_pi() -> int                            # the π-adic valuation
    def ord_Pi() -> int                            # the Π-adic valuation (= ord_π / 1 since Π ramified)
    def to_complex() -> complex                    # numerical embedding for verification

def ord_Pi_mat(M: list[list[Z9Frac]]) -> int       # min over entries
def det_Z9Frac(M) -> Z9Frac
def level(g: list[list[Z9Frac]]) -> int            # ℓ(g) per EP eq 2.2

# Canonical gates as Z9Frac matrices:
def H_matrix() -> list[list[Z9Frac]]
def D_matrix(a: int, b: int, c: int, signs: tuple[int,int,int]) -> list[list[Z9Frac]]
def S_matrix() -> list[list[Z9Frac]]
def T_matrix() -> list[list[Z9Frac]]
```

with unit tests verifying ℓ(H) = 6, ℓ(D) = 0, ℓ(I) = 0, and that ord_Π is a valuation (sums, products behave correctly).

---

## 7. Open questions / followups

- **Q1:** EP cites the algorithm "in a single step using the p-adic Iwasawa decomposition – this is described in [PS] §3.3" (proof of Theorem 2.8(2)). PS §3.3 covers split/inert cases; for ramified Π we need the Iwasawa over Q_Π[ξ] not Q_Π[i]. Is this spelled out anywhere? — Possibly in PS Lubotzky-Phillips-Sarnak references; needs literature check.

- **Q2:** Theorem 4.14 of EP gives the existence of an ε-approximation; the proof goes through Cor 4.5 and Prop 4.7(2). Is there an implicit algorithm in those proofs that we missed? — Re-read carefully.

- **Q3:** What's the asymptotic constant in Phase 5 (rounding)? EP gives 4.236·log_3(1/ε); is that achievable, or does our rounding incur an additional factor? — Empirical Phase 6 question.

- **Q4:** Can the existing zeta9 row-1 LLL be used as-is for Phase 5, or does it need to be generalized?

- **Q5:** Are there alternative "Π-adic Iwasawa" references? Bruhat-Tits and Tits' original papers? Cassels' "Local Fields"? — Phase 4 reading list.
