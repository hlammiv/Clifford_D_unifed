#include "decompose.h"
#include "clifford_cache.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <cassert>
#include <cmath>
#include <climits>
#include <set>
#include <unordered_map>
#include <omp.h>

using namespace std;

// =========================================================================
//  Global runtime flag: enable / disable the local rewrite-rule post-pass
//  applied at the tail of decompose().  Definition lives here; the simplifier
//  itself is defined further down (after decompose()).  HRSA_test.cpp toggles
//  this from main() when the user passes --no-simplify on the command line.
// =========================================================================
bool g_decompose_simplify = true;
// Enable 1-ply look-ahead in trySinglePrefix.  Among prefixes tied at the min
// local D-cost that drop sde_chi by 1, pick the one whose resulting V leaves
// the cheapest follow-up move available.  Doubles the per-peel work but
// avoids the "one bad early greedy choice = ~30 wasted syllables" failure mode.
bool g_decompose_lookahead = true;
// Alternate front-back iteration over the outer x_1 loop in HRSA_bestD.
// 6-cell A/B (2026-05-10) showed alt-order is net negative: it tanked
// θ=0.7854 ε=0.005 by +16 N_D (alt-order=36 vs no-alt=20) by interleaving
// front and back picks, scattering the budget away from the best front
// sub-cluster.  Larger 28-cell ML study (Agent 4) confirms RF-rank gating
// strictly dominates alt-order at every cost budget.  Default OFF; opt-in
// via --alt-order if you want to A/B again.
bool g_hrsa_alt_order = false;
// Mod-3 filter on x_1 candidates (Kalra-Mosca-Valluri 2023 Theorem 3.7).
// When enabled, retain only x_1 candidates with sum(a_1) ≡ 0 (mod 3),
// which is the f(1) ≡ 0 (mod 3) derivative-vanishing condition for χ²
// divisibility. Empirically prunes ~90% of candidates with strong N_D
// bias; 28-cell agent study (memory note hrsa_mod3_filter_pending.md)
// at f=2,3 showed 27/28 cells preserve a top-30% candidate. Default OFF
// pending f=4 validation; opt-in via --mod3-filter on HRSA_tester CLI.
//
// Citation: Kalra, Mosca, Valluri, "Synthesis and Arithmetic of Single
// Qutrit Circuits", arXiv:2311.08696 / Quantum 9, 1647 (2025).
bool g_hrsa_mod3_filter = false;
// Phase 3.5 — RF-augmented sub-cluster gating.  If >0, after entryEnumeration
// rank x_1 candidates by (a) distance to target (existing order) and (b)
// trained Random Forest predicted N_D (rf_features + rf_model).  Iterate the
// UNION of the top-K from each list.  K=0 = disabled (default until f=4
// validation confirms RF generalizes).  Per Agent 4 28-cell back-test:
//   union(top-10, RF-top-10) → D-gap 0.46, 25/28 perfect, ~79% cost.
int g_hrsa_rf_gate = 0;

// =========================================================================
//  Mat3 member methods are now templated and defined inline in decompose.h.
// =========================================================================

// =========================================================================
//  Gate matrices
// =========================================================================

// Helper: zeta_9^k as ringZ9chi (sde=0)
static ringZ9chi zeta9_power(int k) {
    k = ((k % 9) + 9) % 9;
    if (k < 6) {
        return ringZ9chi(ringZ9(1, k), 0);
    }
    // k = 6,7,8: use zeta_9^{6+j} = -zeta_9^j - zeta_9^{3+j}
    int j = k - 6;
    int arr[9] = {0,0,0,0,0,0,0,0,0};
    arr[j] = -1;
    arr[j+3] = -1;
    return ringZ9chi(arr, 0);
}

// omega^k = zeta_9^{3k}
static ringZ9chi omega_power(int k) {
    return zeta9_power(3 * k);
}

Mat3 gateH() {
    // H = (1/(1+2*omega)) * DFT_3
    // 1/(1+2*omega) = (-1 - 2*zeta_9^3)/3
    // H_{jk} = (-1 - 2*zeta_9^3)/3 * omega^{jk}
    //
    // sde_chi(H) = 3 because 1+2*omega = u*chi^3 for unit u in Z[xi]
    // (since sqrt(-3) = i*sqrt(3) and |chi|^3 = sqrt(|chi|^6) relates to 3)
    int ia_arr[9] = {-1, 0, 0, -2, 0, 0, 0, 0, 0};
    ringZ9chi ia(ia_arr, 1);  // (-1-2*zeta_9^3)/3

    Mat3 H;
    for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k)
            H.m[j][k] = ia * omega_power(j * k);
    return H;
}

Mat3 gateD_clifford(int a, int b, int c) {
    // Clifford diagonal: Diag(omega^a, omega^b, omega^c)
    Mat3 D;
    D.m[0][0] = omega_power(a);
    D.m[1][1] = omega_power(b);
    D.m[2][2] = omega_power(c);
    return D;
}

Mat3 gateDgate() {
    // The non-Clifford D gate: Diag(zeta_9, 1, zeta_9^{-1})
    Mat3 D;
    D.m[0][0] = zeta9_power(1);
    D.m[1][1] = ringZ9chi(ringZ9(1), 0);
    D.m[2][2] = zeta9_power(8);  // zeta_9^{-1} = zeta_9^8
    return D;
}

// Full cyclotomic diagonal: Diag(xi^a, xi^b, xi^c)
static Mat3 gateDcyclo(int a, int b, int c) {
    Mat3 D;
    D.m[0][0] = zeta9_power(a);
    D.m[1][1] = zeta9_power(b);
    D.m[2][2] = zeta9_power(c);
    return D;
}

// R = Diag(1, 1, -1)
static Mat3 gateR() {
    Mat3 R;
    R.m[0][0] = ringZ9chi(ringZ9(1), 0);
    R.m[1][1] = ringZ9chi(ringZ9(1), 0);
    R.m[2][2] = ringZ9chi(ringZ9(-1), 0);
    return R;
}

Mat3 gateX() {
    // X = cyclic permutation: |j> -> |j+1 mod 3>
    Mat3 X;
    ringZ9chi one(ringZ9(1), 0);
    X.m[0][2] = one;
    X.m[1][0] = one;
    X.m[2][1] = one;
    return X;
}

// =========================================================================
//  sde_chi computations
// =========================================================================

// sdeChiZ9<T>, sdeChiFull<T>, buildUnitary<T>, and decompose<T> are defined
// as function templates in decompose_impl.h.  At the bottom of this TU we
// explicitly instantiate them for T=int (the HRSA hot path) so the existing
// non-templated callers continue to link without changes.
//
// (See decompose_impl.h for the templated bodies; the cpp_int instantiation
// is emitted by decompose_cli.cpp which #includes the impl header.)

// =========================================================================
//  Decomposition algorithm (following Kalra et al. Theorem 5.7)
// =========================================================================

// The peeling step tries all H * Diag(xi^a1, xi^a2, xi^a3) * R^eps * X^delta
// and finds one that reduces sde_chi of the first column by 1.
//
// From Kalra: sde(H*D*z) = sde(z) + 3 - k, where k = gde of the
// first coordinate of chi^{3+f} * H * D * z.
// We need k >= 4 for sde to decrease by 1.

// Helper: build a single prefix matrix  H * Dcyclo(a1,a2,a3) * R^eps * X^delta
static Mat3 buildPrefix(int a1, int a2, int a3, int eps, int delta,
                         const Mat3& Hmat, const Mat3& Rmat, const Mat3& Xmat) {
    Mat3 prefix = gateDcyclo(a1, a2, a3);
    if (eps == 1) prefix = prefix.mul(Rmat);
    if (delta == 1) prefix = prefix.mul(Xmat);
    else if (delta == 2) prefix = prefix.mul(Xmat).mul(Xmat);
    return Hmat.mul(prefix);
}

// (Original prefix_times_V_00 — now provided as decompose_impl_detail::
// prefix_times_V_00<T> in decompose_impl.h.  Removed here as unused.)

// count_d_from_cyclo / count_d_from_syllable — external linkage so
// decompose_impl.h (in another TU) can call them.  Per Kalra et al.
// (arXiv:2311.08696), R-syllables also cost +1 N_D.
int count_d_from_cyclo(int a1, int a2, int a3) {
    int d = 0;
    if (a1 % 3 != 0) d++;
    if (a2 % 3 != 0) d++;
    if (a3 % 3 != 0) d++;
    return d;
}

int count_d_from_syllable(int a1, int a2, int a3, int eps) {
    return count_d_from_cyclo(a1, a2, a3) + (eps != 0 ? 1 : 0);
}

// =========================================================================
//  Prefix-matrix cache (OPT)
//
//  trySinglePrefix / tryDoublePrefix iterate over 9*9*9*2*3 = 4374 prefix
//  matrices  P(a1,a2,a3,eps,delta) = H * Dcyclo(a1,a2,a3) * R^eps * X^delta.
//  The prefixes are determined entirely by these 5 indices and the fixed
//  gate matrices H, R, X — they do NOT depend on V.  Yet `buildPrefix`
//  was being invoked for every prefix on every peel iteration (and many
//  times more inside `tryDoublePrefix`'s nested loop), each call doing
//  ~4 Mat3 multiplications = ~108 ringZ9chi multiplications.
//
//  Caching the table once (per process) saves an estimated ~470K ringZ9chi
//  multiplies *per peel iteration*, against ~14K mults' worth of "real
//  search work" inside the iteration body.  Single-digit-percent measurement
//  noise notwithstanding, this should be a >5x speedup of decompose() on the
//  general (non-fast-path) branch — and `decompose()` is invoked once per
//  HRSA candidate inside an OpenMP parallel-for in householder_search.cpp,
//  so caching is justified globally.
//
//  Static-local initialization is thread-safe in C++11+ (function-local
//  static "magic statics"), so the table is correctly shared across HRSA's
//  parallel decompose() calls.
//
//  Index layout: idx = ((eps * 3 + delta) * 9 + a1) * 81 + a2 * 9 + a3
//  with eps in {0,1}, delta in {0,1,2}, a1,a2,a3 in {0..8}.  Total 4374.
// =========================================================================
// PrefixEntry is now declared in decompose.h so the templated body in
// decompose_impl.h can see it.
constexpr int N_PREFIX = 4374;  // 2 * 3 * 9 * 9 * 9

namespace {
inline int prefix_index(int a1, int a2, int a3, int eps, int delta) {
    return ((eps * 3 + delta) * 9 + a1) * 81 + a2 * 9 + a3;
}
}  // namespace

// Externally visible: get_prefix_table() — used by the templated decompose
// body in decompose_impl.h.
const std::vector<PrefixEntry>& get_prefix_table() {
    static const std::vector<PrefixEntry> table = []() {
        Mat3 Hmat = gateH();
        Mat3 Rmat = gateR();
        Mat3 Xmat = gateX();
        std::vector<PrefixEntry> tbl(N_PREFIX);
        for (int eps = 0; eps < 2; ++eps)
        for (int delta = 0; delta < 3; ++delta)
        for (int a1 = 0; a1 < 9; ++a1)
        for (int a2 = 0; a2 < 9; ++a2)
        for (int a3 = 0; a3 < 9; ++a3) {
            int idx = prefix_index(a1, a2, a3, eps, delta);
            PrefixEntry& e = tbl[idx];
            e.P = buildPrefix(a1, a2, a3, eps, delta, Hmat, Rmat, Xmat);
            e.d = count_d_from_syllable(a1, a2, a3, eps);
            e.a1 = a1; e.a2 = a2; e.a3 = a3;
            e.eps = eps; e.delta = delta;
        }
        return tbl;
    }();
    return table;
}

// ============================================================
// trySinglePrefix / tryDoublePrefix / decompose / countDgates
// MOVED to decompose_impl.h as function templates.  This TU
// pulls in the impl header and explicitly instantiates the
// int (Mat3) versions at the bottom of the file.
// ============================================================
#if 0  // ---- begin obsolete int-only originals (kept for reference) ----
static bool trySinglePrefix(Mat3& V, int s, DecompResult& result,
                             const Mat3& /*Hmat*/, const Mat3& /*Rmat*/, const Mat3& /*Xmat*/) {
    const std::vector<PrefixEntry>& table = get_prefix_table();
    int best_d = INT_MAX;
    int best_idx = -1;

    // Phase A: locate the min local D-cost among prefixes that reduce sde by 1.
    // Iterate in the original order (eps, delta, a1, a2, a3) so behaviour with
    // look-ahead disabled matches the original greedy bit-exactly.
    for (int idx = 0; idx < N_PREFIX; ++idx) {
        const PrefixEntry& e = table[idx];
        if (e.d >= best_d) continue;

        ringZ9chi new00 = prefix_times_V_00(e.P, V);
        if (sdeChiFull(new00) == s - 1) {
            best_d = e.d;
            best_idx = idx;
            if (best_d == 0 && !g_decompose_lookahead) goto found;
            if (best_d == 0) break;  // with look-ahead, still need Phase B
        }
    }

    if (best_d == INT_MAX) return false;

    // Phase B (look-ahead): among all prefixes tied at best_d that reduce sde,
    // pick the one whose post-application V leaves the cheapest follow-up move.
    // If we've already reached sde 0 (target_s == 0), no follow-up is needed,
    // so just keep best_idx from Phase A.
    if (g_decompose_lookahead && (s - 1) > 0) {
        int target_s = s - 1;
        int best_followup_d = INT_MAX;

        for (int idx = 0; idx < N_PREFIX; ++idx) {
            const PrefixEntry& e = table[idx];
            if (e.d != best_d) continue;

            ringZ9chi new00 = prefix_times_V_00(e.P, V);
            if (sdeChiFull(new00) != target_s) continue;

            // Score this tied winner by the min D-cost of any prefix that
            // drops sde from V_next by 1.  If no such prefix exists, V_next
            // is a dead-end for single-prefix peeling; mark it INT_MAX so
            // we prefer a tied winner with a viable continuation.
            Mat3 V_next = e.P.mul(V);
            int followup_d = INT_MAX;
            for (int idx2 = 0; idx2 < N_PREFIX; ++idx2) {
                const PrefixEntry& e2 = table[idx2];
                if (e2.d >= followup_d) continue;
                ringZ9chi nn00 = prefix_times_V_00(e2.P, V_next);
                if (sdeChiFull(nn00) == target_s - 1) {
                    followup_d = e2.d;
                    if (followup_d == 0) break;
                }
            }

            if (followup_d < best_followup_d) {
                best_followup_d = followup_d;
                best_idx = idx;
            }
        }
    }

    found:
    {
        const PrefixEntry& be = table[best_idx];
        GateStep step;
        step.a0 = be.a1; step.a1 = be.a2; step.a2 = be.a3;
        step.eps = be.eps; step.delta = be.delta; step.has_H = true;
        result.steps.push_back(step);
        result.D_count += be.d;
        V = be.P.mul(V);
    }
    return true;
}

// Try double-prefix: find two prefixes P2*P1 such that sde drops by at least 1.
// Among all pairs that achieve a net reduction, pick the one with the fewest
// total non-Clifford diagonal entries (minimum combined D-gate cost).
//
// (OPT) Like trySinglePrefix, this uses the cached prefix table.  The
// outer-loop reuse is especially valuable here because the inner P2 loop
// would otherwise rebuild 4374 prefix matrices for every outer P1 — i.e.
// 4374×4374 = ~19M buildPrefix invocations in the worst case.
static bool tryDoublePrefix(Mat3& V, int s, DecompResult& result,
                             const Mat3& /*Hmat*/, const Mat3& /*Rmat*/, const Mat3& /*Xmat*/,
                             bool quiet = false) {
    if (!quiet) cout << "  Trying double-prefix at sde_chi=" << s << "..." << endl;

    const std::vector<PrefixEntry>& table = get_prefix_table();

    int best_total_d = INT_MAX;
    int best_new_s = s;
    int best_mid_s = s;
    int best_idx1 = -1, best_idx2 = -1;

    for (int idx1 = 0; idx1 < N_PREFIX; ++idx1) {
        const PrefixEntry& e1 = table[idx1];
        if (e1.d >= best_total_d) continue;  // P1 alone already too expensive

        ringZ9chi mid00 = prefix_times_V_00(e1.P, V);
        int mid_s = sdeChiFull(mid00);

        if (mid_s > s + 1 || mid_s < s - 1) continue;

        Mat3 midV = e1.P.mul(V);
        for (int idx2 = 0; idx2 < N_PREFIX; ++idx2) {
            const PrefixEntry& e2 = table[idx2];
            if (e1.d + e2.d >= best_total_d) continue;

            ringZ9chi new00 = prefix_times_V_00(e2.P, midV);
            int new_s = sdeChiFull(new00);

            if (new_s < s) {
                best_total_d = e1.d + e2.d;
                best_new_s = new_s;
                best_mid_s = mid_s;
                best_idx1 = idx1;
                best_idx2 = idx2;
                if (best_total_d == 0) goto found;
            }
        }
    }

    if (best_total_d == INT_MAX) return false;

    found:
    {
        const PrefixEntry& be1 = table[best_idx1];
        const PrefixEntry& be2 = table[best_idx2];

        GateStep step1;
        step1.a0 = be1.a1; step1.a1 = be1.a2; step1.a2 = be1.a3;
        step1.eps = be1.eps; step1.delta = be1.delta; step1.has_H = true;
        result.steps.push_back(step1);
        result.D_count += be1.d;

        Mat3 midV = be1.P.mul(V);

        GateStep step2;
        step2.a0 = be2.a1; step2.a1 = be2.a2; step2.a2 = be2.a3;
        step2.eps = be2.eps; step2.delta = be2.delta; step2.has_H = true;
        result.steps.push_back(step2);
        result.D_count += be2.d;

        V = be2.P.mul(midV);
        if (!quiet) cout << "  Double-prefix succeeded: sde " << s
             << " -> " << best_mid_s << " -> " << best_new_s << endl;
    }
    return true;
}

DecompResult decompose(Mat3 V, bool quiet) {
    DecompResult result;
    result.D_count = 0;
    result.D_only  = 0;
    result.R_only  = 0;
    result.sde_chi = 0;
    result.sde_chi_final = 0;
    // result.trailing_clifford is default-constructed (zero matrix); replaced
    // below at every successful return path.
    result.success = false;

    // =========================================================================
    //  Fast-path: detect monomial matrices (perm × diagonal ζ₉ phases).
    //  A monomial 3×3 unitary has exactly one nonzero entry per row and column.
    //  D-count comes solely from the non-Clifford (non-ω) diagonal phases.
    // =========================================================================

    // Determine the zeta_9 exponent mod 3 of a unit in Z[zeta_9].
    // Returns 0 if the entry is ±omega^k (Clifford), 1 or 2 otherwise.
    // Uses exact ring arithmetic: multiply by zeta_9^{-j} for j=0..8 and
    // check if the result is ±1 (i.e. rational integer ±1).
    auto unitPhaseMod3 = [](const ringZ9chi& x) -> int {
        // x should be a unit (magnitude 1) in Z[zeta_9] with exp=0.
        // Try x * zeta_9^{-j} for j=0..8. If the product is ±1 (rational integer),
        // then x = ±zeta_9^j, and we return j % 3.
        ringZ9 numer = x.getNumerator();
        for (int j = 0; j < 9; ++j) {
            // Multiply by zeta_9^{-j} = zeta_9^{9-j}
            int conj_exp = (9 - j) % 9;
            ringZ9 zj(1, conj_exp);  // zeta_9^{9-j}
            ringZ9 prod = numer * zj;
            // Check if prod is ±1: that means prod.element[0] = ±1
            // and all other elements are 0.
            bool is_pm1 = true;
            for (int k = 1; k < 6; ++k) {
                if (prod.getTerm(k) != 0) { is_pm1 = false; break; }
            }
            if (is_pm1 && (prod.getTerm(0) == 1 || prod.getTerm(0) == -1)) {
                return j % 3;
            }
        }
        // Fallback: not a simple ±zeta_9^j unit. Shouldn't happen for valid
        // monomial matrices from the gate set, but return -1 to signal failure.
        return -1;
    };

    // Returns total D+R cost (Convention B).  r_out is the R-only contribution
    // (subset of the total) so callers can split for Convention C accounting.
    auto countMonomialD = [&unitPhaseMod3](const Mat3& M, bool& is_monomial, int& r_out) -> int {
        r_out = 0;
        int phases_mod3[3];
        is_monomial = true;
        for (int i = 0; i < 3; ++i) {
            int nz_count = 0;
            for (int j = 0; j < 3; ++j) {
                if (!M.m[i][j].isZero()) {
                    nz_count++;
                    int pm3 = unitPhaseMod3(M.m[i][j]);
                    if (pm3 < 0) { is_monomial = false; return -1; }
                    phases_mod3[i] = pm3;
                }
            }
            if (nz_count != 1) { is_monomial = false; return -1; }
        }
        int p = phases_mod3[0], q = phases_mod3[1], r = phases_mod3[2];
        if (p == 0 && q == 0 && r == 0) {
            // Each entry is ±ω^j (j∈{0,3,6}), i.e. lies in {±1, ±ω, ±ω²}.
            // Two cases:
            //   (a) M is a true Clifford (in our 648-cache) → 0 D-gates.
            //   (b) M = sign × Clifford for some sign matrix in
            //       {±I, ±R, ±R', ±R''} (8 elements; non-trivial signs cost
            //       1 𝒟 gate per Kalra: 𝒟 = diag(±ξ^a, ±ξ^b, ±ξ^c) includes
            //       sign-only variants).
            // Try all 8 sign multiplications and return the smallest D-cost.
            const auto& cliffs = get_clifford_cache().ring_cliffords;
            auto exact_match = [&](const Mat3& M_test) -> bool {
                for (const auto& c : cliffs) {
                    bool eq = true;
                    for (int i = 0; eq && i < 3; ++i)
                        for (int j = 0; eq && j < 3; ++j)
                            if (!(M_test.m[i][j] == c.m[i][j])) eq = false;
                    if (eq) return true;
                }
                return false;
            };
            // (a) M itself in Clifford cache?
            if (exact_match(M)) return 0;
            // (b) Try the 7 non-trivial sign multipliers.
            int best = -1;
            ringZ9 one_z(1);
            ringZ9 neg_z(-1);
            ringZ9chi p1(one_z, 0), m1(neg_z, 0);
            ringZ9chi sign_arr[2] = {p1, m1};
            for (int s0 = 0; s0 < 2; ++s0)
                for (int s1 = 0; s1 < 2; ++s1)
                    for (int s2 = 0; s2 < 2; ++s2) {
                        if (s0 == 0 && s1 == 0 && s2 == 0) continue;  // already tried
                        Mat3 Mtest;
                        for (int i = 0; i < 3; ++i)
                            for (int j = 0; j < 3; ++j) {
                                ringZ9chi sign_factor;
                                if (i == 0) sign_factor = sign_arr[s0];
                                else if (i == 1) sign_factor = sign_arr[s1];
                                else sign_factor = sign_arr[s2];
                                Mtest.m[i][j] = sign_factor * M.m[i][j];
                            }
                        if (exact_match(Mtest)) {
                            // M = (sign_matrix^{-1}) · Mtest  where sign_matrix has
                            // signs (s0,s1,s2) (the inverse is the same sign matrix).
                            // The sign matrix is one 𝒟 gate of the R-only type.
                            if (best < 0 || 1 < best) best = 1;
                        }
                    }
            if (best >= 0) {
                r_out = best;  // The cost is purely R-type (sign-only 𝒟).
                return best;
            }
            // Genuinely not reachable as (sign × Clifford) — fall through.
            is_monomial = false;
            return -1;
        }
        // (p,q,r) ≠ (0,0,0): hardcoded D-pattern heuristic.  All cost is D-type
        // here (no R contribution); r_out stays at 0.
        bool one_gate =
            (p==1&&q==0&&r==2) || (p==2&&q==0&&r==1) ||
            (p==0&&q==1&&r==2) || (p==0&&q==2&&r==1) ||
            (p==1&&q==2&&r==0) || (p==2&&q==1&&r==0);
        return one_gate ? 1 : 2;
    };

    // Check: is V already monomial?
    bool is_mono = false;
    int mono_r = 0;
    int mono_d = countMonomialD(V, is_mono, mono_r);
    if (is_mono) {
        result.D_count = mono_d;
        result.R_only  = mono_r;
        result.D_only  = mono_d - mono_r;
        result.sde_chi = 0;
        result.sde_chi_final = 0;
        result.trailing_clifford = V;  // V is already the residual monomial
        result.success = true;
        return result;
    }

    // (The "diagonal-V fast-path" was REMOVED 2026-05 along with diagSearch.
    //  It used a heuristic D-count formula on the diagonal entries' sde_chi
    //  values, returned NO syllables, and was only meaningful for the
    //  diagSearch heuristic — which itself returned non-unitary V's.  Real
    //  diagonal V's that ARE actually unitary (e.g. coming from HRSA Phase 3)
    //  flow through the general peeling algorithm below, which produces a
    //  proper syllable list.)

    // =========================================================================
    //  Fast-path: detect single-H-layer matrices.
    //  If all 9 entries have magnitude ≈ 1/√3, then V = M_L · H · M_R
    //  where M_L, M_R are monomial (Clifford + possibly D phases).
    //  We compute M_L = V · H† (since H† = H⁻¹) and check if it's monomial.
    //  Then M_R = H† · M_L⁻¹ · V, but since M_L · H · M_R is the full decomp,
    //  D_count = D(M_L) + D(M_R). However, M_L · H · M_R has D_count from
    //  the non-Clifford phases of M_L and M_R.
    //  Actually: V · H† should be monomial (= M_L · H · M_R · H†).
    //  That's NOT M_L in general. Instead: H† · V should equal H† · M_L · H · M_R.
    //  
    //  Simpler: try all 6 permutation matrices P and 3 H-variants (H, H², I):
    //  Check if V = D_L · P · H^k · D_R is satisfied for some diagonal D_L, D_R
    //  and permutation P, with k=0 (monomial, already handled) or k=1.
    //  For k=1: V · (P·H)† should be diagonal. Try all 6 permutations of H.
    // =========================================================================
    {
        // Check if all entries have magnitude ≈ 1/√3
        double target_mag = 1.0 / sqrt(3.0);
        bool all_same_mag = true;
        for (int i = 0; i < 3 && all_same_mag; ++i)
            for (int j = 0; j < 3 && all_same_mag; ++j) {
                double mag = abs(V.m[i][j].toComplexDouble());
                if (abs(mag - target_mag) > 0.01) all_same_mag = false;
            }

        if (all_same_mag) {
            // V is a single-H-layer matrix.
            // Try V · H† and check if it's monomial.
            Mat3 Hmat_local = gateH();
            // Also try with permuted H: H·X, H·X², X·H, X²·H, etc.
            Mat3 Xmat_local = gateX();
            Mat3 X2 = Xmat_local.mul(Xmat_local);

            // Candidates for the "H part": H, X·H, X²·H, H·X, H·X², X·H·X, ...
            // Equivalently: try V · (PH)† for various column permutations of H.
            Mat3 H_variants[6];
            H_variants[0] = Hmat_local;
            H_variants[1] = Hmat_local.mul(Xmat_local);
            H_variants[2] = Hmat_local.mul(X2);
            H_variants[3] = Xmat_local.mul(Hmat_local);
            H_variants[4] = X2.mul(Hmat_local);
            H_variants[5] = Xmat_local.mul(Hmat_local).mul(Xmat_local);

            int best_d = INT_MAX;
            int best_r = 0;
            for (int h = 0; h < 6; ++h) {
                Mat3 Hv_dag = H_variants[h].dagger();
                Mat3 residual = V.mul(Hv_dag);
                bool rm = false;
                int rr = 0;
                int d = countMonomialD(residual, rm, rr);
                if (rm && d < best_d) { best_d = d; best_r = rr; }
            }

            if (best_d < INT_MAX) {
                result.D_count = best_d;
                result.R_only  = best_r;
                result.D_only  = best_d - best_r;
                result.sde_chi = sdeChiFull(V.m[0][0]);
                result.success = true;
                return result;
            }
        }
    }

    // =========================================================================
    //  General case: sde-peeling algorithm
    // =========================================================================
    Mat3 Hmat = gateH();
    Mat3 Rmat = gateR();
    Mat3 Xmat = gateX();

    int s = sdeChiFull(V.m[0][0]);
    result.sde_chi = s;

    if (s == 999) {
        if (!quiet) cout << "Warning: (0,0) entry is zero in decompose" << endl;
        result.sde_chi_final = s;
        result.trailing_clifford = V;
        return result;
    }

    int max_iter = s + 20;
    int iter = 0;

    while (s > 0 && iter < max_iter) {
        iter++;

        // First try single-prefix (fast path)
        if (trySinglePrefix(V, s, result, Hmat, Rmat, Xmat)) {
            s = sdeChiFull(V.m[0][0]);
            continue;
        }

        // Single prefix failed (Kalra obstruction).
        // Try double-prefix: P2 * P1 * V with net sde reduction.
        if (tryDoublePrefix(V, s, result, Hmat, Rmat, Xmat, quiet)) {
            s = sdeChiFull(V.m[0][0]);
            continue;
        }

        // Both failed
        if (!quiet) cout << "Decompose: cannot reduce sde_chi at s=" << s
             << " even with double-prefix" << endl;
        result.sde_chi_final = s;
        result.trailing_clifford = V;
        return result;
    }

    result.success = (s == 0);
    result.sde_chi_final = s;
    result.trailing_clifford = V;  // residual monomial Clifford+D matrix

    // Count D gates in the residual monomial matrix.
    // At sde_chi = 0, V is a unitary over Z[zeta_9] (no denominators).
    // It should be monomial: each row and column has exactly one nonzero entry,
    // which is a power of zeta_9.  The Clifford phases are powers of omega = zeta_9^3.
    // Non-Clifford phases (zeta_9^k where k % 3 != 0) require D gates.
    //
    // The D gate is Diag(zeta9, 1, zeta9^8). Each application changes one
    // diagonal exponent by +1 and another by -1 (mod 9), via X conjugation.
    // So the residual exponents (mod 3) determine the minimum D-count:
    //   all 0 mod 3 → 0 D gates
    //   matches a single D-gate pattern → 1 D gate
    //   otherwise → 2 D gates
    if (result.success) {
        bool resid_mono = false;
        int resid_r = 0;
        int resid_d = countMonomialD(V, resid_mono, resid_r);
        if (resid_mono && resid_d > 0) {
            result.D_count += resid_d;
            result.R_only  += resid_r;
            result.D_only  += (resid_d - resid_r);
        } else if (!resid_mono) {
            // Residual V isn't a recognized monomial/Clifford.  Could be:
            //   (a) Non-monomial (multiple nonzero entries per row/col) — algorithm failure.
            //   (b) Monomial-but-non-Clifford (e.g. diag(-ω,-ω²,1)): each entry is ±ω^k
            //       individually, but the matrix isn't in the 648-Clifford cache.
            //       Implementing it requires D-gates, but the fast-path can't count them.
            // In both cases, the decomposition is incomplete — mark as failure rather
            // than report a misleading N_D=0.  (Was a silent bug pre-2026-05-09.)
            if (!quiet) cout << "Decompose: residual V is non-Clifford after peeling — D-count unknown" << endl;
            result.success = false;
            result.D_count = -1;
        }
    }

    // ---- Local rewrite-rule post-pass ---------------------------------------
    //
    // Runs by default; can be disabled by setting g_decompose_simplify = false
    // (the HRSA_tester binary exposes a `--no-simplify` flag for A/B comparison).
    // The pass preserves the operator product P_n·...·P_1 EXACTLY in ringZ9chi
    // arithmetic (verified after each pass) so the trailing residual and the
    // re-counted residual D-cost above remain correct.
    if (g_decompose_simplify && !result.steps.empty()) {
        // Compute the syllable D-count BEFORE simplification so we can tell how
        // much of result.D_count came from the residual (which simplify_steps
        // does not touch) and update accordingly.
        int prior_syll_d = 0;
        for (const GateStep& s : result.steps)
            prior_syll_d += count_d_from_syllable(s.a0, s.a1, s.a2, s.eps);
        int new_syll_d = simplify_steps(result.steps, /*quiet=*/quiet);
        result.D_count += (new_syll_d - prior_syll_d);
    }

    // Final accounting: split per-step contributions into D-only vs R-only.
    // (D_only and R_only at this point hold only the residual contributions
    // accumulated above; step contributions still need to be added.)
    for (const GateStep& s : result.steps) {
        result.D_only += count_d_from_cyclo(s.a0, s.a1, s.a2);
        result.R_only += (s.eps != 0 ? 1 : 0);
    }

    return result;
}

#endif  // ---- end obsolete int-only originals (replaced by template impl) ----

// =========================================================================
//  Local rewrite-rule simplifier
// =========================================================================
//
// Process model: decompose() emits a list of GateStep entries s_1..s_n where
// each step represents a prefix matrix
//     P_i  =  [H?] · Dcyclo(s_i.a0, s_i.a1, s_i.a2) · R^{s_i.eps} · X^{s_i.delta}
// applied LEFT-multiplicatively to V during peeling.  The full operator the
// step list encodes is therefore the matrix product
//     M_steps  =  P_n · P_{n-1} · ... · P_1
// (i.e. step i+1 sits to the left of step i in the matrix product).
//
// Two invariants the simplifier preserves:
//   (1) M_steps after rewrite equals M_steps before rewrite, EXACTLY in
//       ringZ9chi.  The residual `trailing_clifford` is therefore unchanged.
//   (2) The new total D-count is the sum of count_d_from_cyclo over the new
//       step list.  Returned to the caller, who folds in any residual D-cost.
//
// Verification: at the end we recompute M_steps for the new list and compare
// it via Mat3::equals() to a snapshot of M_steps from before.  Any rule that
// would silently change the operator is caught and aborts the rewrite (the
// step list is reverted in that defensive case — should never trip in
// practice, but is cheap insurance).

// (g_decompose_simplify is defined at the very top of this file.)

// Build  H^h · Dcyclo(a0,a1,a2) · R^eps · X^delta  exactly over ringZ9chi.
// Differs from buildPrefix() in that the H-prefix is OPTIONAL (h ∈ {0,1});
// the rest of the codebase always uses h=1.  We need h=0 prefixes here so
// the simplifier can express H-cancellation results.
static Mat3 buildPrefixHopt(int a0, int a1, int a2, int eps, int delta, bool has_H) {
    static const Mat3 Hmat = gateH();
    static const Mat3 Rmat = gateR();
    static const Mat3 Xmat = gateX();
    Mat3 prefix = gateDcyclo(a0, a1, a2);
    if (eps == 1) prefix = prefix.mul(Rmat);
    if (delta == 1) prefix = prefix.mul(Xmat);
    else if (delta == 2) prefix = prefix.mul(Xmat).mul(Xmat);
    if (has_H) prefix = Hmat.mul(prefix);
    return prefix;
}

Mat3 gateStepPrefix(const GateStep& s) {
    return buildPrefixHopt(s.a0, s.a1, s.a2, s.eps, s.delta, s.has_H);
}

// Compute M_steps = P_n · P_{n-1} · ... · P_1  for the step list `steps`.
static Mat3 computeStepsProduct(const std::vector<GateStep>& steps) {
    // Identity start.
    Mat3 M;
    ringZ9chi one(ringZ9(1), 0);
    M.m[0][0] = one; M.m[1][1] = one; M.m[2][2] = one;
    // Left-multiply by P_1 first, then P_2, ..., so steps[0] ends up on the
    // RIGHT of the product (matches the comment block above).
    for (size_t i = 0; i < steps.size(); ++i)
        M = gateStepPrefix(steps[i]).mul(M);
    return M;
}

// =========================================================================
//  Extended prefix table: every (a0,a1,a2,eps,delta,has_H) tuple paired with
//  its prefix Mat3 and D-cost.  Used as a lookup target for "merged-pair"
//  rewrites: given Q = P_{i+1}·P_i, search this table for an entry whose
//  prefix exactly equals Q (Mat3::equals) and has strictly smaller D-cost
//  than the pair.  Built once per process behind a function-local static.
//
//  Size: 9·9·9·2·3·2 = 8748 entries.  Linear search runs in <1 ms per pair
//  thanks to early-out in Mat3 equality on the first mismatched entry.
// =========================================================================
namespace {
struct ExtPrefixEntry {
    Mat3 P;
    int a0, a1, a2, eps, delta;
    bool has_H;
    int d;
};

const std::vector<ExtPrefixEntry>& get_extended_prefix_table() {
    static const std::vector<ExtPrefixEntry> table = []() {
        std::vector<ExtPrefixEntry> tbl;
        tbl.reserve(8748);
        for (int has_H = 0; has_H < 2; ++has_H)
        for (int eps = 0; eps < 2; ++eps)
        for (int delta = 0; delta < 3; ++delta)
        for (int a0 = 0; a0 < 9; ++a0)
        for (int a1 = 0; a1 < 9; ++a1)
        for (int a2 = 0; a2 < 9; ++a2) {
            ExtPrefixEntry e;
            e.P = buildPrefixHopt(a0, a1, a2, eps, delta, has_H != 0);
            e.a0 = a0; e.a1 = a1; e.a2 = a2;
            e.eps = eps; e.delta = delta;
            e.has_H = (has_H != 0);
            e.d = count_d_from_syllable(a0, a1, a2, eps);
            tbl.push_back(e);
        }
        return tbl;
    }();
    return table;
}

// Look up Q in the extended prefix table; return the index of the *cheapest*
// (lowest D-count) entry whose prefix equals Q exactly, or -1 if none.
int lookup_prefix(const Mat3& Q) {
    const std::vector<ExtPrefixEntry>& tbl = get_extended_prefix_table();
    int best_d = INT_MAX;
    int best_idx = -1;
    for (size_t i = 0; i < tbl.size(); ++i) {
        if (tbl[i].d >= best_d) continue;  // can't improve, skip equals() cost
        if (tbl[i].P.equals(Q)) {
            best_d = tbl[i].d;
            best_idx = (int)i;
            if (best_d == 0) break;  // cheapest possible, done
        }
    }
    return best_idx;
}
} // namespace

// =========================================================================
//  Canonical-form lookup table for global w-window simplification.
//
//  Goal: enable post-pass replacement of any adjacent w-step window
//  (3 <= w <= 6) by a STRICTLY shorter (or strictly cheaper-D) GateStep
//  sequence, when one exists in the canonical short-word library.
//
//  Construction: BFS over GateStep extensions starting from identity, using
//  complex<double> matrix arithmetic for speed and 1e-7 rounding for hash
//  dedup (same scheme as bfs_canonical.cpp).  Each level adds one GateStep
//  to the word; we explore depth 1..MAX_BFS_DEPTH and store, per unique
//  matrix, the Pareto-best (length, d_total) word that reaches it.
//
//  Alphabet: full coverage of single-step prefixes — 9*9*9 (a0,a1,a2)
//  tuples × 2 (eps) × 3 (delta) × 2 (has_H) = 8748 entries.  We dedup the
//  alphabet by Mat3 (1e-7 rounding) before BFS extension.
//
//  Table cap: bounded at MAX_TBL_SIZE entries to keep memory in check; once
//  full, BFS continues iterating (Pareto-update only — no new entries are
//  added).
//
//  Memory: 953K entries × ~165 B = ~155 MB at depth 2 with the full
//  alphabet (a build-time print at the bottom reports the actual count).
//  The table is built once per process behind a function-local static
//  (thread-safe under C++11+).
// =========================================================================
namespace {

// 18 int32 entries (one per real/imag of every CMat3 entry).  Same scheme
// as bfs_canonical.cpp's MatKey: 1e-7 rounding of complex doubles.
using MatKey = std::array<int32_t, 18>;

struct MatKeyHash {
    size_t operator()(const MatKey& k) const noexcept {
        const uint64_t* p = reinterpret_cast<const uint64_t*>(k.data());
        // 18 int32 = 72 bytes = 9 uint64.
        uint64_t h = 1469598103934665603ULL;
        for (int i = 0; i < 9; ++i) {
            h ^= p[i];
            h *= 1099511628211ULL;
        }
        return (size_t)h;
    }
};

static constexpr double KEY_SCALE = 1e7;

// 3x3 complex<double> matrix used purely for fast BFS dedup.
struct CMat3can {
    std::complex<double> m[3][3];
};

static inline CMat3can cmul_can(const CMat3can& A, const CMat3can& B) {
    CMat3can C = {};
    for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 3; ++k) {
            std::complex<double> a = A.m[i][k];
            for (int j = 0; j < 3; ++j)
                C.m[i][j] += a * B.m[k][j];
        }
    return C;
}

static inline MatKey mat_key_can(const CMat3can& M) {
    MatKey k{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            k[6*i + 2*j]     = (int32_t)llround(M.m[i][j].real() * KEY_SCALE);
            k[6*i + 2*j + 1] = (int32_t)llround(M.m[i][j].imag() * KEY_SCALE);
        }
    return k;
}

static MatKey mat_key_from_ringmat(const Mat3& M) {
    CMat3can C = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            C.m[i][j] = M.m[i][j].toComplexDouble();
    return mat_key_can(C);
}

// Build the complex<double> form of a single GateStep prefix (matches
// gateStepPrefix() but in floating point).
static CMat3can gateStepPrefix_cd(const GateStep& s) {
    using cd = std::complex<double>;
    cd om = std::exp(cd(0.0, 2.0 * M_PI / 3.0));
    cd z9 = std::exp(cd(0.0, 2.0 * M_PI / 9.0));

    auto z9pow = [&](int k) {
        k = ((k % 9) + 9) % 9;
        return std::pow(z9, (double)k);
    };

    CMat3can D = {};
    D.m[0][0] = z9pow(s.a0);
    D.m[1][1] = z9pow(s.a1);
    D.m[2][2] = z9pow(s.a2);

    if (s.eps == 1) {
        // R = Diag(1,1,-1)
        D.m[2][2] = -D.m[2][2];
    }

    // Apply X^delta on the right: (D · X^delta)
    //   X.m[0][2]=X.m[1][0]=X.m[2][1]=1  ⇒  (D·X).m[i][j] = D.m[i][(j+1)%3]
    //   ((D·X·X).m[i][j] = D.m[i][(j+2)%3])
    CMat3can DX;
    if (s.delta == 0) {
        DX = D;
    } else {
        DX = {};
        int shift = s.delta;  // delta==1 → +1, delta==2 → +2
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                DX.m[i][j] = D.m[i][(j + shift) % 3];
    }

    if (!s.has_H) return DX;

    // H_{jk} = (-1 - 2*om)/3 * om^{j*k}; |H_{jk}| = 1/sqrt(3).
    cd h_scale = cd(1.0, 0.0) / (cd(1.0, 0.0) + cd(2.0, 0.0) * om);
    CMat3can H = {};
    for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k) {
            int e = (j * k) % 3;
            H.m[j][k] = h_scale * ((e == 0) ? cd(1.0, 0.0) :
                                   (e == 1) ? om : om * om);
        }
    return cmul_can(H, DX);
}

// Lookup table value: shortest known (length, D-cost) word reaching some matrix.
struct CanonicalWord {
    std::vector<GateStep> word;
    int d_total;  // count_d_from_cyclo summed across word
};

// BFS depth (number of GateSteps in the longest stored word).
// We search windows of size w=3..6 in simplify_steps; entries of length
// L < w let us substitute a window with a strictly shorter word.
//
// To keep memory bounded we cap the stored table at MAX_TBL_SIZE entries;
// once full, BFS extension stops adding new entries.  The Pareto update
// path (replace stored word when a cheaper-D word is found at the same
// matrix) still runs even after the cap is hit, so deeper levels can
// still improve the recorded d_total of already-stored matrices.
//
// Empirical sweet spot: depth=2 with full 8748-entry alphabet (a_i in
// {0..8}) builds ~950K canonical entries in ~15 s and finds non-trivial
// *length* reductions on HRSA's output.  Depth=3 with the same alphabet
// runs the level-3 frontier (≈ 950K × 8748 mat-muls = 8B) for the
// Pareto-update pass, which is too slow to be practical at any non-trivial
// alphabet.  Depth=3 with a residue-restricted alphabet (a_i in
// {0,1,2,3,6,7,8}) builds ~5M entries in ~50 s but rarely finds D-cost
// reductions on top of the depth-2 length reductions.  We default to
// depth=2.
static constexpr int MAX_BFS_DEPTH = 2;
static constexpr size_t MAX_TBL_SIZE = 5'000'000;

// Build the BFS-canonical-form lookup once per process (lazy static init).
const std::unordered_map<MatKey, CanonicalWord, MatKeyHash>&
get_canonical_lookup() {
    static const std::unordered_map<MatKey, CanonicalWord, MatKeyHash> table = []() {
        auto t0 = std::chrono::steady_clock::now();

        std::unordered_map<MatKey, CanonicalWord, MatKeyHash> tbl;
        tbl.reserve(1 << 20);  // 1M buckets up front

        // Identity entry: empty word.  We'll never *return* this (replacements
        // would null out a window) but it lets BFS start from identity matrix.
        CMat3can eye = {};
        eye.m[0][0] = std::complex<double>(1.0, 0.0);
        eye.m[1][1] = std::complex<double>(1.0, 0.0);
        eye.m[2][2] = std::complex<double>(1.0, 0.0);
        MatKey eye_key = mat_key_can(eye);
        tbl.emplace(eye_key, CanonicalWord{{}, 0});

        // Build the GateStep alphabet.  Full coverage is 9*9*9*2*3*2 = 8748
        // single-step prefixes — every ringZ9chi prefix that decompose() can
        // emit (HRSA's peeling uses a_i in {0..8} freely).
        std::vector<GateStep> alphabet;
        alphabet.reserve(8748);
        for (int has_H = 0; has_H < 2; ++has_H)
        for (int eps = 0; eps < 2; ++eps)
        for (int delta = 0; delta < 3; ++delta)
        for (int a0 = 0; a0 < 9; ++a0)
        for (int a1 = 0; a1 < 9; ++a1)
        for (int a2 = 0; a2 < 9; ++a2) {
            GateStep s;
            s.has_H = (has_H != 0);
            s.eps = eps; s.delta = delta;
            s.a0 = a0; s.a1 = a1; s.a2 = a2;
            alphabet.push_back(s);
        }

        // Precompute alphabet prefix matrices in CMat3 form + their D-cost.
        std::vector<CMat3can> alpha_P(alphabet.size());
        std::vector<int>      alpha_d(alphabet.size());
        for (size_t i = 0; i < alphabet.size(); ++i) {
            alpha_P[i] = gateStepPrefix_cd(alphabet[i]);
            alpha_d[i] = count_d_from_syllable(alphabet[i].a0,
                                               alphabet[i].a1,
                                               alphabet[i].a2,
                                               alphabet[i].eps);
        }

        // Reduce alphabet by Mat3-deduping: many of the 8748 prefixes give
        // identical CMat3.  Pick the *cheapest* (lowest D-cost) representative
        // per distinct matrix to keep BFS branching minimal.
        std::vector<CMat3can> dedup_P;
        std::vector<GateStep> dedup_step;
        std::vector<int>      dedup_d;
        {
            std::unordered_map<MatKey, int, MatKeyHash> seen_alpha;
            for (size_t i = 0; i < alphabet.size(); ++i) {
                MatKey k = mat_key_can(alpha_P[i]);
                auto it = seen_alpha.find(k);
                if (it == seen_alpha.end()) {
                    seen_alpha[k] = (int)dedup_P.size();
                    dedup_P.push_back(alpha_P[i]);
                    dedup_step.push_back(alphabet[i]);
                    dedup_d.push_back(alpha_d[i]);
                } else {
                    int idx = it->second;
                    if (alpha_d[i] < dedup_d[idx]) {
                        dedup_P[idx] = alpha_P[i];
                        dedup_step[idx] = alphabet[i];
                        dedup_d[idx] = alpha_d[i];
                    }
                }
            }
        }
        size_t alpha_n = dedup_P.size();

        // BFS frontier: matrices at the current level, paired with their words.
        struct Frontier {
            std::vector<CMat3can> M;
            std::vector<std::vector<GateStep>> W;
            std::vector<int> D;
        };
        Frontier frontier;
        frontier.M.push_back(eye);
        frontier.W.push_back({});
        frontier.D.push_back(0);

        for (int level = 1; level <= MAX_BFS_DEPTH; ++level) {
            Frontier next;
            size_t reserve = std::min(frontier.M.size() * alpha_n,
                                       (size_t)2'000'000);
            next.M.reserve(reserve / 4);
            next.W.reserve(reserve / 4);
            next.D.reserve(reserve / 4);

            for (size_t pi = 0; pi < frontier.M.size(); ++pi) {
                const CMat3can& Pmat = frontier.M[pi];
                const std::vector<GateStep>& Pword = frontier.W[pi];
                int Pd = frontier.D[pi];
                for (size_t ai = 0; ai < alpha_n; ++ai) {
                    // Extension semantics: in computeStepsProduct, steps[k] is
                    // multiplied LEFT-of the running product, so the canonical
                    // word [s0, s1, ..., s_{n-1}] produces P_{n-1}·...·P_1·P_0.
                    // We're building words of length `level`; to keep the
                    // semantics consistent with computeStepsProduct, we
                    // multiply alpha_P[ai] on the LEFT of the running matrix
                    // and APPEND alphabet[ai] to the END of the word.
                    CMat3can Q = cmul_can(dedup_P[ai], Pmat);
                    MatKey k = mat_key_can(Q);
                    int newd = Pd + dedup_d[ai];
                    auto it = tbl.find(k);
                    if (it != tbl.end()) {
                        // Already known matrix.  Replace stored word if the
                        // new word is *Pareto-better* on (d_total, length):
                        //   - strictly lower d_total (any length) → take it
                        //   - same d_total, strictly shorter length → take it
                        // (We're optimising primarily for D-cost reduction
                        // since that's what the user-visible N_D metric is.)
                        size_t newl = Pword.size() + 1;
                        bool replace =
                            (newd < it->second.d_total) ||
                            (newd == it->second.d_total &&
                             newl < it->second.word.size());
                        if (replace) {
                            std::vector<GateStep> neww;
                            neww.reserve(newl);
                            neww.insert(neww.end(), Pword.begin(), Pword.end());
                            neww.push_back(dedup_step[ai]);
                            it->second.word = std::move(neww);
                            it->second.d_total = newd;
                        }
                        continue;
                    }
                    // Cap insertion of NEW entries at MAX_TBL_SIZE; we still
                    // continue iterating (Pareto updates above remain useful).
                    if (tbl.size() >= MAX_TBL_SIZE) continue;
                    std::vector<GateStep> neww;
                    neww.reserve(Pword.size() + 1);
                    neww.insert(neww.end(), Pword.begin(), Pword.end());
                    neww.push_back(dedup_step[ai]);
                    tbl.emplace(k, CanonicalWord{neww, newd});
                    next.M.push_back(Q);
                    next.W.push_back(std::move(neww));
                    next.D.push_back(newd);
                }
            }
            frontier = std::move(next);
        }

        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        std::cerr << "[canonical_lookup] built " << tbl.size()
                  << " entries in " << std::fixed << std::setprecision(2)
                  << sec << " s (depth=" << MAX_BFS_DEPTH
                  << ", alphabet=" << alpha_n << ")" << std::endl;
        return tbl;
    }();
    return table;
}

// Look up matrix Q (ringZ9chi) in the canonical lookup; return the canonical
// word if found AND it improves on (orig_len, orig_d) by either:
//   (a) strictly shorter length with d <= orig_d, or
//   (b) strictly lower D-cost with len <= orig_len.
// Returns an empty optional otherwise.  We use a sentinel empty vector + a
// success bool to avoid <optional>.
struct CanonLookupResult {
    bool found;
    std::vector<GateStep> word;
    int d_total;
};

CanonLookupResult canonical_lookup(const Mat3& Q,
                                    int orig_len, int orig_d) {
    const auto& tbl = get_canonical_lookup();
    MatKey k = mat_key_from_ringmat(Q);
    auto it = tbl.find(k);
    if (it == tbl.end()) return {false, {}, 0};
    const CanonicalWord& cw = it->second;
    int new_len = (int)cw.word.size();
    int new_d   = cw.d_total;
    bool better =
        (new_len < orig_len && new_d <= orig_d) ||
        (new_d   < orig_d   && new_len <= orig_len);
    if (!better) return {false, {}, 0};
    return {true, cw.word, new_d};
}

} // namespace

int simplify_steps(std::vector<GateStep>& steps, bool quiet) {
    if (steps.empty()) return 0;

    // Snapshot the operator product BEFORE rewriting for a final equality
    // verification.  Any rule that violates the invariant is caught here and
    // we revert to the snapshot's step list.
    std::vector<GateStep> orig_steps = steps;
    Mat3 M_orig = computeStepsProduct(steps);

    auto step_d = [](const GateStep& s) -> int {
        return count_d_from_syllable(s.a0, s.a1, s.a2, s.eps);
    };

    auto syllable_d_total = [&](const std::vector<GateStep>& v) -> int {
        int d = 0;
        for (const GateStep& s : v) d += step_d(s);
        return d;
    };

    // Identity prefix: Dcyclo(0,0,0)·R^0·X^0 = I, no H.  Everything else has
    // either an H, an X, or a non-trivial diagonal phase, so we only need the
    // pure no-H, eps=0, delta=0, a0=a1=a2=0 case.  (R adds a sign on entry
    // (2,2), X≠I cyclically permutes basis kets, H is the DFT-based Hadamard.)
    auto is_identity_step = [](const GateStep& s) -> bool {
        return !s.has_H && s.eps == 0 && s.delta == 0
            && s.a0 == 0 && s.a1 == 0 && s.a2 == 0;
    };

    // Compute the prefix product of a window steps[i..i+w-1]:
    //   P_window = gateStepPrefix(steps[i+w-1]) · ... · gateStepPrefix(steps[i])
    // Matches the multiplicative semantics of computeStepsProduct().
    auto window_product = [](const std::vector<GateStep>& v,
                              size_t i, size_t w) -> Mat3 {
        Mat3 M;
        ringZ9chi one(ringZ9(1), 0);
        M.m[0][0] = one; M.m[1][1] = one; M.m[2][2] = one;
        for (size_t j = 0; j < w; ++j)
            M = gateStepPrefix(v[i + j]).mul(M);
        return M;
    };

    bool changed = true;
    int passes = 0;
    const int max_passes = 16;  // belt-and-braces; rules monotonically reduce work
    while (changed && passes < max_passes) {
        changed = false;
        ++passes;

        // --- Rule 1: drop identity steps -------------------------------------
        for (size_t i = 0; i < steps.size(); ) {
            if (is_identity_step(steps[i])) {
                steps.erase(steps.begin() + i);
                changed = true;
            } else {
                ++i;
            }
        }

        // --- Rule 2: merge two adjacent purely-diagonal-no-H-no-X steps ------
        // Both purely diagonal ⇒ they commute and combine.  The merged step's
        // diagonal exponents are component-wise sums mod 9; the eps adds mod 2.
        // D-count of the merged step can be strictly less when the new
        // exponents fall on multiples of 3 (the "Clifford" sub-lattice).
        for (size_t i = 0; i + 1 < steps.size(); ) {
            const GateStep& a = steps[i];
            const GateStep& b = steps[i + 1];
            if (!a.has_H && !b.has_H && a.delta == 0 && b.delta == 0) {
                GateStep merged;
                merged.a0   = (a.a0 + b.a0) % 9;
                merged.a1   = (a.a1 + b.a1) % 9;
                merged.a2   = (a.a2 + b.a2) % 9;
                merged.eps   = (a.eps + b.eps) % 2;
                merged.delta = 0;
                merged.has_H = false;
                steps[i] = merged;
                steps.erase(steps.begin() + i + 1);
                changed = true;
                // do NOT increment i: maybe the merged step combines with the
                // new neighbour at position i+1 (now the old i+2).
            } else {
                ++i;
            }
        }

        // --- Rule 3: prefix-table fusion of arbitrary adjacent pairs ---------
        // Q = P_{i+1} · P_i; if Q is itself representable as a SINGLE canonical
        // prefix in the extended table with strictly smaller D-cost than the
        // pair's combined cost, fuse.  Catches H·H cancellations, X·X simpli-
        // fications, R·R cancellations, and assorted diagonal regroupings the
        // first two rules can't reach (since neither relies on H/X cancelling).
        for (size_t i = 0; i + 1 < steps.size(); ) {
            const GateStep& s1 = steps[i];
            const GateStep& s2 = steps[i + 1];
            int pair_d = step_d(s1) + step_d(s2);
            // Cheap pre-filter: if the pair already has D-cost 0, fusing into
            // one step at best leaves it 0 — no D-savings.  But fewer steps is
            // also a win (subsequent passes may use the new neighbour).
            Mat3 Q = gateStepPrefix(s2).mul(gateStepPrefix(s1));
            int idx = lookup_prefix(Q);
            if (idx >= 0) {
                const ExtPrefixEntry& e = get_extended_prefix_table()[idx];
                if (e.d < pair_d || (e.d == pair_d && pair_d == 0)) {
                    GateStep merged;
                    merged.a0 = e.a0; merged.a1 = e.a1; merged.a2 = e.a2;
                    merged.eps = e.eps; merged.delta = e.delta;
                    merged.has_H = e.has_H;
                    steps[i] = merged;
                    steps.erase(steps.begin() + i + 1);
                    changed = true;
                    // do NOT advance i: re-attempt fusion at the same spot.
                    continue;
                }
            }
            ++i;
        }

        // --- Rule 4: w-window canonical-form lookup --------------------------
        // For w from 6 down to 3, scan windows of size w in the step list,
        // compute the window's prefix product, and look it up in the
        // canonical-form lookup table.  If a strictly shorter (or strictly
        // cheaper at same length) word is found, splice it in.
        //
        // Going large-w → small-w lets the bigger windows score the deepest
        // simplifications first (they may collapse 4-6 steps into 1-3 steps),
        // then the smaller-w sweeps clean up remaining locally-reducible
        // pieces.  After each successful replacement we restart the inner loop
        // because positions have shifted.
        for (int w = 6; w >= 3; --w) {
            bool any_at_this_w = true;
            while (any_at_this_w) {
                any_at_this_w = false;
                if ((int)steps.size() < w) break;
                for (size_t i = 0; i + (size_t)w <= steps.size(); ++i) {
                    Mat3 Q = window_product(steps, i, (size_t)w);
                    // Window's own D-cost.
                    int win_d = 0;
                    for (int k = 0; k < w; ++k)
                        win_d += step_d(steps[i + (size_t)k]);
                    CanonLookupResult r = canonical_lookup(Q, w, win_d);
                    if (!r.found) continue;
                    // Splice: remove w steps, insert r.word in their place.
                    steps.erase(steps.begin() + i,
                                steps.begin() + i + (size_t)w);
                    steps.insert(steps.begin() + i,
                                 r.word.begin(), r.word.end());
                    changed = true;
                    any_at_this_w = true;
                    break;  // restart the inner loop at this w
                }
            }
        }
    }

    // --- Verify operator product preserved -----------------------------------
    Mat3 M_new = computeStepsProduct(steps);
    if (!M_new.equals(M_orig)) {
        if (!quiet) cout << "[simplify_steps] verification failed; reverting" << endl;
        steps = orig_steps;
        return syllable_d_total(steps);
    }
    if (!quiet) {
        cout << "[simplify_steps] " << passes << " passes, "
             << orig_steps.size() << " -> " << steps.size() << " steps, "
             << "syll D " << syllable_d_total(orig_steps) << " -> "
             << syllable_d_total(steps) << endl;
    }
    return syllable_d_total(steps);
}

// =========================================================================
//  Direct gate-sequence search for low D-count unitaries
// =========================================================================

// CMat3 is now defined in clifford_cache.h (was duplicated here).

static CMat3 cmul(const CMat3& A, const CMat3& B) {
    CMat3 C = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                C.m[i][j] += A.m[i][k] * B.m[k][j];
    return C;
}

static double frobenius_dist(const CMat3& A, const CMat3& B) {
    double s = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            complex<double> d = A.m[i][j] - B.m[i][j];
            s += d.real()*d.real() + d.imag()*d.imag();
        }
    return sqrt(s);
}

/*
static CMat3 cdagger(const CMat3& A) {
    CMat3 R = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R.m[i][j] = conj(A.m[j][i]);
    return R;
}
*/

// =========================================================================
//  Clifford cache: BFS-built once, reused across all directSearch calls.
//  The Clifford set, its ringZ9chi reification, and the D-gate matrices are
//  independent of theta/epsilon, so building them per-call wasted ~5-10%
//  of each invocation.  The static-local pattern below is thread-safe
//  initialization in C++11+.
// =========================================================================
// CliffordCache struct + get_clifford_cache() are now declared in
// clifford_cache.h so other translation units (e.g. bidir_reify.cpp) can
// access the same canonical BFS-ordered Clifford table.
const CliffordCache& get_clifford_cache() {
    static const CliffordCache cache = []() {
        using cd = std::complex<double>;
        cd om = exp(cd(0, 2.0*M_PI/3.0));
        cd z9 = exp(cd(0, 2.0*M_PI/9.0));

        // Generators (CMat3): H, X, S, S^{-1}.
        CMat3 genH = {};
        cd h_scale = cd(1,0) / (cd(1,0) + cd(2,0)*om);
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) {
                int e = (j*k) % 3;
                genH.m[j][k] = h_scale * ((e==0)?cd(1,0):(e==1)?om:om*om);
            }
        CMat3 genX = {};
        genX.m[0][2] = genX.m[1][0] = genX.m[2][1] = cd(1,0);
        CMat3 genS = {};
        genS.m[0][0] = om; genS.m[1][1] = cd(1,0); genS.m[2][2] = cd(1,0);
        CMat3 genSi = {};
        genSi.m[0][0] = om*om; genSi.m[1][1] = cd(1,0); genSi.m[2][2] = cd(1,0);

        auto mat_key = [](const CMat3& M) -> std::vector<int> {
            std::vector<int> k(18);
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) {
                    k[6*i+2*j]   = (int)round(M.m[i][j].real() * 1e5);
                    k[6*i+2*j+1] = (int)round(M.m[i][j].imag() * 1e5);
                }
            return k;
        };
        std::set<std::vector<int>> seen;
        CMat3 eye = {}; eye.m[0][0] = eye.m[1][1] = eye.m[2][2] = cd(1,0);
        std::vector<CMat3> bfs_q = {eye};
        seen.insert(mat_key(eye));
        CMat3 gens[4] = {genH, genX, genS, genSi};
        std::vector<int> bfs_parent = {-1};
        std::vector<int> bfs_gen    = {-1};
        size_t head = 0;
        while (head < bfs_q.size() && bfs_q.size() < 700) {
            int parent_idx = (int)head;
            CMat3 M = bfs_q[head++];
            for (int g = 0; g < 4; ++g) {
                CMat3 P = cmul(M, gens[g]);
                auto k = mat_key(P);
                if (seen.find(k) == seen.end()) {
                    seen.insert(k);
                    bfs_q.push_back(P);
                    bfs_parent.push_back(parent_idx);
                    bfs_gen.push_back(g);
                }
            }
        }
        int n_cliff = (int)bfs_q.size();

        // ringZ9chi version of every Clifford in BFS order.
        Mat3 ring_gens[4] = { gateH(), gateX(),
                              gateD_clifford(1, 0, 0),
                              gateD_clifford(2, 0, 0) };
        std::vector<Mat3> ring_cliffords(n_cliff);
        Mat3 eye_ring;
        ringZ9 one_z9(1);
        eye_ring.m[0][0] = ringZ9chi(one_z9, 0);
        eye_ring.m[1][1] = ringZ9chi(one_z9, 0);
        eye_ring.m[2][2] = ringZ9chi(one_z9, 0);
        ring_cliffords[0] = eye_ring;
        for (int i = 1; i < n_cliff; ++i) {
            int p = bfs_parent[i];
            int g = bfs_gen[i];
            ring_cliffords[i] = ring_cliffords[p].mul(ring_gens[g]);
        }

        CliffordCache c;
        c.cliffords = std::move(bfs_q);
        c.ring_cliffords = std::move(ring_cliffords);
        c.ring_Dgate[0] = gateDgate();
        c.ring_Dgate[1] = c.ring_Dgate[0].mul(c.ring_Dgate[0]);
        c.Dgate[0] = {};
        c.Dgate[0].m[0][0] = z9; c.Dgate[0].m[1][1] = cd(1,0); c.Dgate[0].m[2][2] = conj(z9);
        c.Dgate[1] = {};
        c.Dgate[1].m[0][0] = z9*z9; c.Dgate[1].m[1][1] = cd(1,0); c.Dgate[1].m[2][2] = conj(z9)*conj(z9);
        c.n_cliff = n_cliff;
        return c;
    }();
    return cache;
}

DirectSearchResult directSearch(double theta, double epsilon, int max_d) {
    using cd = complex<double>;
    DirectSearchResult result;
    result.D_count = -1;
    result.frob_dist = 1e30;
    result.success = false;

    // Target: R_{(0,1)}^Z(theta) = Diag(e^{-itheta/2}, e^{itheta/2}, 1)
    CMat3 target = {};
    target.m[0][0] = polar(1.0, -theta/2.0);
    target.m[1][1] = polar(1.0,  theta/2.0);
    target.m[2][2] = cd(1.0, 0.0);

    // Reuse cached Cliffords + ring versions + Dgate matrices (built once).
    const CliffordCache& cache = get_clifford_cache();
    const std::vector<CMat3>& cliffords      = cache.cliffords;
    const std::vector<Mat3>&  ring_cliffords = cache.ring_cliffords;
    const Mat3*  ring_Dgate = cache.ring_Dgate;
    const CMat3* Dgate      = cache.Dgate;
    const int n_cliff       = cache.n_cliff;

    // Track the WINNING sequence: list of clifford indices interleaved with
    // D-gate exponents.  A k-D-gate sequence has k+1 Cliffords:
    //   V = C_0 · D^{e_0} · C_1 · D^{e_1} · ... · D^{e_{k-1}} · C_k
    // We record win_cliff = [C_0, C_1, ..., C_k]  (size k+1)
    //          win_dexp  = [e_0, e_1, ..., e_{k-1}]  (size k)
    vector<int> win_cliff;
    vector<int> win_dexp;

    cout << "Direct search: " << n_cliff << " Cliffords, eps=" << epsilon << endl;

    // =================================================================
    //  k=0
    // =================================================================
    if (max_d >= 0) {
        int win_i = -1;
        for (int i = 0; i < n_cliff; ++i) {
            double d = frobenius_dist(cliffords[i], target);
            if (d < result.frob_dist) {
                result.frob_dist = d;
                if (d < epsilon) {
                    result.D_count = 0;
                    result.success = true;
                    win_i = i;
                }
            }
        }
        cout << "  k=0: best=" << std::setprecision(15) << result.frob_dist << std::setprecision(6) << endl;
        if (result.success) {
            // Reify V = cliffords[win_i] over ringZ9chi.
            result.V = ring_cliffords[win_i];
            win_cliff = {win_i};
            cout << "  Found at k=0!" << endl;
            return result;
        }
    }

    // =================================================================
    //  k=1: C_L · D^e · C_R  (648 × 2 × 648 = 840K)
    // =================================================================
    if (max_d >= 1) {
        double best1 = 1e30;
        int win_l = -1, win_e = -1, win_r = -1;
        for (int e = 0; e < 2; ++e) {
            vector<CMat3> dc(n_cliff);
            for (int r = 0; r < n_cliff; ++r)
                dc[r] = cmul(Dgate[e], cliffords[r]);
            for (int l = 0; l < n_cliff; ++l)
                for (int r = 0; r < n_cliff; ++r) {
                    CMat3 V = cmul(cliffords[l], dc[r]);
                    double d = frobenius_dist(V, target);
                    if (d < best1) best1 = d;
                    if (d < result.frob_dist) {
                        result.frob_dist = d;
                        if (d < epsilon) {
                            result.D_count = 1;
                            result.success = true;
                            win_l = l; win_e = e; win_r = r;
                        }
                    }
                }
        }
        cout << "  k=1: best=" << std::setprecision(15) << best1 << std::setprecision(6) << endl;
        if (result.success && result.D_count == 1) {
            // Reify V = cliffords[win_l] · Dgate^{win_e} · cliffords[win_r]
            Mat3 v_ring = ring_cliffords[win_l]
                              .mul(ring_Dgate[win_e])
                              .mul(ring_cliffords[win_r]);
            result.V = v_ring;
            win_cliff = {win_l, win_r};
            win_dexp  = {win_e};
            cout << "  Found at k=1!" << endl;
            return result;
        }
    }

    // =================================================================
    //  k=2: LEFT · D^e2 · C2   where LEFT = C0 · D^e1 · C1 (k=1 product)
    //  840K lefts × 1296 rights = ~1.1B (parallelized)
    // =================================================================
    if (max_d >= 2) {
        // Build all k=1 products + track the (e1, c0, c1) decomposition for
        // each so we can reify the winning V over ringZ9chi.
        vector<CMat3> k1;
        k1.reserve(2 * n_cliff * n_cliff);
        struct K1Idx { int c0, e1, c1; };
        vector<K1Idx> k1_idx;
        k1_idx.reserve(2 * n_cliff * n_cliff);
        for (int e = 0; e < 2; ++e) {
            vector<CMat3> dc(n_cliff);
            for (int c1 = 0; c1 < n_cliff; ++c1)
                dc[c1] = cmul(Dgate[e], cliffords[c1]);
            for (int c0 = 0; c0 < n_cliff; ++c0)
                for (int c1 = 0; c1 < n_cliff; ++c1) {
                    k1.push_back(cmul(cliffords[c0], dc[c1]));
                    k1_idx.push_back({c0, e, c1});
                }
        }
        int n_k1 = (int)k1.size();

        // Right factors: D^e2 · C2 with index tracking.
        vector<CMat3> rights;
        rights.reserve(2 * n_cliff);
        struct RightIdx { int e2, c2; };
        vector<RightIdx> right_idx;
        right_idx.reserve(2 * n_cliff);
        for (int e2 = 0; e2 < 2; ++e2)
            for (int c2 = 0; c2 < n_cliff; ++c2) {
                rights.push_back(cmul(Dgate[e2], cliffords[c2]));
                right_idx.push_back({e2, c2});
            }
        int n_right = (int)rights.size();

        cout << "  k=2: " << n_k1 << " × " << n_right << " = "
             << (long long)n_k1*n_right << endl;

        // Per-thread tracking of best winning indices under reduction.
        // Early-exit flag: once any thread finds d < epsilon we can stop the
        // remaining outer iterations since success only requires *some* hit.
        const int n_threads = omp_get_max_threads();
        vector<double> tl_best(n_threads, 1e30);
        vector<int>    tl_l(n_threads, -1);
        vector<int>    tl_r(n_threads, -1);
        double best2 = 1e30;
        std::atomic<bool> found2(false);
        #pragma omp parallel for schedule(dynamic, 64) reduction(min:best2)
        for (int l = 0; l < n_k1; ++l) {
            if (found2.load(std::memory_order_relaxed)) continue;
            int tid = omp_get_thread_num();
            for (int r = 0; r < n_right; ++r) {
                CMat3 V = cmul(k1[l], rights[r]);
                double d = frobenius_dist(V, target);
                if (d < best2) best2 = d;
                if (d < tl_best[tid] && d < epsilon) {
                    tl_best[tid] = d;
                    tl_l[tid] = l;
                    tl_r[tid] = r;
                    found2.store(true, std::memory_order_relaxed);
                }
            }
        }
        // Merge per-thread bests.
        int win_l_k2 = -1, win_r_k2 = -1;
        double win_d_k2 = 1e30;
        for (int t = 0; t < n_threads; ++t)
            if (tl_l[t] >= 0 && tl_best[t] < win_d_k2) {
                win_d_k2 = tl_best[t];
                win_l_k2 = tl_l[t];
                win_r_k2 = tl_r[t];
            }
        result.frob_dist = min(result.frob_dist, best2);
        if (best2 < epsilon) { result.D_count = 2; result.success = true; }
        cout << "  k=2: best=" << std::setprecision(15) << best2 << std::setprecision(6) << endl;
        if (result.success && result.D_count == 2) {
            // Reify V = cliffords[c0]·Dgate^{e1}·cliffords[c1]·Dgate^{e2}·cliffords[c2]
            const K1Idx& K = k1_idx[win_l_k2];
            const RightIdx& R = right_idx[win_r_k2];
            Mat3 v_ring = ring_cliffords[K.c0]
                              .mul(ring_Dgate[K.e1])
                              .mul(ring_cliffords[K.c1])
                              .mul(ring_Dgate[R.e2])
                              .mul(ring_cliffords[R.c2]);
            result.V = v_ring;
            win_cliff = {K.c0, K.c1, R.c2};
            win_dexp  = {K.e1, R.e2};
            cout << "  Found at k=2!" << endl;
            return result;
        }

        // =================================================================
        //  k=3: meet-in-the-middle using k=1 products
        //  V = LEFT · D^e2 · RIGHT  where LEFT, RIGHT are k=1 products.
        //  Rearrange: D^e2 · RIGHT = LEFT† · target
        //  Hash all "D^e2 · k1" products, then for each LEFT compute LEFT†·target
        //  and look up nearest neighbor in the hash.
        // =================================================================
        if (max_d >= 3) {
            cout << "  k=3: building hash of " << n_k1*2 << " right halves..." << endl;

            double grid = max(epsilon * 0.5, 0.005);
            auto hash_key = [&grid](cd v00, cd v11) -> long long {
                int r0 = (int)round(v00.real() / grid);
                int i0 = (int)round(v00.imag() / grid);
                int r1 = (int)round(v11.real() / grid);
                int i1 = (int)round(v11.imag() / grid);
                return ((long long)(r0+5000) * 10001 + (i0+5000)) * 100020001LL
                     + ((long long)(r1+5000) * 10001 + (i1+5000));
            };

            // Build hash: D^e2 · k1_product
            unordered_multimap<long long, int> rh_hash;
            vector<CMat3> rh_vec;
            rh_vec.reserve(n_k1 * 2);
            for (int e2 = 0; e2 < 2; ++e2)
                for (int r = 0; r < n_k1; ++r) {
                    CMat3 rh = cmul(Dgate[e2], k1[r]);
                    int idx = (int)rh_vec.size();
                    rh_vec.push_back(rh);
                    rh_hash.emplace(hash_key(rh.m[0][0], rh.m[1][1]), idx);
                }

            cout << "  k=3: searching " << n_k1 << " left halves..." << endl;
            // Track winning sequence so V can be reified on success.
            // Each (l, idx) pair encodes V = k1[l] · rh_vec[idx]; rh_vec layout
            // is e2-major: idx = e2 * n_k1 + r, where (c0,e1,c1) = k1_idx[l]
            // is the LEFT and (c2,e3,c3) = k1_idx[r] is the right-k1 inside rh_vec.
            struct K3Win { double dist; int l; int idx; };
            K3Win win = { 1e30, -1, -1 };

            // Early-exit flag: once any thread finds a hit within epsilon, others
            // skip remaining work.  This keeps best-effort enumeration semantics
            // for the "no hit" path (full search) while making the success path fast.
            std::atomic<bool> found3(false);
            #pragma omp parallel
            {
                K3Win local = { 1e30, -1, -1 };
                #pragma omp for schedule(dynamic, 256) nowait
                for (int l = 0; l < n_k1; ++l) {
                    if (found3.load(std::memory_order_relaxed)) continue;
                    // L† · target
                    CMat3 Lt = {};
                    for (int i = 0; i < 3; ++i)
                        for (int j = 0; j < 3; ++j)
                            Lt.m[i][j] = conj(k1[l].m[j][i]);
                    CMat3 goal = cmul(Lt, target);

                    // Check hash neighbors
                    for (int dr0 = -1; dr0 <= 1; ++dr0)
                    for (int di0 = -1; di0 <= 1; ++di0)
                    for (int dr1 = -1; dr1 <= 1; ++dr1)
                    for (int di1 = -1; di1 <= 1; ++di1) {
                        cd g00 = goal.m[0][0] + cd(dr0*grid, di0*grid);
                        cd g11 = goal.m[1][1] + cd(dr1*grid, di1*grid);
                        long long hk = hash_key(g00, g11);
                        auto range = rh_hash.equal_range(hk);
                        for (auto it = range.first; it != range.second; ++it) {
                            double d = frobenius_dist(goal, rh_vec[it->second]);
                            if (d < local.dist) {
                                local.dist = d;
                                local.l = l;
                                local.idx = it->second;
                                if (d < epsilon) found3.store(true, std::memory_order_relaxed);
                            }
                        }
                    }
                }
                #pragma omp critical
                {
                    if (local.dist < win.dist) win = local;
                }
            }

            double best3 = win.dist;
            result.frob_dist = min(result.frob_dist, best3);
            cout << "  k=3: best=" << std::setprecision(15) << best3 << std::setprecision(6) << endl;
            if (best3 < epsilon && win.l >= 0 && win.idx >= 0) {
                // Recover (c0,e1,c1) for LEFT and (e2; c2,e3,c3) for RIGHT.
                int e2 = win.idx / n_k1;
                int r  = win.idx % n_k1;
                K1Idx L = k1_idx[win.l];
                K1Idx R = k1_idx[r];
                // Reify V over ringZ9chi:
                // V = C_L.c0 · D^{L.e1} · C_L.c1 · D^{e2} · C_R.c0 · D^{R.e1} · C_R.c1
                Mat3 v_ring = ring_cliffords[L.c0]
                                  .mul(ring_Dgate[L.e1])
                                  .mul(ring_cliffords[L.c1])
                                  .mul(ring_Dgate[e2])
                                  .mul(ring_cliffords[R.c0])
                                  .mul(ring_Dgate[R.e1])
                                  .mul(ring_cliffords[R.c1]);
                result.V = v_ring;
                result.D_count = 3;
                result.success = true;
                win_cliff = {L.c0, L.c1, R.c0, R.c1};
                win_dexp  = {L.e1, e2,   R.e1};
                cout << "  Found at k=3! (l=" << win.l << ", idx=" << win.idx
                     << ", c=[" << L.c0 << "," << L.c1 << "," << R.c0 << "," << R.c1
                     << "], e=[" << L.e1 << "," << e2 << "," << R.e1 << "])" << endl;
                return result;
            }
        }
    }

    if (!result.success) {
        cout << "  No solution found up to D-count=" << max_d
             << " (best=" << result.frob_dist << ")" << endl;
    }
    return result;
}


// =========================================================================
//  diagSearch + findClosestLatticePoints REMOVED 2026-05.
//
//  diagSearch was a heuristic that:
//    (a) had a buffer-overflow bug in findClosestLatticePoints (declared
//        int arr[6] but ringZ9::ringZ9(int arr[9]) reads 9 ints; slots
//        6..8 came from stack garbage; reduce() then subtracted that
//        garbage from the canonical slots → corrupted ringZ9 elements).
//    (b) returned NON-UNITARY V's (no |·|² = 3^{2f} constraint) even
//        absent the overflow bug.
//    (c) decompose(V) on those non-unitary V's took the diagonal-V
//        fast-path returning a heuristic D-count with NO syllables; the
//        "circuit" was never actually constructed.
//
//  The reported V Frobenius was therefore not the implementable circuit's
//  accuracy (since no circuit was constructed). HRSA's Phase 1 (direct
//  search) and Phase 3 (Householder) both produce honest exactly-unitary
//  results and remain.
// =========================================================================

// =========================================================================
// Templated-decompose bridge (2026-05-13 refactor).
//
// Pull in the templated bodies and emit explicit instantiations for the int
// path so external translation units (HRSA_test, householder_search, etc.)
// link against the same exact symbol that the original non-templated
// `DecompResult decompose(Mat3, bool)` resolved to.
//
// The cpp_int instantiation is emitted by decompose_cli.cpp (which also
// #includes decompose_impl.h with T=cpp_int).
// =========================================================================
#include "decompose_impl.h"

// Explicit instantiations.
template DecompResultBase<int> decompose<int>(Mat3Base<int> V, bool quiet);
template int sdeChiFull<int>(const ringZ9chiBase<int>& x);
template int sdeChiZ9<int>(ringZ9Base<int> a);
template Mat3Base<int> buildUnitary<int>(const std::array<ringZ9chiBase<int>,3>& u);

// Re-introduce countDgates (now uses the templated int instantiation).
int countDgates(const std::array<ringZ9chi,3>& u) {
    Mat3 V = buildUnitary<int>(u);
    DecompResult dr = decompose<int>(V);
    return dr.D_count;
}
