#ifndef DECOMPOSE_IMPL_H
#define DECOMPOSE_IMPL_H

// decompose_impl.h
//
// Templated implementation of decompose<T>(), sdeChiFull<T>(), sdeChiZ9<T>(),
// and buildUnitary<T>().  Included by:
//   - decompose.cpp     (explicit-instantiates the int version for HRSA's
//                         hot path and other in-tree callers)
//   - decompose_cli.cpp  (instantiates the cpp_int version for SK post-processing)
//
// Non-templated helpers (get_prefix_table, simplify_steps, directSearch,
// get_clifford_cache, gateH/X/Dgate/Dcyclo, count_d_from_*) live in
// decompose.cpp and are referenced via the declarations in decompose.h.

#include "decompose.h"
#include "clifford_cache.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <set>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// Public knobs (defined in decompose.cpp).
extern bool g_decompose_simplify;
extern bool g_decompose_lookahead;

// PrefixEntry / get_prefix_table / count_d_from_cyclo / count_d_from_syllable
// are declared in decompose.h.

// =========================================================================
//  sde_chi (templated)
// =========================================================================
template<typename T>
int sdeChiZ9(ringZ9Base<T> a) {
    if (a.isZero()) return 999;
    int s = a.sdeChi();
    if (s < 6) return s;
    if (!a.isDivisibleByInt(3)) return 6;
    ringZ9Base<T> a_div3 = a / T(3);
    return 6 + sdeChiZ9<T>(a_div3);
}

template<typename T>
int sdeChiFull(const ringZ9chiBase<T>& x) {
    if (x.isZero()) return 0;
    ringZ9Base<T> numer = x.getNumerator();
    int f = x.getExp();
    if (f == 0) return sdeChiZ9<T>(numer);
    int ell = sdeChiZ9<T>(numer);
    return 6 * f - ell;
}

// =========================================================================
//  buildUnitary (templated)
// =========================================================================
template<typename T>
Mat3Base<T> buildUnitary(const std::array<ringZ9chiBase<T>,3>& u) {
    Mat3Base<T> H;
    ringZ9chiBase<T> one(ringZ9Base<T>(1), 0);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            ringZ9chiBase<T> uiuj = u[i] * u[j].complexConj();
            H.m[i][j] = (i == j ? one - uiuj : ringZ9chiBase<T>() - uiuj);
        }
    Mat3Base<T> V;
    for (int j = 0; j < 3; ++j) {
        V.m[0][j] = H.m[1][j];
        V.m[1][j] = H.m[0][j];
        V.m[2][j] = H.m[2][j];
    }
    return V;
}

// =========================================================================
//  Templated body helpers (internal — not exposed)
// =========================================================================
namespace decompose_impl_detail {

// Compute (prefix * V)[0][0] without building the full product.
// prefix is int-storage (Mat3), V is T-storage (Mat3Base<T>).  We promote
// prefix entries to ringZ9chiBase<T> on the fly via the cross-template ctor.
template<typename T>
inline ringZ9chiBase<T> prefix_times_V_00(const Mat3& prefix, const Mat3Base<T>& V) {
    ringZ9chiBase<T> result;
    for (int k = 0; k < 3; ++k) {
        ringZ9chiBase<T> p_kT(prefix.m[0][k]);  // int → T
        result = result + p_kT * V.m[k][0];
    }
    return result;
}

// prefix(int) * V(T) → Mat3Base<T>
template<typename T>
inline Mat3Base<T> prefix_times_V(const Mat3& prefix, const Mat3Base<T>& V) {
    Mat3Base<T> C;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) {
                ringZ9chiBase<T> p_ikT(prefix.m[i][k]);
                C.m[i][j] = C.m[i][j] + p_ikT * V.m[k][j];
            }
    return C;
}

// Compare entry-wise:  M[i][j] (T-storage) against C[i][j] (int-storage)
// without materializing a temporary Mat3Base<T> copy of C.  Hot-path used by
// countMonomialD's Clifford-cache match check.  For T==int we leverage the
// existing operator==; for T==cpp_int we walk the 6 coefficients comparing
// each (cpp_int) entry to the corresponding int promoted on the fly.
template<typename T>
inline bool ringchi_eq_int(const ringZ9chiBase<T>& a, const ringZ9chi& b) {
    if (a.getExp() != b.getExp()) return false;
    for (int i = 0; i < 6; ++i)
        if (a.getTerm(i) != T(b.getTerm(i))) return false;
    return true;
}

template<typename T>
inline bool mat3_eq_int(const Mat3Base<T>& A, const Mat3& B) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (!ringchi_eq_int<T>(A.m[i][j], B.m[i][j])) return false;
    return true;
}

// =========================================================================
//  trySinglePrefix / tryDoublePrefix (templated)
// =========================================================================
template<typename T>
bool trySinglePrefix(Mat3Base<T>& V, int s, DecompResultBase<T>& result) {
    const std::vector<PrefixEntry>& table = get_prefix_table();
    constexpr int N_PREFIX = 4374;
    int best_d = INT_MAX;
    int best_idx = -1;

    // Phase A: locate the min D-cost prefix that reduces sde_chi by 1.
    // Parallel min-find: each thread keeps its local (d, idx) winner, combine
    // at the end via critical section. Loses the early-exit on best_d==0
    // and the e.d-based pruning, but the work per iteration is cheap (one
    // ringZ9chi multiply + one sdeChi call), so the wash is positive at
    // 4374 / nthreads chunks.
    #pragma omp parallel
    {
        int local_best_d = INT_MAX;
        int local_best_idx = -1;
        #pragma omp for nowait schedule(static)
        for (int idx = 0; idx < N_PREFIX; ++idx) {
            const PrefixEntry& e = table[idx];
            if (e.d >= local_best_d) continue;
            ringZ9chiBase<T> new00 = prefix_times_V_00<T>(e.P, V);
            if (sdeChiFull<T>(new00) == s - 1) {
                local_best_d = e.d;
                local_best_idx = idx;
            }
        }
        #pragma omp critical
        {
            if (local_best_d < best_d || (local_best_d == best_d && local_best_idx >= 0 && local_best_idx < best_idx)) {
                best_d = local_best_d;
                best_idx = local_best_idx;
            }
        }
    }

    if (best_d == INT_MAX) return false;

    if (g_decompose_lookahead && (s - 1) > 0) {
        int target_s = s - 1;
        int best_followup_d = INT_MAX;

        // Phase B: among prefixes tied at best_d that reduce sde by 1, pick
        // the one whose follow-up has the cheapest D-cost. The outer 4374
        // loop is independent across idx; inner 4374 follow-up scan stays
        // serial inside each thread to keep the work granularity reasonable.
        #pragma omp parallel
        {
            int local_best_followup_d = INT_MAX;
            int local_best_idx = best_idx;  // start from Phase A winner
            #pragma omp for nowait schedule(dynamic, 64)
            for (int idx = 0; idx < N_PREFIX; ++idx) {
                const PrefixEntry& e = table[idx];
                if (e.d != best_d) continue;

                ringZ9chiBase<T> new00 = prefix_times_V_00<T>(e.P, V);
                if (sdeChiFull<T>(new00) != target_s) continue;

                Mat3Base<T> V_next = prefix_times_V<T>(e.P, V);
                int followup_d = INT_MAX;
                for (int idx2 = 0; idx2 < N_PREFIX; ++idx2) {
                    const PrefixEntry& e2 = table[idx2];
                    if (e2.d >= followup_d) continue;
                    ringZ9chiBase<T> nn00 = prefix_times_V_00<T>(e2.P, V_next);
                    if (sdeChiFull<T>(nn00) == target_s - 1) {
                        followup_d = e2.d;
                        if (followup_d == 0) break;
                    }
                }

                if (followup_d < local_best_followup_d) {
                    local_best_followup_d = followup_d;
                    local_best_idx = idx;
                }
            }
            #pragma omp critical
            {
                if (local_best_followup_d < best_followup_d) {
                    best_followup_d = local_best_followup_d;
                    best_idx = local_best_idx;
                }
            }
        }
    }

    {
        const PrefixEntry& be = table[best_idx];
        GateStep step;
        step.a0 = be.a1; step.a1 = be.a2; step.a2 = be.a3;
        step.eps = be.eps; step.delta = be.delta; step.has_H = true;
        result.steps.push_back(step);
        result.D_count += be.d;
        V = prefix_times_V<T>(be.P, V);
    }
    return true;
}

template<typename T>
bool tryDoublePrefix(Mat3Base<T>& V, int s, DecompResultBase<T>& result,
                     bool quiet = false) {
    if (!quiet) std::cout << "  Trying double-prefix at sde_chi=" << s << "..." << std::endl;
    const std::vector<PrefixEntry>& table = get_prefix_table();
    constexpr int N_PREFIX = 4374;

    int best_total_d = INT_MAX;
    int best_new_s = s;
    int best_mid_s = s;
    int best_idx1 = -1, best_idx2 = -1;

    // Parallelize the outer 4374-iter loop; inner 4374-iter stays serial per
    // thread. 4374² = ~19M iter pairs in the worst case, big speedup target.
    // Each thread keeps its local best, combine via critical at the end.
    #pragma omp parallel
    {
        int local_best_total_d = INT_MAX;
        int local_best_new_s = s;
        int local_best_mid_s = s;
        int local_best_idx1 = -1, local_best_idx2 = -1;

        #pragma omp for nowait schedule(dynamic, 32)
        for (int idx1 = 0; idx1 < N_PREFIX; ++idx1) {
            const PrefixEntry& e1 = table[idx1];
            if (e1.d >= local_best_total_d) continue;

            ringZ9chiBase<T> mid00 = prefix_times_V_00<T>(e1.P, V);
            int mid_s = sdeChiFull<T>(mid00);

            if (mid_s > s + 1 || mid_s < s - 1) continue;

            Mat3Base<T> midV = prefix_times_V<T>(e1.P, V);
            for (int idx2 = 0; idx2 < N_PREFIX; ++idx2) {
                const PrefixEntry& e2 = table[idx2];
                if (e1.d + e2.d >= local_best_total_d) continue;

                ringZ9chiBase<T> new00 = prefix_times_V_00<T>(e2.P, midV);
                int new_s = sdeChiFull<T>(new00);

                if (new_s < s) {
                    local_best_total_d = e1.d + e2.d;
                    local_best_new_s = new_s;
                    local_best_mid_s = mid_s;
                    local_best_idx1 = idx1;
                    local_best_idx2 = idx2;
                    if (local_best_total_d == 0) break;  // best possible; exit inner loop
                }
            }
        }

        #pragma omp critical
        {
            if (local_best_total_d < best_total_d) {
                best_total_d = local_best_total_d;
                best_new_s = local_best_new_s;
                best_mid_s = local_best_mid_s;
                best_idx1 = local_best_idx1;
                best_idx2 = local_best_idx2;
            }
        }
    }

    if (best_total_d == INT_MAX) return false;

    {
        const PrefixEntry& be1 = table[best_idx1];
        const PrefixEntry& be2 = table[best_idx2];

        GateStep step1;
        step1.a0 = be1.a1; step1.a1 = be1.a2; step1.a2 = be1.a3;
        step1.eps = be1.eps; step1.delta = be1.delta; step1.has_H = true;
        result.steps.push_back(step1);
        result.D_count += be1.d;

        Mat3Base<T> midV = prefix_times_V<T>(be1.P, V);

        GateStep step2;
        step2.a0 = be2.a1; step2.a1 = be2.a2; step2.a2 = be2.a3;
        step2.eps = be2.eps; step2.delta = be2.delta; step2.has_H = true;
        result.steps.push_back(step2);
        result.D_count += be2.d;

        V = prefix_times_V<T>(be2.P, midV);
        if (!quiet) std::cout << "  Double-prefix succeeded: sde " << s
             << " -> " << best_mid_s << " -> " << best_new_s << std::endl;
    }
    return true;
}

}  // namespace decompose_impl_detail

// =========================================================================
//  decompose (templated body)
// =========================================================================
template<typename T>
DecompResultBase<T> decompose(Mat3Base<T> V, bool quiet) {
    using namespace decompose_impl_detail;

    DecompResultBase<T> result;
    result.D_count = 0;
    result.D_only  = 0;
    result.R_only  = 0;
    result.sde_chi = 0;
    result.sde_chi_final = 0;
    result.success = false;

    // -- unitPhaseMod3: determine zeta_9 exponent mod 3 of a unit in Z[zeta_9].
    auto unitPhaseMod3 = [](const ringZ9chiBase<T>& x) -> int {
        ringZ9Base<T> numer = x.getNumerator();
        for (int j = 0; j < 9; ++j) {
            int conj_exp = (9 - j) % 9;
            ringZ9Base<T> zj(1, conj_exp);
            ringZ9Base<T> prod = numer * zj;
            bool is_pm1 = true;
            for (int k = 1; k < 6; ++k) {
                if (prod.getTerm(k) != T(0)) { is_pm1 = false; break; }
            }
            if (is_pm1 && (prod.getTerm(0) == T(1) || prod.getTerm(0) == T(-1))) {
                return j % 3;
            }
        }
        return -1;
    };

    // -- countMonomialD: returns total D+R cost (Convention B); is_monomial / r_out outputs.
    auto countMonomialD = [&unitPhaseMod3](const Mat3Base<T>& M, bool& is_monomial, int& r_out) -> int {
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
            // Check against the 648 Cliffords.  ring_cliffords are int-storage;
            // promote each to Mat3Base<T> for the comparison.
            const auto& cliffs = get_clifford_cache().ring_cliffords;
            auto exact_match = [&](const Mat3Base<T>& M_test) -> bool {
                // Compare against each int-storage Clifford directly; avoids
                // 648 per-call cast_mat3 copies on the int path.
                for (const auto& c : cliffs) {
                    if (mat3_eq_int<T>(M_test, c)) return true;
                }
                return false;
            };
            if (exact_match(M)) return 0;
            int best = -1;
            ringZ9Base<T> one_z(1);
            ringZ9Base<T> neg_z(-1);
            ringZ9chiBase<T> p1(one_z, 0), m1(neg_z, 0);
            ringZ9chiBase<T> sign_arr[2] = {p1, m1};
            for (int s0 = 0; s0 < 2; ++s0)
                for (int s1 = 0; s1 < 2; ++s1)
                    for (int s2 = 0; s2 < 2; ++s2) {
                        if (s0 == 0 && s1 == 0 && s2 == 0) continue;
                        Mat3Base<T> Mtest;
                        for (int i = 0; i < 3; ++i)
                            for (int j = 0; j < 3; ++j) {
                                ringZ9chiBase<T> sign_factor;
                                if (i == 0) sign_factor = sign_arr[s0];
                                else if (i == 1) sign_factor = sign_arr[s1];
                                else sign_factor = sign_arr[s2];
                                Mtest.m[i][j] = sign_factor * M.m[i][j];
                            }
                        if (exact_match(Mtest)) {
                            if (best < 0 || 1 < best) best = 1;
                        }
                    }
            if (best >= 0) {
                r_out = best;
                return best;
            }
            is_monomial = false;
            return -1;
        }
        bool one_gate =
            (p==1&&q==0&&r==2) || (p==2&&q==0&&r==1) ||
            (p==0&&q==1&&r==2) || (p==0&&q==2&&r==1) ||
            (p==1&&q==2&&r==0) || (p==2&&q==1&&r==0);
        return one_gate ? 1 : 2;
    };

    // Fast-path: V already monomial?
    bool is_mono = false;
    int mono_r = 0;
    int mono_d = countMonomialD(V, is_mono, mono_r);
    if (is_mono) {
        result.D_count = mono_d;
        result.R_only  = mono_r;
        result.D_only  = mono_d - mono_r;
        result.sde_chi = 0;
        result.sde_chi_final = 0;
        result.trailing_clifford = V;
        result.success = true;
        return result;
    }

    // Fast-path: single-H-layer (|entries| ≈ 1/√3)
    {
        double target_mag = 1.0 / std::sqrt(3.0);
        bool all_same_mag = true;
        for (int i = 0; i < 3 && all_same_mag; ++i)
            for (int j = 0; j < 3 && all_same_mag; ++j) {
                double mag = std::abs(V.m[i][j].toComplexDouble());
                if (std::abs(mag - target_mag) > 0.01) all_same_mag = false;
            }

        if (all_same_mag) {
            // Build H-variants in int; convert to T for the residual check.
            Mat3 Hmat_local = gateH();
            Mat3 Xmat_local = gateX();
            Mat3 X2 = Xmat_local.mul(Xmat_local);

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
                Mat3Base<T> Hv_dag_T = cast_mat3<T, int>(H_variants[h].dagger());
                Mat3Base<T> residual = V.mul(Hv_dag_T);
                bool rm = false;
                int rr = 0;
                int d = countMonomialD(residual, rm, rr);
                if (rm && d < best_d) { best_d = d; best_r = rr; }
            }

            if (best_d < INT_MAX) {
                result.D_count = best_d;
                result.R_only  = best_r;
                result.D_only  = best_d - best_r;
                result.sde_chi = sdeChiFull<T>(V.m[0][0]);
                result.success = true;
                return result;
            }
        }
    }

    // General sde-peeling.
    int s = sdeChiFull<T>(V.m[0][0]);
    result.sde_chi = s;

    if (s == 999) {
        if (!quiet) std::cout << "Warning: (0,0) entry is zero in decompose" << std::endl;
        result.sde_chi_final = s;
        result.trailing_clifford = V;
        return result;
    }

    int max_iter = s + 20;
    int iter = 0;

    while (s > 0 && iter < max_iter) {
        iter++;
        if (trySinglePrefix<T>(V, s, result)) {
            s = sdeChiFull<T>(V.m[0][0]);
            continue;
        }
        if (tryDoublePrefix<T>(V, s, result, quiet)) {
            s = sdeChiFull<T>(V.m[0][0]);
            continue;
        }
        if (!quiet) std::cout << "Decompose: cannot reduce sde_chi at s=" << s
             << " even with double-prefix" << std::endl;
        result.sde_chi_final = s;
        result.trailing_clifford = V;
        return result;
    }

    result.success = (s == 0);
    result.sde_chi_final = s;
    result.trailing_clifford = V;

    if (result.success) {
        bool resid_mono = false;
        int resid_r = 0;
        int resid_d = countMonomialD(V, resid_mono, resid_r);
        if (resid_mono && resid_d > 0) {
            result.D_count += resid_d;
            result.R_only  += resid_r;
            result.D_only  += (resid_d - resid_r);
        } else if (!resid_mono) {
            if (!quiet) std::cout << "Decompose: residual V is non-Clifford after peeling — D-count unknown" << std::endl;
            result.success = false;
            result.D_count = -1;
        }
    }

    // simplify_steps is int-only (it manipulates the int-coefficient GateStep
    // list — the syllable prefixes themselves are always int).  Applies in
    // both int and big-int decompose paths.
    if (g_decompose_simplify && !result.steps.empty()) {
        int prior_syll_d = 0;
        for (const GateStep& st : result.steps)
            prior_syll_d += count_d_from_syllable(st.a0, st.a1, st.a2, st.eps);
        int new_syll_d = simplify_steps(result.steps, quiet);
        result.D_count += (new_syll_d - prior_syll_d);
    }

    for (const GateStep& st : result.steps) {
        result.D_only += count_d_from_cyclo(st.a0, st.a1, st.a2);
        result.R_only += (st.eps != 0 ? 1 : 0);
    }

    return result;
}

#endif  // DECOMPOSE_IMPL_H
