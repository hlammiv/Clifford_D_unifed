#ifndef DECOMPOSE_H
#define DECOMPOSE_H

#include <array>
#include <vector>
#include <string>
#include "cyclotomic_int9.h"
#include "Z9chi.h"

/**
 * @file decompose.h
 * @brief Exact synthesis of a unitary in U(3, Z[zeta_9, 1/3]) into a product
 *        of Clifford+D generators, following the sde-reduction approach of
 *        Kalra et al. and the C+R decomposition of Gustafson et al.
 *
 * 2026-05-13: Templated rewrite.  Mat3Base<T> stores entries of ringZ9chiBase<T>.
 *   - `using Mat3    = Mat3Base<int>`            : HRSA hot path (existing callers).
 *   - `using Mat3Big = Mat3Base<cpp_int>`        : decompose_tool post-processing of
 *     SK-assembled exact unitaries, whose coefficients can reach O(3^f) with f≥40
 *     and thus overflow int.
 *
 * The function template `decompose<T>(Mat3Base<T> V, ...)` works for both
 * int and cpp_int instantiations.  External non-templated callers see
 * `decompose(Mat3 V, ...)` via the default typedef and template argument
 * deduction; the int instantiation is provided by decompose.cpp.  Translation
 * units that need a non-int instantiation (e.g. decompose_cli.cpp for cpp_int)
 * should `#include "decompose_impl.h"` to pull in the template definition.
 */

/// A 3x3 matrix over ringZ9chiBase<T>.
template<typename T>
struct Mat3Base {
    ringZ9chiBase<T> m[3][3];

    Mat3Base() {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                m[i][j] = ringZ9chiBase<T>();
    }

    Mat3Base mul(const Mat3Base& B) const {
        Mat3Base C;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                    C.m[i][j] = C.m[i][j] + m[i][k] * B.m[k][j];
        return C;
    }

    Mat3Base dagger() const {
        Mat3Base D;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                D.m[i][j] = m[j][i].complexConj();
        return D;
    }

    bool equals(const Mat3Base& B) const {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                if (!(m[i][j] == B.m[i][j])) return false;
        return true;
    }

    void print() const {
        for (int i = 0; i < 3; ++i) {
            std::cout << "[ ";
            for (int j = 0; j < 3; ++j) {
                m[i][j].print();
                if (j < 2) std::cout << ", ";
            }
            std::cout << " ]" << std::endl;
        }
    }
};

// Public typedefs.
using Mat3    = Mat3Base<int>;
using Mat3Big = Mat3Base<boost::multiprecision::cpp_int>;

// Cross-storage conversion: build a Mat3Base<T_dst> from a Mat3Base<T_src> by
// copying each ringZ9chiBase entry through its cross-template ctor.
//
// The same-type overload below short-circuits to a single Mat3 copy (no
// per-element promotion), so the int-only hot path pays only a struct copy
// instead of 9 ringZ9chi copies via the SFINAE'd cross-template ctor path.
template<typename T_dst, typename T_src>
inline typename std::enable_if<!std::is_same<T_dst, T_src>::value, Mat3Base<T_dst>>::type
cast_mat3(const Mat3Base<T_src>& src) {
    Mat3Base<T_dst> dst;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            dst.m[i][j] = ringZ9chiBase<T_dst>(src.m[i][j]);
    return dst;
}

template<typename T_dst, typename T_src>
inline typename std::enable_if<std::is_same<T_dst, T_src>::value, const Mat3Base<T_dst>&>::type
cast_mat3(const Mat3Base<T_src>& src) {
    return src;
}

/// One step in the gate sequence: H D(a0,a1,a2) Dgate^eps X^delta
struct GateStep {
    int a0, a1, a2;
    int eps;
    int delta;
    bool has_H;
};

/// Result of the decomposition.
template<typename T>
struct DecompResultBase {
    std::vector<GateStep> steps;
    int D_count;
    int D_only;
    int R_only;
    int sde_chi;
    int sde_chi_final;
    Mat3Base<T> trailing_clifford;
    bool success;
};

using DecompResult    = DecompResultBase<int>;
using DecompResultBig = DecompResultBase<boost::multiprecision::cpp_int>;

// ---- Fixed gate matrices (int storage; cheap to cast to T at call sites) ----

Mat3 gateH();
Mat3 gateD_clifford(int a, int b, int c);
Mat3 gateDgate();
Mat3 gateX();

// ---- sde_chi for ringZ9chi (templated) ----

template<typename T>
int sdeChiZ9(ringZ9Base<T> a);

template<typename T>
int sdeChiFull(const ringZ9chiBase<T>& x);

// ---- Decomposition (templated) ----

template<typename T>
DecompResultBase<T> decompose(Mat3Base<T> V, bool quiet = false);

// gateStepPrefix returns an int-storage Mat3 (syllable prefixes are
// always int-coefficient regardless of the input unitary's precision).
Mat3 gateStepPrefix(const GateStep& s);

int simplify_steps(std::vector<GateStep>& steps, bool quiet = true);

template<typename T>
Mat3Base<T> buildUnitary(const std::array<ringZ9chiBase<T>,3>& u);

int countDgates(const std::array<ringZ9chi,3>& u);

// ---- Direct search (int-only; uses tiny-coefficient Cliffords + D gates) ----

struct DirectSearchResult {
    Mat3 V;
    int D_count;
    double frob_dist;
    bool success;
};

DirectSearchResult directSearch(double theta, double epsilon, int max_d);

// ---- Shared cache types referenced by both decompose.cpp (int) and
// ---- decompose_impl.h (templated body).
//
// The 4374-entry prefix table holds the int-storage prefix matrices
// H · Dcyclo(a1,a2,a3) · R^eps · X^delta along with their D-cost.  The
// templated decompose body pulls these from the table and promotes each
// entry to ringZ9chiBase<T> on the fly via cast_mat3<T,int>.
struct PrefixEntry {
    Mat3 P;
    int d;
    int a1, a2, a3, eps, delta;
};

const std::vector<PrefixEntry>& get_prefix_table();
int count_d_from_cyclo(int a1, int a2, int a3);
int count_d_from_syllable(int a1, int a2, int a3, int eps);

#endif
