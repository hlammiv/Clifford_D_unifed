// e0_net_dump.cpp — dump the inverse-closed sign-extended Clifford set used as
// the Solovay-Kitaev epsilon_0-net.
//
// HISTORY
// -------
// Originally this binary emitted exactly N = 8 * 648 = 5184 elements of the
// form (sign · Clifford).  That set is NOT closed under matrix inverse:
// (sign · C)^dagger = C^dagger · sign, which is in general (Clifford · sign'),
// i.e. a *right* sign-extension, not in the original (left-extension) family.
// In the empirical SK driver only ~44% of the 5184 elements had their inverse
// in the same set, which caused ~1e-6 drift in reverse-words built from the
// "inverse lookup" (sk_driver.py:build_inverse_lookup).
//
// CURRENT BEHAVIOUR (default)
// ---------------------------
// 1. Build the 5184 sign-extended Cliffords.
// 2. Iteratively add their Hermitian conjugates until the set is closed.
// 3. Write the closed set to /tmp/e0_net_closed.txt (or the path passed on
//    argv).  Each line keeps the same column layout as the original dump:
//        idx sign_pattern clifford_idx  <18 complex doubles row-major>  <54 ints>
//    For elements that arose from the inverse-closure pass and are not of the
//    form (sign · Clifford), sign_pattern = -1 and clifford_idx = -1 are
//    written as sentinels (existing Python loaders parse parts[1]/parts[2]
//    as ints — they still parse, just need to handle the sentinel value).
//
// LEGACY MODE
// -----------
// Pass --unclosed (anywhere on argv) to fall back to the original
// 5184-element dump.  Default output path in that mode is
// /tmp/e0_net_5184.txt.

#include "clifford_cache.h"
#include "decompose.h"
#include "cyclotomic_int9.h"
#include "Z9chi.h"
#include <complex>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <vector>

using cd = std::complex<double>;

// ---------------------------------------------------------------------------
// Element representation
// ---------------------------------------------------------------------------

struct NetElem {
    Mat3 V;            // exact ringZ9chi 3x3 (denom_exp = 0 for all entries)
    cd   M[3][3];      // complex-double cache for output / verification
    int  sign_pattern; // [0..7] for original (sign·Clifford), -1 for inv-closure
    int  clifford_idx; // [0..N) for original, -1 for inv-closure additions
};

// Build the canonical hash key: 54 int64s = 6 numerator terms * 9 matrix
// entries.  (All entries have denom_exp = 0 in this set: signs and Cliffords
// are denom-0, and Hermitian-conjugate (= elementwise GaloisAut(-1) + transpose)
// preserves denom_exp.)
static std::vector<int64_t> key_of(const Mat3& V) {
    std::vector<int64_t> k;
    k.reserve(54 + 9);  // entries + per-entry denom_exp (paranoia)
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const ringZ9chi& e = V.m[i][j];
            ringZ9 num = e.getNumerator();
            for (int t = 0; t < 6; ++t) k.push_back((int64_t)num.getTerm(t));
            k.push_back((int64_t)e.getExp());  // safety net; should be 0 here
        }
    }
    return k;
}

// Fill a CMat3-shaped buffer from a ringZ9chi Mat3 by per-entry conversion.
static void mat3_to_complex(const Mat3& V, cd out[3][3]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            out[i][j] = V.m[i][j].toComplexDouble();
}

static void write_elem(std::ofstream& f, int idx, const NetElem& e) {
    f << idx << " " << e.sign_pattern << " " << e.clifford_idx;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            f << " " << e.M[i][j].real() << " " << e.M[i][j].imag();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            ringZ9 num = e.V.m[i][j].getNumerator();
            for (int k = 0; k < 6; ++k)
                f << " " << num.getTerm(k);
        }
    }
    f << "\n";
}

int main(int argc, char* argv[]) {
    bool unclosed = false;
    const char* path_arg = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--unclosed") == 0) {
            unclosed = true;
        } else if (path_arg == nullptr) {
            path_arg = argv[i];
        } else {
            fprintf(stderr, "usage: %s [--unclosed] [output_path]\n", argv[0]);
            return 1;
        }
    }
    const char* out_path = path_arg ? path_arg
                                    : (unclosed ? "/tmp/e0_net_5184.txt"
                                                : "/tmp/e0_net_closed.txt");

    std::ofstream f(out_path);
    if (!f) { fprintf(stderr, "cannot open %s\n", out_path); return 1; }
    f << std::setprecision(17);

    const CliffordCache& cache = get_clifford_cache();
    const int N = (int)cache.cliffords.size();
    fprintf(stderr, "Clifford cache has %d entries; building initial %d-element set\n",
            N, 8 * N);

    // -----------------------------------------------------------------------
    // 1) Build the 5184 sign-extended Cliffords.
    // -----------------------------------------------------------------------
    std::vector<NetElem> elems;
    elems.reserve(8 * N);
    for (int sp = 0; sp < 8; ++sp) {
        for (int ci = 0; ci < N; ++ci) {
            NetElem e;
            e.sign_pattern = sp;
            e.clifford_idx = ci;

            // ringZ9chi Mat3: V[i][j] = (s_i ? -1 : +1) * ring_cliffords[ci].m[i][j]
            const Mat3& rc = cache.ring_cliffords[ci];
            ringZ9chi zero;
            for (int i = 0; i < 3; ++i) {
                int sgn = ((sp >> i) & 1) ? -1 : +1;
                for (int j = 0; j < 3; ++j) {
                    ringZ9chi t = rc.m[i][j];
                    e.V.m[i][j] = (sgn == -1) ? (zero - t) : t;
                }
            }
            // Complex-double cache (matches the original dump's M[i][j] exactly).
            cd s[3];
            for (int k = 0; k < 3; ++k)
                s[k] = ((sp >> k) & 1) ? cd(-1, 0) : cd(1, 0);
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    e.M[i][j] = s[i] * cache.cliffords[ci].m[i][j];
            elems.push_back(std::move(e));
        }
    }

    // -----------------------------------------------------------------------
    // Legacy --unclosed path: just write the 5184 and exit.
    // -----------------------------------------------------------------------
    if (unclosed) {
        f << "# e0_net_5184 (legacy --unclosed): 8 sign patterns x " << N
          << " Cliffords = " << (8 * N) << " elements\n";
        f << "# columns: idx sign_pattern clifford_idx  "
          << "Re00 Im00 Re01 Im01 ... Re22 Im22  "
          << "V_00_a0 V_00_a1 ... V_22_a5\n";
        int idx = 0;
        for (const NetElem& e : elems) write_elem(f, idx++, e);
        f.close();
        fprintf(stderr, "Wrote %d entries (legacy mode) to %s\n", idx, out_path);
        return 0;
    }

    // -----------------------------------------------------------------------
    // 2) Iteratively close under matrix inverse (= Hermitian conjugate, since
    //    every entry is a unitary).
    // -----------------------------------------------------------------------
    std::map<std::vector<int64_t>, int> seen;  // key -> index in elems
    for (size_t i = 0; i < elems.size(); ++i) {
        auto k = key_of(elems[i].V);
        // Dedup the initial set too — there shouldn't be collisions among the
        // 5184 distinct (sp, ci) pairs, but be defensive.
        if (seen.find(k) == seen.end()) {
            seen[k] = (int)i;
        }
    }
    const size_t initial_distinct = seen.size();
    fprintf(stderr, "Initial distinct elements: %zu (expected %d)\n",
            initial_distinct, 8 * N);

    // Worklist BFS: pop element, compute V^dagger, add if new.  Repeat until
    // every newly-added element has been processed.  For a finite group closed
    // under inverse this terminates in one pass over a unitary set, but we use
    // an iterative worklist to handle any sequencing.
    size_t cursor = 0;
    while (cursor < elems.size()) {
        const Mat3 V = elems[cursor].V;  // copy; dagger() is const but we mutate elems below
        Mat3 Vd = V.dagger();
        auto kd = key_of(Vd);
        if (seen.find(kd) == seen.end()) {
            NetElem ne;
            ne.V = Vd;
            mat3_to_complex(Vd, ne.M);
            ne.sign_pattern = -1;   // sentinel: inverse-closure addition
            ne.clifford_idx = -1;
            seen[kd] = (int)elems.size();
            elems.push_back(std::move(ne));
        }
        ++cursor;
    }

    fprintf(stderr, "Inverse-closed size: %zu (added %zu via closure)\n",
            elems.size(), elems.size() - initial_distinct);

    // -----------------------------------------------------------------------
    // 3) Output.
    // -----------------------------------------------------------------------
    f << "# e0_net_closed: inverse-closure of the 8 x " << N
      << " sign-extended Clifford set; total " << elems.size() << " elements\n";
    f << "# columns: idx sign_pattern clifford_idx  "
      << "Re00 Im00 Re01 Im01 ... Re22 Im22  "
      << "V_00_a0 V_00_a1 ... V_22_a5\n";
    f << "# NOTE: sign_pattern = -1 and clifford_idx = -1 are sentinels for\n"
         "#       elements added by the inverse-closure pass (not of the form\n"
         "#       sign * Clifford).\n";
    int idx = 0;
    for (const NetElem& e : elems) write_elem(f, idx++, e);
    f.close();
    fprintf(stderr, "Wrote %d entries to %s\n", idx, out_path);
    return 0;
}
