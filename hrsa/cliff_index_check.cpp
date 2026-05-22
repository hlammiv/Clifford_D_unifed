// cliff_index_check.cpp - Sanity check that bidir_bfs's BFS Clifford order
// matches decompose.cpp's get_clifford_cache() Clifford order.
//
// Builds the qutrit Clifford set inline using the same logic as
// build_cliffords() in bidir_bfs.cpp (BFS over generators {H, X, S, S^{-1}})
// and compares each numerical CMat3 to the numerical evaluation of the
// corresponding ringZ9chi entry in get_clifford_cache().ring_cliffords.
//
// Reports:
//   - max element-wise complex-distance over all entries
//   - number of mismatches with distance > 1e-9
//   - exit 0 if max distance < 1e-9, else exit 1

#include "clifford_cache.h"
#include "decompose.h"

#include <complex>
#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <set>
#include <vector>

using namespace std;
using cd = complex<double>;

static CMat3 cmul_local(const CMat3& A, const CMat3& B) {
    CMat3 C = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                C.m[i][j] += A.m[i][k] * B.m[k][j];
    return C;
}

// Build the qutrit Clifford set inline using the same generators / dedup
// as bidir_bfs.cpp's build_cliffords().
static void build_cliffords_bidir_style(vector<CMat3>& cliffords) {
    cd om = exp(cd(0, 2.0*M_PI/3.0));

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

    auto mat_key = [](const CMat3& M) -> vector<int> {
        vector<int> k(18);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                k[6*i+2*j]   = (int)round(M.m[i][j].real() * 1e5);
                k[6*i+2*j+1] = (int)round(M.m[i][j].imag() * 1e5);
            }
        return k;
    };
    set<vector<int>> seen;
    CMat3 eye = {}; eye.m[0][0] = eye.m[1][1] = eye.m[2][2] = cd(1,0);
    cliffords = {eye};
    seen.insert(mat_key(eye));
    CMat3 gens[4] = {genH, genX, genS, genSi};
    size_t head = 0;
    while (head < cliffords.size() && cliffords.size() < 700) {
        CMat3 M = cliffords[head++];
        for (int g = 0; g < 4; ++g) {
            CMat3 P = cmul_local(M, gens[g]);
            auto k = mat_key(P);
            if (seen.find(k) == seen.end()) {
                seen.insert(k);
                cliffords.push_back(P);
            }
        }
    }
}

static double cmat_frob_dist(const CMat3& A, const CMat3& B) {
    double s = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            cd d = A.m[i][j] - B.m[i][j];
            s += d.real()*d.real() + d.imag()*d.imag();
        }
    return sqrt(s);
}

// Convert ringZ9chi Mat3 to numerical CMat3 by evaluating each entry at
// zeta_9 -> exp(2 pi i / 9).
static CMat3 ring_to_numerical(const Mat3& M) {
    CMat3 R = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R.m[i][j] = M.m[i][j].toComplexDouble();
    return R;
}

int main() {
    // 1. Build bidir-style Clifford set inline.
    vector<CMat3> bidir_cliffs;
    build_cliffords_bidir_style(bidir_cliffs);
    cout << "bidir-style |C| = " << bidir_cliffs.size() << "\n";

    // 2. Get decompose's cache.
    const CliffordCache& cache = get_clifford_cache();
    cout << "decompose cache  |C| = " << cache.cliffords.size() << "\n";

    if (bidir_cliffs.size() != cache.cliffords.size()) {
        cout << "FAIL: cardinality mismatch.\n";
        return 1;
    }
    int n = (int)bidir_cliffs.size();

    // 3. Compare each (i): bidir_cliffs[i] vs numerical eval of ring_cliffords[i].
    //    Also compare bidir_cliffs[i] vs cache.cliffords[i] (same numerical builder).
    double max_d_ring = 0.0;
    double max_d_num  = 0.0;
    int n_bad_ring = 0;
    int n_bad_num  = 0;
    for (int i = 0; i < n; ++i) {
        CMat3 ring_eval = ring_to_numerical(cache.ring_cliffords[i]);
        double d_ring = cmat_frob_dist(bidir_cliffs[i], ring_eval);
        double d_num  = cmat_frob_dist(bidir_cliffs[i], cache.cliffords[i]);
        if (d_ring > max_d_ring) max_d_ring = d_ring;
        if (d_num  > max_d_num)  max_d_num  = d_num;
        if (d_ring > 1e-9) ++n_bad_ring;
        if (d_num  > 1e-9) ++n_bad_num;
        if (d_ring > 1e-9 && n_bad_ring < 5) {
            cout << "  mismatch at i=" << i << ": d_ring=" << setprecision(15) << d_ring << "\n";
        }
    }
    cout << "max ‖bidir[i] - ringEval(cache.ring_cliffords[i])‖_F = "
         << setprecision(15) << max_d_ring << "  (bad: " << n_bad_ring << ")\n";
    cout << "max ‖bidir[i] - cache.cliffords[i]‖_F                  = "
         << setprecision(15) << max_d_num  << "  (bad: " << n_bad_num  << ")\n";

    if (max_d_ring < 1e-9 && max_d_num < 1e-9) {
        cout << "OK: indexing is direct (no permutation needed).\n";
        return 0;
    } else {
        cout << "FAIL: indexings do not match — a permutation map would be required.\n";
        return 1;
    }
}
