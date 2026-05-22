// rf_features.cpp — feature extraction for the sub-cluster RF predictor.
// Mirrors the Python features_for_a + sub-cluster aggregation in rf_train.py.
//
// Feature layout (21 features, must match rf_model.h ordering):
//   [ 0] x1_q          [ 8] x1_l1_4       [16] x1_nz
//   [ 1] x1_q_2        [ 9] x1_l1_min     [17] n_in_cluster
//   [ 2] x1_q_4        [10] x1_linf       [18] f
//   [ 3] x1_q_min      [11] x1_sgn_sum    [19] x2_l1_mean
//   [ 4] x1_q_max      [12] x1_sgn_sum_w  [20] x2_l1_min
//   [ 5] x1_q_sum      [13] x1_mod3_sum   [21] x2_q_mean   <-- not present? check
//   [ 6] x1_l1         [14]               [22] x2_q_min
//   [ 7] x1_l1_2       [15]
//
// Order in the joblib metadata (per Python):
//   x1_q, x1_q_2, x1_q_4, x1_q_min, x1_q_max, x1_q_sum,
//   x1_l1, x1_l1_2, x1_l1_4, x1_l1_min,
//   x1_linf, x1_sgn_sum, x1_sgn_sum_w, x1_mod3_sum, x1_nz,
//   n_in_cluster, f,
//   x2_l1_mean, x2_l1_min, x2_q_mean, x2_q_min
// = 21 features total.

#include "rf_features.h"
#include "rf_model.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace {

// q-form: q(a) = sum a_k^2 - (a_0 a_3 + a_1 a_4 + a_2 a_5)
// (Verified algebraically against Tr(a*conj(a))/6 for Z[zeta_9].)
inline int q_form(const int* a) {
    return a[0]*a[0] + a[1]*a[1] + a[2]*a[2] + a[3]*a[3] + a[4]*a[4] + a[5]*a[5]
         - (a[0]*a[3] + a[1]*a[4] + a[2]*a[5]);
}

inline int abs_int(int x) { return x < 0 ? -x : x; }

// L1 weight in basis {1, zeta, ..., zeta^5}.
inline int l1_form(const int* a) {
    return abs_int(a[0]) + abs_int(a[1]) + abs_int(a[2]) +
           abs_int(a[3]) + abs_int(a[4]) + abs_int(a[5]);
}

// Compute the Galois conjugate sigma_r(x) for x = sum a_k zeta_9^k.
// Mirrors ringZ9::GaloisAut(r) but operates directly on a 6-int array
// (avoids constructing a ringZ9 + reducing).
// Reduction: zeta_9^6 = -1 - zeta_9^3 (from Phi_9 = x^6 + x^3 + 1).
inline void galois_action(const int* a, int r, int* out) {
    int swap[9] = {0,0,0,0,0,0,0,0,0};
    for (int k = 0; k < 6; ++k) {
        int idx = ((r * k) % 9 + 9) % 9;
        swap[idx] += a[k];
    }
    out[0] = swap[0] - swap[6];
    out[1] = swap[1] - swap[7];
    out[2] = swap[2] - swap[8];
    out[3] = swap[3] - swap[6];
    out[4] = swap[4] - swap[7];
    out[5] = swap[5] - swap[8];
}

inline int python_mod3(int x) {
    return ((x % 3) + 3) % 3;
}

} // anon namespace

void rf_extract_features(const ringZ9& x1,
                          const std::vector<ringZ9>& x2_list,
                          int f, int n_in_cluster,
                          double* feats) {
    int a1[6], a2[6], a4[6];
    for (int k = 0; k < 6; ++k) a1[k] = x1.getTerm(k);
    galois_action(a1, 2, a2);
    galois_action(a1, 4, a4);

    int q1     = q_form(a1);
    int q1_2   = q_form(a2);
    int q1_4   = q_form(a4);
    int q1_min = std::min({q1, q1_2, q1_4});
    int q1_max = std::max({q1, q1_2, q1_4});
    int q1_sum = q1 + q1_2 + q1_4;
    int l1     = l1_form(a1);
    int l1_2   = l1_form(a2);
    int l1_4   = l1_form(a4);
    int l1_min = std::min({l1, l1_2, l1_4});
    int linf   = std::max({abs_int(a1[0]), abs_int(a1[1]), abs_int(a1[2]),
                           abs_int(a1[3]), abs_int(a1[4]), abs_int(a1[5])});
    int sgn_sum   = a1[0]+a1[1]+a1[2]+a1[3]+a1[4]+a1[5];
    int sgn_sum_w = 0*a1[0]+1*a1[1]+2*a1[2]+3*a1[3]+4*a1[4]+5*a1[5];
    int mod3_sum  = python_mod3(sgn_sum);
    int nz = (a1[0]!=0)+(a1[1]!=0)+(a1[2]!=0)+(a1[3]!=0)+(a1[4]!=0)+(a1[5]!=0);

    feats[ 0] = q1;
    feats[ 1] = q1_2;
    feats[ 2] = q1_4;
    feats[ 3] = q1_min;
    feats[ 4] = q1_max;
    feats[ 5] = q1_sum;
    feats[ 6] = l1;
    feats[ 7] = l1_2;
    feats[ 8] = l1_4;
    feats[ 9] = l1_min;
    feats[10] = linf;
    feats[11] = sgn_sum;
    feats[12] = sgn_sum_w;
    feats[13] = mod3_sum;
    feats[14] = nz;
    feats[15] = n_in_cluster;
    feats[16] = f;

    // x_2 summary stats
    if (x2_list.empty()) {
        feats[17] = feats[18] = feats[19] = feats[20] = 0.0;
        return;
    }
    double x2_l1_sum = 0.0, x2_q_sum = 0.0;
    int    x2_l1_min_v = 0x7fffffff, x2_q_min_v = 0x7fffffff;
    for (const auto& x2 : x2_list) {
        int b[6];
        for (int k = 0; k < 6; ++k) b[k] = x2.getTerm(k);
        int l = l1_form(b);
        int q = q_form(b);
        x2_l1_sum += l;
        x2_q_sum  += q;
        if (l < x2_l1_min_v) x2_l1_min_v = l;
        if (q < x2_q_min_v)  x2_q_min_v  = q;
    }
    double n = (double)x2_list.size();
    feats[17] = x2_l1_sum / n;
    feats[18] = (double)x2_l1_min_v;
    feats[19] = x2_q_sum / n;
    feats[20] = (double)x2_q_min_v;
}

double rf_score_subcluster(const ringZ9& x1,
                            const std::vector<ringZ9>& x2_list,
                            int f, int n_in_cluster) {
    double feats[RF_N_FEATURES];
    rf_extract_features(x1, x2_list, f, n_in_cluster, feats);
    return rf_predict(feats);
}
