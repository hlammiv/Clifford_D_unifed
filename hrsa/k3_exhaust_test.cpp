// k3_exhaust_test.cpp - Exhaustive depth-3 enumeration over qutrit Cliff+D circuits.
// Computes true min Frobenius distance from R^Z(theta) to any C0·D^e1·C1·D^e2·C2·D^e3·C3.
// Bypasses the MITM hash to verify whether cum_3 < cum_2.
//
// Usage:  ./k3_exhaust_test <theta>
//
// Cost: ~840K × 1.7M = 1.4e12 cmul+frob ops. ~2-3 hours on 14 cores.

#include <complex>
#include <vector>
#include <set>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include <atomic>

using namespace std;
using cd = complex<double>;

struct CMat3 { cd m[3][3]; };

static CMat3 cmul(const CMat3& A, const CMat3& B) {
    CMat3 C = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                C.m[i][j] += A.m[i][k] * B.m[k][j];
    return C;
}

static double frob_dist(const CMat3& A, const CMat3& B) {
    double s = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            cd d = A.m[i][j] - B.m[i][j];
            s += d.real()*d.real() + d.imag()*d.imag();
        }
    return sqrt(s);
}

int main(int argc, char** argv) {
    if (argc < 2) { cerr << "usage: k3_exhaust_test <theta>\n"; return 1; }
    double theta = atof(argv[1]);

    // target = R^Z(theta) = diag(e^{-itheta/2}, e^{itheta/2}, 1)
    CMat3 target = {};
    target.m[0][0] = polar(1.0, -theta/2.0);
    target.m[1][1] = polar(1.0,  theta/2.0);
    target.m[2][2] = cd(1.0, 0.0);

    // BFS-build the qutrit Clifford group from {H, X, S, S^{-1}}
    cd om = exp(cd(0, 2.0*M_PI/3.0));
    cd z9 = exp(cd(0, 2.0*M_PI/9.0));

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
    vector<CMat3> cliffords = {eye};
    seen.insert(mat_key(eye));
    CMat3 gens[4] = {genH, genX, genS, genSi};
    size_t head = 0;
    while (head < cliffords.size() && cliffords.size() < 700) {
        CMat3 M = cliffords[head++];
        for (int g = 0; g < 4; ++g) {
            CMat3 P = cmul(M, gens[g]);
            auto k = mat_key(P);
            if (seen.find(k) == seen.end()) {
                seen.insert(k);
                cliffords.push_back(P);
            }
        }
    }
    int n_cliff = (int)cliffords.size();
    cerr << "Built " << n_cliff << " Cliffords.\n";

    CMat3 Dgate[2];
    Dgate[0] = {}; Dgate[0].m[0][0] = z9;     Dgate[0].m[1][1] = cd(1,0); Dgate[0].m[2][2] = conj(z9);
    Dgate[1] = {}; Dgate[1].m[0][0] = z9*z9;  Dgate[1].m[1][1] = cd(1,0); Dgate[1].m[2][2] = conj(z9)*conj(z9);

    // k1[idx] = C0 · D^e · C1
    int n_k1 = 2 * n_cliff * n_cliff;
    vector<CMat3> k1((size_t)n_k1);
    cerr << "Building " << n_k1 << " depth-1 products...\n";
    #pragma omp parallel for collapse(2) schedule(static)
    for (int e = 0; e < 2; ++e)
        for (int c0 = 0; c0 < n_cliff; ++c0) {
            CMat3 dc;
            for (int c1 = 0; c1 < n_cliff; ++c1) {
                dc = cmul(Dgate[e], cliffords[c1]);
                int idx = (e * n_cliff + c0) * n_cliff + c1;
                k1[(size_t)idx] = cmul(cliffords[c0], dc);
            }
        }

    // rh[idx] = D^e2 · k1[r],  with idx = e2*n_k1 + r,  size = 2*n_k1
    int n_rh = 2 * n_k1;
    vector<CMat3> rh((size_t)n_rh);
    cerr << "Building " << n_rh << " right halves...\n";
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < n_rh; ++idx) {
        int e2 = idx / n_k1;
        int r  = idx % n_k1;
        rh[(size_t)idx] = cmul(Dgate[e2], k1[(size_t)r]);
    }

    long long total_ops = (long long)n_k1 * (long long)n_rh;
    cerr << "Exhaustive depth-3 search: " << total_ops << " ops ("
         << (double)total_ops / 1e12 << "T).\n";
    cerr << "Estimated wall on " << omp_get_max_threads() << " cores at 80ns/op: ~"
         << (total_ops * 80e-9 / omp_get_max_threads()) << "s\n";

    double t0 = omp_get_wtime();
    double global_best = 1e30;
    long long progress = 0;
    long long progress_step = max(1, n_k1 / 100);
    #pragma omp parallel
    {
        double local_best = 1e30;
        #pragma omp for schedule(dynamic, 64)
        for (int l = 0; l < n_k1; ++l) {
            const CMat3& kl = k1[(size_t)l];
            for (int idx = 0; idx < n_rh; ++idx) {
                CMat3 V = cmul(kl, rh[(size_t)idx]);
                double d = frob_dist(V, target);
                if (d < local_best) local_best = d;
            }
            // Loose progress reporting
            #pragma omp atomic
            ++progress;
            if (progress % progress_step == 0) {
                #pragma omp critical
                cerr << "\r  progress: " << (progress * 100 / n_k1) << "%   " << flush;
            }
        }
        #pragma omp critical
        if (local_best < global_best) global_best = local_best;
    }
    double wall = omp_get_wtime() - t0;
    cerr << "\n";

    cout << "theta=" << theta
         << " best_3_exhaustive=" << setprecision(15) << global_best
         << " wall=" << setprecision(3) << wall << "s" << endl;
    return 0;
}
