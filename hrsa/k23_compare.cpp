// k23_compare.cpp - Find depth-2 best and depth-3 best matrices, compare them.
//
// Usage:  ./k23_compare <theta>
//
// Reports:
//  - V_2 (matrix, m[i][j]) and the (c0, e1, c1, e2, c2) word indices
//  - V_3 (matrix, m[i][j]) and the (c0, e1, c1, e2, c2, e3, c3) word indices
//  - Frobenius difference ||V_2 - V_3||_F
//  - Whether they are entry-wise equal (machine precision)

#include <complex>
#include <vector>
#include <set>
#include <unordered_map>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>

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

static void print_mat(const string& label, const CMat3& M) {
    cout << label << ":\n";
    cout << setprecision(15);
    for (int i = 0; i < 3; ++i) {
        cout << "  [ ";
        for (int j = 0; j < 3; ++j) {
            cout << "(" << M.m[i][j].real() << ", " << M.m[i][j].imag() << ")";
            if (j < 2) cout << ", ";
        }
        cout << " ]\n";
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { cerr << "usage: k23_compare <theta>\n"; return 1; }
    double theta = atof(argv[1]);

    CMat3 target = {};
    target.m[0][0] = polar(1.0, -theta/2.0);
    target.m[1][1] = polar(1.0,  theta/2.0);
    target.m[2][2] = cd(1.0, 0.0);

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

    // Build depth-1 products: k1[idx] = C0 · D^e · C1, with index tracking
    int n_k1 = 2 * n_cliff * n_cliff;
    vector<CMat3> k1((size_t)n_k1);
    struct K1Idx { int c0, e1, c1; };
    vector<K1Idx> k1_idx((size_t)n_k1);
    for (int e = 0; e < 2; ++e)
        for (int c0 = 0; c0 < n_cliff; ++c0)
            for (int c1 = 0; c1 < n_cliff; ++c1) {
                int idx = (e * n_cliff + c0) * n_cliff + c1;
                k1[(size_t)idx] = cmul(cliffords[c0], cmul(Dgate[e], cliffords[c1]));
                k1_idx[(size_t)idx] = {c0, e, c1};
            }
    cerr << "Built " << n_k1 << " depth-1 products.\n";

    // ===== Depth-2 best (full enumeration with index tracking) =====
    // V_2 = k1[l] · D^e2 · C_2  =  k1[l] · rights[r], where rights[r] = D^e2 · C_2.
    int n_right = 2 * n_cliff;
    vector<CMat3> rights((size_t)n_right);
    struct RightIdx { int e2, c2; };
    vector<RightIdx> right_idx((size_t)n_right);
    for (int e2 = 0; e2 < 2; ++e2)
        for (int c2 = 0; c2 < n_cliff; ++c2) {
            int idx = e2 * n_cliff + c2;
            rights[(size_t)idx] = cmul(Dgate[e2], cliffords[c2]);
            right_idx[(size_t)idx] = {e2, c2};
        }
    cerr << "Built " << n_right << " right factors. Searching depth-2...\n";

    struct K2Win { double dist; int l; int r; };
    K2Win win2 = { 1e30, -1, -1 };
    #pragma omp parallel
    {
        K2Win local = { 1e30, -1, -1 };
        #pragma omp for schedule(dynamic, 64)
        for (int l = 0; l < n_k1; ++l) {
            for (int r = 0; r < n_right; ++r) {
                CMat3 V = cmul(k1[(size_t)l], rights[(size_t)r]);
                double d = frob_dist(V, target);
                if (d < local.dist) {
                    local.dist = d; local.l = l; local.r = r;
                }
            }
        }
        #pragma omp critical
        if (local.dist < win2.dist) win2 = local;
    }

    K1Idx L2 = k1_idx[(size_t)win2.l];
    RightIdx R2 = right_idx[(size_t)win2.r];
    CMat3 V_2 = cmul(k1[(size_t)win2.l], rights[(size_t)win2.r]);

    cout << "\n========== DEPTH-2 BEST ==========\n";
    cout << "Frobenius distance to target: " << setprecision(15) << win2.dist << "\n";
    cout << "Word: C[" << L2.c0 << "] · D^" << (L2.e1+1) << " · C[" << L2.c1
         << "] · D^" << (R2.e2+1) << " · C[" << R2.c2 << "]\n";
    print_mat("V_2", V_2);

    // ===== Depth-3 best (MITM with index tracking) =====
    // V_3 = k1[l] · rh[idx], rh[idx] = D^e2 · k1[r], idx = e2*n_k1 + r
    cerr << "Searching depth-3 (MITM)...\n";

    double grid = 0.005;
    auto hash_key = [&grid](cd v00, cd v11) -> long long {
        int r0 = (int)round(v00.real() / grid);
        int i0 = (int)round(v00.imag() / grid);
        int r1 = (int)round(v11.real() / grid);
        int i1 = (int)round(v11.imag() / grid);
        return ((long long)(r0+5000) * 10001 + (i0+5000)) * 100020001LL
             + ((long long)(r1+5000) * 10001 + (i1+5000));
    };
    int n_rh = 2 * n_k1;
    vector<CMat3> rh_vec((size_t)n_rh);
    unordered_multimap<long long, int> rh_hash;
    rh_hash.reserve((size_t)n_rh);
    for (int idx = 0; idx < n_rh; ++idx) {
        int e2 = idx / n_k1;
        int r = idx % n_k1;
        rh_vec[(size_t)idx] = cmul(Dgate[e2], k1[(size_t)r]);
        rh_hash.emplace(hash_key(rh_vec[(size_t)idx].m[0][0], rh_vec[(size_t)idx].m[1][1]), idx);
    }

    struct K3Win { double dist; int l; int idx; };
    K3Win win3 = { 1e30, -1, -1 };
    #pragma omp parallel
    {
        K3Win local = { 1e30, -1, -1 };
        #pragma omp for schedule(dynamic, 256)
        for (int l = 0; l < n_k1; ++l) {
            CMat3 Lt = {};
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    Lt.m[i][j] = conj(k1[(size_t)l].m[j][i]);
            CMat3 goal = cmul(Lt, target);
            for (int dr0 = -1; dr0 <= 1; ++dr0)
            for (int di0 = -1; di0 <= 1; ++di0)
            for (int dr1 = -1; dr1 <= 1; ++dr1)
            for (int di1 = -1; di1 <= 1; ++di1) {
                cd g00 = goal.m[0][0] + cd(dr0*grid, di0*grid);
                cd g11 = goal.m[1][1] + cd(dr1*grid, di1*grid);
                long long hk = hash_key(g00, g11);
                auto range = rh_hash.equal_range(hk);
                for (auto it = range.first; it != range.second; ++it) {
                    double d = frob_dist(goal, rh_vec[(size_t)it->second]);
                    if (d < local.dist) {
                        local.dist = d;
                        local.l = l;
                        local.idx = it->second;
                    }
                }
            }
        }
        #pragma omp critical
        if (local.dist < win3.dist) win3 = local;
    }

    if (win3.l < 0) {
        cout << "\n========== DEPTH-3 BEST ==========\n";
        cout << "MITM hash found nothing (probably need wider grid).\n";
        return 0;
    }

    int e2_3 = win3.idx / n_k1;
    int r_3  = win3.idx % n_k1;
    K1Idx L3 = k1_idx[(size_t)win3.l];
    K1Idx R3 = k1_idx[(size_t)r_3];
    CMat3 V_3 = cmul(k1[(size_t)win3.l], rh_vec[(size_t)win3.idx]);

    cout << "\n========== DEPTH-3 BEST (MITM) ==========\n";
    cout << "Frobenius distance to target: " << setprecision(15) << win3.dist << "\n";
    cout << "Word: C[" << L3.c0 << "] · D^" << (L3.e1+1) << " · C[" << L3.c1
         << "] · D^" << (e2_3+1) << " · C[" << R3.c0 << "] · D^" << (R3.e1+1)
         << " · C[" << R3.c1 << "]\n";
    print_mat("V_3", V_3);

    // ===== Compare V_2 and V_3 =====
    cout << "\n========== COMPARISON ==========\n";
    double diff = frob_dist(V_2, V_3);
    cout << "||V_2 - V_3||_F = " << setprecision(15) << diff << "\n";
    cout << "best_2 = " << setprecision(15) << win2.dist << "\n";
    cout << "best_3 = " << setprecision(15) << win3.dist << "\n";
    cout << "best_3 - best_2 = " << setprecision(15) << (win3.dist - win2.dist) << "\n";

    if (diff < 1e-12) {
        cout << "==> V_2 and V_3 are the SAME matrix (entry-wise equal).\n";
    } else {
        cout << "==> V_2 and V_3 are DIFFERENT matrices.\n";
        cout << "    Per-entry differences:\n" << setprecision(6);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                cd d = V_2.m[i][j] - V_3.m[i][j];
                cout << "    m[" << i << "][" << j << "]: re=" << d.real()
                     << " im=" << d.imag() << "\n";
            }
    }
    return 0;
}
