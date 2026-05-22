// tube_pruned_bfs.cpp - BFS with FORWARD tube(δ_keep) pruning, to see if tighter
// tube(δ_tight) matrices emerge at deeper K despite the forward pruning.
//
// Algorithm:
//   G[0] = Cliffords (full).
//   G[K+1] built only from PARENTS in tube(δ_keep) ∩ G[K]; new matrices added
//   to G[K+1] only if they pass tube(δ_keep).
// Per level, report tube counts at multiple δ and best Frob to test angles
// for {full, tube(0.5), tube(0.3), tube(0.1)}.
//
// Usage: ./tube_pruned_bfs [delta_keep=0.5] [max_depth=7]

#include <complex>
#include <vector>
#include <set>
#include <unordered_map>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <atomic>
#include <mutex>
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

static double frob_dist_to_RZ(const CMat3& V, double theta) {
    cd t00 = polar(1.0, -theta/2.0);
    cd t11 = polar(1.0,  theta/2.0);
    double s = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            cd ref = (i==j) ? (i==0?t00:(i==1?t11:cd(1,0))) : cd(0,0);
            cd d = V.m[i][j] - ref;
            s += d.real()*d.real() + d.imag()*d.imag();
        }
    return sqrt(s);
}

static double tube_score(const CMat3& V) {
    double max_off = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (i != j) {
                double mag = abs(V.m[i][j]);
                if (mag > max_off) max_off = mag;
            }
    cd diff22 = V.m[2][2] - cd(1.0, 0.0);
    double dist22 = sqrt(diff22.real()*diff22.real() + diff22.imag()*diff22.imag());
    return max(max_off, dist22);
}

using MatKey = std::string;
static MatKey make_key(const CMat3& V) {
    constexpr double S = 1e7;
    char buf[72];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            int64_t r = (int64_t)round(V.m[i][j].real() * S);
            int64_t im = (int64_t)round(V.m[i][j].imag() * S);
            uint64_t packed = ((uint64_t)(uint32_t)r << 32) | (uint64_t)(uint32_t)im;
            memcpy(buf + 8*(3*i+j), &packed, 8);
        }
    return MatKey(buf, 72);
}

int main(int argc, char** argv) {
    double delta_keep = (argc > 1) ? atof(argv[1]) : 0.5;
    int MAX_K = (argc > 2) ? atoi(argv[2]) : 7;
    cerr << "delta_keep=" << delta_keep << "  max_depth=" << MAX_K << "\n";

    // Build Cliffords.
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
    CMat3 genS = {}; genS.m[0][0] = om; genS.m[1][1] = cd(1,0); genS.m[2][2] = cd(1,0);
    CMat3 genSi = {}; genSi.m[0][0] = om*om; genSi.m[1][1] = cd(1,0); genSi.m[2][2] = cd(1,0);

    auto mat_key_old = [](const CMat3& M) -> vector<int> {
        vector<int> k(18);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                k[6*i+2*j]   = (int)round(M.m[i][j].real() * 1e5);
                k[6*i+2*j+1] = (int)round(M.m[i][j].imag() * 1e5);
            }
        return k;
    };
    set<vector<int>> seen_init;
    CMat3 eye = {}; eye.m[0][0] = eye.m[1][1] = eye.m[2][2] = cd(1,0);
    vector<CMat3> cliffords = {eye};
    seen_init.insert(mat_key_old(eye));
    CMat3 gens[4] = {genH, genX, genS, genSi};
    size_t head = 0;
    while (head < cliffords.size() && cliffords.size() < 700) {
        CMat3 M = cliffords[head++];
        for (int g = 0; g < 4; ++g) {
            CMat3 P = cmul(M, gens[g]);
            auto k = mat_key_old(P);
            if (seen_init.find(k) == seen_init.end()) {
                seen_init.insert(k);
                cliffords.push_back(P);
            }
        }
    }
    int n_cliff = (int)cliffords.size();
    cerr << "Built " << n_cliff << " Cliffords.\n";

    CMat3 Dgate[2];
    Dgate[0] = {}; Dgate[0].m[0][0] = z9;     Dgate[0].m[1][1] = cd(1,0); Dgate[0].m[2][2] = conj(z9);
    Dgate[1] = {}; Dgate[1].m[0][0] = z9*z9;  Dgate[1].m[1][1] = cd(1,0); Dgate[1].m[2][2] = conj(z9)*conj(z9);

    // BFS with tube-pruned forward.
    vector<vector<CMat3>> G(MAX_K + 1);
    unordered_map<MatKey, int> seen_global;  // dedup ANY matrix already seen at any prior level
    seen_global.reserve(20000000);

    // Level 0: full Cliffords.
    G[0] = cliffords;
    for (const auto& V : G[0]) seen_global.emplace(make_key(V), 0);

    vector<double> thetas = {2.14, 1.0053, 1.5708, 0.5027};
    vector<double> deltas = {0.05, 0.1, 0.2, 0.3, 0.5, 1.0};

    auto report_level = [&](int K, double wall) {
        const auto& gK = G[K];
        size_t total = gK.size();
        cout << "\n--- K=" << K << " ---  total |G_K| = " << total
             << "  wall=" << fixed << setprecision(2) << wall << "s\n";
        cout << "tube counts within G_K (NEW matrices at this level):\n";
        cout << "  δ        count        fraction\n";
        for (double d : deltas) {
            size_t cnt = 0;
            for (const auto& V : gK)
                if (tube_score(V) < d) ++cnt;
            cout << "  " << setw(7) << fixed << setprecision(3) << d
                 << setw(13) << cnt
                 << setw(15) << fixed << setprecision(5)
                 << (total > 0 ? (double)cnt / (double)total : 0.0) << "\n";
        }
        cout << "\nbest Frob in G_K  vs  tube(δ=0.1) ∩ G_K  (per target θ):\n";
        cout << "  theta       full           tube0.1        diff\n";
        for (double th : thetas) {
            double bf = 1e30, bt01 = 1e30;
            for (const auto& V : gK) {
                double f = frob_dist_to_RZ(V, th);
                if (f < bf) bf = f;
                if (tube_score(V) < 0.1 && f < bt01) bt01 = f;
            }
            cout << "  " << setw(8) << fixed << setprecision(3) << th
                 << "  " << setw(13) << setprecision(10) << bf
                 << "  " << setw(13) << setprecision(10) << (bt01 > 1e29 ? -1.0 : bt01)
                 << "  " << setw(11) << setprecision(3) << (bt01 > 1e29 ? -1.0 : (bt01 - bf)) << "\n";
        }
        // Track if any tube(0.1) matrix appeared at this level.
        size_t nt01 = 0;
        for (const auto& V : gK) if (tube_score(V) < 0.1) ++nt01;
        if (nt01 > 0) {
            cout << "  *** tube(δ=0.1) MATRICES APPEAR AT K=" << K << " (count=" << nt01 << ") ***\n";
        }
    };

    double t0 = omp_get_wtime();
    report_level(0, omp_get_wtime() - t0);

    for (int K = 1; K <= MAX_K; ++K) {
        double tlevel = omp_get_wtime();

        // Parents: tube(δ_keep) ∩ G[K-1] for K >= 2, or full G[K-1] for K=1
        // (avoid losing the K=1 best by pruning at K=0 where the diag-only Cliffords would be missed).
        vector<size_t> parent_idx;
        if (K == 1) {
            parent_idx.resize(G[K-1].size());
            for (size_t i = 0; i < G[K-1].size(); ++i) parent_idx[i] = i;
        } else {
            for (size_t i = 0; i < G[K-1].size(); ++i)
                if (tube_score(G[K-1][i]) < delta_keep) parent_idx.push_back(i);
        }
        cerr << "Building G[" << K << "] from " << parent_idx.size() << " tube-keep parents...\n";

        const auto& prev = G[K-1];
        vector<CMat3>& gK = G[K];
        gK.reserve(8 * parent_idx.size());
        std::mutex mu;
        atomic<long long> processed(0);
        long long total_tasks = (long long)parent_idx.size() * 2;
        long long progress_step = max((long long)1, total_tasks / 20);

        #pragma omp parallel
        {
            vector<pair<MatKey, CMat3>> staging;
            staging.reserve(128);
            #pragma omp for schedule(dynamic, 16)
            for (long long task = 0; task < total_tasks; ++task) {
                size_t pidx = parent_idx[(size_t)(task / 2)];
                int e = (int)(task % 2);
                const CMat3& V = prev[pidx];
                for (int c = 0; c < n_cliff; ++c) {
                    CMat3 M = cmul(V, cmul(Dgate[e], cliffords[c]));
                    // Forward tube prune: skip if not in tube(δ_keep).
                    if (K >= 2 && tube_score(M) >= delta_keep) continue;
                    MatKey key = make_key(M);
                    staging.emplace_back(std::move(key), M);
                    if (staging.size() >= 64) {
                        lock_guard<mutex> lk(mu);
                        for (auto& p : staging) {
                            auto it = seen_global.find(p.first);
                            if (it == seen_global.end()) {
                                seen_global.emplace(p.first, K);
                                gK.push_back(p.second);
                            }
                        }
                        staging.clear();
                    }
                }
                long long pp = ++processed;
                if (pp % progress_step == 0) {
                    cerr << "\r  progress: " << (pp * 100 / total_tasks)
                         << "%  G_K_size=" << gK.size() << flush;
                }
            }
            if (!staging.empty()) {
                lock_guard<mutex> lk(mu);
                for (auto& p : staging) {
                    auto it = seen_global.find(p.first);
                    if (it == seen_global.end()) {
                        seen_global.emplace(p.first, K);
                        gK.push_back(p.second);
                    }
                }
            }
        }
        cerr << "\n";
        double wall = omp_get_wtime() - tlevel;
        report_level(K, wall);
    }
    return 0;
}
