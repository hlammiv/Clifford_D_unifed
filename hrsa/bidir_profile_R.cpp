// bidir_bfs.cpp - Bidirectional BFS to approximate R^Z(theta) at depth K_f + K_b.
//
// Forward: from identity, applying forward gates {D^e · C}, build G_{K_f}.
// Backward: from R^Z(theta), applying inverse gates, build H_{K_b}.
// Match: for each V_f ∈ G_{K_f}, find closest W_b ∈ H_{K_b} by Frobenius distance.
// The min over pairs (V_f, W_b) ‖V_f - W_b‖_F = best achievable Frob error of a
// (K_f + K_b)-depth circuit approximating R^Z(theta).
//
// Path reification: each level stores per-matrix Provenance (parent_idx, parent_level,
// e, c) so the actual circuit (sequence of (D^e · C_c) gates) can be reconstructed
// for any winning forward / backward pair.
//
// Usage: ./bidir_bfs [theta=2.14] [K_f=4] [K_b=4]
//
// Memory budget: at K_f=K_b=4, |G|+|H| ≈ 6.8M matrices × 240 B = ~1.6 GB. Fine.
// Provenance overhead is 16 B per matrix; at 20 M (K=5) that is ~320 MB extra.

#include <complex>
#include <vector>
#include <set>
#include <unordered_map>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <cstring>
#include <atomic>
#include <mutex>
#include <utility>
#include <omp.h>
#include <memory>

using namespace std;
using cd = complex<double>;

struct CMat3 { cd m[3][3]; };

// Provenance: each non-root matrix M at level K was produced as
//   forward:  M = parent · D^e · C_c            (parent = G[K-1].mats[parent_idx])
//   backward: M = parent · cliff_inv[c] · D_inv^e (parent = H[K-1].mats[parent_idx])
// For roots (K==0), parent_idx is repurposed to encode the root identity:
//   forward roots: parent_level=-1, parent_idx = clifford index, e=-1, c=-1
//   backward root: parent_level=-1, parent_idx = 0,             e=-1, c=-1
struct Provenance {
    int parent_idx;
    int parent_level;
    int e;
    int c;
};

struct BFSLevel {
    vector<CMat3> mats;
    vector<Provenance> prov;  // parallel to mats
};

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

// Build qutrit Clifford group (~648 elements) and Dgate[3].
static void build_cliffords(vector<CMat3>& cliffords, CMat3 Dgate[3]) {
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
    cliffords = {eye};
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
    Dgate[0] = {}; Dgate[0].m[0][0] = z9;     Dgate[0].m[1][1] = cd(1,0); Dgate[0].m[2][2] = conj(z9);
    Dgate[1] = {}; Dgate[1].m[0][0] = z9*z9;  Dgate[1].m[1][1] = cd(1,0); Dgate[1].m[2][2] = conj(z9)*conj(z9);
    Dgate[2] = {}; Dgate[2].m[0][0] = cd(1,0); Dgate[2].m[1][1] = cd(1,0); Dgate[2].m[2][2] = cd(-1,0);
}

// Forward BFS: V_K = V_{K-1} · D^e · C. Returns G[0..K_max].
// Each M produced records its parent (in the previous level), the chosen
// D-power e, and the chosen Clifford index c.
static void bfs_forward(int K_max,
                        const vector<CMat3>& cliffords,
                        const CMat3 Dgate[3],
                        vector<BFSLevel>& G) {
    int n_cliff = (int)cliffords.size();
    G.assign(K_max + 1, BFSLevel{});
    G[0].mats = cliffords;
    G[0].prov.resize(cliffords.size());
    for (int i = 0; i < n_cliff; ++i)
        G[0].prov[i] = Provenance{ i, -1, -1, -1 };

    // Sharded dedup: split the seen set across N_SHARDS to allow parallel
    // insertions.  Shard chosen by hash(key) % N_SHARDS.  Each shard has its
    // own mutex + map + accumulator vectors.  At end of each level, merge.
    constexpr int N_SHARDS = 64;
    struct Shard {
        unordered_map<MatKey, int> seen;
        vector<CMat3> mats;
        vector<Provenance> prov;
        std::mutex mu;
    };
    vector<unique_ptr<Shard>> shards(N_SHARDS);
    for (int i = 0; i < N_SHARDS; ++i) shards[i] = std::make_unique<Shard>();
    auto pick_shard = [](const MatKey& k) -> int {
        return (int)(std::hash<MatKey>{}(k) % N_SHARDS);
    };
    // Seed level-0 cliffords into shards.
    for (const auto& V : G[0].mats) {
        MatKey k = make_key(V);
        int s = pick_shard(k);
        shards[s]->seen.emplace(k, 0);
    }
    for (int K = 1; K <= K_max; ++K) {
        const auto& prev = G[K-1].mats;
        auto& gK = G[K].mats;
        auto& pK = G[K].prov;
        gK.reserve(8 * prev.size());
        pK.reserve(8 * prev.size());
        // Per-shard accumulators get reset at each level (they hold ONLY this
        // level's new additions).
        for (auto& sp : shards) { sp->mats.clear(); sp->prov.clear(); }
        #pragma omp parallel
        {
            // Per-thread, per-shard staging: batch up items destined for each
            // shard, periodically flush.
            vector<vector<pair<pair<MatKey, CMat3>, Provenance>>> tl_stage(N_SHARDS);
            #pragma omp for schedule(dynamic, 16)
            for (long long task = 0; task < (long long)prev.size() * 3; ++task) {
                int p_idx = (int)(task / 3);
                int e = (int)(task % 3);
                const CMat3& V = prev[(size_t)p_idx];
                for (int c = 0; c < n_cliff; ++c) {
                    CMat3 M = cmul(V, cmul(Dgate[e], cliffords[c]));
                    MatKey key = make_key(M);
                    int s = pick_shard(key);
                    tl_stage[s].push_back({{std::move(key), M}, Provenance{p_idx, K-1, e, c}});
                    // Flush this shard if buffer full.
                    if (tl_stage[s].size() >= 32) {
                        std::lock_guard<std::mutex> lk(shards[s]->mu);
                        for (auto& it : tl_stage[s]) {
                            if (shards[s]->seen.find(it.first.first) == shards[s]->seen.end()) {
                                shards[s]->seen.emplace(it.first.first, K);
                                shards[s]->mats.push_back(it.first.second);
                                shards[s]->prov.push_back(it.second);
                            }
                        }
                        tl_stage[s].clear();
                    }
                }
            }
            // Final flush per shard.
            for (int s = 0; s < N_SHARDS; ++s) {
                if (tl_stage[s].empty()) continue;
                std::lock_guard<std::mutex> lk(shards[s]->mu);
                for (auto& it : tl_stage[s]) {
                    if (shards[s]->seen.find(it.first.first) == shards[s]->seen.end()) {
                        shards[s]->seen.emplace(it.first.first, K);
                        shards[s]->mats.push_back(it.first.second);
                        shards[s]->prov.push_back(it.second);
                    }
                }
            }
        }
        // Merge shard accumulators into level's gK/pK (single-threaded, but
        // just push_back — fast).
        for (auto& sp : shards) {
            for (auto& M : sp->mats) gK.push_back(M);
            for (auto& p : sp->prov) pK.push_back(p);
        }
        cerr << "  forward K=" << K << "  |G_K|=" << gK.size() << "\n";
    }
}

// Backward BFS: from R^Z(theta), expand using inverse gates.
// Backward step: W → W · C^{-1} · D^{-e}.
// Each new matrix records (parent_idx, e, c) where the gate applied was
// (cliff_inv[c], D_inv^e).
// ==============================================================
//  Profile main — builds forward set, sweeps N target rotations,
//  records winning forward state per θ, dumps CSV.
// ==============================================================

struct Metrics {
    double off_diag_l2;
    double max_off;
    double min_diag;
};

static Metrics compute_metrics(const CMat3& V) {
    Metrics m{};
    m.min_diag = 1e30;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double a = abs(V.m[i][j]);
            if (i == j) {
                if (a < m.min_diag) m.min_diag = a;
            } else {
                m.off_diag_l2 += a * a;
                if (a > m.max_off) m.max_off = a;
            }
        }
    m.off_diag_l2 = sqrt(m.off_diag_l2);
    return m;
}

int main(int argc, char** argv) {
    int K = (argc > 1) ? atoi(argv[1]) : 5;
    int N = (argc > 2) ? atoi(argv[2]) : 100;
    string out_path = (argc > 3) ? argv[3] : "/tmp/bidir_profile.csv";

    cerr << "Building Cliffords..." << endl;
    vector<CMat3> cliffords;
    CMat3 Dgate[3];
    build_cliffords(cliffords, Dgate);
    cerr << "  " << cliffords.size() << " cliffords" << endl;

    cerr << "Forward BFS at K=" << K << "..." << endl;
    vector<BFSLevel> G(K + 1);
    G[0].mats = cliffords;
    G[0].prov.resize(cliffords.size(), {0, -1, -1, -1});
    bfs_forward(K, cliffords, Dgate, G);

    // Flatten all forward states into one vector with global index = level offset + idx
    vector<CMat3> all;
    vector<int>   all_level;
    for (int k = 0; k <= K; ++k) {
        for (size_t i = 0; i < G[k].mats.size(); ++i) {
            all.push_back(G[k].mats[i]);
            all_level.push_back(k);
        }
    }
    cerr << "Forward set size: " << all.size() << endl;

    cerr << "Computing metrics..." << endl;
    vector<Metrics> mets(all.size());
    #pragma omp parallel for schedule(dynamic, 1024)
    for (size_t i = 0; i < all.size(); ++i)
        mets[i] = compute_metrics(all[i]);

    cerr << "Sweeping " << N << " thetas..." << endl;
    vector<int>    won_count(all.size(), 0);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int t = 0; t < N; ++t) {
        double theta = 2.0 * M_PI * (t + 0.5) / N;
        CMat3 R = {};
        R.m[0][0] = polar(1.0, -theta/2.0);
        R.m[1][1] = polar(1.0,  theta/2.0);
        R.m[2][2] = cd(1, 0);
        double best = 1e30; int win = -1;
        for (size_t i = 0; i < all.size(); ++i) {
            double d = frob_dist(all[i], R);
            if (d < best) { best = d; win = (int)i; }
        }
        #pragma omp atomic
        won_count[win]++;
    }

    int n_winners = 0;
    for (int wc : won_count) if (wc > 0) n_winners++;
    cerr << "n_winning_states = " << n_winners << "  (out of " << all.size() << ")" << endl;
    cerr << "Writing CSV to " << out_path << "..." << endl;
    ofstream f(out_path);
    f << "idx,level,off_diag_l2,max_off,min_diag,won_count\n";
    for (size_t i = 0; i < all.size(); ++i) {
        f << i << "," << all_level[i] << ","
          << mets[i].off_diag_l2 << "," << mets[i].max_off
          << "," << mets[i].min_diag << "," << won_count[i] << "\n";
    }
    cerr << "Done." << endl;
    return 0;
}
