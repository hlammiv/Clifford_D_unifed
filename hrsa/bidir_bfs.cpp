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
#include <cstring>
#include <atomic>
#include <mutex>
#include <utility>
#include <omp.h>

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
static CMat3 cdagger(const CMat3& A) {
    CMat3 D = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            D.m[i][j] = conj(A.m[j][i]);
    return D;
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

// Build qutrit Clifford group (~648 elements) and Dgate[2].
static void build_cliffords(vector<CMat3>& cliffords, CMat3 Dgate[2]) {
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
}

// Forward BFS: V_K = V_{K-1} · D^e · C. Returns G[0..K_max].
// Each M produced records its parent (in the previous level), the chosen
// D-power e, and the chosen Clifford index c.
static void bfs_forward(int K_max,
                        const vector<CMat3>& cliffords,
                        const CMat3 Dgate[2],
                        vector<BFSLevel>& G) {
    int n_cliff = (int)cliffords.size();
    G.assign(K_max + 1, BFSLevel{});
    G[0].mats = cliffords;
    G[0].prov.resize(cliffords.size());
    for (int i = 0; i < n_cliff; ++i)
        G[0].prov[i] = Provenance{ i, -1, -1, -1 };
    unordered_map<MatKey, int> seen;
    seen.reserve(20000000);
    for (const auto& V : G[0].mats) seen.emplace(make_key(V), 0);
    for (int K = 1; K <= K_max; ++K) {
        const auto& prev = G[K-1].mats;
        auto& gK = G[K].mats;
        auto& pK = G[K].prov;
        gK.reserve(8 * prev.size());
        pK.reserve(8 * prev.size());
        std::mutex mu;
        #pragma omp parallel
        {
            // Stage (key, mat, prov) before merging into shared dedup.
            struct Item { MatKey key; CMat3 M; Provenance pr; };
            vector<Item> staging;
            staging.reserve(128);
            #pragma omp for schedule(dynamic, 16)
            for (long long task = 0; task < (long long)prev.size() * 2; ++task) {
                int p_idx = (int)(task / 2);
                int e = (int)(task % 2);
                const CMat3& V = prev[(size_t)p_idx];
                for (int c = 0; c < n_cliff; ++c) {
                    CMat3 M = cmul(V, cmul(Dgate[e], cliffords[c]));
                    MatKey key = make_key(M);
                    Provenance pr{ p_idx, K-1, e, c };
                    staging.push_back(Item{ std::move(key), M, pr });
                    if (staging.size() >= 64) {
                        lock_guard<mutex> lk(mu);
                        for (auto& it : staging) {
                            if (seen.find(it.key) == seen.end()) {
                                seen.emplace(it.key, K);
                                gK.push_back(it.M);
                                pK.push_back(it.pr);
                            }
                        }
                        staging.clear();
                    }
                }
            }
            if (!staging.empty()) {
                lock_guard<mutex> lk(mu);
                for (auto& it : staging) {
                    if (seen.find(it.key) == seen.end()) {
                        seen.emplace(it.key, K);
                        gK.push_back(it.M);
                        pK.push_back(it.pr);
                    }
                }
            }
        }
        cerr << "  forward K=" << K << "  |G_K|=" << gK.size() << "\n";
    }
}

// Backward BFS: from R^Z(theta), expand using inverse gates.
// Backward step: W → W · C^{-1} · D^{-e}.
// Each new matrix records (parent_idx, e, c) where the gate applied was
// (cliff_inv[c], D_inv^e).
static void bfs_backward(int K_max,
                         const vector<CMat3>& cliffords,
                         const CMat3 Dgate[2],
                         const CMat3& target,
                         vector<BFSLevel>& H,
                         unordered_map<MatKey, int>& seen) {
    int n_cliff = (int)cliffords.size();
    // Inverse of Cliffords: use dagger (since unitary).
    vector<CMat3> cliff_inv(n_cliff);
    for (int i = 0; i < n_cliff; ++i) cliff_inv[i] = cdagger(cliffords[i]);
    // Inverse of Dgate^e: D^{-e}. Diagonal inverse = conjugate.
    CMat3 Dgate_inv[2];
    Dgate_inv[0] = cdagger(Dgate[0]);  // = D^{-1}
    Dgate_inv[1] = cdagger(Dgate[1]);  // = D^{-2}

    H.assign(K_max + 1, BFSLevel{});
    H[0].mats = {target};
    H[0].prov = { Provenance{ 0, -1, -1, -1 } };
    seen.clear();
    seen.reserve(10000000);
    seen.emplace(make_key(target), 0);

    for (int K = 1; K <= K_max; ++K) {
        const auto& prev = H[K-1].mats;
        auto& hK = H[K].mats;
        auto& pK = H[K].prov;
        hK.reserve(8 * prev.size());
        pK.reserve(8 * prev.size());
        std::mutex mu;
        #pragma omp parallel
        {
            struct Item { MatKey key; CMat3 M; Provenance pr; };
            vector<Item> staging;
            staging.reserve(128);
            #pragma omp for schedule(dynamic, 16)
            for (long long task = 0; task < (long long)prev.size() * 2; ++task) {
                int p_idx = (int)(task / 2);
                int e = (int)(task % 2);
                const CMat3& W = prev[(size_t)p_idx];
                // Backward step: W → W · C_inv · D_inv^e
                // (i.e., undoing the forward step V → V · D^e · C)
                for (int c = 0; c < n_cliff; ++c) {
                    CMat3 M = cmul(W, cmul(cliff_inv[c], Dgate_inv[e]));
                    MatKey key = make_key(M);
                    Provenance pr{ p_idx, K-1, e, c };
                    staging.push_back(Item{ std::move(key), M, pr });
                    if (staging.size() >= 64) {
                        lock_guard<mutex> lk(mu);
                        for (auto& it : staging) {
                            if (seen.find(it.key) == seen.end()) {
                                seen.emplace(it.key, K);
                                hK.push_back(it.M);
                                pK.push_back(it.pr);
                            }
                        }
                        staging.clear();
                    }
                }
            }
            if (!staging.empty()) {
                lock_guard<mutex> lk(mu);
                for (auto& it : staging) {
                    if (seen.find(it.key) == seen.end()) {
                        seen.emplace(it.key, K);
                        hK.push_back(it.M);
                        pK.push_back(it.pr);
                    }
                }
            }
        }
        cerr << "  backward K=" << K << "  |H_K|=" << hK.size() << "\n";
    }
}

// Walk parent pointers from G[level].prov[idx] back to G[0].
// Returns the list of (e, c) pairs in FORWARD order, i.e. the gates
// (D^{e_1}·C_{c_1}), (D^{e_2}·C_{c_2}), ... that, when right-multiplied in
// order onto the starting Clifford G[0].mats[start], yield G[level].mats[idx].
// Also fills `start_clifford_idx` with the G[0] entry index.
static vector<pair<int,int>> reconstruct_forward_path(int level, int idx,
                                                      const vector<BFSLevel>& G,
                                                      int& start_clifford_idx) {
    vector<pair<int,int>> rev;
    int cur_lvl = level;
    int cur_idx = idx;
    while (cur_lvl > 0) {
        const Provenance& p = G[cur_lvl].prov[cur_idx];
        rev.emplace_back(p.e, p.c);
        cur_lvl = p.parent_level;
        cur_idx = p.parent_idx;
    }
    // Now cur_lvl == 0; cur_idx is the root Clifford index.
    start_clifford_idx = cur_idx;
    // rev was emitted from the leaf back to the root, so reverse.
    vector<pair<int,int>> fwd(rev.rbegin(), rev.rend());
    return fwd;
}

// Walk parent pointers from H[level].prov[idx] back to H[0].
// Returns the list of (e, c) pairs in the order the BACKWARD BFS applied them
// (i.e. step 1 first). These gates were applied as (cliff_inv[c], D_inv^e):
//   H[K].mats[idx] = target · (cliff_inv[c_1] · D_inv^{e_1})
//                           · (cliff_inv[c_2] · D_inv^{e_2}) · ...
// To compose with the forward expansion (which uses (D^e · C_c)), the caller
// must REVERSE this list — see main().
static vector<pair<int,int>> reconstruct_backward_path(int level, int idx,
                                                       const vector<BFSLevel>& H) {
    vector<pair<int,int>> rev;
    int cur_lvl = level;
    int cur_idx = idx;
    while (cur_lvl > 0) {
        const Provenance& p = H[cur_lvl].prov[cur_idx];
        rev.emplace_back(p.e, p.c);
        cur_lvl = p.parent_level;
        cur_idx = p.parent_idx;
    }
    // rev[0] is the LAST backward step (closest to leaf); reverse so rev[0]
    // becomes the FIRST step (closest to target).
    vector<pair<int,int>> ordered(rev.rbegin(), rev.rend());
    return ordered;
}

// Build the constructed circuit:
//   constructed = (G[0] starting Clifford)
//                 · (forward gates in order)
//                 · (REVERSED backward gates as forward-direction (D^e·C_c) pairs)
// On a true match V_f = W_b, this equals target exactly.
// On approximate match, ‖constructed − target‖_F = ‖V_f − W_b‖_F because:
//   V_f ≈ W_b
//   W_b = target · [cliff_inv[c1]·D_inv^{e1}] · ... · [cliff_inv[cK]·D_inv^{eK}]
//   ⇒ target = W_b · [D^{eK}·cliff[cK]] · ... · [D^{e1}·cliff[c1]]
//            = W_b · (reversed backward path applied as forward gates)
//   ⇒ constructed = V_f · (reversed backward gates) ≈ target.
static CMat3 build_constructed(int start_clifford_idx,
                               const vector<pair<int,int>>& fwd_gates,
                               const vector<pair<int,int>>& bwd_gates_in_order,
                               const vector<CMat3>& cliffords,
                               const CMat3 Dgate[2]) {
    CMat3 cur = cliffords[start_clifford_idx];
    for (const auto& p : fwd_gates) {
        cur = cmul(cur, cmul(Dgate[p.first], cliffords[p.second]));
    }
    // Apply backward gates in REVERSE, as forward-direction (D^e · C_c) pairs.
    for (auto it = bwd_gates_in_order.rbegin(); it != bwd_gates_in_order.rend(); ++it) {
        cur = cmul(cur, cmul(Dgate[it->first], cliffords[it->second]));
    }
    return cur;
}

static void print_path(const char* label, const vector<pair<int,int>>& gates) {
    cout << label << "=[";
    for (size_t i = 0; i < gates.size(); ++i) {
        if (i) cout << ",";
        cout << gates[i].first << ":" << gates[i].second;
    }
    cout << "]";
}

int main(int argc, char** argv) {
    double theta = (argc > 1) ? atof(argv[1]) : 2.14;
    int K_f = (argc > 2) ? atoi(argv[2]) : 4;
    int K_b = (argc > 3) ? atoi(argv[3]) : 4;
    cerr << "theta=" << theta << "  K_f=" << K_f << "  K_b=" << K_b << "\n";

    vector<CMat3> cliffords;
    CMat3 Dgate[2];
    build_cliffords(cliffords, Dgate);
    cerr << "Built " << cliffords.size() << " Cliffords.\n";

    CMat3 target = {};
    target.m[0][0] = polar(1.0, -theta/2.0);
    target.m[1][1] = polar(1.0,  theta/2.0);
    target.m[2][2] = cd(1.0, 0.0);

    vector<BFSLevel> G, H;
    cerr << "=== Forward BFS ===\n";
    double t0 = omp_get_wtime();
    bfs_forward(K_f, cliffords, Dgate, G);
    double t_fwd = omp_get_wtime() - t0;

    unordered_map<MatKey, int> seen_b;
    cerr << "=== Backward BFS ===\n";
    t0 = omp_get_wtime();
    bfs_backward(K_b, cliffords, Dgate, target, H, seen_b);
    double t_back = omp_get_wtime() - t0;

    // Forward direct check: best Frob from any G_K_f matrix to target.
    double fwd_best = 1e30;
    for (int k = 0; k <= K_f; ++k)
        for (const auto& V : G[k].mats) {
            double d = frob_dist(V, target);
            if (d < fwd_best) fwd_best = d;
        }

    // Bidirectional pairwise check: for each V_f ∈ G_{K_f}, find closest W_b ∈ H_{K_b}.
    cerr << "=== Bidirectional intersect ===\n";
    t0 = omp_get_wtime();
    int exact_hits = 0;
    double best_exact_frob = 1e30;
    unordered_map<MatKey, size_t> g_hash;
    g_hash.reserve(G[K_f].mats.size());
    for (size_t i = 0; i < G[K_f].mats.size(); ++i)
        g_hash.emplace(make_key(G[K_f].mats[i]), i);
    for (size_t j = 0; j < H[K_b].mats.size(); ++j) {
        MatKey k = make_key(H[K_b].mats[j]);
        auto it = g_hash.find(k);
        if (it != g_hash.end()) {
            ++exact_hits;
            const CMat3& V_f = G[K_f].mats[it->second];
            double d = frob_dist(V_f, target);
            if (d < best_exact_frob) best_exact_frob = d;
        }
    }
    cerr << "  exact (key-equal) hits: " << exact_hits << "\n";
    double t_intersect = omp_get_wtime() - t0;

    cerr << "  approximate match scan...\n";
    t0 = omp_get_wtime();
    double grid = 0.5;  // tune
    auto coarse_key = [&](const CMat3& M) -> uint64_t {
        uint64_t h = 1469598103934665603ULL;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                int32_t r = (int32_t)round(M.m[i][j].real() / grid);
                int32_t im = (int32_t)round(M.m[i][j].imag() / grid);
                uint64_t v = ((uint64_t)(uint32_t)r << 32) | (uint64_t)(uint32_t)im;
                h ^= v; h *= 1099511628211ULL;
            }
        return h;
    };
    unordered_map<uint64_t, vector<size_t>> h_buckets;
    h_buckets.reserve(H[K_b].mats.size());
    for (size_t j = 0; j < H[K_b].mats.size(); ++j) {
        h_buckets[coarse_key(H[K_b].mats[j])].push_back(j);
    }
    double best_meet_frob = 1e30;
    size_t best_v_idx = 0, best_w_idx = 0;
    bool best_found = false;
    #pragma omp parallel
    {
        double local_best = 1e30;
        size_t local_v = 0, local_w = 0;
        bool local_found = false;
        #pragma omp for schedule(dynamic, 1024)
        for (size_t i = 0; i < G[K_f].mats.size(); ++i) {
            uint64_t k = coarse_key(G[K_f].mats[i]);
            auto it = h_buckets.find(k);
            if (it == h_buckets.end()) continue;
            for (size_t j : it->second) {
                double d = frob_dist(G[K_f].mats[i], H[K_b].mats[j]);
                if (d < local_best) {
                    local_best = d; local_v = i; local_w = j; local_found = true;
                }
            }
        }
        #pragma omp critical
        if (local_found && local_best < best_meet_frob) {
            best_meet_frob = local_best; best_v_idx = local_v; best_w_idx = local_w;
            best_found = true;
        }
    }
    double t_approx = omp_get_wtime() - t0;

    cerr << "\n";
    cout << "=== RESULTS ===\n";
    cout << "theta = " << setprecision(6) << theta << "\n";
    cout << "K_f = " << K_f << ", K_b = " << K_b << " (total path depth = " << (K_f + K_b) << ")\n";
    cout << "|G_K_f| = " << G[K_f].mats.size() << "\n";
    cout << "|H_K_b| = " << H[K_b].mats.size() << "\n";
    cout << "Forward-only best Frob to R^Z(theta): " << setprecision(15) << fwd_best << "\n";
    cout << "Exact (matrix-equal) intersect hits: " << exact_hits << "\n";
    cout << "Approximate-match best ‖V_f - W_b‖_F: " << setprecision(15) << best_meet_frob << "\n";
    cout << "  (this is the achievable Frob error of a depth-" << (K_f+K_b)
         << " circuit approximating R^Z(theta))\n";
    cout << "wall: forward=" << setprecision(2) << fixed << t_fwd
         << "s  backward=" << t_back
         << "s  exact=" << t_intersect
         << "s  approx=" << t_approx << "s  TOTAL=" << (t_fwd+t_back+t_intersect+t_approx) << "s\n";

    // Path reification on the winning approximate-match pair.
    if (best_found) {
        int start_cliff = -1;
        vector<pair<int,int>> fwd_gates =
            reconstruct_forward_path(K_f, (int)best_v_idx, G, start_cliff);
        vector<pair<int,int>> bwd_gates =
            reconstruct_backward_path(K_b, (int)best_w_idx, H);

        // Compose the reversed-backward gate list (what gets right-multiplied
        // onto V_f to recover target).
        vector<pair<int,int>> bwd_reversed(bwd_gates.rbegin(), bwd_gates.rend());

        CMat3 constructed = build_constructed(start_cliff, fwd_gates, bwd_gates,
                                              cliffords, Dgate);
        double verify = frob_dist(constructed, target);

        cout << setprecision(6) << defaultfloat;
        cout << "WIN: depth=" << (K_f + K_b)
             << " frob=" << best_meet_frob
             << " start_clifford=" << start_cliff << " ";
        print_path("forward", fwd_gates);
        cout << " ";
        print_path("backward", bwd_reversed);
        cout << "\n";
        cout << "verify: ||constructed - target||_F = "
             << setprecision(15) << verify << "\n";
    } else {
        cout << "WIN: (no approximate match found in coarse buckets)\n";
    }
    return 0;
}
