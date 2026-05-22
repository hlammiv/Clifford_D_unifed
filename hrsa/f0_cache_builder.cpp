// f0_cache_builder.cpp — enumerate the f=0 subgroup of the qutrit Clifford+D
// group via exact-arithmetic BFS, recording the minimum D-count for each
// reachable element.  Output: a binary cache file consumable by decompose().
//
// Strategy:
//   - Start from {I} at D-count 0.
//   - At each level k, expand the frontier by left-multiplying by
//       generators G = {H, X, S, S^{-1}, D, D^{-1}}.
//   - Cliffords (H, X, S, S^{-1}) cost 0 D-gates; D, D^{-1} cost +1.
//   - Hash each new element exactly (using ringZ9chi coefficient arrays)
//     and dedup.  Save the minimum D-count discovered.
//   - Stop at K_MAX D-gates (CLI arg), or when level produces no new elements.
//   - Filter results to f=0 (every entry has exp==0) and write to disk.
//
// Parallelism: per-level expansion is OMP-parallelized over the frontier.
// Each thread accumulates new (matrix, D-count) candidates locally; merges
// into the global hash with a coarse mutex (one bulk merge per level).

#include "decompose.h"
#include "Z9chi.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <omp.h>
#include <unordered_map>
#include <vector>

using namespace std;

// 60-byte canonical key per ringZ9chi entry: 9 int32 numerator + 1 int32 exp.
// Mat3 → 9 entries × 60 bytes = 540 bytes per matrix key.  Stored as raw bytes
// for fast hashing.
struct Mat3Key {
    int32_t data[3 * 3 * 10];  // 90 ints = 360 bytes

    bool operator==(const Mat3Key& o) const {
        return memcmp(data, o.data, sizeof(data)) == 0;
    }
};

namespace std {
    template<> struct hash<Mat3Key> {
        size_t operator()(const Mat3Key& k) const noexcept {
            // FNV-1a over the byte block
            size_t h = 1469598103934665603ull;
            const unsigned char* p = reinterpret_cast<const unsigned char*>(k.data);
            for (size_t i = 0; i < sizeof(k.data); ++i) {
                h ^= p[i];
                h *= 1099511628211ull;
            }
            return h;
        }
    };
}

static Mat3Key encode_mat3(const Mat3& M) {
    Mat3Key k{};
    int idx = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            for (int t = 0; t < 9; ++t)
                k.data[idx++] = (int32_t)M.m[i][j].getTerm(t);
            k.data[idx++] = (int32_t)M.m[i][j].getExp();
        }
    return k;
}

static bool is_f0(const Mat3& M) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (M.m[i][j].getExp() != 0) return false;
    return true;
}

static Mat3 identity() {
    Mat3 I;
    ringZ9 one_z9(1);
    I.m[0][0] = ringZ9chi(one_z9, 0);
    I.m[1][1] = ringZ9chi(one_z9, 0);
    I.m[2][2] = ringZ9chi(one_z9, 0);
    return I;
}

int main(int argc, char** argv) {
    int K_MAX = (argc > 1) ? atoi(argv[1]) : 6;
    string out_path = (argc > 2) ? argv[2] : "/tmp/f0_cache.bin";
    cout << "f0_cache_builder: K_MAX=" << K_MAX << " out=" << out_path << endl;

    // Build generators.  Match the existing 648-Clifford cache convention:
    // {H, X, Z, Z^{-1}} where Z = diag(ω, 1, 1) = gateD_clifford(1,0,0).
    // Using S = diag(1, ω, ω²) instead generates only a subgroup of size 108.
    //
    // Also include R = diag(1, 1, -1) (qutrit reflection): R appears as a
    // "free" gate in HRSA's syllable structure (H·D·R^ε·X^δ) and is a Clifford
    // in the Clifford+R framework.  Without R, the f=0 subgroup of ⟨H,X,Z,D⟩
    // is only 1458 elements and excludes sign-variant unitaries that HRSA's
    // Householder construction can produce.
    Mat3 Hm = gateH();
    Mat3 Xm = gateX();
    Mat3 Zm = gateD_clifford(1, 0, 0);     // Z = diag(ω, 1, 1)
    Mat3 Zinv = gateD_clifford(2, 0, 0);   // Z^{-1} = diag(ω², 1, 1)
    Mat3 Dm = gateDgate();                  // D = diag(ζ_9, 1, ζ_9^{-1})
    Mat3 Dinv = Dm.dagger();                // D^{-1}
    // R = diag(1, 1, -1) — encoded as ringZ9chi (-1 at coef 0, exp 0).
    Mat3 Rm;
    int one_arr[9] = {1,0,0,0,0,0,0,0,0};
    int neg_arr[9] = {-1,0,0,0,0,0,0,0,0};
    Rm.m[0][0] = ringZ9chi(ringZ9(one_arr), 0);
    Rm.m[1][1] = ringZ9chi(ringZ9(one_arr), 0);
    Rm.m[2][2] = ringZ9chi(ringZ9(neg_arr), 0);

    vector<Mat3> cliff_gens = {Hm, Xm, Zm, Zinv, Rm};  // 0-D-cost generators
    vector<Mat3> dgate_gens = {Dm, Dinv};               // +1-D-cost generators

    // Global cache: hash → min D-count.  Also holds a parallel vector of Mat3
    // for serialization at the end.
    unordered_map<Mat3Key, int> cache;
    vector<Mat3> elems;  // parallel to insertion order

    auto t0 = chrono::steady_clock::now();
    auto add_to_cache = [&](const Mat3& M, int d) -> bool {
        Mat3Key k = encode_mat3(M);
        auto it = cache.find(k);
        if (it == cache.end()) {
            cache.emplace(k, d);
            elems.push_back(M);
            return true;
        }
        if (d < it->second) it->second = d;
        return false;
    };

    // Helper: close a set of seed matrices under Clifford multiplication.
    // All elements get assigned D-count = d_target.  Returns elements newly
    // added to the cache at this D-count level.
    auto close_under_cliff = [&](vector<Mat3> seeds, int d_target) -> vector<Mat3> {
        vector<Mat3> all_at_level;
        for (auto& s : seeds) {
            if (add_to_cache(s, d_target)) all_at_level.push_back(s);
        }
        size_t cursor = 0;
        while (cursor < all_at_level.size()) {
            size_t end = all_at_level.size();
            int n_threads = omp_get_max_threads();
            vector<vector<Mat3>> tl_new(n_threads);
            #pragma omp parallel for schedule(dynamic, 256)
            for (size_t i = cursor; i < end; ++i) {
                int tid = omp_get_thread_num();
                for (const auto& g : cliff_gens) {
                    Mat3 W = g.mul(all_at_level[i]);
                    tl_new[tid].push_back(std::move(W));
                }
            }
            for (int tid = 0; tid < n_threads; ++tid) {
                for (auto& W : tl_new[tid]) {
                    if (add_to_cache(W, d_target)) {
                        all_at_level.push_back(std::move(W));
                    }
                }
            }
            cursor = end;
        }
        return all_at_level;
    };

    Mat3 I = identity();
    cout << "Phase 1: enumerating D=0 (Clifford group)..." << endl;
    vector<Mat3> level_d = close_under_cliff({I}, 0);
    cout << "  D=0: " << level_d.size() << " elements (cache=" << cache.size()
         << ") elapsed=" << chrono::duration<double>(chrono::steady_clock::now()-t0).count() << "s" << endl;

    for (int k = 1; k <= K_MAX; ++k) {
        cout << "Phase 2: enumerating D=" << k << "..." << endl;
        // Apply D / D^{-1} to every element in level_d (D-count k-1) to seed
        // the next level.
        vector<Mat3> seeds;
        seeds.reserve(level_d.size() * dgate_gens.size());
        size_t prev = level_d.size();
        int n_threads = omp_get_max_threads();
        vector<vector<Mat3>> tl_seeds(n_threads);
        #pragma omp parallel for schedule(dynamic, 256)
        for (size_t i = 0; i < prev; ++i) {
            int tid = omp_get_thread_num();
            for (const auto& g : dgate_gens) {
                Mat3 W = g.mul(level_d[i]);
                tl_seeds[tid].push_back(std::move(W));
            }
        }
        for (int tid = 0; tid < n_threads; ++tid)
            for (auto& s : tl_seeds[tid]) seeds.push_back(std::move(s));

        // Close the seeds under Clifford multiplication at D-count = k.
        // close_under_cliff dedupes against the global cache (so any element
        // that was previously discovered at lower D-count is not re-added).
        level_d = close_under_cliff(std::move(seeds), k);
        cout << "  D=" << k << ": " << level_d.size() << " new elements (cache=" << cache.size()
             << ") elapsed=" << chrono::duration<double>(chrono::steady_clock::now()-t0).count() << "s" << endl;

        if (level_d.empty()) {
            cout << "  converged at D=" << k << endl;
            break;
        }
    }

    auto t1 = chrono::steady_clock::now();
    double wall = chrono::duration<double>(t1-t0).count();
    cout << "BFS done.  total cache size=" << cache.size()
         << "  wall=" << wall << "s" << endl;

    // Filter to f=0 elements.
    vector<pair<Mat3, int>> f0;
    for (size_t i = 0; i < elems.size(); ++i) {
        if (is_f0(elems[i])) {
            Mat3Key k = encode_mat3(elems[i]);
            f0.emplace_back(elems[i], cache[k]);
        }
    }
    cout << "f=0 elements: " << f0.size()
         << " (out of " << cache.size() << ")" << endl;

    // D-count histogram for the f=0 subset
    vector<int> hist(K_MAX + 2, 0);
    for (const auto& [M, d] : f0) {
        if (d >= 0 && d < (int)hist.size()) hist[d]++;
    }
    cout << "D-count histogram (f=0 only):" << endl;
    for (int d = 0; d < (int)hist.size(); ++d)
        if (hist[d]) cout << "  D=" << d << ": " << hist[d] << " elements" << endl;

    // Write cache to disk.  Format:
    //   uint64_t  count
    //   for each entry:
    //     int32_t  d_count
    //     int32_t[90]  Mat3Key.data
    {
        ofstream f(out_path, ios::binary);
        uint64_t n = f0.size();
        f.write(reinterpret_cast<const char*>(&n), sizeof(n));
        for (const auto& [M, d] : f0) {
            int32_t d32 = d;
            f.write(reinterpret_cast<const char*>(&d32), sizeof(d32));
            Mat3Key k = encode_mat3(M);
            f.write(reinterpret_cast<const char*>(k.data), sizeof(k.data));
        }
    }
    cout << "Wrote cache: " << out_path << endl;

    return 0;
}
