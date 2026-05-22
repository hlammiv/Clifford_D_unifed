// bfs_canonical.cpp — BFS-with-hash-dedup canonical-form exploration of the
// qutrit Clifford+D group.
//
// Goal
// ----
// Existing brute-force enumerators in this directory (k3_exhaust_test.cpp,
// k23_compare.cpp) can reach K = 3 D-gates but no further: the word count
// (2*648)^K already explodes by K = 4. Many of those words, however, evaluate
// to the SAME matrix — so if we deduplicate by matrix identity (not by word)
// we can push the depth further.
//
// We build:
//      G[0] = Cliffords                                     (~648 matrices)
//      G[K] = { V · D^e · C  :  V in G[K-1], e in {1,2}, C in Cliffords,
//                                   AND  M not in any earlier level }
//
// At each level we record the best Frobenius distance to a list of target
// rotations R^Z(theta_i) and emit one JSONL line per level.
//
// Hash: we round all 18 reals (9 complex entries) to 1e-7. Collisions are
// resolved with full-Frobenius equality (strict < 1e-9).
//
// Parallelism: per-thread local hash; merged into the global set under a
// critical section after each thread's slice of the work.
//
// CLI
// ---
//      ./bfs_canonical
//          --theta 2.14 [--theta 1.0053 ...]
//          --max-depth 8
//          --memory-gb 32
//          --output /tmp/bfs.jsonl
//          [--resume <prefix>]
//
// Build
// -----
//      g++ -O3 -flto -fopenmp -Wall -pedantic bfs_canonical.cpp -o bfs_canonical
//
// Smoke test (must pass — these reproduce existing baselines):
//      |G_0|         = 648
//      cum_1(2.14)   ≈ 0.46... (matches k1 best)
//      cum_2(2.14)   = 0.45936236249623...
//      cum_3(2.14)   = 0.45936236249623... (same matrix as best_2)

#include <complex>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <array>
#include <chrono>
#include <atomic>
#include <omp.h>

#ifdef __linux__
#include <sys/resource.h>
#endif

using namespace std;
using cd = complex<double>;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

struct CMat3 {
    cd m[3][3];
};

static inline CMat3 cmul(const CMat3& A, const CMat3& B) {
    CMat3 C = {};
    for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 3; ++k) {
            cd a = A.m[i][k];
            for (int j = 0; j < 3; ++j)
                C.m[i][j] += a * B.m[k][j];
        }
    return C;
}

static inline double frob_dist(const CMat3& A, const CMat3& B) {
    double s = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            cd d = A.m[i][j] - B.m[i][j];
            s += d.real()*d.real() + d.imag()*d.imag();
        }
    return sqrt(s);
}

// 18 int32 entries (one per real/imag of every CMat3 entry). Rounded so two
// matrices that agree to 1e-7 produce the same key.
//
// 144 bits packed into 5 uint64s (we use 144 bits = 18 * 8 bits would be too
// little, so we use 18 int32 ⇒ 576-bit key). For the unordered_set we pass
// the 18 ints as an array<int32,18> with a custom hash.
using MatKey = array<int32_t, 18>;

struct MatKeyHash {
    size_t operator()(const MatKey& k) const noexcept {
        // FNV-1a style mix on the raw bytes.
        const uint64_t* p = reinterpret_cast<const uint64_t*>(k.data());
        // 18 int32 = 72 bytes = 9 × uint64.
        uint64_t h = 1469598103934665603ULL;
        for (int i = 0; i < 9; ++i) {
            h ^= p[i];
            h *= 1099511628211ULL;
        }
        return (size_t)h;
    }
};

static constexpr double KEY_SCALE = 1e7;  // 1e-7 rounding granularity.

static inline MatKey mat_key(const CMat3& M) {
    MatKey k{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            // llround to handle potential overflow gracefully even though
            // unitary entries are bounded by 1.
            k[6*i + 2*j]     = (int32_t)llround(M.m[i][j].real() * KEY_SCALE);
            k[6*i + 2*j + 1] = (int32_t)llround(M.m[i][j].imag() * KEY_SCALE);
        }
    return k;
}

// Resolve hash-key collisions with a real Frobenius check.
static inline bool mat_eq(const CMat3& A, const CMat3& B) {
    return frob_dist(A, B) < 1e-9;
}

// ---------------------------------------------------------------------------
// Word tracking — for every matrix in G_K we store the parent matrix index in
// G_{K-1}, the D-gate exponent (0 ⇒ D^1, 1 ⇒ D^2) and the Clifford index used
// to extend. This lets us reconstruct the gate word for the best matrix.
// ---------------------------------------------------------------------------

struct Provenance {
    int32_t parent;       // index into G_{K-1}; -1 for K=0 (Clifford itself)
    int32_t cliff;        // Clifford index; for K=0 this is the Clifford idx.
    int8_t  e;            // 0 ⇒ D^1, 1 ⇒ D^2; ignored for K=0.
};

// One BFS level: parallel arrays (matrices, provenance, hash keys).
struct Level {
    vector<CMat3>      mats;
    vector<Provenance> prov;
};

// ---------------------------------------------------------------------------
// Memory accounting
// ---------------------------------------------------------------------------

static double resident_gb() {
#ifdef __linux__
    // /proc/self/status VmRSS gives KB.
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return 0.0;
    char buf[256];
    long kb = 0;
    while (fgets(buf, sizeof(buf), f)) {
        if (strncmp(buf, "VmRSS:", 6) == 0) {
            sscanf(buf + 6, "%ld", &kb);
            break;
        }
    }
    fclose(f);
    return kb / (1024.0 * 1024.0);
#else
    return 0.0;
#endif
}

// Memory-budget bookkeeping: estimated bytes per matrix held in (mats vector,
// prov vector, hash entry). CMat3 = 9*16 = 144B, Provenance ~12B, hash entry
// ~ MatKey(72) + bucket pointer (~8B) + load-factor slack. Around 256B/elem
// in practice. (Currently informational only; the live RSS is read from
// /proc/self/status above.)

// ---------------------------------------------------------------------------
// JSON helpers (tiny, no external lib — use ostream with setprecision(15)).
// ---------------------------------------------------------------------------

// Convert a (level K, index inside G_K) to a printable gate word, walking
// parents back through G_0..G_{K-1}.
static string reconstruct_word(int K, int idx,
                               const vector<Level>& G) {
    // Walk parents to depth 0.
    vector<const Provenance*> chain;
    int k = K;
    int i = idx;
    while (k >= 0) {
        chain.push_back(&G[(size_t)k].prov[(size_t)i]);
        if (k == 0) break;
        i = G[(size_t)k].prov[(size_t)i].parent;
        --k;
    }
    // chain[0] = level K (last extension), ..., chain.back() = level 0
    // (initial Clifford). Reverse to print left-to-right:
    //      C[c0] · D^e1 · C[c1] · D^e2 · C[c2] · ...
    string s;
    s.reserve(64);
    // Initial Clifford from level 0:
    const Provenance* c0 = chain.back();
    s += "C[";
    s += to_string(c0->cliff);
    s += "]";
    // Walk forward in the chain (skip the level-0 entry we just printed):
    for (int t = (int)chain.size() - 2; t >= 0; --t) {
        const Provenance* p = chain[(size_t)t];
        s += " · D^";
        s += to_string(p->e + 1);
        s += " · C[";
        s += to_string(p->cliff);
        s += "]";
    }
    return s;
}

// ---------------------------------------------------------------------------
// Resume-from-disk support.
// We dump (mats, prov) for G_K to <prefix>.level<K>.bin once a level is
// finished if memory is large enough to warrant it.
// ---------------------------------------------------------------------------

static bool save_level(const string& prefix, int K, const Level& g) {
    string path = prefix + ".level" + to_string(K) + ".bin";
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    uint64_t n = g.mats.size();
    fwrite(&n, sizeof(n), 1, f);
    if (n) {
        fwrite(g.mats.data(), sizeof(CMat3), (size_t)n, f);
        fwrite(g.prov.data(), sizeof(Provenance), (size_t)n, f);
    }
    fclose(f);
    return true;
}

static bool load_level(const string& prefix, int K, Level& g) {
    string path = prefix + ".level" + to_string(K) + ".bin";
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    uint64_t n = 0;
    if (fread(&n, sizeof(n), 1, f) != 1) { fclose(f); return false; }
    g.mats.resize((size_t)n);
    g.prov.resize((size_t)n);
    if (n) {
        if (fread(g.mats.data(), sizeof(CMat3), (size_t)n, f) != (size_t)n) {
            fclose(f); return false;
        }
        if (fread(g.prov.data(), sizeof(Provenance), (size_t)n, f) != (size_t)n) {
            fclose(f); return false;
        }
    }
    fclose(f);
    return true;
}

// ---------------------------------------------------------------------------
// Build the qutrit Clifford group (same generators / BFS as the existing
// k3_exhaust_test / k23_compare programs).
// ---------------------------------------------------------------------------

static vector<CMat3> build_cliffords() {
    cd om = exp(cd(0, 2.0 * M_PI / 3.0));

    CMat3 genH = {};
    cd h_scale = cd(1, 0) / (cd(1, 0) + cd(2, 0) * om);
    for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k) {
            int e = (j * k) % 3;
            genH.m[j][k] = h_scale * ((e == 0) ? cd(1, 0) :
                                      (e == 1) ? om : om * om);
        }
    CMat3 genX = {};
    genX.m[0][2] = genX.m[1][0] = genX.m[2][1] = cd(1, 0);
    CMat3 genS = {};
    genS.m[0][0] = om;       genS.m[1][1] = cd(1, 0); genS.m[2][2] = cd(1, 0);
    CMat3 genSi = {};
    genSi.m[0][0] = om * om; genSi.m[1][1] = cd(1, 0); genSi.m[2][2] = cd(1, 0);
    CMat3 gens[4] = {genH, genX, genS, genSi};

    CMat3 eye = {};
    eye.m[0][0] = eye.m[1][1] = eye.m[2][2] = cd(1, 0);

    unordered_set<MatKey, MatKeyHash> seen;
    vector<CMat3> cliffords = {eye};
    seen.insert(mat_key(eye));
    size_t head = 0;
    while (head < cliffords.size() && cliffords.size() < 1000) {
        CMat3 M = cliffords[head++];
        for (int g = 0; g < 4; ++g) {
            CMat3 P = cmul(M, gens[g]);
            MatKey k = mat_key(P);
            if (!seen.count(k)) {
                seen.insert(k);
                cliffords.push_back(P);
            }
        }
    }
    return cliffords;
}

// ---------------------------------------------------------------------------
// Main.
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    // -------- CLI parse --------
    vector<double> thetas;
    int     max_depth = 4;
    double  memory_gb = 32.0;
    string  output    = "/tmp/bfs.jsonl";
    string  resume_prefix;
    string  save_prefix;        // by default no checkpoints
    double  checkpoint_min_gb = 8.0;

    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        auto need = [&](int k) {
            if (i + k >= argc) {
                cerr << "missing value after " << a << "\n";
                exit(2);
            }
        };
        if (a == "--theta")          { need(1); thetas.push_back(atof(argv[++i])); }
        else if (a == "--max-depth") { need(1); max_depth = atoi(argv[++i]); }
        else if (a == "--memory-gb") { need(1); memory_gb = atof(argv[++i]); }
        else if (a == "--output")    { need(1); output    = argv[++i]; }
        else if (a == "--resume")    { need(1); resume_prefix = argv[++i]; }
        else if (a == "--save")      { need(1); save_prefix   = argv[++i]; }
        else if (a == "--help" || a == "-h") {
            cout <<
"Usage: bfs_canonical [options]\n"
"  --theta T           target angle (repeatable)\n"
"  --max-depth K       maximum BFS depth in D-gates (default 4)\n"
"  --memory-gb B       memory budget in GB (default 32)\n"
"  --output PATH       JSONL output path (default /tmp/bfs.jsonl)\n"
"  --save PREFIX       checkpoint G_K to PREFIX.level<K>.bin once memory > 8 GB\n"
"  --resume PREFIX     resume from previously saved checkpoints\n";
            return 0;
        } else {
            cerr << "unknown arg: " << a << "\n"; return 2;
        }
    }
    if (thetas.empty()) {
        cerr << "must supply at least one --theta\n";
        return 2;
    }

    // -------- Targets --------
    vector<CMat3> targets((size_t)thetas.size());
    for (size_t t = 0; t < thetas.size(); ++t) {
        CMat3 T = {};
        T.m[0][0] = polar(1.0, -thetas[t] / 2.0);
        T.m[1][1] = polar(1.0,  thetas[t] / 2.0);
        T.m[2][2] = cd(1.0, 0.0);
        targets[t] = T;
    }
    // cum_K[t] = best Frobenius distance over G_0..G_K, for theta_t.
    vector<double> cum_best(thetas.size(), 1e30);
    // Per-theta best (level K, idx into G_K) achieving cum_best.
    vector<pair<int,int>> cum_best_at(thetas.size(), {-1,-1});

    // -------- Open JSONL output --------
    ofstream out(output, ios::app);
    if (!out) {
        cerr << "failed to open output: " << output << "\n"; return 2;
    }
    out << setprecision(15);

    // -------- D gates --------
    cd z9 = exp(cd(0, 2.0 * M_PI / 9.0));
    CMat3 Dgate[2];
    Dgate[0] = {};
    Dgate[0].m[0][0] = z9;        Dgate[0].m[1][1] = cd(1, 0); Dgate[0].m[2][2] = conj(z9);
    Dgate[1] = {};
    Dgate[1].m[0][0] = z9 * z9;   Dgate[1].m[1][1] = cd(1, 0); Dgate[1].m[2][2] = conj(z9) * conj(z9);

    // -------- Cliffords --------
    auto t_clifford_0 = chrono::steady_clock::now();
    vector<CMat3> cliffords = build_cliffords();
    int n_cliff = (int)cliffords.size();
    auto t_clifford_1 = chrono::steady_clock::now();
    double t_cliff_s = chrono::duration<double>(t_clifford_1 - t_clifford_0).count();
    cerr << "Built " << n_cliff << " Cliffords in " << fixed << setprecision(2)
         << t_cliff_s << " s.\n";

    // -------- Prepare the global dedup set --------
    // Maps mat_key -> (level, index) so we can detect "already seen at any
    // earlier level" and resolve hash-collision via mat_eq.
    struct GlobalEntry { int32_t level; int32_t idx; };
    unordered_map<MatKey, GlobalEntry, MatKeyHash> seen_global;

    // -------- Initialise / resume G_0 --------
    vector<Level> G;
    G.reserve((size_t)max_depth + 1);

    Level g0;
    if (!resume_prefix.empty() && load_level(resume_prefix, 0, g0)) {
        cerr << "resume: loaded G_0 (" << g0.mats.size() << " matrices)\n";
    } else {
        g0.mats = cliffords;
        g0.prov.resize((size_t)n_cliff);
        for (int i = 0; i < n_cliff; ++i) {
            g0.prov[(size_t)i] = {(int32_t)-1, (int32_t)i, (int8_t)0};
        }
    }
    // Insert G_0 entries into the global dedup set and update cum_best.
    for (size_t i = 0; i < g0.mats.size(); ++i) {
        seen_global[mat_key(g0.mats[i])] = {(int32_t)0, (int32_t)i};
        for (size_t t = 0; t < thetas.size(); ++t) {
            double d = frob_dist(g0.mats[i], targets[t]);
            if (d < cum_best[t]) {
                cum_best[t] = d;
                cum_best_at[t] = {0, (int)i};
            }
        }
    }
    G.push_back(std::move(g0));

    // -------- Emit level-0 record --------
    {
        auto t_now = chrono::steady_clock::now();
        double wall_s = chrono::duration<double>(t_now - t_clifford_0).count();
        out << "{\"K\": 0";
        out << ", \"G_K_count\": " << G[0].mats.size();
        out << ", \"G_total\": "   << G[0].mats.size();
        out << ", \"cum_K\": {";
        for (size_t t = 0; t < thetas.size(); ++t) {
            if (t) out << ", ";
            out << "\"" << thetas[t] << "\": " << cum_best[t];
        }
        out << "}";
        out << ", \"wall_seconds\": " << wall_s;
        out << ", \"memory_gb\": "    << resident_gb();
        out << ", \"best_word_at_" << thetas[0] << "\": \""
            << reconstruct_word(cum_best_at[0].first,
                                cum_best_at[0].second, G) << "\"";
        out << "}\n" << flush;
    }

    // Try to resume higher levels too (if checkpoints exist).
    if (!resume_prefix.empty()) {
        for (int K = 1; K <= max_depth; ++K) {
            Level gK;
            if (!load_level(resume_prefix, K, gK)) break;
            for (size_t i = 0; i < gK.mats.size(); ++i) {
                seen_global[mat_key(gK.mats[i])] = {(int32_t)K, (int32_t)i};
                for (size_t t = 0; t < thetas.size(); ++t) {
                    double d = frob_dist(gK.mats[i], targets[t]);
                    if (d < cum_best[t]) {
                        cum_best[t] = d;
                        cum_best_at[t] = {K, (int)i};
                    }
                }
            }
            cerr << "resume: loaded G_" << K << " (" << gK.mats.size() << ")\n";
            G.push_back(std::move(gK));
        }
    }

    // ===========================================================================
    // BFS loop.  Each iteration extends G_{K-1} → G_K.
    // ===========================================================================

    int n_threads = omp_get_max_threads();
    cerr << "Using " << n_threads << " OpenMP threads.\n";

    for (int K = (int)G.size(); K <= max_depth; ++K) {
        const Level& prev = G[(size_t)K - 1];
        size_t n_prev = prev.mats.size();
        cerr << "BFS level " << K << ": expanding " << n_prev
             << " parents × 2 D-exponents × " << n_cliff
             << " Cliffords = " << (n_prev * 2ULL * (size_t)n_cliff) << " candidates.\n";

        // -------- Inline-dedup expansion --------
        // Each thread maintains a tiny staging buffer (BATCH entries). When
        // full, the thread enters a critical section, looks each candidate up
        // in seen_global, and (if new) inserts into seen_global + gK. This
        // amortises lock overhead while keeping peak memory bounded:
        // unique candidates land in gK as we go, duplicates are dropped
        // immediately, so per-thread RAM stays at O(BATCH).
        Level gK;
        gK.mats.reserve((size_t)n_prev);   // rough lower bound
        gK.prov.reserve((size_t)n_prev);

        constexpr size_t BATCH = 64;

        // Memory budget abort flag.
        atomic<bool> aborted(false);

        // We split work over (parent index v, exponent e). Each task expands
        // n_cliff Cliffords in a tight inner loop. Dynamic schedule keeps
        // load even (early parents may dedup more).
        long long total_tasks = (long long)n_prev * 2;
        long long tasks_done = 0;
        long long progress_step = max((long long)1, total_tasks / 50);

        // Lambda that flushes a thread-local batch into seen_global + gK
        // under a critical section.
        auto flush_batch = [&](array<CMat3, BATCH>& bmats,
                               array<MatKey, BATCH>& bkeys,
                               array<Provenance, BATCH>& bprov,
                               size_t n) {
            if (n == 0) return;
            #pragma omp critical(seen_global)
            {
                for (size_t i = 0; i < n; ++i) {
                    auto it = seen_global.find(bkeys[i]);
                    if (it != seen_global.end()) {
                        // Hash-collision check: confirm it's truly the same
                        // matrix.
                        int lvl = it->second.level;
                        int idx = it->second.idx;
                        const CMat3& other = (lvl == K)
                            ? gK.mats[(size_t)idx]
                            : G[(size_t)lvl].mats[(size_t)idx];
                        if (mat_eq(bmats[i], other)) {
                            continue;  // genuine duplicate
                        }
                        // Hash collision but different matrix — should be
                        // unreachable at 1e-7 rounding for Clifford+D
                        // matrices. Log and skip (treat as duplicate).
                        cerr << "  WARNING: hash collision with distinct "
                                "matrix at level " << K
                             << " (other at lvl=" << lvl
                             << " idx=" << idx << "); skipping.\n";
                        continue;
                    }
                    int32_t new_idx = (int32_t)gK.mats.size();
                    gK.mats.push_back(bmats[i]);
                    gK.prov.push_back(bprov[i]);
                    seen_global[bkeys[i]] = {(int32_t)K, new_idx};
                }
            }
        };

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            array<CMat3, BATCH>      bmats;
            array<MatKey, BATCH>     bkeys;
            array<Provenance, BATCH> bprov;
            size_t bn = 0;

            #pragma omp for schedule(dynamic, 32)
            for (long long t = 0; t < total_tasks; ++t) {
                if (aborted.load(memory_order_relaxed)) continue;
                int v = (int)(t / 2);
                int e = (int)(t % 2);
                CMat3 VD = cmul(prev.mats[(size_t)v], Dgate[e]);
                for (int c = 0; c < n_cliff; ++c) {
                    CMat3 M = cmul(VD, cliffords[(size_t)c]);
                    MatKey k = mat_key(M);
                    bmats[bn] = M;
                    bkeys[bn] = k;
                    bprov[bn] = {(int32_t)v, (int32_t)c, (int8_t)e};
                    ++bn;
                    if (bn == BATCH) {
                        flush_batch(bmats, bkeys, bprov, bn);
                        bn = 0;
                    }
                }

                // Crude progress + memory check.
                long long done;
                #pragma omp atomic capture
                done = ++tasks_done;
                if (tid == 0 && (done % progress_step == 0)) {
                    double rss = resident_gb();
                    cerr << "\r  K=" << K << " progress="
                         << (done * 100 / total_tasks) << "%  "
                         << "RSS=" << fixed << setprecision(2) << rss
                         << " GB    " << flush;
                    if (rss > memory_gb) {
                        cerr << "\n  memory budget exceeded ("
                             << rss << " GB > " << memory_gb << " GB), aborting.\n";
                        aborted.store(true, memory_order_relaxed);
                    }
                }
            }

            // Flush any leftover entries from this thread's batch.
            flush_batch(bmats, bkeys, bprov, bn);
            bn = 0;
        }
        cerr << "\n";

        // -------- Update cum_best --------
        // Parallel reduction over G_K.
        size_t n_K = gK.mats.size();
        for (size_t t = 0; t < thetas.size(); ++t) {
            double local_best = cum_best[t];
            int    local_idx  = -1;
            #pragma omp parallel
            {
                double th_best = local_best;
                int    th_idx  = -1;
                #pragma omp for schedule(static)
                for (long long i = 0; i < (long long)n_K; ++i) {
                    double d = frob_dist(gK.mats[(size_t)i], targets[t]);
                    if (d < th_best) { th_best = d; th_idx = (int)i; }
                }
                #pragma omp critical
                if (th_best < local_best) { local_best = th_best; local_idx = th_idx; }
            }
            if (local_best < cum_best[t]) {
                cum_best[t] = local_best;
                cum_best_at[t] = {K, local_idx};
            }
        }

        // -------- G_total accounting --------
        size_t G_total = 0;
        for (auto& g : G) G_total += g.mats.size();
        G_total += n_K;

        // Push G_K into the global vector so reconstruct_word() can walk it
        // before we emit the JSON line.
        G.push_back(std::move(gK));

        // -------- Emit JSONL record --------
        auto t_K_1 = chrono::steady_clock::now();
        double wall_s = chrono::duration<double>(t_K_1 - t_clifford_0).count();
        double rss = resident_gb();
        out << "{\"K\": " << K;
        out << ", \"G_K_count\": " << n_K;
        out << ", \"G_total\": "   << G_total;
        out << ", \"cum_K\": {";
        for (size_t t = 0; t < thetas.size(); ++t) {
            if (t) out << ", ";
            out << "\"" << thetas[t] << "\": " << cum_best[t];
        }
        out << "}";
        out << ", \"wall_seconds\": " << wall_s;
        out << ", \"memory_gb\": "    << rss;
        out << ", \"best_word_at_" << thetas[0] << "\": \""
            << reconstruct_word(cum_best_at[0].first,
                                cum_best_at[0].second, G) << "\"";
        out << ", \"aborted\": " << (aborted.load() ? "true" : "false");
        out << "}\n" << flush;

        // -------- Optional checkpoint --------
        if (!save_prefix.empty() && rss > checkpoint_min_gb) {
            if (save_level(save_prefix, K, G.back())) {
                cerr << "  checkpoint: saved G_" << K
                     << " to " << save_prefix << ".level" << K << ".bin\n";
            } else {
                cerr << "  checkpoint failed for G_" << K << "\n";
            }
        }

        if (aborted.load()) {
            cerr << "stopping at level " << K << " (memory budget).\n";
            break;
        }
        if (rss > memory_gb) {
            cerr << "stopping after level " << K
                 << " (RSS " << rss << " GB > budget " << memory_gb << " GB).\n";
            break;
        }
    }

    // -------- Final summary to stderr --------
    cerr << "\n========== final ==========\n";
    for (size_t t = 0; t < thetas.size(); ++t) {
        cerr << "theta=" << thetas[t]
             << "  cum_best=" << setprecision(15) << cum_best[t]
             << "  level=" << cum_best_at[t].first
             << "  word=" << reconstruct_word(cum_best_at[t].first,
                                              cum_best_at[t].second, G)
             << "\n";
    }
    return 0;
}
