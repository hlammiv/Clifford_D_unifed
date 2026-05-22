// beam_search.cpp - Priority-queue beam search over qutrit Clifford+D circuits.
//
// For a given target R^Z(theta), maintains a beam of the top-N matrices closest
// (in Frobenius norm) to the target.  At each depth K, each beam member V is
// expanded as V * D^e * C for e in {1,2}, C in the qutrit Clifford group.  The
// best BEAM_WIDTH candidates by Frobenius distance survive.  We report the
// running minimum (cum_K) per depth for each theta.
//
// Single-target beam: one independent beam per theta.  Per-theta JSONL output.
//
// Build (matches the existing project flags):
//   g++ -O3 -flto -fopenmp -Wall -pedantic beam_search.cpp -o beam_search
//
// Usage:
//   ./beam_search --theta 2.14 [--theta 1.0053] \
//                 --max-depth 16 --beam-width 100000 \
//                 --output /tmp/beam.jsonl
//
// Per-level output (appended to the JSONL file, one line per (theta, K)):
//   {"K":4,"theta":2.14,"beam_size":100000,"best_so_far":0.31415,
//    "wall_seconds":12.3,"best_word_at_K":"D^1 C[123] D^2 C[45] ..."}
//
// Notes:
//   * The Clifford group / Dgate / cmul / frob_dist scaffolding mirrors
//     k3_exhaust_test.cpp and k23_compare.cpp so behaviour is consistent.
//   * Top-N selection uses a fixed-size max-heap (std::priority_queue with the
//     default less<> comparator gives a max-heap on the distance, so the worst
//     entry sits at top() and is the cheapest to evict when the heap is full).
//   * Parallelism: OMP partitions the beam and each thread maintains its own
//     local top-N heap.  Heaps are merged after the parallel region.
//   * Word tracking: each beam member stores its full word as a vector of
//     uint16_t step codes (step = e * n_cliff + c, with e in {0,1}).  Memory
//     at beam=100K, depth=20 is ~4 MB total, so we just store words inline.

#include <complex>
#include <vector>
#include <set>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <queue>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <omp.h>

using std::vector;
using std::string;
using cd = std::complex<double>;

struct CMat3 { cd m[3][3]; };

static inline CMat3 cmul(const CMat3& A, const CMat3& B) {
    CMat3 C = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            cd s(0, 0);
            for (int k = 0; k < 3; ++k) s += A.m[i][k] * B.m[k][j];
            C.m[i][j] = s;
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
    return std::sqrt(s);
}

// Beam member: matrix + word (encoded as step codes).  We store the parent
// index and the latest step separately so we can rebuild words cheaply, but
// because the beam can be permuted between levels we just keep the full word.
struct BeamElt {
    CMat3 V;
    double dist;
    vector<uint16_t> word;  // length == current depth K; step = e*n_cliff + c
};

// Max-heap entry for top-N selection: order by dist (largest at top()).
// We hold an INDEX into a backing pool so we don't shuffle big BeamElts.
struct HeapEnt {
    double dist;
    uint32_t idx;
    bool operator<(const HeapEnt& o) const { return dist < o.dist; } // max-heap
};

// Build the qutrit Clifford group from {H, X, S, S^{-1}} via BFS on cmul.
// Same construction as k3_exhaust_test.cpp / k23_compare.cpp.
static vector<CMat3> build_cliffords() {
    cd om = std::exp(cd(0, 2.0*M_PI/3.0));

    CMat3 genH = {};
    cd h_scale = cd(1,0) / (cd(1,0) + cd(2,0)*om);
    for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k) {
            int e = (j*k) % 3;
            genH.m[j][k] = h_scale * ((e==0) ? cd(1,0) : (e==1) ? om : om*om);
        }
    CMat3 genX = {};
    genX.m[0][2] = genX.m[1][0] = genX.m[2][1] = cd(1,0);
    CMat3 genS = {};
    genS.m[0][0] = om;     genS.m[1][1] = cd(1,0); genS.m[2][2] = cd(1,0);
    CMat3 genSi = {};
    genSi.m[0][0] = om*om; genSi.m[1][1] = cd(1,0); genSi.m[2][2] = cd(1,0);

    auto mat_key = [](const CMat3& M) -> vector<int> {
        vector<int> k(18);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                k[6*i+2*j]   = (int)std::round(M.m[i][j].real() * 1e5);
                k[6*i+2*j+1] = (int)std::round(M.m[i][j].imag() * 1e5);
            }
        return k;
    };
    std::set<vector<int>> seen;
    CMat3 eye = {}; eye.m[0][0] = eye.m[1][1] = eye.m[2][2] = cd(1,0);
    vector<CMat3> cliffords = {eye};
    seen.insert(mat_key(eye));
    CMat3 gens[4] = {genH, genX, genS, genSi};
    size_t head = 0;
    while (head < cliffords.size() && cliffords.size() < 1000) {
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
    return cliffords;
}

static CMat3 make_target(double theta) {
    CMat3 t = {};
    t.m[0][0] = std::polar(1.0, -theta/2.0);
    t.m[1][1] = std::polar(1.0,  theta/2.0);
    t.m[2][2] = cd(1.0, 0.0);
    return t;
}

// Decode a beam member's word into "D^e C[c] ..." form.
static string word_to_string(const vector<uint16_t>& word, int n_cliff) {
    std::ostringstream os;
    for (size_t i = 0; i < word.size(); ++i) {
        int code = word[i];
        int e = code / n_cliff;
        int c = code % n_cliff;
        if (i) os << " ";
        os << "D^" << (e+1) << " C[" << c << "]";
    }
    return os.str();
}

// JSON-escape a string (only " and \ matter in our outputs since words are ASCII).
static string json_escape(const string& s) {
    string r;
    r.reserve(s.size());
    for (char c : s) {
        if (c == '"' || c == '\\') r.push_back('\\');
        r.push_back(c);
    }
    return r;
}

struct CliArgs {
    vector<double> thetas;
    int max_depth = 16;
    size_t beam_width = 100000;
    string output = "/tmp/beam.jsonl";
};

static bool parse_args(int argc, char** argv, CliArgs& a) {
    for (int i = 1; i < argc; ++i) {
        string s = argv[i];
        auto take_d = [&](double& out) {
            if (i + 1 >= argc) return false;
            out = std::atof(argv[++i]);
            return true;
        };
        auto take_i = [&](int& out) {
            if (i + 1 >= argc) return false;
            out = std::atoi(argv[++i]);
            return true;
        };
        auto take_s = [&](string& out) {
            if (i + 1 >= argc) return false;
            out = argv[++i];
            return true;
        };
        if (s == "--theta") {
            double v;
            if (!take_d(v)) return false;
            a.thetas.push_back(v);
        } else if (s == "--max-depth") {
            if (!take_i(a.max_depth)) return false;
        } else if (s == "--beam-width") {
            int v; if (!take_i(v)) return false;
            if (v <= 0) return false;
            a.beam_width = (size_t)v;
        } else if (s == "--output") {
            if (!take_s(a.output)) return false;
        } else if (s == "-h" || s == "--help") {
            return false;
        } else {
            std::cerr << "unknown arg: " << s << "\n";
            return false;
        }
    }
    return !a.thetas.empty();
}

static void usage(const char* prog) {
    std::cerr <<
        "usage: " << prog << " --theta <T> [--theta <T> ...] \\\n"
        "         --max-depth <K> --beam-width <N> --output <file.jsonl>\n";
}

// Run beam search for one theta.  Appends one JSONL line per depth to `out`.
static void run_beam(double theta, int max_depth, size_t beam_width,
                     const vector<CMat3>& cliffords, const CMat3 Dgate[2],
                     std::ostream& out) {
    const int n_cliff = (int)cliffords.size();
    const CMat3 target = make_target(theta);
    const int n_steps = 2 * n_cliff;  // (e, c) combinations expanding each member

    // Precompute D^e * C for the inner expansion: dc[e*n_cliff + c] = D^(e+1) * C
    vector<CMat3> dc((size_t)n_steps);
    for (int e = 0; e < 2; ++e)
        for (int c = 0; c < n_cliff; ++c)
            dc[(size_t)(e * n_cliff + c)] = cmul(Dgate[e], cliffords[c]);

    // Initial beam at K=0: the identity.  We do not emit a K=0 JSONL line
    // because the task asks for K = 1..max_depth.
    CMat3 eye = {}; eye.m[0][0] = eye.m[1][1] = eye.m[2][2] = cd(1,0);
    vector<BeamElt> beam;
    beam.push_back({eye, frob_dist(eye, target), {}});

    double cum_best = beam[0].dist;
    string cum_best_word = "";  // empty word == identity at K=0

    for (int K = 1; K <= max_depth; ++K) {
        double t0 = omp_get_wtime();
        const int in_size = (int)beam.size();
        const size_t total_cands = (size_t)in_size * (size_t)n_steps;

        // We will produce up to beam_width survivors.  Each thread keeps a
        // local max-heap of size <= beam_width over indices into a
        // thread-local pool of "candidate" entries (CMat3 + dist + parent_idx
        // + step_code).  After the parallel region, merge.

        struct Cand {
            CMat3 V;
            double dist;
            uint32_t parent;  // index into `beam`
            uint16_t step;    // e * n_cliff + c
        };

        const int nT = omp_get_max_threads();
        vector<vector<Cand>>            tl_pool(nT);
        vector<std::priority_queue<HeapEnt>> tl_heap(nT);
        for (int t = 0; t < nT; ++t) {
            tl_pool[t].reserve(std::min(beam_width + 16,
                                        (size_t)((total_cands / (size_t)nT) + 16)));
        }

        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            auto& pool = tl_pool[tid];
            auto& heap = tl_heap[tid];

            // Distribute beam-member expansions across threads.
            #pragma omp for schedule(static)
            for (int b = 0; b < in_size; ++b) {
                const CMat3& V = beam[(size_t)b].V;
                for (int s = 0; s < n_steps; ++s) {
                    CMat3 M = cmul(V, dc[(size_t)s]);
                    double d = frob_dist(M, target);

                    if (heap.size() < beam_width) {
                        Cand c{M, d, (uint32_t)b, (uint16_t)s};
                        uint32_t pi = (uint32_t)pool.size();
                        pool.push_back(std::move(c));
                        heap.push({d, pi});
                    } else if (d < heap.top().dist) {
                        // Reuse the slot of the entry we are evicting so the
                        // pool stays bounded by beam_width.
                        uint32_t pi = heap.top().idx;
                        heap.pop();
                        pool[pi] = Cand{M, d, (uint32_t)b, (uint16_t)s};
                        heap.push({d, pi});
                    }
                }
            }
        }

        // Merge per-thread heaps into a single max-heap of size <= beam_width.
        // Re-emit (dist, global_pool_idx) into a flat global pool so survivors
        // don't reference per-thread storage.
        std::priority_queue<HeapEnt> gheap;
        vector<Cand> gpool;
        gpool.reserve(beam_width + 16);
        for (int t = 0; t < nT; ++t) {
            auto& pool = tl_pool[t];
            auto& heap = tl_heap[t];
            while (!heap.empty()) {
                HeapEnt e = heap.top(); heap.pop();
                const Cand& c = pool[e.idx];
                if (gheap.size() < beam_width) {
                    uint32_t pi = (uint32_t)gpool.size();
                    gpool.push_back(c);
                    gheap.push({c.dist, pi});
                } else if (c.dist < gheap.top().dist) {
                    uint32_t pi = gheap.top().idx;
                    gheap.pop();
                    gpool[pi] = c;
                    gheap.push({c.dist, pi});
                }
            }
        }

        // Build the new beam from the merged pool.  Sort survivors by dist
        // ascending so beam[0] is always the closest.
        const size_t survivors = gpool.size();
        vector<BeamElt> new_beam;
        new_beam.reserve(survivors);
        for (size_t i = 0; i < survivors; ++i) {
            const Cand& c = gpool[i];
            BeamElt elt;
            elt.V = c.V;
            elt.dist = c.dist;
            elt.word = beam[c.parent].word;
            elt.word.push_back(c.step);
            new_beam.push_back(std::move(elt));
        }
        std::sort(new_beam.begin(), new_beam.end(),
                  [](const BeamElt& a, const BeamElt& b){ return a.dist < b.dist; });
        beam = std::move(new_beam);

        // Update running minimum.  cum_best may be at this depth or earlier.
        double depth_best = beam.empty() ? 1e30 : beam[0].dist;
        string depth_best_word = beam.empty() ? string()
            : word_to_string(beam[0].word, n_cliff);
        if (depth_best < cum_best) {
            cum_best = depth_best;
            cum_best_word = depth_best_word;
        }

        double wall = omp_get_wtime() - t0;

        // Emit JSONL: one line per (theta, K).
        out << "{"
            << "\"K\":" << K
            << ",\"theta\":" << std::setprecision(15) << theta
            << ",\"beam_size\":" << beam.size()
            << ",\"best_so_far\":" << std::setprecision(15) << cum_best
            << ",\"depth_best\":" << std::setprecision(15) << depth_best
            << ",\"wall_seconds\":" << std::setprecision(6) << wall
            << ",\"best_word_at_K\":\"" << json_escape(depth_best_word) << "\""
            << ",\"best_word_so_far\":\"" << json_escape(cum_best_word) << "\""
            << "}\n";
        out.flush();

        std::cerr << "  theta=" << std::setprecision(6) << theta
                  << " K=" << K
                  << " beam=" << beam.size()
                  << " depth_best=" << std::setprecision(10) << depth_best
                  << " cum_best=" << std::setprecision(10) << cum_best
                  << " wall=" << std::setprecision(3) << wall << "s\n";
    }
}

int main(int argc, char** argv) {
    CliArgs args;
    if (!parse_args(argc, argv, args)) {
        usage(argv[0]);
        return 1;
    }

    vector<CMat3> cliffords = build_cliffords();
    int n_cliff = (int)cliffords.size();
    std::cerr << "Built " << n_cliff << " Cliffords.\n";

    cd z9 = std::exp(cd(0, 2.0*M_PI/9.0));
    CMat3 Dgate[2];
    Dgate[0] = {}; Dgate[0].m[0][0] = z9;       Dgate[0].m[1][1] = cd(1,0); Dgate[0].m[2][2] = std::conj(z9);
    Dgate[1] = {}; Dgate[1].m[0][0] = z9*z9;    Dgate[1].m[1][1] = cd(1,0); Dgate[1].m[2][2] = std::conj(z9)*std::conj(z9);

    std::ofstream out(args.output, std::ios::app);
    if (!out) {
        std::cerr << "cannot open output file: " << args.output << "\n";
        return 2;
    }

    std::cerr << "max_depth=" << args.max_depth
              << " beam_width=" << args.beam_width
              << " thetas=" << args.thetas.size()
              << " threads=" << omp_get_max_threads()
              << "\n";

    for (double theta : args.thetas) {
        std::cerr << "==== theta=" << std::setprecision(15) << theta << " ====\n";
        run_beam(theta, args.max_depth, args.beam_width, cliffords, Dgate, out);
    }
    return 0;
}
