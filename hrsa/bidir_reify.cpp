// bidir_reify.cpp - Implementation of run_bidir() and reify_bidir().
//
// CLIFFORD INDEX ACCESS: we use the previously exposed
// get_clifford_cache() (declared in clifford_cache.h, defined in
// decompose.cpp). The cache contains both the numerical CMat3 Cliffords
// and their ringZ9chi twins in the same BFS order that bidir_bfs.cpp
// uses internally. We therefore index directly with the (start_clifford, c)
// integers parsed from the bidir_bfs WIN line — no permutation needed.
// A spot-check validates the first few entries match (see verify_index_alignment).
//
// STEPS EMISSION: BidirCircuit.steps is left EMPTY. Each (e, c) pair in the
// forward / backward index lists encodes  (D^e · C_c)  where C_c is an
// arbitrary Clifford produced by BFS, NOT one of HRSA's canonical
// "[H?] · D(a0,a1,a2) · D^eps · X^delta" prefixes. There is no faithful
// way to encode a generic monomial Clifford into a single GateStep without
// extending the GateStep schema (and v_validate.py only checks N_D and the
// V matrix, not the syllable list). Callers can still recover the gate
// sequence from BidirCircuit.start_clifford / forward / backward.
//
// The HRSA dispatcher emits the BidirCircuit's V matrix and N_D into the
// uniform-schema JSON; the syllables[] field is null for bidir circuits.

#include "bidir_reify.h"
#include "clifford_cache.h"

#include <cctype>
#include <cerrno>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

using cd = std::complex<double>;

// Lightweight Mat3 frobenius distance vs a target diagonal R^Z(theta),
// computed by evaluating each ringZ9chi entry numerically.
double mat3_frob_to_RZ(const Mat3& V, double theta) {
    cd target[3][3] = {};
    target[0][0] = std::polar(1.0, -theta/2.0);
    target[1][1] = std::polar(1.0,  theta/2.0);
    target[2][2] = cd(1.0, 0.0);
    double s = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            cd v = V.m[i][j].toComplexDouble();
            cd d = v - target[i][j];
            s += d.real()*d.real() + d.imag()*d.imag();
        }
    return std::sqrt(s);
}

// Build the first few BFS-ordered Cliffords using identical logic to
// bidir_bfs.cpp's build_cliffords(). Used only to spot-check that the
// shared get_clifford_cache() is in the same order. Returns the first
// `n_first` matrices.
std::vector<CMat3> build_first_n_cliffords_bidir_style(int n_first) {
    cd om = std::exp(cd(0, 2.0*M_PI/3.0));
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

    auto cmul_local = [](const CMat3& A, const CMat3& B) {
        CMat3 C = {};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                    C.m[i][j] += A.m[i][k] * B.m[k][j];
        return C;
    };
    auto mat_key = [](const CMat3& M) {
        std::vector<int> k(18);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                k[6*i+2*j]   = (int)std::round(M.m[i][j].real() * 1e5);
                k[6*i+2*j+1] = (int)std::round(M.m[i][j].imag() * 1e5);
            }
        return k;
    };
    std::set<std::vector<int>> seen;
    CMat3 eye = {};
    eye.m[0][0] = eye.m[1][1] = eye.m[2][2] = cd(1,0);
    std::vector<CMat3> q = {eye};
    seen.insert(mat_key(eye));
    CMat3 gens[4] = {genH, genX, genS, genSi};
    size_t head = 0;
    while (head < q.size() && (int)q.size() < n_first) {
        CMat3 M = q[head++];
        for (int g = 0; g < 4; ++g) {
            CMat3 P = cmul_local(M, gens[g]);
            auto k = mat_key(P);
            if (seen.find(k) == seen.end()) {
                seen.insert(k);
                q.push_back(P);
            }
        }
    }
    if ((int)q.size() > n_first) q.resize(n_first);
    return q;
}

// Numerical evaluation of a Mat3 via toComplexDouble.
CMat3 ring_to_cmat(const Mat3& M) {
    CMat3 R = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R.m[i][j] = M.m[i][j].toComplexDouble();
    return R;
}

double cmat3_frob(const CMat3& A, const CMat3& B) {
    double s = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            cd d = A.m[i][j] - B.m[i][j];
            s += d.real()*d.real() + d.imag()*d.imag();
        }
    return std::sqrt(s);
}

// Spot-check that our shared cache's first N Cliffords match the BFS
// ordering used by bidir_bfs. Run once on first reify call.
void verify_index_alignment_once() {
    static bool checked = false;
    if (checked) return;
    checked = true;

    constexpr int N_CHECK = 5;
    const CliffordCache& cache = get_clifford_cache();
    if ((int)cache.ring_cliffords.size() < N_CHECK) {
        std::cerr << "bidir_reify: cache too small ("
                  << cache.ring_cliffords.size()
                  << " Cliffords); skipping alignment spot-check.\n";
        return;
    }
    auto first = build_first_n_cliffords_bidir_style(N_CHECK);
    double max_d = 0.0;
    for (int i = 0; i < N_CHECK; ++i) {
        CMat3 ring_eval = ring_to_cmat(cache.ring_cliffords[i]);
        double d = cmat3_frob(first[i], ring_eval);
        if (d > max_d) max_d = d;
    }
    if (max_d > 1e-6) {
        std::cerr << "bidir_reify: WARNING — Clifford index alignment spot-check "
                     "max Frob = " << max_d
                  << " (> 1e-6); BFS orders may not match! "
                     "Reified V will be wrong.\n";
    } else {
        std::cerr << "bidir_reify: alignment spot-check OK (max Frob "
                  << max_d << " over first " << N_CHECK << " entries).\n";
    }
}

// Parse a "forward=[e:c,e:c,...]" or "backward=[e:c,e:c,...]" segment
// out of the WIN line. `key` is the prefix before "=[".
bool parse_path(const std::string& line, const std::string& key,
                std::vector<std::pair<int,int>>& out) {
    out.clear();
    std::string needle = key + "=[";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) return false;
    pos += needle.size();
    size_t end = line.find(']', pos);
    if (end == std::string::npos) return false;
    std::string body = line.substr(pos, end - pos);
    if (body.empty()) return true;  // empty path is legal
    std::stringstream ss(body);
    std::string token;
    while (std::getline(ss, token, ',')) {
        size_t colon = token.find(':');
        if (colon == std::string::npos) return false;
        try {
            int e = std::stoi(token.substr(0, colon));
            int c = std::stoi(token.substr(colon + 1));
            out.emplace_back(e, c);
        } catch (...) {
            return false;
        }
    }
    return true;
}

bool parse_int_field(const std::string& line, const std::string& key, int& out) {
    std::string needle = key + "=";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) return false;
    pos += needle.size();
    size_t end = pos;
    while (end < line.size() && (std::isdigit((unsigned char)line[end])
                                  || line[end] == '-')) ++end;
    if (end == pos) return false;
    try {
        out = std::stoi(line.substr(pos, end - pos));
    } catch (...) {
        return false;
    }
    return true;
}

bool parse_double_field(const std::string& line, const std::string& key, double& out) {
    std::string needle = key + "=";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) return false;
    pos += needle.size();
    size_t end = pos;
    while (end < line.size() && (std::isdigit((unsigned char)line[end])
                                  || line[end] == '.' || line[end] == '-'
                                  || line[end] == '+' || line[end] == 'e'
                                  || line[end] == 'E')) ++end;
    if (end == pos) return false;
    try {
        out = std::stod(line.substr(pos, end - pos));
    } catch (...) {
        return false;
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// reify_bidir: walk the index path through the shared ringZ9chi cache.
// ---------------------------------------------------------------------------
BidirCircuit reify_bidir(int start_clifford,
                          const std::vector<std::pair<int,int>>& forward,
                          const std::vector<std::pair<int,int>>& backward,
                          double theta) {
    BidirCircuit bc;
    bc.start_clifford = start_clifford;
    bc.forward = forward;
    bc.backward = backward;
    bc.N_D = (int)forward.size() + (int)backward.size();

    verify_index_alignment_once();

    const CliffordCache& cache = get_clifford_cache();
    const int n_cliff = cache.n_cliff;

    auto in_range = [n_cliff](int idx) {
        return idx >= 0 && idx < n_cliff;
    };
    // bidir_bfs prints e as the Dgate[] array index (0 or 1), where
    //   Dgate[0] = D^1,  Dgate[1] = D^2.
    auto e_ok = [](int e) { return e >= 0 && e <= 1; };

    if (!in_range(start_clifford)) {
        std::cerr << "bidir_reify: start_clifford=" << start_clifford
                  << " out of range [0," << n_cliff << ").\n";
        return bc;
    }
    for (auto& p : forward) {
        if (!e_ok(p.first) || !in_range(p.second)) {
            std::cerr << "bidir_reify: bad forward gate (e="
                      << p.first << ", c=" << p.second << ").\n";
            return bc;
        }
    }
    for (auto& p : backward) {
        if (!e_ok(p.first) || !in_range(p.second)) {
            std::cerr << "bidir_reify: bad backward gate (e="
                      << p.first << ", c=" << p.second << ").\n";
            return bc;
        }
    }

    // V = ring_cliffords[start_clifford]
    //     · ∏_(e,c)∈forward  (ring_Dgate[e] · ring_cliffords[c])
    //     · ∏_(e,c)∈backward (ring_Dgate[e] · ring_cliffords[c])
    // (e is the Dgate[] array index, 0-based.)
    Mat3 V = cache.ring_cliffords[start_clifford];
    for (const auto& p : forward) {
        V = V.mul(cache.ring_Dgate[p.first]).mul(cache.ring_cliffords[p.second]);
    }
    for (const auto& p : backward) {
        V = V.mul(cache.ring_Dgate[p.first]).mul(cache.ring_cliffords[p.second]);
    }
    bc.V = V;

    // Compute v_f = max getExp() over V's entries (ringZ9chi denominator
    // exponent; matches the f_common convention in HRSA_test.cpp's JSON
    // emitter).
    int v_f = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            int e = V.m[i][j].getExp();
            if (e > v_f) v_f = e;
        }
    bc.v_f = v_f;

    bc.frob_to_target = mat3_frob_to_RZ(V, theta);
    bc.steps.clear();   // explicitly empty — see file header note.
    bc.valid = true;
    return bc;
}

// ---------------------------------------------------------------------------
// run_bidir: fork + popen ./bidir_bfs and parse the WIN line.
// ---------------------------------------------------------------------------
BidirCircuit run_bidir(double theta, int K_f, int K_b, double target_eps) {
    BidirCircuit bc;

    // Locate bidir_bfs binary. We assume it sits next to HRSA_tester (current
    // working directory if invoked from unified/hrsa/, else fail).
    // Try a couple of paths in order: ./bidir_bfs, then absolute install path.
    const char* candidates[] = {
        "./bidir_bfs",
        "/home/hlamm/Desktop/efficent_gates/unified/hrsa/bidir_bfs",
    };
    std::string bin;
    for (const char* p : candidates) {
        if (access(p, X_OK) == 0) { bin = p; break; }
    }
    if (bin.empty()) {
        std::cerr << "run_bidir: cannot locate bidir_bfs binary "
                     "(tried ./bidir_bfs and absolute path).\n";
        return bc;
    }

    std::ostringstream cmd;
    cmd << bin << " " << theta << " " << K_f << " " << K_b
        << " 2>/dev/null";

    FILE* fp = popen(cmd.str().c_str(), "r");
    if (!fp) {
        std::cerr << "run_bidir: popen failed: " << std::strerror(errno) << "\n";
        return bc;
    }

    std::string win_line;
    std::string verify_line;
    char buf[8192];
    while (std::fgets(buf, sizeof(buf), fp)) {
        std::string line(buf);
        // Strip trailing newline for cleaner parsing
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
            line.pop_back();
        if (line.rfind("WIN:", 0) == 0) win_line = line;
        else if (line.rfind("verify:", 0) == 0) verify_line = line;
    }
    int rc = pclose(fp);
    if (rc != 0) {
        std::cerr << "run_bidir: bidir_bfs exited with status " << rc << "\n";
        // continue anyway — partial output may still contain the WIN line
    }

    if (win_line.empty()) {
        std::cerr << "run_bidir: no WIN line in bidir_bfs output.\n";
        return bc;
    }
    if (win_line.find("(no approximate match found") != std::string::npos) {
        std::cerr << "run_bidir: bidir_bfs reported no approximate match.\n";
        return bc;
    }

    // Parse fields: depth=, frob=, start_clifford=, forward=[..], backward=[..]
    int depth = -1;
    double frob = -1.0;
    int start_cliff = -1;
    std::vector<std::pair<int,int>> fwd, bwd;
    parse_int_field(win_line, "depth", depth);
    parse_double_field(win_line, "frob", frob);
    parse_int_field(win_line, "start_clifford", start_cliff);
    bool fp_ok = parse_path(win_line, "forward", fwd);
    bool bp_ok = parse_path(win_line, "backward", bwd);

    if (start_cliff < 0 || !fp_ok || !bp_ok) {
        std::cerr << "run_bidir: failed to parse WIN line: " << win_line << "\n";
        return bc;
    }

    // Reify against the shared ringZ9chi cache.
    BidirCircuit out = reify_bidir(start_cliff, fwd, bwd, theta);
    if (!out.valid) {
        return bc;
    }

    // Cross-check the numerical Frob (ours vs bidir's) — should agree up to
    // ~1e-6 because bidir works in double-precision and so does our
    // toComplexDouble evaluation.
    if (frob >= 0.0) {
        double diff = std::fabs(out.frob_to_target - frob);
        if (diff > 1e-3) {
            std::cerr << "run_bidir: WARNING — frob mismatch: bidir reports "
                      << frob << " but reified V gives " << out.frob_to_target
                      << " (delta=" << diff << ").\n";
        }
    }

    // Reported depth should equal forward.size() + backward.size() = N_D.
    if (depth >= 0 && depth != out.N_D) {
        std::cerr << "run_bidir: WARNING — bidir reports depth=" << depth
                  << " but parsed forward+backward=" << out.N_D << "\n";
    }

    (void)verify_line;  // could parse the constructed-vs-target check, optional.
    (void)target_eps;   // unused; the dispatcher decides accept/reject.
    return out;
}
