// esa_v_check.cpp — does ESA's enumeration cover HRSA's V?
//
// Reads /tmp/hrsa_v_numerators.txt (written by hrsa_v_inspect) which has
// V[i][j].getNumerator() as 9 ringZ9 coefficients per entry.  Then calls
// ESA's fullDiagEnumeration / fullX3Enumeration with the SAME args ESA
// internally uses (negated theta, max_f=4, etc.) and checks whether each
// of the 6 components ESA actually enumerates (V[0][0], V[1][1], V[2][2]
// for diagonals; V[0][1], V[0][2], V[1][2] from lookup) is present.
//
// Run twice: once with max_candidates=0 (exhaustive — tests "is the
// enumeration logic correct?") and once with max_candidates=500 (default
// — tests "is the top-500 truncation losing the V we need?").
//
// V[1][0], V[2][0], V[2][1] are NOT enumerated by ESA — they're computed
// from solveSystem in exhaustiveCompleteUnitary, so we don't test them.

#include "exhaustive_search.h"
#include "cyclotomic_int9.h"
#include "Z9chi.h"
#include <atomic>
#include <fstream>
#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <sstream>

using namespace std;

std::atomic<bool> interrupted(false);

// Construct ringZ9 from 9 coefficients (canonical form has a6=a7=a8=0,
// but we accept any since the constructor reduces).
static ringZ9 build_ring(const array<int,9>& a){
    int arr[9] = {a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]};
    return ringZ9(arr);
}

static bool ring_eq(const ringZ9& a, const ringZ9& b){
    for(int k = 0; k < 6; ++k){
        if(a.getTerm(k) != b.getTerm(k)) return false;
    }
    return true;
}

static bool find_in(const ringZ9& target, const vector<ringZ9>& vec){
    for(const ringZ9& r : vec){
        if(ring_eq(target, r)) return true;
    }
    return false;
}

int main(int argc, char** argv){
    string path = argc > 1 ? argv[1] : "/tmp/hrsa_v_numerators.txt";
    ifstream fin(path);
    if(!fin){ cerr << "Cannot open " << path << "\n"; return 1; }

    // Parse: skip comment line, then 9 lines of "i j exp a0..a8"
    array<array<ringZ9,3>,3> V;
    int v_exp = -1;
    double theta = 0, eps = 0;
    string line;
    while(getline(fin, line)){
        if(line.empty()) continue;
        if(line[0] == '#'){
            // Parse: # theta=X epsilon=Y hrsa_max_f=Z v_exp=E
            size_t p = line.find("theta=");
            if(p != string::npos) theta = stod(line.substr(p+6));
            p = line.find("epsilon=");
            if(p != string::npos) eps = stod(line.substr(p+8));
            p = line.find("v_exp=");
            if(p != string::npos) v_exp = stoi(line.substr(p+6));
            continue;
        }
        istringstream ss(line);
        int i, j, e;
        array<int,9> a;
        ss >> i >> j >> e;
        for(int k = 0; k < 9; ++k) ss >> a[k];
        V[i][j] = build_ring(a);
        if(e != v_exp) cerr << "WARN: V[" << i << "][" << j << "] exp=" << e
                            << " differs from v_exp=" << v_exp << "\n";
    }

    cout << "Read V at theta=" << theta << " eps=" << eps
         << " v_exp(=ESA's f)=" << v_exp << "\n";

    // Diagnostic dump of every V entry's filter-relevant properties.
    const int    f_pow_sq_i = [&]{ int p=1; for(int k=0;k<2*v_exp;++k)p*=3; return p; }();
    const double f_pow_sq   = (double)f_pow_sq_i;
    const double eps_sq     = eps * eps;
    cout << "\nf_pow_sq = " << f_pow_sq << "  eps_sq = " << eps_sq
         << "  f_pow_sq*eps_sq = " << (f_pow_sq * eps_sq) << "\n";
    cout << "(diag bound: |x|^2 <= f_pow_sq.  off-diag bound: |x|^2 <= f_pow_sq*eps_sq)\n";
    cout << "\nV entry diagnostic dump:\n";
    cout << "  i j  | abs_val_sq      quad     sdeChi  !div3?  passes_diag  passes_offdiag\n";
    for(int i = 0; i < 3; ++i){
        for(int j = 0; j < 3; ++j){
            const ringZ9& v = V[i][j];
            double absq = v.abs_val_sq();
            int    q    = v.quad();
            int    sde  = v.sdeChi();
            // !div3: at least one coef not divisible by 3
            bool not_all_div3 = false;
            for(int k = 0; k < 6; ++k){
                if(v.getTerm(k) % 3 != 0){ not_all_div3 = true; break; }
            }
            bool passes_diag    = (absq <= f_pow_sq + 1.0) && (q <= f_pow_sq_i) && not_all_div3;
            bool passes_offdiag = (absq <= f_pow_sq * eps_sq + 1.0) && (q <= f_pow_sq_i) && not_all_div3;
            char buf[256];
            snprintf(buf, sizeof(buf),
                "  %d %d  | %12.4f  %8d  %5d   %3s     %4s         %4s\n",
                i, j, absq, q, sde,
                not_all_div3 ? "yes" : "no",
                passes_diag    ? "YES" : "NO",
                passes_offdiag ? "YES" : "NO");
            cout << buf;
        }
    }

    // Also dump the loop-bound A that fullX3Enumeration would use.
    cout << "\nfullX3Enumeration outer loop bound A:\n";
    cout << "  A_unitarity = 4*(f_pow_sq - minQ)  -- from fullDiagEnumeration result\n";
    cout << "  A_epsilon   = 4*f_pow_sq*eps_sq + FPRL ≈ "
         << (4.0*f_pow_sq*eps_sq + 1.0) << "  (Fix A' optimization)\n";
    cout << "  A = min(A_unitarity, A_epsilon)\n";
    cout << "Form-sum = 4*quad.  Outer loop accepts only when 4*quad <= A.\n";
    cout << "Off-diagonals with principal |x|^2 small but quad large (Kalra form averages\n";
    cout << "over Galois embeddings) get filtered by 4*quad <= A_epsilon = "
         << (4.0*f_pow_sq*eps_sq + 1.0) << "  even though |x|^2 itself passes.\n";
    cout.flush();
    if(argc > 2 && string(argv[2]) == "diag-only") return 0;

    // ESAWithSorting negates theta internally — match that convention.
    const double theta_int = -theta;
    const int    f         = v_exp;

    // findMinQ on cands[0]/cands[2] determines fullX3Enumeration's minQ.
    // For exhaustive presence-check we'll use minQ=0 to get the widest
    // possible lookup (nothing pruned).  ESA's actual minQ comes from the
    // candidate set, which depends on the truncation, so we test both.

    auto check_components = [&](const char* label, size_t max_cands){
        cout << "\n=== " << label << "  (fullDiagEnumeration max_candidates="
             << max_cands << ") ===\n";

        array<vector<ringZ9>,3> cands;
        fullDiagEnumeration(cands, theta_int, eps, f, max_cands);

        cout << "  cands sizes: " << cands[0].size() << " "
             << cands[1].size() << " " << cands[2].size() << "\n";

        // Diagonal entries: V[0][0]=x_1, V[1][1]=y_2, V[2][2]=z_3
        cout << "  V[0][0] (x_1) in cands[0]: "
             << (find_in(V[0][0], cands[0]) ? "FOUND" : "MISSING") << "\n";
        cout << "  V[1][1] (y_2) in cands[1]: "
             << (find_in(V[1][1], cands[1]) ? "FOUND" : "MISSING") << "\n";
        cout << "  V[2][2] (z_3) in cands[2]: "
             << (find_in(V[2][2], cands[2]) ? "FOUND" : "MISSING") << "\n";

        // For lookup we need a minQ.  Use minQ=0 → exhaustive lookup
        // (matches what ESA would do with these candidate sets if their
        // minQ ends up as 0).  ESA's actual minQ is computed in-line:
        //   minQ = min(findMinQ(cands[0], f), findMinQ(cands[2], f))
        int minQ = 0;
        if(!cands[0].empty() && !cands[2].empty()){
            minQ = min(findMinQ(cands[0], f), findMinQ(cands[2], f));
        }
        cout << "  minQ for fullX3Enumeration = " << minQ << "\n";

        vector<ringZ9> lookup;
        fullX3Enumeration(lookup, f, minQ, eps);
        cout << "  lookup size: " << lookup.size() << "\n";

        // Off-diagonal entries enumerated through lookup: V[0][1]=x_2,
        // V[0][2]=x_3, V[1][2]=y_3
        cout << "  V[0][1] (x_2) in lookup:    "
             << (find_in(V[0][1], lookup) ? "FOUND" : "MISSING") << "\n";
        cout << "  V[0][2] (x_3) in lookup:    "
             << (find_in(V[0][2], lookup) ? "FOUND" : "MISSING") << "\n";
        cout << "  V[1][2] (y_3) in lookup:    "
             << (find_in(V[1][2], lookup) ? "FOUND" : "MISSING") << "\n";
    };

    check_components("EXHAUSTIVE",  /*max_cands=*/0);
    check_components("DEFAULT-500", /*max_cands=*/500);

    return 0;
}
