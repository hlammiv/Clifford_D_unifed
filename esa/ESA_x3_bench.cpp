// ESA_x3_bench.cpp — micro-benchmark of fullX3Enumeration in isolation.
//
// fullX3Enumeration is the second of ESA's two enumeration steps.  It walks
// a 6-dimensional integer lattice and filters candidates by abs_val_sq.
// Fix A' (2026-05) tightens the loop bound to match the inner filter; this
// benchmark exercises the function alone to confirm the impact, since the
// other step (fullDiagEnumeration) is unaffected by A'.
//
// Usage: ./ESA_x3_bench <f> <epsilon> [M]
//
// Reports: wall-time (sec), output size, and the loop's effective A bound.

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <atomic>
#include <csignal>
#include <cmath>

#include "exhaustive_search.h"
#include "cyclotomic_int9.h"

std::atomic<bool> interrupted(false);

using namespace std;
using clk = chrono::steady_clock;

int main(int argc, char* argv[]) {
    signal(SIGINT, handleCtrlC);

    if (argc < 3) {
        cerr << "Usage: ./ESA_x3_bench <f> <epsilon> [M]" << endl;
        return 1;
    }
    int    f       = atoi(argv[1]);
    double epsilon = atof(argv[2]);
    int    M       = (argc >= 4) ? atoi(argv[3]) : 0;

    // Print what bounds Fix A' picks (so we can sanity-check expected speedup).
    double f_pow_sq = pow(3.0, 2 * f);
    int A_unitarity = 4 * (static_cast<int>(f_pow_sq) - M);
    int A_epsilon   = static_cast<int>(ceil(4.0 * f_pow_sq * epsilon * epsilon)) + 1;
    int A_used      = min(A_unitarity, A_epsilon);

    cout << fixed << setprecision(3);
    cout << "f=" << f
         << " epsilon=" << epsilon
         << " M=" << M
         << "  A_unitarity=" << A_unitarity
         << "  A_epsilon=" << A_epsilon
         << "  A_used=" << A_used << "\n";

    vector<ringZ9> lookup;
    auto t0 = clk::now();
    fullX3Enumeration(lookup, f, M, epsilon);
    auto t1 = clk::now();
    double secs = chrono::duration<double>(t1 - t0).count();

    cout << "fullX3Enumeration: " << secs << "s, "
         << "|lookup|=" << lookup.size() << "\n";
    return 0;
}
