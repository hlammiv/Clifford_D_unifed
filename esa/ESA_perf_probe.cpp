// ESA_perf_probe.cpp — instrument the two enumeration steps so we can see
// concretely what makes ESA() blow up at f=3.
//
// Reports:
//   - cands[0..2] sizes from fullDiagEnumeration
//   - lookup size from fullX3Enumeration (which is uncapped today)
//   - time spent in each step
//   - estimated bytes in the lookup table
//
// Usage: ./ESA_perf_probe <theta> <epsilon> <f_max>
//
// Walks f from 0 to f_max, printing one line per f. f=1, f=2 are skipped to
// match ESA()'s actual behavior.

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <array>
#include <cmath>
#include <cstdlib>
#include <atomic>
#include <csignal>

#include "exhaustive_search.h"
#include "cyclotomic_int9.h"

std::atomic<bool> interrupted(false);

using namespace std;
using clk = chrono::steady_clock;

static double secs(clk::time_point a, clk::time_point b) {
    return chrono::duration<double>(b - a).count();
}

int main(int argc, char* argv[]) {
    signal(SIGINT, handleCtrlC);

    if (argc != 4) {
        cerr << "Usage: ./ESA_perf_probe <theta> <epsilon> <f_max>" << endl;
        return 1;
    }
    double theta_user = atof(argv[1]);
    double epsilon    = atof(argv[2]);
    int    f_max      = atoi(argv[3]);

    // Match the post-fix ESA() entry: theta_internal = -theta_user.
    double theta = -theta_user;

    cout << fixed << setprecision(4);
    cout << "theta_user=" << theta_user << " epsilon=" << epsilon << "\n";
    cout << "f  diag_t  |c0|  |c1|  |c2|  x3_t  |lookup|  bytes_lookup\n";

    int f = 0;
    while (f <= f_max) {
        if (interrupted) break;

        array<vector<ringZ9>, 3> cands;
        auto t0 = clk::now();
        fullDiagEnumeration(cands, theta, epsilon, f, /*max_candidates=*/500);
        auto t1 = clk::now();

        // Match ESA() inputs to fullX3Enumeration: minQ = min(findMinQ(cands[0],f), findMinQ(cands[2],f))
        // We don't have findMinQ exposed, so approximate by min q over each set.
        auto min_q = [](const vector<ringZ9>& v) {
            int best = INT32_MAX;
            for (const auto& z : v) {
                int q = z.quad();
                if (q < best) best = q;
            }
            return best == INT32_MAX ? 0 : best;
        };
        int minQ = min(min_q(cands[0]), min_q(cands[2]));

        vector<ringZ9> lookup;
        auto t2 = clk::now();
        fullX3Enumeration(lookup, f, minQ, epsilon);
        auto t3 = clk::now();

        size_t lookup_bytes = lookup.capacity() * sizeof(ringZ9);

        cout << f
             << "  " << secs(t0, t1)
             << "  " << cands[0].size()
             << "  " << cands[1].size()
             << "  " << cands[2].size()
             << "  " << secs(t2, t3)
             << "  " << lookup.size()
             << "  " << lookup_bytes
             << "\n";

        f = (f == 0) ? 3 : f + 1;
    }

    return 0;
}
