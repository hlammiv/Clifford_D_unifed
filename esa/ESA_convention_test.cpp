// ESA_convention_test.cpp
//
// FAST convention sanity check for ESA.  ESA's full search (ESA(),
// ESAWithSorting()) is too slow for verification — even at f=3 it can
// burn minutes and gigabytes.  Instead, we directly exercise the
// candidate-filter step (`fullDiagEnumeration`) which is the place where
// theta convention enters the algorithm.
//
// What we verify:
//   After the post-2026-05 convention fix, ESA's PUBLIC entry points
//   apply `theta = -theta;` so that input theta means target =
//   Diag(e^{-i theta/2}, e^{+i theta/2}, 1) (canonical/paper convention).
//   We mirror that reconciliation here: pass theta_internal = -theta_user
//   to fullDiagEnumeration and check that:
//     cands[0] contains an element close to e^{-i theta_user / 2}  (canonical V[0][0])
//     cands[1] contains an element close to e^{+i theta_user / 2}  (canonical V[1][1])
//     cands[2] contains an element close to 1                       (canonical V[2][2])
//
// If ANY of those targets has no nearby candidate, the convention fix is
// wrong.  This test runs in milliseconds because fullDiagEnumeration at
// small f is cheap; the whole search-completion machinery is bypassed.
//
// Output (one line per check, plus a summary):
//
//   ESA_CONV theta=<float> entry=<00|11|22> closest=<float>+i<float>
//                          target=<float>+i<float> dist=<float> result=<PASS|FAIL>
//   ESA_CONV theta=<float> overall=<PASS|FAIL>
//
// Exit codes:
//   0 = all three entries pass
//   1 = at least one entry fails (convention bug)
//   3 = bad usage

#include <iostream>
#include <iomanip>
#include <cmath>
#include <complex>
#include <array>
#include <vector>
#include <cstdlib>
#include <atomic>
#include <csignal>
#include <limits>

#include "exhaustive_search.h"
#include "cyclotomic_int9.h"
#include "Z9chi.h"

std::atomic<bool> interrupted(false);

using namespace std;

// Find the candidate in `cands` whose toComplexDouble()/3^f is closest to
// `target`. Returns the closest distance and stores the closest complex
// value in *closest_out (or NaN+NaN i if cands is empty).
static double closest_to(const vector<ringZ9>& cands, complex<double> target,
                         double f_pow, complex<double>* closest_out) {
    double best = numeric_limits<double>::infinity();
    *closest_out = complex<double>(numeric_limits<double>::quiet_NaN(),
                                   numeric_limits<double>::quiet_NaN());
    for (const ringZ9& c : cands) {
        complex<double> z = c.toComplexDouble() / f_pow;
        double d = abs(z - target);
        if (d < best) { best = d; *closest_out = z; }
    }
    return best;
}

int main(int argc, char* argv[]) {
    signal(SIGINT, handleCtrlC);

    if (argc != 4) {
        cerr << "Usage: ./ESA_convention_test <theta_user> <epsilon> <f>\n"
             << "  Verifies that fullDiagEnumeration's candidate filter, after\n"
             << "  the unified theta = -theta reconciliation, produces candidates\n"
             << "  near canonical V[i][i] for i = 0, 1, 2.\n";
        return 3;
    }

    double theta_user = atof(argv[1]);
    double epsilon    = atof(argv[2]);
    int    f          = atoi(argv[3]);

    // Mirror the reconciliation done at the top of ESA() / ESAWithSorting().
    double theta_internal = -theta_user;

    array<vector<ringZ9>, 3> cands;
    fullDiagEnumeration(cands, theta_internal, epsilon, f, /*max_candidates=*/500);

    double f_pow = pow(3.0, f);

    // Canonical targets.
    complex<double> tgt[3] = {
        complex<double>(cos(theta_user / 2), -sin(theta_user / 2)),  // e^{-i theta/2}
        complex<double>(cos(theta_user / 2),  sin(theta_user / 2)),  // e^{+i theta/2}
        complex<double>(1.0, 0.0)
    };
    const char* labels[3] = {"00", "11", "22"};

    cout << fixed << setprecision(8);

    bool all_pass = true;
    for (int i = 0; i < 3; ++i) {
        complex<double> closest;
        double dist = closest_to(cands[i], tgt[i], f_pow, &closest);
        bool pass = (dist <= epsilon + 1e-9);
        if (!pass) all_pass = false;

        cout << "ESA_CONV theta=" << theta_user
             << " entry=" << labels[i]
             << " closest=" << closest.real() << "+i" << closest.imag()
             << " target="  << tgt[i].real()  << "+i" << tgt[i].imag()
             << " dist="    << dist
             << " cands_size=" << cands[i].size()
             << " result=" << (pass ? "PASS" : "FAIL")
             << endl;
    }

    cout << "ESA_CONV theta=" << theta_user
         << " overall=" << (all_pass ? "PASS" : "FAIL") << endl;

    return all_pass ? 0 : 1;
}
