// ESA_e2e_test.cpp — end-to-end timing of ESA() with the post-fix-A'/B/D code.
//
// Runs ESA(theta, epsilon, max_f) (which steps f = 0 then 3, 4, ...), times
// it, and verifies the returned V approximates the canonical target
// Diag(e^{-iθ/2}, e^{+iθ/2}, 1) within Frobenius distance epsilon.
//
// Usage: ./ESA_e2e_test <theta> <epsilon> <max_f>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <complex>
#include <array>
#include <cstdlib>
#include <atomic>
#include <csignal>

#include "exhaustive_search.h"
#include "cyclotomic_int9.h"
#include "Z9chi.h"

std::atomic<bool> interrupted(false);

using namespace std;
using clk = chrono::steady_clock;

int main(int argc, char* argv[]) {
    signal(SIGINT, handleCtrlC);

    if (argc != 4) {
        cerr << "Usage: ./ESA_e2e_test <theta> <epsilon> <max_f>" << endl;
        return 1;
    }
    double theta   = atof(argv[1]);
    double epsilon = atof(argv[2]);
    int    max_f   = atoi(argv[3]);

    auto t0 = clk::now();
    array<ringZ9chi, 9> V = ESA(theta, epsilon, max_f);
    auto t1 = clk::now();
    double secs = chrono::duration<double>(t1 - t0).count();

    bool all_zero = true;
    for (int k = 0; k < 9; ++k) {
        if (abs(V[k].toComplexDouble()) > 1e-12) { all_zero = false; break; }
    }

    if (all_zero) {
        cout << fixed << setprecision(4)
             << "ESA_E2E theta=" << theta << " eps=" << epsilon
             << " max_f=" << max_f << " wall=" << secs
             << "s result=NO_SOLUTION" << endl;
        return 2;
    }

    // Canonical target.
    complex<double> target[3][3] = {};
    target[0][0] = complex<double>(cos(theta/2), -sin(theta/2));
    target[1][1] = complex<double>(cos(theta/2),  sin(theta/2));
    target[2][2] = complex<double>(1.0, 0.0);

    double frob_sq = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            complex<double> diff = target[i][j] - V[i + 3*j].toComplexDouble();
            frob_sq += norm(diff);
        }
    double frob = sqrt(frob_sq);

    cout << fixed << setprecision(8)
         << "ESA_E2E theta=" << theta << " eps=" << epsilon
         << " max_f=" << max_f << " wall=" << setprecision(3) << secs << "s "
         << "frob=" << setprecision(8) << frob
         << " result=" << (frob < epsilon ? "PASS" : "FAIL") << endl;

    // Print V matrix for inspection.
    cout << "V =" << endl;
    for (int i = 0; i < 3; ++i) {
        cout << "  ";
        for (int j = 0; j < 3; ++j) {
            complex<double> v = V[i + 3*j].toComplexDouble();
            cout << "(" << setw(10) << v.real() << ", " << setw(10) << v.imag() << ")  ";
        }
        cout << endl;
    }
    cout << "target =" << endl;
    for (int i = 0; i < 3; ++i) {
        cout << "  ";
        for (int j = 0; j < 3; ++j) {
            cout << "(" << setw(10) << target[i][j].real() << ", " << setw(10) << target[i][j].imag() << ")  ";
        }
        cout << endl;
    }
    return frob < epsilon ? 0 : 1;
}
