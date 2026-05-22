// ringZ9_bench.cpp — micro-benchmark for ringZ9::operator*.
//
// Measures wall time and operations/sec on a fixed input set so refactor
// iterations are comparable.  Uses repeated multiplies on a pool of random
// elements; the result is summed into a sink to defeat dead-code elimination.

#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <atomic>
#include <csignal>

#include "cyclotomic_int9.h"

std::atomic<bool> interrupted(false);

using namespace std;
using clk = chrono::steady_clock;

int main(int argc, char** argv) {
    int seed     = (argc >= 2) ? atoi(argv[1]) : 42;
    int n_pool   = (argc >= 3) ? atoi(argv[2]) : 4096;
    long n_iter  = (argc >= 4) ? atol(argv[3]) : 50'000'000;
    int max_coef = (argc >= 5) ? atoi(argv[4]) : 5;

    mt19937 rng(seed);
    uniform_int_distribution<int> dist(-max_coef, max_coef);
    vector<ringZ9> pool;
    pool.reserve(n_pool);
    for (int i = 0; i < n_pool; ++i) {
        int arr[9] = {0};
        for (int k = 0; k < 6; ++k) arr[k] = dist(rng);
        pool.emplace_back(arr);
    }

    cout << "ringZ9_bench: seed=" << seed
         << " pool=" << n_pool
         << " iter=" << n_iter
         << " max_coef=" << max_coef << "\n";

    // Sum slot 0 across products to keep the optimizer honest.
    long long sink = 0;
    auto t0 = clk::now();
    size_t mask = static_cast<size_t>(n_pool) - 1;
    if ((mask & (mask + 1)) != 0) {
        cerr << "n_pool must be a power of 2\n"; return 1;
    }
    for (long n = 0; n < n_iter; ++n) {
        size_t i = static_cast<size_t>(n) & mask;
        size_t j = static_cast<size_t>(n * 1103515245ULL + 12345) & mask;
        ringZ9 p = pool[i] * pool[j];
        sink += p.getTerm(0);
    }
    auto t1 = clk::now();
    double secs = chrono::duration<double>(t1 - t0).count();

    cout << fixed << setprecision(3)
         << "wall=" << secs << "s  "
         << "rate=" << (n_iter / secs / 1e6) << " Mops/s  "
         << "sink=" << sink << "\n";
    return 0;
}
