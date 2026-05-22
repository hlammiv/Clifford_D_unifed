// lookup_count.cpp — count-only probe of fullX3Enumeration.
//
// Uses ATOMIC counters instead of materializing a vector — memory is O(1).
// Tells us how many elements WOULD be in lookup at the given (f, minQ, epsilon).
//
// Usage: ./lookup_count <f> <minQ> <epsilon>
//
// Filter logic mirrors fullX3Enumeration's inner body, copy-pasted to avoid
// modifying the production code path.  If the production code drifts, this
// probe needs to drift with it.

#include "cyclotomic_int9.h"
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <omp.h>

constexpr double FPRL = 0.0001;

namespace {
const double JFIX_RCOS[4] = {
     0.766044443118978,
     0.17364817766693041,
    -0.5,
    -0.9396926207859083
};
const double JFIX_RSIN[4] = {
     0.6427876096865393,
     0.984807753012208,
     0.8660254037844387,
     0.3420201433256689
};
}

static int three_power(int n){ int p = 1; for(int i = 0; i < n; ++i) p *= 3; return p; }

int main(int argc, char** argv){
    if(argc != 4){ fprintf(stderr, "Usage: %s <f> <minQ> <epsilon>\n", argv[0]); return 1; }
    const int    f       = atoi(argv[1]);
    const int    M       = atoi(argv[2]);
    const double epsilon = atof(argv[3]);

    const double f_pow_sq = pow(3, 2*f);
    const double eps_sq   = epsilon * epsilon;
    const int    A        = 4 * (three_power(2*f) - M);
    const int    max_a3   = (int)ceil(sqrt(A / 3.0));

    fprintf(stderr, "f=%d minQ=%d eps=%g  A=%d  max_a3=%d  (f_pow_sq=%g eps²·f_pow²=%g)\n",
            f, M, epsilon, A, max_a3, f_pow_sq, eps_sq * f_pow_sq);

    std::atomic<long long> n_pass_absq{0};       // pass abs_sq filter
    std::atomic<long long> n_pass_q{0};          // also pass q-form
    std::atomic<long long> n_pass_galois2{0};    // also pass |σ_2|² ≤ f_pow²
    std::atomic<long long> n_pass_galois4{0};    // also pass |σ_4|² ≤ f_pow²
    std::atomic<long long> n_pass_div3{0};       // also pass !div3 (= final lookup count)

    #pragma omp parallel for schedule(dynamic, 1) collapse(2) default(none) \
        shared(JFIX_RCOS, JFIX_RSIN, n_pass_absq, n_pass_q, \
               n_pass_galois2, n_pass_galois4, n_pass_div3) \
        firstprivate(max_a3, A, f_pow_sq, eps_sq)
    for(int a3 = -max_a3; a3 <= max_a3; ++a3){
        for(int a4 = -200; a4 <= 200; ++a4){       // collapse needs uniform bound
            const int budget_a3 = A - 3*a3*a3;
            if(budget_a3 < 0) continue;
            const int max_a4 = (int)ceil(sqrt((double)budget_a3));
            if(std::abs(a4) > max_a4) continue;
            const int budget_a4 = budget_a3 - 3*a4*a4;
            if(budget_a4 < 0) continue;

            const int max_a5 = (int)ceil(sqrt((double)budget_a4));
            for(int a5 = -max_a5; a5 <= max_a5; ++a5){
                const int budget_a5 = budget_a4 - 3*a5*a5;
                if(budget_a5 < 0) continue;

                const int max_b0 = (int)ceil(sqrt((double)budget_a5));
                for(int b0 = -max_b0; b0 <= max_b0; ++b0){
                    const int budget_b0 = budget_a5 - b0*b0;
                    if((b0 + a3) % 2 != 0 || budget_b0 < 0) continue;
                    const int a0 = (b0 + a3) / 2;

                    const int max_b1 = (int)ceil(sqrt((double)budget_b0));
                    for(int b1 = -max_b1; b1 <= max_b1; ++b1){
                        const int budget_b1 = budget_b0 - b1*b1;
                        if((b1 + a4) % 2 != 0 || budget_b1 < 0) continue;
                        const int a1 = (b1 + a4) / 2;

                        const int max_b2 = (int)ceil(sqrt((double)budget_b1));
                        for(int b2 = -max_b2; b2 <= max_b2; ++b2){
                            if((b2 + a5) % 2 != 0) continue;
                            const int a2 = (b2 + a5) / 2;

                            const double re = a0 + JFIX_RCOS[0]*a1 + JFIX_RCOS[1]*a2 + JFIX_RCOS[2]*a3 + JFIX_RCOS[3]*(a4+a5);
                            const double im =      JFIX_RSIN[0]*a1 + JFIX_RSIN[1]*a2 + JFIX_RSIN[2]*a3 + JFIX_RSIN[3]*(a4-a5);
                            const double abs_sq = re*re + im*im;
                            if(abs_sq > f_pow_sq * eps_sq + FPRL) continue;
                            n_pass_absq.fetch_add(1, std::memory_order_relaxed);

                            const int q = a0*a0 + a1*a1 + a2*a2 + a3*a3 + a4*a4 + a5*a5
                                        - a0*a3 - a1*a4 - a2*a5;
                            if((double)q > f_pow_sq + FPRL) continue;
                            n_pass_q.fetch_add(1, std::memory_order_relaxed);

                            const double re2 = a0 + JFIX_RCOS[1]*a1 + JFIX_RCOS[3]*a2 + JFIX_RCOS[2]*a3 + JFIX_RCOS[0]*(a4+a5);
                            const double im2 =      JFIX_RSIN[1]*a1 + JFIX_RSIN[3]*a2 - JFIX_RSIN[2]*a3 + JFIX_RSIN[0]*(a5-a4);
                            if(re2*re2 + im2*im2 > f_pow_sq + FPRL) continue;
                            n_pass_galois2.fetch_add(1, std::memory_order_relaxed);

                            const double re4 = a0 + JFIX_RCOS[3]*a1 + JFIX_RCOS[0]*a2 + JFIX_RCOS[2]*a3 + JFIX_RCOS[1]*(a4+a5);
                            const double im4 =      JFIX_RSIN[3]*a1 - JFIX_RSIN[0]*a2 + JFIX_RSIN[2]*a3 + JFIX_RSIN[1]*(a5-a4);
                            if(re4*re4 + im4*im4 > f_pow_sq + FPRL) continue;
                            n_pass_galois4.fetch_add(1, std::memory_order_relaxed);

                            const bool all_div3 = (a0%3==0) && (a1%3==0) && (a2%3==0)
                                               && (a3%3==0) && (a4%3==0) && (a5%3==0);
                            const bool all_zero = (a0|a1|a2|a3|a4|a5) == 0;
                            if(all_div3 && !all_zero) continue;
                            n_pass_div3.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                }
            }
        }
    }

    printf("pass abs_sq:    %12lld\n",  (long long)n_pass_absq);
    printf("+ pass q:       %12lld  (cumulative)\n", (long long)n_pass_q);
    printf("+ pass |σ_2|²:  %12lld  (cumulative)\n", (long long)n_pass_galois2);
    printf("+ pass |σ_4|²:  %12lld  (cumulative)\n", (long long)n_pass_galois4);
    printf("+ pass !div3:   %12lld  (= final lookup count)\n", (long long)n_pass_div3);
    return 0;
}
