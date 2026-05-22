// hrsa_v_inspect.cpp — for a given (theta, epsilon, max_f), run HRSA Phase 3,
// build V, and report V's actual sde_chi / sde_3 so we can compare to what
// ESA would need (max_f >= sde_3 of V) to find the same answer.

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <atomic>
#include <csignal>

#include "householder_search.h"
#include "decompose.h"
#include "cyclotomic_int9.h"
#include "Z9chi.h"

using namespace std;

std::atomic<bool> interrupted(false);
// (handleCtrlC is defined in householder_search.cpp; we just need our own
// `interrupted` symbol here.)

extern void handleCtrlC(int);

int main(int argc, char** argv){
    signal(SIGINT, handleCtrlC);
    if (argc != 4){ cerr << "Usage: theta epsilon max_f\n"; return 1; }
    double theta   = atof(argv[1]);
    double epsilon = atof(argv[2]);
    int    max_f   = atoi(argv[3]);

    // HRSA Phase 3 directly (no Direct, no diagSearch).
    array<ringZ9chi,3> u = HRSA(theta, epsilon, max_f, 1.0);
    bool found = !(u[0].isZero() && u[1].isZero() && u[2].isZero());
    if (!found){ cout << "HRSA: NO_SOLUTION\n"; return 2; }

    int u_exp = u[0].getExp();
    cout << "HRSA Householder vector u found at exp = " << u_exp
         << " (so u_i has denom 3^" << u_exp << ")\n";

    Mat3 V = buildUnitary(u);

    cout << "\nV entries' exp (sde_3 of each entry):\n";
    int max_v_exp = 0;
    for (int i = 0; i < 3; ++i){
        for (int j = 0; j < 3; ++j){
            int e = V.m[i][j].getExp();
            if (e > max_v_exp) max_v_exp = e;
        }
    }
    cout << "  max V[i][j].getExp() = " << max_v_exp
         << "  (i.e. V is in U(3, Z[ζ_9, 1/3^" << max_v_exp << "]))\n";

    cout << "\nIndividual entries' sde_chi (=6f for canonical-denom-3^f entries):\n";
    for (int i = 0; i < 3; ++i){
        for (int j = 0; j < 3; ++j){
            cout << "  V[" << i << "][" << j << "] exp=" << V.m[i][j].getExp()
                 << " sde_chi=" << sdeChiFull(V.m[i][j]) << "\n";
        }
    }

    DecompResult dr = decompose(V, /*quiet=*/true);
    cout << "\ndecompose(V): success=" << (dr.success?"true":"false")
         << " D_count=" << dr.D_count
         << " sde_chi_initial(V[0][0])=" << dr.sde_chi
         << " sde_chi_final=" << dr.sde_chi_final
         << " syllables=" << dr.steps.size() << "\n";

    cout << "\nFor ESA to find V via its strict search at fixed-f:\n"
         << "  ESA must set max_f >= " << max_v_exp << " (V lives in U(3, Z[ζ_9, 1/3^" << max_v_exp << "]))\n"
         << "  HRSA's Householder vector u was at f=" << u_exp << ".\n";

    // Dump V's 9 entry numerators (each entry has denom 3^max_v_exp)
    // so the ESA-side probe can search for them in the enumeration outputs.
    // Format per line: "i j exp a0 a1 a2 a3 a4 a5 a6 a7 a8"
    // (a0..a8 are the 9 coefficients of the ringZ9 numerator; ESA uses
    // a canonical 6-coefficient reduced form so a6..a8 should be 0 for
    // matchable entries.)
    FILE* fp = fopen("/tmp/hrsa_v_numerators.txt", "w");
    if (fp){
        fprintf(fp, "# theta=%g epsilon=%g hrsa_max_f=%d v_exp=%d\n",
                theta, epsilon, max_f, max_v_exp);
        for (int i = 0; i < 3; ++i){
            for (int j = 0; j < 3; ++j){
                ringZ9 num = V.m[i][j].getNumerator();
                int e = V.m[i][j].getExp();
                fprintf(fp, "%d %d %d", i, j, e);
                for (int k = 0; k < 9; ++k){
                    fprintf(fp, " %d", num.getTerm(k));
                }
                fprintf(fp, "\n");
            }
        }
        fclose(fp);
        cout << "\nV numerators dumped to /tmp/hrsa_v_numerators.txt\n";
    }
    return 0;
}
