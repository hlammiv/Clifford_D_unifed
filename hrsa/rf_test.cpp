// rf_test.cpp - sanity check that C++ rf_predict matches Python rf.predict for one known input.
#include "rf_model.h"
#include <cstdio>

int main(){
    // Features from rf_export.py sanity check (first sub-cluster).
    // Python rf.predict expected: 44.7516
    double feats[RF_N_FEATURES] = {
        3.0, 4.0, 914.0, 914.0, 914.0, 914.0, 914.0, 2742.0,
        65.0, 71.0, 84.0, 65.0, 17.0, -23.0, -106.0, 1.0, 6.0,
        40.5, 37.0, 397.5, 313.0
    };
    double pred = rf_predict(feats);
    printf("C++ rf_predict = %.4f\n", pred);
    printf("Python expected = 44.7516\n");
    printf("Match (within 1e-3): %s\n", (pred > 44.74 && pred < 44.77) ? "YES" : "NO");
    return 0;
}
