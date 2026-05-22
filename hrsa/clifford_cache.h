// clifford_cache.h - public interface to the BFS-built qutrit Clifford cache.
//
// The Clifford set (~648 elements), its ringZ9chi reification, and the D-gate
// matrices are independent of theta/epsilon, so they are built once via a
// thread-safe static-local initializer (in decompose.cpp) and exposed here so
// that other translation units (e.g. bidir_reify.cpp) can right-multiply onto
// the same canonical (BFS-ordered) Clifford table.
//
// IMPORTANT: BFS order matches that of bidir_bfs.cpp's build_cliffords()
// because both use identical generators {H, X, S, S^{-1}} in the same order
// and the same dedup-via-rounded-key logic. See cliff_index_check.cpp for the
// runtime check.

#pragma once

#include <complex>
#include <vector>
#include "decompose.h"  // Mat3

// Fast 3x3 complex matrix type used by the BFS / direct search.
// Definition kept in sync with the duplicate inside decompose.cpp.
struct CMat3 {
    std::complex<double> m[3][3];
};

struct CliffordCache {
    std::vector<CMat3> cliffords;       // BFS-ordered, ~648 entries
    std::vector<Mat3>  ring_cliffords;  // ringZ9chi version, same order
    Mat3  ring_Dgate[2];                // {D, D^2} as ringZ9chi
    CMat3 Dgate[2];                     // {D, D^2} as complex<double>
    int n_cliff;
};

const CliffordCache& get_clifford_cache();
