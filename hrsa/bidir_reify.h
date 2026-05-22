// bidir_reify.h - Wrap bidir_bfs as a subprocess and reify the winning
// (start_clifford, forward, backward) index path back over ringZ9chi using
// decompose.cpp's BFS-built Clifford+D cache (exposed via clifford_cache.h).
//
// Used by the HRSA dispatcher (HRSA_test.cpp) as a Phase 2 between the
// direct k-D-gate search and the full Householder search.
//
// Approach overview:
//   * run_bidir(theta, K_f, K_b, target_eps): forks ./bidir_bfs as a child,
//     captures its stdout, parses the "WIN: ..." and "verify: ..." lines,
//     then reifies the index path into a ringZ9chi unitary via reify_bidir().
//   * reify_bidir(start_clifford, forward, backward, theta): given the parsed
//     indices, multiplies through ring_cliffords and ring_Dgate from
//     get_clifford_cache() to get an exact ringZ9chi V matrix; computes
//     N_D, ringZ9chi denominator exponent v_f, and Frobenius distance to
//     the target R^Z(theta).
//
// Note: the BFS Clifford ordering used by bidir_bfs.cpp (build_cliffords())
// must match the BFS ordering used by decompose.cpp (get_clifford_cache()).
// Both build with generators {H, X, S, S^{-1}} in that order with identical
// dedup logic; cliff_index_check.cpp verifies this at runtime.
// reify_bidir() additionally performs a lightweight in-process spot-check
// (compares the first 3 ring_cliffords against a freshly built CMat3 BFS
// prefix) before any reification happens.

#pragma once

#include "decompose.h"  // for Mat3, GateStep
#include <string>
#include <utility>
#include <vector>

struct BidirCircuit {
    int start_clifford;             ///< index into the 648-element BFS Clifford array
    std::vector<std::pair<int,int>> forward;   ///< (e, c) pairs, e ∈ {1,2}, c ∈ [0, 648)
    std::vector<std::pair<int,int>> backward;  ///< (e, c) pairs (already in forward order)
    Mat3 V;                          ///< reified ringZ9chi unitary
    std::vector<GateStep> steps;     ///< placeholder/empty (see implementation note)
    int N_D;                         ///< total D-count = forward.size() + backward.size()
    int v_f;                         ///< ringZ9chi denominator exponent (max getExp() over V's entries)
    double frob_to_target;           ///< ‖V − R^Z(theta)‖_F (numerical, evaluated from ringZ9chi)
    bool valid;                      ///< true if subprocess succeeded and V is well-formed

    BidirCircuit() : start_clifford(-1), N_D(0), v_f(-1),
                     frob_to_target(1e30), valid(false) {}
};

// Run bidir_bfs as a subprocess for given (theta, K_f, K_b), parse the WIN line,
// reify V over ringZ9chi using the existing decompose.cpp Clifford+D cache.
// `target_eps` is informational only (the dispatcher decides accept/reject).
// Returns a BidirCircuit with valid=true on success, valid=false otherwise.
BidirCircuit run_bidir(double theta, int K_f, int K_b, double target_eps);

// Helper: given start_clifford + forward + backward indices, reify V using
// ring_cliffords + ring_Dgate from decompose.cpp's cache. Returns a completed
// BidirCircuit (V, N_D, v_f, frob_to_target).
BidirCircuit reify_bidir(int start_clifford,
                          const std::vector<std::pair<int,int>>& forward,
                          const std::vector<std::pair<int,int>>& backward,
                          double theta);
