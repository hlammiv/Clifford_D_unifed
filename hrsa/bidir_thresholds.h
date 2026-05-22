// bidir_thresholds.h - Calibration table mapping (theta, K=K_f=K_b) to the
// best Frobenius distance ‖V_f - W_b‖_F achievable by bidir_bfs at that depth.
// Used by the integrated bidir+HRSA dispatcher to pick the smallest K such
// that bidir is guaranteed (worst-case over theta) to find a circuit at
// Frob < eps; if no calibrated K suffices, the dispatcher falls back to HRSA.
//
// STATUS (2026-05-09): PARTIAL / STUB.
// The Wave 1C calibration sweep over 8 thetas at K∈{4,5} could not be run by
// the calibration agent because the Bash sandbox denied execution of the
// bidir_bfs binary. The table below is populated only with the three
// datapoints already known prior to calibration (theta = 2.14):
//
//   K=4 (depth 8) : Frob ≈ 0.253
//   K=5 (depth 10): Frob ≈ 0.185285112566574  (verified from /tmp/bidir_K5.log)
//   K=6 (depth 12): Frob ≈ 0.096              (recorded; not routinely calibrated, ~5h cost)
//
// All other (theta, K) cells are flagged with the sentinel value
// FROB_UNCALIBRATED (= -1.0). Callers MUST treat any negative entry as
// "no data" and should NOT use FROB_WORST until the table is filled in.
//
// To complete calibration, re-run Wave 1C with sandbox permission to invoke
// /home/hlamm/Desktop/efficent_gates/unified/hrsa/bidir_bfs:
//   for theta in 0.5 0.7854 1.0053 1.5708 2.0106 2.14 2.5133 2.879; do
//     for K in 4 5; do
//       OMP_NUM_THREADS=14 ./bidir_bfs $theta $K $K > /tmp/bidir_${theta}_${K}.log 2>&1
//     done
//   done
// then grep "Approximate-match best" from each log and fill FROB_TABLE below.

#pragma once

namespace bidir_thresholds {

constexpr double FROB_UNCALIBRATED = -1.0;

constexpr int N_THETA = 8;
constexpr double THETAS[N_THETA] = {
    0.5, 0.7854, 1.0053, 1.5708, 2.0106, 2.14, 2.5133, 2.879
};

// Frob[theta_idx][K_idx] for K_idx ∈ {0=K=4, 1=K=5}.
// Entries with FROB_UNCALIBRATED have not yet been measured.
constexpr double FROB_TABLE[N_THETA][2] = {
    { FROB_UNCALIBRATED, FROB_UNCALIBRATED },  // theta=0.5
    { FROB_UNCALIBRATED, FROB_UNCALIBRATED },  // theta=0.7854 (π/4)
    { FROB_UNCALIBRATED, FROB_UNCALIBRATED },  // theta=1.0053
    { FROB_UNCALIBRATED, FROB_UNCALIBRATED },  // theta=1.5708 (π/2)
    { FROB_UNCALIBRATED, FROB_UNCALIBRATED },  // theta=2.0106
    { 0.253,             0.185285112566574 },  // theta=2.14   (worst-case k=2; from prior runs)
    { FROB_UNCALIBRATED, FROB_UNCALIBRATED },  // theta=2.5133
    { FROB_UNCALIBRATED, FROB_UNCALIBRATED },  // theta=2.879  (11π/12)
};

// Worst-case (max-over-theta) Frob at each K. Conservative threshold:
// "if eps > FROB_WORST[K_idx], bidir at depth 2K is guaranteed to win."
//
// CURRENT VALUES are placeholders derived from the only calibrated theta
// (=2.14, the empirical worst case from earlier exploratory runs). Once
// the full sweep above is filled in, recompute these as max over the row.
// Until then, callers should be aware that FROB_WORST may UNDERESTIMATE
// the true worst-case (we have no data for the other 7 thetas).
constexpr double FROB_WORST[2] = {
    0.253,             // K=4 worst (from theta=2.14 only — UPPER BOUND ASSUMED, NOT VERIFIED)
    0.185285112566574, // K=5 worst (from theta=2.14 only — UPPER BOUND ASSUMED, NOT VERIFIED)
};

// For lookups: given eps, returns the smallest K such that worst-case Frob < eps.
// Returns -1 if no K in our table satisfies the condition (use HRSA fallback).
inline int min_K_for_eps(double eps) {
    if (eps > FROB_WORST[0]) return 4;   // depth 8 sufficient
    if (eps > FROB_WORST[1]) return 5;   // depth 10 sufficient
    return -1;                           // need deeper than K=5 → fall back to HRSA
}

// K=6 (depth 12) datapoint at theta=2.14: Frob ≈ 0.096.
// Not part of the routine table due to ~5h wall cost per run; kept here as
// a reference for callers considering K=6 as an opt-in deeper search.
constexpr double FROB_K6_THETA214 = 0.096;

} // namespace bidir_thresholds
