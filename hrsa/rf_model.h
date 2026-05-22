// Auto-generated from rf_subcluster_predictor.joblib.  Do not edit by hand.
#pragma once
#include <array>

#define RF_N_TREES 80
#define RF_N_FEATURES 21

// Feature names (in order):
//  [ 0] f
//  [ 1] n_in_cluster
//  [ 2] x1_q
//  [ 3] x1_q_2
//  [ 4] x1_q_4
//  [ 5] x1_q_min
//  [ 6] x1_q_max
//  [ 7] x1_q_sum
//  [ 8] x1_l1
//  [ 9] x1_l1_2
//  [10] x1_l1_4
//  [11] x1_l1_min
//  [12] x1_linf
//  [13] x1_sgn_sum
//  [14] x1_sgn_sum_w
//  [15] x1_mod3_sum
//  [16] x1_nz
//  [17] x2_l1_mean
//  [18] x2_l1_min
//  [19] x2_q_mean
//  [20] x2_q_min

// Predict N_D for a sub-cluster.  feats[] must be RF_N_FEATURES doubles.
double rf_predict(const double* feats);

