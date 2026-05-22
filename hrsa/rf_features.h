// rf_features.h — feature extraction for the sub-cluster RF predictor.
// Pairs with rf_model.h / rf_model.cpp (auto-generated).
#pragma once
#include <vector>
#include "cyclotomic_int9.h"

// Compute the 21-feature vector used by the trained RF, given:
//   x1: the integer ringZ9 numerator of the sub-cluster's x_1 candidate
//   x2_list: the matching x_2 candidates (one per (x_1, x_2) sub-cluster)
//   f: u-denominator level (e.g. 2 for HRSA(f=2), 4 for V-denom=4)
//   n_in_cluster: number of valid (x_1, x_2, x_3) triples observed (for the trained cluster_size feature)
// Output: feats[] must be at least RF_N_FEATURES (=21) doubles.
void rf_extract_features(const ringZ9& x1,
                          const std::vector<ringZ9>& x2_list,
                          int f, int n_in_cluster,
                          double* feats);

// Convenience: compute features then call rf_predict.
double rf_score_subcluster(const ringZ9& x1,
                            const std::vector<ringZ9>& x2_list,
                            int f, int n_in_cluster);
