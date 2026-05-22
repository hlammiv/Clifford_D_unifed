#pragma once
#include <complex>
#include <vector>
#include <string>

// Use the same CMat3 layout as bidir_bfs.cpp (a struct with cd m[3][3]).
// To avoid a circular dependency, this header forward-declares CMat3 and the
// implementation uses memcpy / fixed sizes.
struct CMat3 { std::complex<double> m[3][3]; };

namespace bfs_cache {

// Save a vector<CMat3> to <path>. Returns true on success.
bool save_mats(const std::string& path, const std::vector<CMat3>& mats);

// Load a vector<CMat3> from <path>. Returns true on success.
// Resizes `mats` to the loaded count.
bool load_mats(const std::string& path, std::vector<CMat3>& mats);

// Returns true if a cache file exists at <path>.
bool exists(const std::string& path);

} // namespace bfs_cache
