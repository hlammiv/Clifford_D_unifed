# Paths to Z3
#Z3_DIR := ./z3
#Z3_INCLUDE := $(Z3_DIR)/include
#Z3_BIN := $(Z3_DIR)/bin

# Compiler and flags
CXX := g++
CXXFLAGS := -O3 -flto -fno-fat-lto-objects -fopenmp -Wall -Werror -pedantic #-I$(Z3_INCLUDE) -I.
#LDFLAGS := -L$(Z3_BIN) -lz3

# Source and target
SRC := HRSA_test.cpp cyclotomic_int9.cpp Z9chi.cpp householder_search.cpp decompose.cpp bidir_reify.cpp rf_features.cpp rf_model.cpp
TARGET := HRSA_tester

# Standalone CLI tool that wraps decompose() so non-C++ drivers (e.g. the
# Python zeta9 wrapper) can invoke it.  Links only the algorithm + ring
# code — no HRSA / householder_search dependency.
CLI_SRC := decompose_cli.cpp decompose.cpp cyclotomic_int9.cpp Z9chi.cpp
CLI_TARGET := decompose_tool

# Build rule
all: $(TARGET) $(CLI_TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@ #$(LDFLAGS)

$(CLI_TARGET): $(CLI_SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Standalone BFS-with-hash-dedup canonical-form explorer (single .cpp).
bfs_canonical: bfs_canonical.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Standalone bidirectional-BFS approximator (called as a subprocess by
# HRSA_tester's Phase 2 / bidir_reify).  Single .cpp, no algorithm deps.
bidir_bfs: bidir_bfs.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Indexing-bridge sanity check: verifies bidir_bfs's BFS Clifford order
# matches decompose's get_clifford_cache().cliffords order entry-for-entry.
cliff_index_check: cliff_index_check.cpp decompose.cpp cyclotomic_int9.cpp Z9chi.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

# f=0 Clifford+D subgroup BFS enumerator (one-time cache build).
f0_cache_builder: f0_cache_builder.cpp decompose.cpp cyclotomic_int9.cpp Z9chi.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

# Bidir forward set profiler — for pruning analysis.
bidir_profile: bidir_profile.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Run rule
run: $(TARGET)
	./$(TARGET)

# Clean rule
clean:
	rm -f $(TARGET) $(CLI_TARGET)

bidir_profile_R: bidir_profile_R.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Sanity test: verifies C++ rf_predict matches Python sklearn output for a known input.
rf_test: rf_test.cpp rf_model.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

# Dump the 5184 sign-extended Clifford set as ε_0-net for Solovay-Kitaev.
e0_net_dump: e0_net_dump.cpp decompose.cpp cyclotomic_int9.cpp Z9chi.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@
