# Paths to Z3
#Z3_DIR := ./z3
#Z3_INCLUDE := $(Z3_DIR)/include
#Z3_BIN := $(Z3_DIR)/bin

# Compiler and flags
CXX := g++
CXXFLAGS := -Wall -Werror -pedantic -O2 -march=native -fopenmp #-I$(Z3_INCLUDE) -I. 
#LDFLAGS := -L$(Z3_BIN) -lz3

# Source and target
SRC := ESA_test.cpp cyclotomic_int9.cpp Z9chi.cpp exhaustive_search.cpp
TARGET := ESA_tester

# Diagnostic build with cascade counters in the inner-loop hot path.
# Identical sources, compiled with -DESA_DIAG so DIAG_INC() emits atomics.
DIAG_TARGET := ESA_diag

# Convention sanity-check driver: uses ESA() (starts at f=0) so it finishes
# in seconds rather than minutes.  See ESA_convention_test.cpp.
CONV_SRC := ESA_convention_test.cpp cyclotomic_int9.cpp Z9chi.cpp exhaustive_search.cpp
CONV_TARGET := ESA_convention_test

# Pre-flight unit test for ringZ9 (cyclotomic_int9.cpp). Run before any Fix-E
# refactor of operator* / reduce(); rerun after each refactor step.
RING_SRC := ringZ9_unit_test.cpp cyclotomic_int9.cpp
RING_TARGET := ringZ9_unit_test

# Build rule
all: $(TARGET) $(CONV_TARGET) $(RING_TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@ #$(LDFLAGS)

$(DIAG_TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -DESA_DIAG $^ -o $@

$(CONV_TARGET): $(CONV_SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(RING_TARGET): $(RING_SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Run rule
run: $(TARGET)
	./$(TARGET)

# Clean rule
clean:
	rm -f $(TARGET) $(DIAG_TARGET) $(CONV_TARGET) $(RING_TARGET)
