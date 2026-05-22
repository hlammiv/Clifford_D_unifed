# Compiler and flags
CXX := g++
CXXFLAGS := -Wall -Werror -pedantic -g -fsanitize=undefined

# Source and target
SRC := ESA_test.cpp cyclotomic_int9.cpp Z9chi.cpp exhaustive_search.cpp
TARGET := ESA_tester

# Build rule
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@ #$(LDFLAGS)

# Run rule
run: $(TARGET)
	./$(TARGET)

# Clean rule
clean:
	rm -f $(TARGET)
