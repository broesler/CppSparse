#==============================================================================
#    File: makefile
# Created: 2024-10-24 17:45
#  Author: Bernie Roesler
#
#  Description: Build executables and tests for CSparse++.
#==============================================================================

# Set the compiler options
CC = clang++
CFLAGS = -Wall -pedantic -std=c++20

BREW = /opt/homebrew

SRC_DIR := src
INCL_DIR := include
TEST_DIR := test
DEMO_DIR := demo

OPT := -I$(INCL_DIR)
INCL := $(wildcard $(INCL_DIR)/*.h) $(wildcard $(TEST_DIR)/*.h)

# Source files, object files, and executables
SRC := $(wildcard $(SRC_DIR)/*.cpp)
SRC := $(filter-out $(SRC_DIR)/pybind11_wrapper.cpp, $(SRC))
OBJ := $(SRC:%.cpp=%.o)

# Test source files
TEST_SRC := $(wildcard $(TEST_DIR)/*.cpp)
TEST_SRC := $(filter-out $(TEST_DIR)/test_printing.cpp, $(TEST_SRC))
TEST_OBJ := $(TEST_SRC:%.cpp=%.o)  # object files go in demo dir

# Demo source files
DEMO_SRC := $(wildcard $(DEMO_DIR)/*.cpp)
DEMO_EXEC := $(notdir $(DEMO_SRC:.cpp=))  # executables go in top-level dir
DEMO_OBJ := $(DEMO_SRC:%.cpp=%.o)  # object files go in demo dir

info :
	@echo "INCL: $(INCL)"
	@echo "SRC: $(SRC)"
	@echo "OBJ: $(OBJ)"
	@echo "TEST_SRC: $(TEST_SRC)"
	@echo "TEST_OBJ: $(TEST_OBJ)"
	@echo "DEMO_SRC: $(DEMO_SRC)"
	@echo "DEMO_EXEC: $(DEMO_EXEC)"
	@echo "DEMO_OBJ: $(DEMO_OBJ)"

# -----------------------------------------------------------------------------
#        Make options 
# -----------------------------------------------------------------------------
test: OPT += -I$(TEST_DIR) -I$(BREW)/include
test: LDLIBS = -L$(BREW)/lib -lcatch2 -lCatch2Main
# test: CFLAGS += -O3
test: test_csparse

debug: CFLAGS += -DDEBUG -glldb #-Og
debug: CFLAGS += -fno-inline -fsanitize=address -fno-omit-frame-pointer
debug: test demos

run_debug: debug
	LSAN_OPTIONS="suppressions=$(abspath suppressions.sup)" ./test_csparse

.PHONY: demos
demos: CFLAGS += -O3 -DNDEBUG  # optimize and disable asserts
demos: $(DEMO_EXEC)

.PHONY: run_demos
run_demos: demos
run_demos: $(DEMO_EXEC)  # ensure demos are built before running
	- ./demo1 < ./data/t1
	- ./demo2 './data/t1'  # make sure reading from filename works
	- ./demo2 < ./data/ash219
	- ./demo2 < ./data/bcsstk01
	- ./demo2 < ./data/fs_183_1
	- ./demo2 < ./data/mbeacxc
	- ./demo2 < ./data/west0067
	- ./demo2 < ./data/lp_afiro
	- ./demo2 < ./data/bcsstk16

# -----------------------------------------------------------------------------
#         Compile and Link
# -----------------------------------------------------------------------------
test_csparse: % : $(TEST_OBJ) $(OBJ)
	$(CC) $(CFLAGS) $(OPT) -o $@ $^ $(LDLIBS)

$(DEMO_EXEC): % : $(DEMO_DIR)/%.o $(OBJ)
	$(CC) $(CFLAGS) $(OPT) -o $@ $^

# Objects depend on source and headers
$(SRC_DIR)/%.o : $(SRC_DIR)/%.cpp $(INCL)
	$(CC) $(CFLAGS) $(OPT) -c $< -o $@

$(TEST_DIR)/%.o : $(TEST_DIR)/%.cpp $(INCL)
	$(CC) $(CFLAGS) $(OPT) -c $< -o $@

$(DEMO_DIR)/%.o : $(DEMO_DIR)/%.cpp $(INCL)
	$(CC) $(CFLAGS) $(OPT) -c $< -o $@

# clean up
.PHONY: depend clean
clean:
	rm -f *~
	rm -f $(SRC_DIR)/*.o
	rm -f $(TEST_DIR)/*.o
	rm -f $(DEMO_DIR)/*.o
	rm -rf *.dSYM/
	rm -f test_csparse
	rm -f $(DEMO_EXEC)

#==============================================================================
#==============================================================================
