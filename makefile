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
INCL := $(wildcard $(INCL_DIR)/*.h) \
		$(wildcard $(TEST_DIR)/*.h) \
		$(wildcard $(DEMO_DIR)/*.h)

INCL := $(filter-out $(wildcard $(INCL_DIR)/pybind11*.h), $(INCL))

# NOTE: all executables go in top-level dir, object files in source dir
# Source files, object files, and executables
SRC := $(wildcard $(SRC_DIR)/*.cpp)
SRC := $(filter-out $(wildcard $(SRC_DIR)/pybind11*.cpp), $(SRC))
OBJ := $(SRC:%.cpp=%.o)

# Test source files
TEST_SRC := $(wildcard $(TEST_DIR)/*.cpp)
TEST_SRC := $(filter-out $(TEST_DIR)/test_printing.cpp, $(TEST_SRC))
TEST_OBJ := $(TEST_SRC:%.cpp=%.o)

# Demo source files
ALL_DEMO_SRC := $(wildcard $(DEMO_DIR)/*.cpp)
ALL_DEMO_OBJ := $(ALL_DEMO_SRC:%.cpp=%.o)

DEMO_EXEC_SRC := $(wildcard $(DEMO_DIR)/demo[1-9]*.cpp)
DEMO_EXEC := $(notdir $(DEMO_EXEC_SRC:.cpp=))
DEMO_HELPER_OBJ := $(filter-out $(DEMO_EXEC_SRC:.cpp=.o), $(ALL_DEMO_OBJ))

info :
	@echo "INCL: $(INCL)"
	@echo "SRC: $(SRC)"
	@echo "OBJ: $(OBJ)"
	@echo "TEST_SRC: $(TEST_SRC)"
	@echo "TEST_OBJ: $(TEST_OBJ)"
	@echo "ALL_DEMO_SRC: $(ALL_DEMO_SRC)"
	@echo "ALL_DEMO_OBJ: $(ALL_DEMO_OBJ)"
	@echo "DEMO_EXEC_SRC: $(DEMO_EXEC_SRC)"
	@echo "DEMO_EXEC: $(DEMO_EXEC)"
	@echo "DEMO_HELPER_OBJ: $(DEMO_HELPER_OBJ)"


# -----------------------------------------------------------------------------
#        Make options 
# -----------------------------------------------------------------------------
test: OPT += -I$(TEST_DIR) -I$(BREW)/include
test: LDLIBS = -L$(BREW)/lib -lcatch2 -lCatch2Main
# test: CFLAGS += -O3
test: test_csparse

debug: CFLAGS += -DDEBUG -glldb -O0 # no optimization
debug: CFLAGS += -fno-inline -fsanitize=address -fno-omit-frame-pointer
debug: test demos

run_debug: debug
	LSAN_OPTIONS="suppressions=$(abspath suppressions.sup)" ./test_csparse

.PHONY: demos
demos: CFLAGS += -O3 -DNDEBUG  # optimize and disable asserts
demos: $(DEMO_EXEC)

.PHONY: profile
profile: CFLAGS += -O2 -g  # optimize and add debug symbols
profile: $(DEMO_EXEC)

.PHONY: run_demos
run_demos: CFLAGS += -O3 -DNDEBUG  # optimize and disable asserts
run_demos: demos
run_demos: $(DEMO_EXEC)  # ensure demos are built before running
	- ./demo1 './data/t1'
	- ./demo2 './data/t1'
	- ./demo2 './data/ash219'
	- ./demo2 './data/bcsstk01'
	- ./demo2 './data/fs_183_1'
	- ./demo2 './data/mbeacxc'
	- ./demo2 './data/west0067'
	- ./demo2 './data/lp_afiro'
	- ./demo2 './data/bcsstk16'
	- ./demo3 './data/bcsstk01'
	- ./demo3 './data/bcsstk16'


# -----------------------------------------------------------------------------
#         Compile and Link
# -----------------------------------------------------------------------------
test_csparse: % : $(TEST_OBJ) $(OBJ)
	$(CC) $(CFLAGS) $(OPT) -o $@ $^ $(LDLIBS)

$(DEMO_EXEC): % : $(DEMO_DIR)/%.o $(DEMO_HELPER_OBJ) $(OBJ)
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
