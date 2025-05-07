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
DEMO_DIR := demo

OPT := -I$(INCL_DIR)
INCL := $(wildcard $(INCL_DIR)/*.h)

# Source files, object files, and executables
SRC_BASE := test_csparse utils coo csc amd cholesky qr lu solve example_matrices
SRC := $(addprefix $(SRC_DIR)/, $(addsuffix .cpp, $(SRC_BASE)))
OBJ := $(SRC:%.cpp=%.o)

# Demo source files
DEMO_SRC := $(wildcard $(DEMO_DIR)/*.cpp)
DEMO_EXEC := $(notdir $(DEMO_SRC:.cpp=))  # executables go in top-level dir
DEMO_OBJ := $(DEMO_SRC:%.cpp=%.o)  # object files go in demo dir

CORE_OBJ := $(filter-out $(SRC_DIR)/test_csparse.o, $(OBJ))

info :
	@echo "INCL: $(INCL)"
	@echo "SRC: $(SRC)"
	@echo "OBJ: $(OBJ)"
	@echo "DEMO_SRC: $(DEMO_SRC)"
	@echo "DEMO_EXEC: $(DEMO_EXEC)"
	@echo "DEMO_OBJ: $(DEMO_OBJ)"
	@echo "CORE_OBJ: $(CORE_OBJ)"

# -----------------------------------------------------------------------------
#        Make options 
# -----------------------------------------------------------------------------
test: OPT += -I$(BREW)/include 
test: LDLIBS = -L$(BREW)/lib -lcatch2 -lCatch2Main
# test: CFLAGS += -O3
test: test_csparse

debug: CFLAGS += -DDEBUG -glldb #-Og
debug: CFLAGS += -fno-inline -fsanitize=address
debug: test demos

.PHONY: demos
demos: $(DEMO_EXEC)

.PHONY: run_demos
run_demos: $(DEMO_EXEC)  # ensure demos are built before running
	- ./demo1 < ./data/t1

# -----------------------------------------------------------------------------
#         Compile and Link
# -----------------------------------------------------------------------------
test_csparse: % : $(SRC_DIR)/%.o $(OBJ)
	$(CC) $(CFLAGS) $(OPT) -o $@ $^ $(LDLIBS)

$(DEMO_EXEC): % : $(DEMO_DIR)/%.o $(CORE_OBJ)
	$(CC) $(CFLAGS) $(OPT) -o $@ $^

# Objects depend on source and headers
$(SRC_DIR)/%.o : $(SRC_DIR)/%.cpp $(INCL)
	$(CC) $(CFLAGS) $(OPT) -c $< -o $@

$(DEMO_DIR)/%.o : $(DEMO_DIR)/%.cpp $(INCL)
	$(CC) $(CFLAGS) $(OPT) -c $< -o $@

# clean up
.PHONY: depend clean
clean:
	rm -f *~
	rm -f $(SRC_DIR)/*.o
	rm -f $(DEMO_DIR)/*.o
	rm -rf *.dSYM/
	rm -f test_csparse
	rm -f $(DEMO_EXEC)

#==============================================================================
#==============================================================================
