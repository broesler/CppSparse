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

# BREW := $(shell brew --prefix)  # FIXME adds space? run command only once
BREW = /opt/homebrew

SRC_DIR := src
INCL_DIR := include

OPT := -I$(INCL_DIR)

INCL := $(wildcard $(INCL_DIR)/*.h)
SRC_BASE := test_csparse utils coo csc cholesky qr solve
SRC := $(addprefix $(SRC_DIR)/, $(addsuffix .cpp, $(SRC_BASE)))
OBJ := $(SRC:%.cpp=%.o)

info :
	@echo "INCL: $(INCL)"
	@echo "SRC: $(SRC)"

# -----------------------------------------------------------------------------
#        Make options 
# -----------------------------------------------------------------------------
all: test_csparse

test: OPT += -I$(BREW)/include 
test: LDLIBS = -L$(BREW)/lib -lcatch2 -lCatch2Main
test: CFLAGS += -glldb #-fsanitize=address #-Og 
test: test_csparse

debug: CFLAGS += -DDEBUG -glldb -Og -fno-inline -fsanitize=address,leak
debug: all


# -----------------------------------------------------------------------------
#         Compile and Link
# -----------------------------------------------------------------------------
test_csparse: % : $(SRC_DIR)/%.o $(OBJ)
	$(CC) $(CFLAGS) $(OPT) -o $@ $^ $(LDLIBS)

# Objects depend on source and headers
$(SRC_DIR)/%.o : $(SRC_DIR)/%.cpp $(INCL)
	$(CC) $(CFLAGS) $(OPT) -c $< -o $@

# clean up
.PHONY: depend clean
clean:
	rm -f *~
	rm -f $(SRC_DIR)/*.o
	rm -rf *.dSYM/
	rm -f test_csparse
	rm -f test_stdvector

#==============================================================================
#==============================================================================
