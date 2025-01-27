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

INCL := $(wildcard *.h)
SRC := test_csparse.cpp utils.cpp coo.cpp csc.cpp decomposition.cpp
OBJ := $(SRC:%.cpp=%.o)

info :
	@echo "SRC: $(SRC)"

GAXPY_SRC := gaxpy_perf.cpp $(filter-out test_csparse.cpp, $(SRC))
GAXPY_OBJ := $(GAXPY_SRC:%.cpp=%.o)

LUSOLVE_SRC := lusolve_perf.cpp $(filter-out test_csparse.cpp, $(SRC))
LUSOLVE_OBJ := $(LUSOLVE_SRC:%.cpp=%.o)

# -----------------------------------------------------------------------------
#        Make options 
# -----------------------------------------------------------------------------
all: test_csparse gaxpy_perf gatxpy_perf

test: OPT = -I$(BREW)/include 
test: LDLIBS = -L$(BREW)/lib -lcatch2 -lCatch2Main
test: CFLAGS += -glldb #-fsanitize=address #-Og 
test: test_csparse

gaxpy_perf: CFLAGS += -O3

gatxpy_perf: CFLAGS += -DGATXPY -O3 

# lusolve_perf: CFLAGS += -O3

debug: CFLAGS += -DDEBUG -glldb -Og -fno-inline -fsanitize=address,leak
debug: all


# -----------------------------------------------------------------------------
#         Compile and Link
# -----------------------------------------------------------------------------
test_csparse: % : %.o $(OBJ)
	$(CC) $(CFLAGS) $(OPT) -o $@ $^ $(LDLIBS)

gaxpy_perf: $(GAXPY_OBJ)
	$(CC) $(CFLAGS) $(OPT) -o $@ $^

gatxpy_perf: $(GAXPY_OBJ)
	$(CC) $(CFLAGS) $(OPT) -o $@ $^

lusolve_perf: $(LUSOLVE_OBJ)
	$(CC) $(CFLAGS) $(OPT) -o $@ $^

# Objects depend on source and headers
%.o : %.cpp $(INCL)
	$(CC) $(CFLAGS) $(OPT) -c $< -o $@

# clean up
.PHONY: depend clean
clean:
	rm -f *~
	rm -f *.o
	rm -rf *.dSYM/
	rm -f test_csparse
	rm -f test_stdvector
	rm -f gaxpy_perf
	rm -f gatxpy_perf
	rm -f lusolve_perf

#==============================================================================
#==============================================================================
