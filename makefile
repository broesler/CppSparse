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
SRC := $(wildcard *.cpp)
SRC := $(filter-out test_stdvector.cpp gaxpy_perf.cpp, $(SRC))
OBJ := $(SRC:%.cpp=%.o)

GAXPY_SRC := $(filter-out test_stdvector.cpp test_csparse.cpp, $(wildcard *.cpp))
GAXPY_OBJ := $(GAXPY_SRC:%.cpp=%.o)

# -----------------------------------------------------------------------------
#        Make options 
# -----------------------------------------------------------------------------
all: test_csparse gaxpy_perf

test: OPT = -I$(BREW)/include 
test: LDLIBS = -L$(BREW)/lib -lcatch2 -lCatch2Main
test: CFLAGS += -glldb #-Og #-fsanitize=address
test: test_csparse

# gaxpy_perf: CFLAGS += -glldb -fno-inline -fsanitize=address
gaxpy_perf: CFLAGS += -O3

debug: CFLAGS += -DDEBUG -glldb -Og -fno-inline -fsanitize=address,leak
debug: all


# -----------------------------------------------------------------------------
#         Compile and Link
# -----------------------------------------------------------------------------
test_csparse: % : %.o $(OBJ)
	$(CC) $(CFLAGS) $(OPT) -o $@ $^ $(LDLIBS)

gaxpy_perf: $(GAXPY_OBJ)
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

#==============================================================================
#==============================================================================
