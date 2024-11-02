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
SRC := $(filter-out test_stdvector.cpp, $(SRC))
OBJ := $(SRC:%.cpp=%.o)

# -----------------------------------------------------------------------------
#        Make options 
# -----------------------------------------------------------------------------
all: test_csparse

test: OPT = -I$(BREW)/include 
test: LDLIBS = -L$(BREW)/lib -lcatch2 -lCatch2Main
test: test_csparse

debug: CFLAGS += -DDEBUG -glldb -fno-inline -fsanitize=address,leak
debug: all

# -----------------------------------------------------------------------------
#         Compile and Link
# -----------------------------------------------------------------------------
test_csparse: % : %.o $(OBJ)
	$(CC) $(CFLAGS) $(OPT) -o $@ $^ $(LDLIBS)

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

#==============================================================================
#==============================================================================
