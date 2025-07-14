#==============================================================================
#    File: makefile
# Created: 2024-10-24 17:45
#  Author: Bernie Roesler
#
#  Description: Build executables and tests for CSparse++.
#==============================================================================

# Set the build options
BUILD_TYPE ?= Release
BUILD_DIR := build/$(BUILD_TYPE)

CXX ?= clang++
JOBS = 8

CMAKE_CONFIG_ARGS := -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DCMAKE_CXX_COMPILER=$(CXX)
CMAKE_BUILD_ARGS := --build $(BUILD_DIR) --config $(BUILD_TYPE) -j${JOBS}

.PHONY: all lib tests python demos install depend clean superclean

DEMO_EXEC := demo1 demo2 demo3

# -----------------------------------------------------------------------------
#         Targets
# -----------------------------------------------------------------------------
all: lib tests python demos

# Build the C++ library
lib:
	mkdir -p $(BUILD_DIR)
	cmake -S . -B $(BUILD_DIR) $(CMAKE_CONFIG_ARGS)
	cmake $(CMAKE_BUILD_ARGS) --target csparse_lib

# Build the C++ tests
tests: lib
	cmake $(CMAKE_BUILD_ARGS) --target test_csparse

# Build the python module
python: lib
	cmake $(CMAKE_BUILD_ARGS) --target csparse
	cmake --install $(BUILD_DIR)

# Build the C++ demos
demos: lib
	cmake $(CMAKE_BUILD_ARGS) --target $(DEMO_EXEC)

# debug: CFLAGS := $(CFLAGS) -DDEBUG -glldb -O0  # no optimization
# debug: CFLAGS += -fno-inline -fno-omit-frame-pointer #-fsanitize=address

# run_debug: debug
# 	LSAN_OPTIONS="suppressions=$(abspath suppressions.sup)" ./test_csparse

# .PHONY: profile
# profile: CFLAGS := $(CFLAGS) -O2 -g  # optimize and add debug symbols
# profile: $(DEMO_EXEC)

.PHONY: run_demos
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


# clean up
clean:
	rm -f *~
	rm -f $(BUILD_DIR)/*.o
	rm -f $(BUILD_DIR)/*.a
	rm -f $(BUILD_DIR)/*.so
	rm -rf $(BUILD_DIR)/*.dSYM
	rm -f test_csparse
	rm -f $(DEMO_EXEC)

# clean up
superclean: clean
	rm -rf $(BUILD_DIR)

#==============================================================================
#==============================================================================
