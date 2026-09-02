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

BREW_LLVM := $(shell brew --prefix llvm 2>/dev/null)
ifneq ($(BREW_LLVM),)
	CXX := $(BREW_LLVM)/bin/clang++
else
	CXX ?= clang++
endif

$(info >>> Using CXX = $(CXX))

JOBS = 8

CMAKE_CONFIG_ARGS := -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	-DCMAKE_CXX_COMPILER=$(CXX) \
	-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
	-DCMAKE_CXX_SCAN_FOR_MODULES=OFF \
	-DCMAKE_PREFIX_PATH=$(shell python -m pybind11 --cmakedir)

CMAKE_BUILD_ARGS := --build $(BUILD_DIR) --config $(BUILD_TYPE) -j${JOBS}

ifdef USE_ASAN
	CMAKE_CONFIG_ARGS += -DUSE_ASAN=$(USE_ASAN)
endif

.PHONY: all lib tests python demos install depend clean

DEMO_EXEC := demo1 demo2 demo3

# -----------------------------------------------------------------------------
#         Targets
# -----------------------------------------------------------------------------
all: lib tests python demos

$(BUILD_DIR)/build.ninja: CmakeLists.txt
	cmake -S . -B $(BUILD_DIR) -G Ninja $(CMAKE_CONFIG_ARGS)

# Build the C++ library
lib: $(BUILD_DIR)/build.ninja
	cmake $(CMAKE_BUILD_ARGS) --target csparse_lib

# Build the C++ tests
tests: lib
	cmake $(CMAKE_BUILD_ARGS) --target test_csparse

# Run the tests with LSAN options
# run_debug_tests: tests
# 	LSAN_OPTIONS="suppressions=$(abspath suppressions.sup)" ./test_csparse

# Build the C++ demos
demos: lib
	cmake $(CMAKE_BUILD_ARGS) --target $(DEMO_EXEC)

.PHONY: run_demos
run_demos: demos  # ensure demos are built before running
	- ./$(BUILD_DIR)/demo1 './data/t1'
	- ./$(BUILD_DIR)/demo2 './data/t1'
	- ./$(BUILD_DIR)/demo2 './data/ash219'
	- ./$(BUILD_DIR)/demo2 './data/bcsstk01'
	- ./$(BUILD_DIR)/demo2 './data/fs_183_1'
	- ./$(BUILD_DIR)/demo2 './data/mbeacxc'
	- ./$(BUILD_DIR)/demo2 './data/west0067'
	- ./$(BUILD_DIR)/demo2 './data/lp_afiro'
	- ./$(BUILD_DIR)/demo2 './data/bcsstk16'
	- ./$(BUILD_DIR)/demo3 './data/bcsstk01'
	- ./$(BUILD_DIR)/demo3 './data/bcsstk16'

# Build the python module
python:
	uv sync --all-packages --extra dev
	uv pip install --no-build-isolation -e ./python

# clean up
clean:
	rm -rf build/
	find . -type d -name '__pycache__' -exec rm -rf {} \+
	find . -type d -name '*.egg-info' -exec rm -rf {} \+
	find . -type f -name "*.so" -delete

#==============================================================================
#==============================================================================
