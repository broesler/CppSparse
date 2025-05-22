#==============================================================================
#    File: CMakeLists.txt
# Created: 2025-02-05 14:08
#  Author: Bernie Roesler
#
#  Description: Build makefile for CSparse++ with pybind11 bindings.
#==============================================================================
# CMakeLists.txt
cmake_minimum_required(VERSION 3.31)  # Or higher

project(CSparsePythonBindings)

# Ensure C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # Use strict standard compliance

find_package(pybind11 CONFIG REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

set(BASENAMES
	utils
	sparse_matrix
	coo
	csc
	fillreducing
	cholesky
	lu
	qr
	solve
	example_matrices
)

set(SOURCES "")
set(HEADERS include/csparse.h include/types.h)

foreach(BASENAME ${BASENAMES})
    list(APPEND SOURCES "src/${BASENAME}.cpp")
    list(APPEND HEADERS "include/${BASENAME}.h")
endforeach()

add_library(csc_lib ${SOURCES} ${HEADERS})
target_include_directories(csc_lib PUBLIC include)

# Add the python module itself
file(GLOB PYBIND11_SRC src/pybind11*.cpp)
list(APPEND PYBIND11_SRC demo/demo.cpp)

pybind11_add_module(csparse_module ${PYBIND11_SRC})
target_include_directories(csparse_module PRIVATE include demo)
target_link_libraries(csparse_module PRIVATE csc_lib)

# Explicit compiler options for both the library and the Python module
target_compile_options(csc_lib PRIVATE -Wall -pedantic)
target_compile_options(csparse_module PRIVATE -Wall -pedantic)

# Build the Python module in the build directory
set_target_properties(
    csparse_module
    PROPERTIES
    OUTPUT_NAME csparse
    MODULE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build
)

# Install the python module in the package directory
install(
    TARGETS csparse_module
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/python/csparse
    COMPONENT python
)

# TODO
# * Add install target for the C++ library
# * Incorporate my own makefile for building and running the C++ tests
# * Run the Python tests with pytest

#==============================================================================
#==============================================================================
