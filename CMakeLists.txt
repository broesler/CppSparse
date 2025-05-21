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

find_package(pybind11 REQUIRED)
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
    pybind11_conversion
)

set(SOURCES "")
set(HEADERS include/csparse.h include/types.h)

foreach(BASENAME ${BASENAMES})
    list(APPEND SOURCES "src/${BASENAME}.cpp")
    list(APPEND HEADERS "include/${BASENAME}.h")
endforeach()

message(STATUS "PYBIND11_INCLUDE_DIRS: ${PYBIND11_INCLUDE_DIRS}")

add_library(csc_lib ${SOURCES})
target_include_directories(
    csc_lib
    PUBLIC
    include
    /Users/bernardroesler/miniconda3/envs/dev311/lib/python3.11/site-packages/pybind11/include
    ${PYBIND11_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

pybind11_add_module(csparse_module src/pybind11_wrapper.cpp)

target_link_libraries(csparse_module PRIVATE csc_lib ${PYBIND11_LIBRARIES})
target_include_directories(
    csparse_module 
    PRIVATE
    include
    ${PYBIND11_INCLUDE_DIRS}
    /Users/bernardroesler/miniconda3/envs/dev311/lib/python3.11/site-packages/pybind11/include
    ${Python3_INCLUDE_DIRS}
)

# Add flags for Debug build type
# if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#     # Default debug flags (like -g and -O0 from CMAKE_CXX_FLAGS_DEBUG)
#     # We want to enable sanitizers for Debug builds
#     if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
#         message(STATUS "Enabling AddressSanitizer for Debug build.")
#         # Add sanitizer flags
#         set(ASAN_FLAGS "-fsanitize=address -fno-omit-frame-pointer -O1 -g")
#         set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${ASAN_FLAGS}")
#         set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} ${ASAN_FLAGS}")
#         set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} ${ASAN_FLAGS}")
#     endif()
# endif()

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
