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

find_package(pybind11 REQUIRED) # Find pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

set(BASENAMES utils coo csc cholesky solve)

set(SOURCES "")
set(HEADERS include/csparse.h include/types.h)

foreach(BASENAME ${BASENAMES})
    list(APPEND SOURCES "src/${BASENAME}.cpp")
    list(APPEND HEADERS "include/${BASENAME}.h")
endforeach()

add_library(csc_lib ${SOURCES} ${HEADERS})
target_include_directories(csc_lib PUBLIC include)

pybind11_add_module(csparse_module src/pybind11_wrapper.cpp)

target_link_libraries(csparse_module PRIVATE csc_lib)

# Ensure C++20 standard -- doesn't work?
# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)  # Use strict standard compliance

# Explicit compiler options for both the library and the Python module
target_compile_options(csc_lib PRIVATE --std=c++20 -Wall -pedantic)
target_compile_options(csparse_module PRIVATE --std=c++20 -Wall -pedantic)

# Set Python output directory
set_target_properties(
    csparse_module
    PROPERTIES
    OUTPUT_NAME csparse
    MODULE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python
)

