#==============================================================================
#    File: CMakeLists.txt
# Created: 2025-02-05 14:08
#  Author: Bernie Roesler
#
#  Description: Build makefile for CSparse++ with pybind11 bindings.
#==============================================================================
# CMakeLists.txt
cmake_minimum_required(VERSION 3.31)  # Or higher

project(CSparse LANGUAGES CXX)

# Ensure C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # Use strict standard compliance

add_compile_options(
    -Wall
    -pedantic
    -Wextra
)

# Set debugging flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG -fno-inline -fno-omit-frame-pointer")

# Set ASAN options
option(USE_ASAN "Enable AddressSanitizer" OFF)

if (USE_ASAN)
    if (UNIX AND NOT APPLE)
        set(ENV{ASAN_OPTIONS} "detect_leaks=1")
    endif()
    set(ASAN_FLAGS "-fsanitize=address")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${ASAN_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} ${ASAN_FLAGS}")
    message(STATUS "AddressSanitizer enabled.")
endif()

# Check for required packages
find_package(Catch2 3 QUIET)

if (NOT Catch2_FOUND)
    message(STATUS "Catch2 not found in default paths, checking homebrew...")
    set(CATCH2_ROOT_HINT "/opt/homebrew")
    find_package(Catch2 3 QUIET
        PATHS ${CATCH2_ROOT_HINT}
        PATH_SUFFIXES lib/cmake/Catch2
        NO_DEFAULT_PATH
    )
endif()

if (NOT Catch2_FOUND)
    message(STATUS "Catch2 not found. Fetching with FetchContent...")
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.6.0  # Or latest stable
    )
    FetchContent_MakeAvailable(Catch2)
endif()

if (TARGET Catch2::Catch2WithMain)
    message(STATUS "Catch2::Catch2WithMain target exists.")
else()
    message(FATAL_ERROR "Catch2::Catch2WithMain target not found.")
endif()


find_package(pybind11 CONFIG REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# -----------------------------------------------------------------------------
#         Core Sources and Headers
# -----------------------------------------------------------------------------
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
    list(APPEND SOURCES src/${BASENAME}.cpp)
    list(APPEND HEADERS include/${BASENAME}.h)
endforeach()

# -----------------------------------------------------------------------------
#         Main C++ Library
# -----------------------------------------------------------------------------
add_library(csparse_lib ${SOURCES} ${HEADERS})
target_include_directories(csparse_lib PUBLIC include)

# -----------------------------------------------------------------------------
#         Python Module with Pybind11
# -----------------------------------------------------------------------------
# Add the python module itself
file(GLOB PYBIND11_SRC src/pybind11*.cpp)
list(APPEND PYBIND11_SRC demo/demo.cpp)

pybind11_add_module(csparse ${PYBIND11_SRC})
target_include_directories(csparse PRIVATE include demo)
target_link_libraries(csparse PRIVATE csparse_lib)

# Build the Python module in the build directory
set_target_properties(
    csparse
    PROPERTIES
    OUTPUT_NAME csparse
    MODULE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build
)

# Install the python module in the package directory
install(
    TARGETS csparse
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/python/csparse
    COMPONENT python
)

# -----------------------------------------------------------------------------
#         C++ Tests with Catch2
# -----------------------------------------------------------------------------
# Gather all test source files
file(GLOB TEST_SOURCES test/test_*.cpp test/test_helpers.cpp)
file(GLOB TEST_HEADERS test/test_helpers.h)

add_executable(test_csparse ${TEST_SOURCES} ${TEST_HEADERS} ${SOURCES} ${HEADERS})
target_include_directories(test_csparse PRIVATE include test)
target_link_libraries(test_csparse PRIVATE csparse_lib Catch2::Catch2WithMain)
set_target_properties(test_csparse PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# Register with CTest
enable_testing()
include(CTest)
add_test(NAME CppTests COMMAND test_csparse)
# Ensure ./test_csparse is run from C++Sparse directory for relative paths
set_tests_properties(CppTests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# ------------------------------------------------------------------------------
#       C++ Demos: demo1, demo2, etc.
# ------------------------------------------------------------------------------
file(GLOB DEMO_SRC demo/demo[1-9]*.cpp)
file(GLOB DEMO_HELPERS demo/demo*.cpp)
list(REMOVE_ITEM DEMO_HELPERS ${DEMO_SRC})  # helpers = all demo/*.cpp not demo1-9

foreach(DEMO_FILE ${DEMO_SRC})
    get_filename_component(DEMO_NAME ${DEMO_FILE} NAME_WE)
    add_executable(${DEMO_NAME} ${DEMO_FILE} ${DEMO_HELPERS} ${SOURCES} ${HEADERS})
    target_include_directories(${DEMO_NAME} PRIVATE include demo)
    target_link_libraries(${DEMO_NAME} PRIVATE csparse_lib)
    set_target_properties(${DEMO_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endforeach()


#==============================================================================
#==============================================================================
