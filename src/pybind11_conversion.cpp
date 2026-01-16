/*==============================================================================
 *     File: pybind11_conversion.cpp
 *  Created: 2025-05-20 20:22
 *   Author: Bernie Roesler
 *
 *  Description: Conversion functions for pybind11 wrapper.
 *
 *============================================================================*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <vector>

#include "pybind11_conversion.h"
#include "csparse.h"

namespace py = pybind11;


py::object scipy_from_coo(const cs::COOMatrix& A) 
{
    py::module_ sparse = py::module_::import("scipy.sparse");

    auto data = py::cast(A.data());
    auto row = py::cast(A.row());
    auto col = py::cast(A.col());

    // Create the SciPy coo_array
    py::object scipy_coo_array = sparse.attr("coo_array")(
        py::make_tuple(data, py::make_tuple(row, col)),
        py::arg("shape") = A.shape()
    );

    return scipy_coo_array;
}


py::object scipy_from_csc(const cs::CSCMatrix& A) 
{
    py::module_ sparse = py::module_::import("scipy.sparse");

    auto data = py::cast(A.data());
    auto indices = py::cast(A.indices());
    auto indptr = py::cast(A.indptr());

    // Create the SciPy CSC matrix
    py::object scipy_csc_array = sparse.attr("csc_array")(
        py::make_tuple(data, indices, indptr),
        py::arg("shape") = A.shape()
    );

    return scipy_csc_array;
}


cs::CSCMatrix csc_from_scipy(const py::object& obj)
{
    py::object A;

    // Check if it's already a csc_array, or convertible to one
    if (py::hasattr(obj, "tocsc")) {
        A = obj.attr("tocsc")(); // Convert to CSC if not already
    } else {
        throw std::runtime_error("Input object is not convertible to a "
            "SciPy CSC matrix (missing .tocsc() method).");
    }

    // Verify it has the expected attributes
    if (!py::hasattr(A, "data") ||
        !py::hasattr(A, "indices") ||
        !py::hasattr(A, "indptr") ||
        !py::hasattr(A, "shape"))
    {
        throw std::runtime_error("Converted object is not a valid "
                "SciPy CSC matrix (missing data, indices, indptr, or shape).");
    }

    try {
        // Cast SciPy attributes to py::array_t and then to std::vector
        // Our std::vector type caster will handle the numpy.ndarray -> std::vector conversion automatically.
        auto data = A.attr("data").cast<std::vector<double>>();
        auto indices = A.attr("indices").cast<std::vector<cs::csint>>();
        auto indptr = A.attr("indptr").cast<std::vector<cs::csint>>();

        // Get shape
        auto shape = A.attr("shape").cast<cs::Shape>();

        // Construct the C++ CSCMatrix using the loaded data
        // The 'value' member is the target CSCMatrix object
        return cs::CSCMatrix(data, indices, indptr, shape);

    } catch (const py::cast_error& e) {
        std::cerr << "Error in SciPy CSC to C++ CSCMatrix cast: "
            << e.what() << std::endl;
        throw e;

    } catch (const py::error_already_set& e) {
        std::cerr << "Python error in SciPy CSC to C++ CSCMatrix cast: "
            << e.what() << std::endl;
        PyErr_Print();  // print Python traceback
        throw e;
    }
}


cs::COOMatrix coo_from_scipy(const py::object& obj)
{
    py::object A;

    // Check if it's already a coo_array, or convertible to one
    if (py::hasattr(obj, "tocoo")) {
        A = obj.attr("tocoo")(); // Convert to COO if not already
    } else {
        throw std::runtime_error("Input object is not convertible to a "
            "SciPy COO matrix (missing .tocoo() method).");
    }

    // Verify it has the expected attributes
    if (!py::hasattr(A, "data") ||
        !py::hasattr(A, "row") ||
        !py::hasattr(A, "col") ||
        !py::hasattr(A, "shape"))
    {
        throw std::runtime_error("Converted object is not a valid "
                "SciPy COO matrix (missing data, row, col, or shape).");
    }

    try {
        // Cast SciPy attributes to py::array_t and then to std::vector
        // Our std::vector type caster will handle the numpy.ndarray -> std::vector conversion automatically.
        auto data = A.attr("data").cast<std::vector<double>>();
        auto row = A.attr("row").cast<std::vector<cs::csint>>();
        auto col = A.attr("col").cast<std::vector<cs::csint>>();

        // Get shape
        auto shape = A.attr("shape").cast<cs::Shape>();

        // Construct the C++ COOMatrix using the loaded data
        // The 'value' member is the target COOMatrix object
        return cs::COOMatrix(data, row, col, shape);

    } catch (const py::cast_error& e) {
        std::cerr << "Error in SciPy COO to C++ COOMatrix cast: "
            << e.what() << std::endl;
        throw e;

    } catch (const py::error_already_set& e) {
        std::cerr << "Python error in SciPy COO to C++ COOMatrix cast: "
            << e.what() << std::endl;
        PyErr_Print();  // print Python traceback
        throw e;
    }
}


/*==============================================================================
 *============================================================================*/
