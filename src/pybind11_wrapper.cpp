/*==============================================================================
 *     File: pybind11_wrapper.cpp
 *  Created: 2025-02-05 14:05
 *   Author: Bernie Roesler
 *
 *  Description: Wrap the CSparse module with pybind11.
 *
 *============================================================================*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// TODO might need to write additional bindings for numpy arrays
// #include <pybind11/numpy.h>

#include "csparse.h"

namespace py = pybind11;


PYBIND11_MODULE(csparse, m) {
    m.doc() = "CSparse module for sparse matrix operations.";

    // COOMatrix class
    py::class_<cs::COOMatrix>(m, "COOMatrix")
        .def(py::init<>())
        .def(py::init<
                const std::vector<double>&,
                const std::vector<cs::csint>&,
                const std::vector<cs::csint>&,
                cs::csint,
                cs::csint
            >())
        .def(py::init<cs::csint, cs::csint, cs::csint>())
        // TODO how to handle a file pointer?
        // .def(py::init<std::istream&>())
        .def_static("random", &cs::COOMatrix::random,
                py::arg("M"),
                py::arg("N"),
                py::arg("density")=0.1,
                py::arg("seed")=0
            )
        .def(py::init<const cs::CSCMatrix&>())
        .def("nnz", &cs::COOMatrix::nnz)
        .def("nzmax", &cs::COOMatrix::nzmax)
        .def("shape", &cs::COOMatrix::shape)
        //
        .def("row", &cs::COOMatrix::row)
        .def("column", &cs::COOMatrix::column)
        .def("data", &cs::COOMatrix::data)
        //
        .def("assign", py::overload_cast
                        <cs::csint, cs::csint, double>(&cs::COOMatrix::assign))
        //
        .def("compress", &cs::COOMatrix::compress)
        .def("tocsc", &cs::COOMatrix::tocsc)
        .def("toarray", &cs::COOMatrix::toarray, py::arg("order")='F')
        //
        .def("transpose", &cs::COOMatrix::transpose)
        .def("T", &cs::COOMatrix::T)
        //
        .def("dot", &cs::COOMatrix::dot)
        .def("__mul__", &cs::COOMatrix::dot);
        //
        // TODO how to handle printing
        // .def("__str__", &cs::COOMatrix::str);

        // TODO Need separate C++ function using py::tuple for set/getitem
        // .def("__setitem__", &cs::COOMatrix::assign)
        // .def("__getitem__", &cs::COOMatrix::assign)

    // CSCMatrix class
    py::class_<cs::CSCMatrix>(m, "CSCMatrix")
        .def(py::init<>())
        .def(py::init<
                const std::vector<double>&,
                const std::vector<cs::csint>&,
                const std::vector<cs::csint>&,
                const cs::Shape&
            >())
        .def(py::init<cs::csint, cs::csint, cs::csint, bool >(),
                py::arg("M"),
                py::arg("N"), 
                py::arg("nzmax")=0,
                py::arg("values")=true
            )
        .def(py::init<const cs::COOMatrix&>())
        .def(py::init<const std::vector<double>&, cs::csint, cs::csint>())
        //
        .def("nnz", &cs::CSCMatrix::nnz)
        .def("nzmax", &cs::CSCMatrix::nzmax)
        .def("shape", &cs::CSCMatrix::shape)
        //
        .def("indptr", &cs::CSCMatrix::indptr)
        .def("indices", &cs::CSCMatrix::indices)
        .def("data", &cs::CSCMatrix::data)
        //
        .def("to_canonical", &cs::CSCMatrix::to_canonical)
        .def("has_sorted_indices", &cs::CSCMatrix::has_sorted_indices)
        .def("has_canonical_format", &cs::CSCMatrix::has_canonical_format)
        .def("is_symmetric", &cs::CSCMatrix::is_symmetric)
        //
        // .def("__call__", &cs::CSCMatrix::operator())
        .def("assign", py::overload_cast
                        <cs::csint, cs::csint, double>(&cs::CSCMatrix::assign))
        .def("assign", py::overload_cast<
                        const std::vector<cs::csint>&,
                        const std::vector<cs::csint>&,
                        const std::vector<double>&>(&cs::CSCMatrix::assign))
        .def("assign", py::overload_cast<
                        const std::vector<cs::csint>&,
                        const std::vector<cs::csint>&,
                        const cs::CSCMatrix&>(&cs::CSCMatrix::assign))
        //
        .def("tocoo", &cs::CSCMatrix::tocoo)
        .def("toarray", &cs::CSCMatrix::toarray, py::arg("order")='F')
        //
        .def("transpose", &cs::CSCMatrix::transpose)
        .def("T", &cs::CSCMatrix::T)
        //
        .def("band", py::overload_cast<cs::csint, cs::csint>
                        (&cs::CSCMatrix::band, py::const_))
        //
        .def("gaxpy", &cs::CSCMatrix::gaxpy)
        .def("gaxpy_row", &cs::CSCMatrix::gaxpy_row)
        .def("gaxpy_col", &cs::CSCMatrix::gaxpy_col)
        .def("gaxpy_block", &cs::CSCMatrix::gaxpy_block)
        .def("gatxpy", &cs::CSCMatrix::gatxpy)
        .def("gatxpy_row", &cs::CSCMatrix::gatxpy_row)
        .def("gatxpy_col", &cs::CSCMatrix::gatxpy_col)
        .def("gatxpy_block", &cs::CSCMatrix::gatxpy_block)
        .def("sym_gaxpy", &cs::CSCMatrix::sym_gaxpy)
        //
        .def("scale", &cs::CSCMatrix::scale)
        //
        .def("dot", py::overload_cast<const double>(&cs::CSCMatrix::dot, py::const_))
        .def("dot", py::overload_cast<const std::vector<double>&>(&cs::CSCMatrix::dot, py::const_))
        .def("dot", py::overload_cast<const cs::CSCMatrix&>(&cs::CSCMatrix::dot, py::const_))
        .def("__mul__", py::overload_cast<const double>(&cs::CSCMatrix::dot, py::const_))
        .def("__mul__", py::overload_cast<const std::vector<double>&>(&cs::CSCMatrix::dot, py::const_))
        .def("__mul__", py::overload_cast<const cs::CSCMatrix&>(&cs::CSCMatrix::dot, py::const_))
        //
        .def("add", &cs::CSCMatrix::add)
        .def("__add__", &cs::CSCMatrix::add)
        //
        .def("permute", &cs::CSCMatrix::permute)
        .def("symperm", &cs::CSCMatrix::symperm)
        .def("permute_transpose", &cs::CSCMatrix::permute_transpose)
        .def("permute_rows", &cs::CSCMatrix::permute_rows)
        .def("permute_cols", &cs::CSCMatrix::permute_cols)
        //
        .def("norm", &cs::CSCMatrix::norm)
        .def("fronorm", &cs::CSCMatrix::fronorm)
        //
        .def("slice", &cs::CSCMatrix::slice)
        .def("index", &cs::CSCMatrix::index)
        .def("add_empty_top", &cs::CSCMatrix::add_empty_top)
        .def("add_empty_bottom", &cs::CSCMatrix::add_empty_bottom)
        .def("add_empty_left", &cs::CSCMatrix::add_empty_left)
        .def("add_empty_right", &cs::CSCMatrix::add_empty_right)
        //
        .def("sum_rows", &cs::CSCMatrix::sum_rows)
        .def("sum_cols", &cs::CSCMatrix::sum_cols);

}

/*==============================================================================
 *============================================================================*/
