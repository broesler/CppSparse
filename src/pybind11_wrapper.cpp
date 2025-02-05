/*==============================================================================
 *     File: pybind11_wrapper.cpp
 *  Created: 2025-02-05 14:05
 *   Author: Bernie Roesler
 *
 *  Description: Wrap the CSparse module with pybind11.
 *
 *============================================================================*/

#include <pybind11/pybind11.h>

#include "csparse.h"

namespace py = pybind11;

PYBIND11_MODULE(CSparse, m) {
    m.doc() = "CSparse module for sparse matrix operations.";

    // Matrix class
    py::class_<CSCMatrix>(m, "CSCMatrix")
        .def(py::init<>())
        .def(py::init<const std::vector<double>&, const std::vector<csint>&, const std::vector<csint>&>())
        .def("shape", &CSCMatrix::shape)
        .def("scale", &CSCMatrix::scale)
        .def("compress", &CSCMatrix::compress)
        .def("gaxpy", &CSCMatrix::gaxpy)
        .def("sym_gaxpy", &CSCMatrix::sym_gaxpy)
        .def("transpose", &CSCMatrix::transpose)
        .def("T", &CSCMatrix::T)
        .def("dot", &CSCMatrix::dot)
        .def("to_coo", &CSCMatrix::tocoo)
        .def("to_csc", &CSCMatrix::tocsc)
        .def("to_dense", &CSCMatrix::todense)
        .def("to_numpy", &CSCMatrix::tonumpy)
        .def("__setitem__", &CSCMatrix::assign)
        .def("__call__", &CSCMatrix::assign)
        .def("__mul__", &CSCMatrix::dot)
        .def("__add__", &CSCMatrix::add);
        // .def("__sub__", &CSCMatrix::sub)
        // .def("__repr__", &CSCMatrix::repr)
        // .def("__str__", &CSCMatrix::str);

    // COOMatrix class
    py::class_<COOMatrix>(m, "COOMatrix")
        .def(py::init<>())
        .def(py::init<const std::vector<double>&, const std::vector<csint>&, const std::vector<csint>&>())
        .def("shape", &COOMatrix::shape)
        .def("compress", &COOMatrix::compress)
        .def("tocsc", &COOMatrix::tocsc);
}

/*==============================================================================
 *============================================================================*/
