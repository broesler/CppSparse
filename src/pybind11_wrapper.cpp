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

    // Register the enum class 'AMDOrder'
    py::enum_<cs::AMDOrder>(m, "AMDOrder")
        .value("Natural", cs::AMDOrder::Natural)
        .value("APlusAT", cs::AMDOrder::APlusAT)
        .value("ATANoDenseRows", cs::AMDOrder::ATANoDenseRows)
        .value("ATA", cs::AMDOrder::ATA)
        .export_values();

    // Bind the Symbolic struct
    py::class_<cs::Symbolic>(m, "Symbolic")
        // Expose the members of the struct as attributes in Python
        .def(py::init<>())  // Default constructor
        .def_readwrite("p_inv", &cs::Symbolic::p_inv)
        .def_readwrite("q", &cs::Symbolic::q)
        .def_readwrite("parent", &cs::Symbolic::parent)
        .def_readwrite("cp", &cs::Symbolic::cp)
        .def_readwrite("leftmost", &cs::Symbolic::leftmost)
        .def_readwrite("m2", &cs::Symbolic::m2)
        .def_readwrite("lnz", &cs::Symbolic::lnz)
        .def_readwrite("unz", &cs::Symbolic::unz);

    // Bind the QRResult struct
    py::class_<cs::QRResult>(m, "QRResult")
        .def_readwrite("V", &cs::QRResult::V)
        .def_readwrite("beta", &cs::QRResult::beta)
        .def_readwrite("R", &cs::QRResult::R)
        // Add __iter__ for unpacking in Python: V, beta, R = qr(A, S)
        .def("__iter__", [](const cs::QRResult& res) {
            auto* v = new std::vector<py::object>{
                py::cast(res.V),
                py::cast(res.beta),
                py::cast(res.R)
            };
            return py::make_iterator(v->begin(), v->end(), 
                    py::return_value_policy::take_ownership);
        }, py::keep_alive<0, 1>());

    //--------------------------------------------------------------------------
    //        COOMatrix class
    //--------------------------------------------------------------------------
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
        .def_property_readonly("nnz", &cs::COOMatrix::nnz)
        .def_property_readonly("nzmax", &cs::COOMatrix::nzmax)
        .def_property_readonly("shape", [](const cs::COOMatrix& A) {
            auto s = A.shape();
            return std::make_tuple(s[0], s[1]);
        })
        //
        .def_property_readonly("row", &cs::COOMatrix::row)
        .def_property_readonly("column", &cs::COOMatrix::column)
        .def_property_readonly("data", &cs::COOMatrix::data)
        //
        .def("assign", py::overload_cast
                        <cs::csint, cs::csint, double>(&cs::COOMatrix::assign))
        .def("__setitem__", [](cs::COOMatrix& A, py::tuple t, double v) {
            cs::csint i = t[0].cast<cs::csint>();
            cs::csint j = t[1].cast<cs::csint>();
            A.assign(i, j, v);
        })
        // TODO handle assigning to vectors
        //
        .def("compress", &cs::COOMatrix::compress)
        .def("tocsc", &cs::COOMatrix::tocsc)
        .def("toarray", &cs::COOMatrix::toarray, py::arg("order")='F')
        //
        .def("transpose", &cs::COOMatrix::transpose)
        .def_property_readonly("T", &cs::COOMatrix::T)
        //
        .def("dot", &cs::COOMatrix::dot)
        .def("__mul__", &cs::COOMatrix::dot);
        //
        // TODO how to handle printing
        // .def("__str__", &cs::COOMatrix::str);

    //--------------------------------------------------------------------------
    //        CSCMatrix class
    //--------------------------------------------------------------------------
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
        .def_property_readonly("nnz", &cs::CSCMatrix::nnz)
        .def_property_readonly("nzmax", &cs::CSCMatrix::nzmax)
        .def_property_readonly("shape", [](const cs::CSCMatrix& A) {
            auto s = A.shape();
            return std::make_tuple(s[0], s[1]);
        })
        //
        .def_property_readonly("indptr", &cs::CSCMatrix::indptr)
        .def_property_readonly("indices", &cs::CSCMatrix::indices)
        .def_property_readonly("data", &cs::CSCMatrix::data)
        //
        .def("to_canonical", &cs::CSCMatrix::to_canonical)
        .def_property_readonly("has_sorted_indices", &cs::CSCMatrix::has_sorted_indices)
        .def_property_readonly("has_canonical_format", &cs::CSCMatrix::has_canonical_format)
        .def_property_readonly("is_symmetric", &cs::CSCMatrix::is_symmetric)
        //
        .def("__call__", py::overload_cast<cs::csint, cs::csint>(&cs::CSCMatrix::operator(), py::const_))
        // TODO handle slices
        .def("__getitem__", [](cs::CSCMatrix& A, py::tuple t) {
            cs::csint i = t[0].cast<cs::csint>();
            cs::csint j = t[1].cast<cs::csint>();
            return A(i, j);
        })
        //
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
        .def("__setitem__", [](cs::CSCMatrix& A, py::tuple t, double v) {
            // TODO handle slices
            cs::csint i = t[0].cast<cs::csint>();
            cs::csint j = t[1].cast<cs::csint>();
            A.assign(i, j, v);
        })
        //
        .def("tocoo", &cs::CSCMatrix::tocoo)
        .def("toarray", &cs::CSCMatrix::toarray, py::arg("order")='F')
        //
        .def("transpose", &cs::CSCMatrix::transpose, py::arg("values")=true)
        .def_property_readonly("T", &cs::CSCMatrix::T)
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
        .def("__matmul__", py::overload_cast<const double>(&cs::CSCMatrix::dot, py::const_))
        .def("__matmul__", py::overload_cast<const std::vector<double>&>(&cs::CSCMatrix::dot, py::const_))
        .def("__matmul__", py::overload_cast<const cs::CSCMatrix&>(&cs::CSCMatrix::dot, py::const_))
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

    //--------------------------------------------------------------------------
    //        Decomposition Functions
    //--------------------------------------------------------------------------
    m.def("schol", &cs::schol,
            py::arg("A"),
            py::arg("ordering")=cs::AMDOrder::Natural,
            py::arg("use_postorder")=false);
    m.def("symbolic_cholesky", &cs::symbolic_cholesky);
    m.def("chol", &cs::chol,
            py::arg("A"),
            py::arg("S"),
            py::arg("drop_tol")=0.0);
    m.def("leftchol", &cs::leftchol);
    m.def("rechol", &cs::rechol);

    m.def("sqr", &cs::sqr,
            py::arg("A"),
            py::arg("order")=cs::AMDOrder::Natural);
    m.def("qr", &cs::qr);

    //--------------------------------------------------------------------------
    //      Solve functions
    //--------------------------------------------------------------------------
    m.def("lsolve", &cs::lsolve);
    m.def("usolve", &cs::usolve);
    m.def("lsolve_opt", &cs::lsolve_opt);
    m.def("usolve_opt", &cs::usolve_opt);
}

/*==============================================================================
 *============================================================================*/
