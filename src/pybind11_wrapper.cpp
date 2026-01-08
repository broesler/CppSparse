/*==============================================================================
 *     File: pybind11_wrapper.cpp
 *  Created: 2025-02-05 14:05
 *   Author: Bernie Roesler
 *
 *  Description: Wrap the CSparse module with pybind11.
 *
 *============================================================================*/

#include <array>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "pybind11_conversion.h"
#include "csparse.h"
#include "demo.h"

namespace py = pybind11;


PYBIND11_MODULE(csparse, m) {
    m.doc() = "C++Sparse module for sparse matrix operations.";

    //--------------------------------------------------------------------------
    //        Structs
    //--------------------------------------------------------------------------
    py::class_<cs::Problem>(m, "Problem")
        .def(py::init<>())
        .def_property_readonly("A", [](const cs::Problem& p) { return scipy_from_csc(p.A); })
        .def_property_readonly("C", [](const cs::Problem& p) { return scipy_from_csc(p.C); })
        .def_readonly("is_sym", &cs::Problem::is_sym)
        .def_readonly("b", &cs::Problem::b)
        // .def_readwrite("x", &cs::Problem::x)
        // .def_readwrite("resid", &cs::Problem::resid)
        .def_static("from_matrix",
            &cs::Problem::from_matrix,
            py::arg("T"),
            py::arg("droptol")=0
        );

    py::class_<cs::CholResult>(m, "CholResult")
        .def_property_readonly("L", [](const cs::CholResult& res) { return scipy_from_csc(res.L); })
        .def_property_readonly("p", [](const cs::CholResult& res) { return cs::inv_permute(res.p_inv); })
        .def_readonly("parent", &cs::CholResult::parent)
        .def("__iter__", [](const cs::CholResult& res) {
            py::object result = py::make_tuple(
                scipy_from_csc(res.L),
                cs::inv_permute(res.p_inv),
                res.parent
            );
            return py::make_iterator(result);
        });

    // Bind the QRResult struct
    py::class_<cs::QRResult>(m, "QRResult")
        .def_property_readonly("V", [](const cs::QRResult& qr) { return scipy_from_csc(qr.V); })
        .def_readonly("beta", &cs::QRResult::beta)
        .def_property_readonly("R", [](const cs::QRResult& qr) { return scipy_from_csc(qr.R); })
        .def_property_readonly("p", [](const cs::QRResult& qr) { return cs::inv_permute(qr.p_inv); })
        .def_readonly("q", &cs::QRResult::q)
        // Add the __iter__ method to make it unpackable
        .def("__iter__", [](const cs::QRResult& qr) {
            py::object result = py::make_tuple(
                scipy_from_csc(qr.V),
		        qr.beta,
		        scipy_from_csc(qr.R),
		        cs::inv_permute(qr.p_inv),
		        qr.q
            );
            return py::make_iterator(result);
        });

    // Bind the LUResult struct
    py::class_<cs::LUResult>(m, "LUResult")
        .def_property_readonly("L", [](const cs::LUResult& lu) { return scipy_from_csc(lu.L); })
        .def_property_readonly("U", [](const cs::LUResult& lu) { return scipy_from_csc(lu.U); })
        .def_property_readonly("p", [](const cs::LUResult& lu) { return cs::inv_permute(lu.p_inv); })
        .def_readonly("q", &cs::LUResult::q)
        .def("__iter__", [](const cs::LUResult& lu) {
            py::object result = py::make_tuple(
                scipy_from_csc(lu.L),
                scipy_from_csc(lu.U),
                cs::inv_permute(lu.p_inv),
                lu.q
            );
            return py::make_iterator(result);
        });

    // Bind the MaxMatch struct
    py::class_<cs::MaxMatch>(m, "MaxMatch")
        .def_readonly("jmatch", &cs::MaxMatch::jmatch)
        .def_readonly("imatch", &cs::MaxMatch::imatch)
        .def("__iter__", [](const cs::MaxMatch& res) {
            py::object result = py::make_tuple(res.jmatch, res.imatch);
            return py::make_iterator(result);
        });

    // Bind the DMPermResult struct
    py::class_<cs::DMPermResult>(m, "DMPermResult")
        .def_readonly("p", &cs::DMPermResult::p)
        .def_readonly("q", &cs::DMPermResult::q)
        .def_readonly("r", &cs::DMPermResult::r)
        .def_readonly("s", &cs::DMPermResult::s)
        .def_property_readonly("cc", [](const cs::DMPermResult& res) { return array_to_numpy(res.cc); })
        .def_property_readonly("rr", [](const cs::DMPermResult& res) { return array_to_numpy(res.rr); })
        .def_readonly("Nb", &cs::DMPermResult::Nb)
        .def("__iter__", [](const cs::DMPermResult& res) {
            py::object result = py::make_tuple(
                res.p,
                res.q,
                res.r,
                res.s,
                array_to_numpy(res.cc),
                array_to_numpy(res.rr)
            );
            return py::make_iterator(result);
        });

    // Bind the SCCResult struct
    py::class_<cs::SCCResult>(m, "SCCResult")
        .def_readonly("p", &cs::SCCResult::p)
        .def_readonly("r", &cs::SCCResult::r)
        .def_readonly("Nb", &cs::SCCResult::Nb)
        .def("__iter__", [](const cs::SCCResult& res) {
            py::object result = py::make_tuple(res.p, res.r, res.Nb);
            return py::make_iterator(result);
        });

    //--------------------------------------------------------------------------
    //        COOMatrix class
    //--------------------------------------------------------------------------
    py::class_<cs::COOMatrix>(m, "COOMatrix")
        .def(py::init<>())
        .def(py::init<
            const std::vector<double>&,
            const std::vector<cs::csint>&,
            const std::vector<cs::csint>&,
            const cs::Shape>(),
            py::arg("data"),
            py::arg("row"),
            py::arg("col"),
            py::arg("shape")=cs::Shape{0, 0}
        )
        .def(py::init<const cs::Shape&, cs::csint>())
        .def_static("from_file", &cs::COOMatrix::from_file, py::arg("filename"))
        .def_static("random",
            &cs::COOMatrix::random,
            py::arg("M"),
            py::arg("N"),
            py::arg("density")=0.1,
            py::arg("seed")=0
        )
        .def(py::init<const cs::CSCMatrix&>())
        .def_property_readonly("nnz", &cs::COOMatrix::nnz)
        .def_property_readonly("nzmax", &cs::COOMatrix::nzmax)
        .def_property_readonly("shape",
            [](const cs::COOMatrix& A) {
                cs::Shape s = A.shape();
                return std::make_tuple(s[0], s[1]);
            }
        )
        //
        .def_property_readonly("data", &cs::COOMatrix::data)
        .def_property_readonly("row", &cs::COOMatrix::row)
        .def_property_readonly("col", &cs::COOMatrix::col)
        //
        .def("insert", py::overload_cast<cs::csint, cs::csint, double>(&cs::COOMatrix::insert))
        .def("__setitem__",
            [](cs::COOMatrix& A, py::tuple t, double v) {
                cs::csint i = t[0].cast<cs::csint>();
                cs::csint j = t[1].cast<cs::csint>();
                A.insert(i, j, v);
            }
        )
        //
        .def("compress", &cs::COOMatrix::compress)
        .def("tocsc", &cs::COOMatrix::tocsc)
        .def("toscipy",
            [](const cs::COOMatrix& self) {
                return scipy_from_coo(self);
            }
        )
        .def("to_dense_vector", &cs::COOMatrix::to_dense_vector, py::arg("order")='F')
        .def("toarray", &sparse_to_ndarray<cs::COOMatrix>, py::arg("order")='C')
        //
        .def("transpose", &cs::COOMatrix::transpose)
        .def_property_readonly("T", &cs::COOMatrix::T)
        //
        .def("dot", &cs::COOMatrix::dot)
        .def("__matmul__", &cs::COOMatrix::dot)
        //
        .def("__repr__", [](const cs::COOMatrix& A) {
            return A.to_string(false);  // don't print all elements
        })
        .def("__str__", &cs::COOMatrix::to_string,
            py::arg("verbose")=true,
            py::arg("threshold")=1000
        );

    // -------------------------------------------------------------------------
    //         ItemProxy Class
    // -------------------------------------------------------------------------
    py::class_<cs::CSCMatrix::ItemProxy>(m, "ItemProxy")
        .def("__float__", [](const cs::CSCMatrix::ItemProxy& self) {
                return static_cast<double>(self);
        })
        .def("__mul__", [](const cs::CSCMatrix::ItemProxy& self, double v) {
                return static_cast<double>(self) * v;
            }, py::is_operator()
        )
        .def("__rmul__", [](const cs::CSCMatrix::ItemProxy& self, double v) {
                return static_cast<double>(self) * v;
            }, py::is_operator()
        )
        .def("__iadd__",
            [](cs::CSCMatrix::ItemProxy& self, double v) { return self += v; },
            py::is_operator()
        )
        .def("__isub__",
            [](cs::CSCMatrix::ItemProxy& self, double v) { return self -= v; },
            py::is_operator()
        )
        .def("__imul__",
            [](cs::CSCMatrix::ItemProxy& self, double v) { return self *= v; },
            py::is_operator()
        )
        .def("__idiv__",
            [](cs::CSCMatrix::ItemProxy& self, double v) { return self /= v; },
            py::is_operator()
        );

    //--------------------------------------------------------------------------
    //        CSCMatrix class
    //--------------------------------------------------------------------------
    py::class_<cs::CSCMatrix>(m, "CSCMatrix")
        .def(py::init<>())
        .def(py::init<
            const std::vector<double>&,
            const std::vector<cs::csint>&,
            const std::vector<cs::csint>&,
            const cs::Shape&>()
        )
        .def(py::init<const cs::Shape&, cs::csint, bool>(),
            py::arg("shape"),
            py::arg("nzmax")=0,
            py::arg("values")=true
        )
        .def(py::init<const cs::COOMatrix&>())
        .def(py::init<const std::vector<double>&, const cs::Shape&, const char>(),
            py::arg("A"),
            py::arg("shape"),
            py::arg("order")='F'
        )
        //
        .def_property_readonly("nnz", &cs::CSCMatrix::nnz)
        .def_property_readonly("nzmax", &cs::CSCMatrix::nzmax)
        .def_property_readonly("shape",
            [](const cs::CSCMatrix& A) {
                cs::Shape s = A.shape();
                return std::make_tuple(s[0], s[1]);
            }
        )
        //
        .def_property_readonly("indptr", &cs::CSCMatrix::indptr)
        .def_property_readonly("indices", &cs::CSCMatrix::indices)
        .def_property_readonly("data", &cs::CSCMatrix::data)
        //
        .def("dropzeros", &cs::CSCMatrix::dropzeros)
        .def("droptol", &cs::CSCMatrix::droptol, py::arg("tol")=1e-15)
        .def("to_canonical", &cs::CSCMatrix::to_canonical)
        .def("toscipy",
            [](const cs::CSCMatrix& self) {
                return scipy_from_csc(self);
            }
        )
        //
        .def_property_readonly("has_sorted_indices", &cs::CSCMatrix::has_sorted_indices)
        .def_property_readonly("has_canonical_format", &cs::CSCMatrix::has_canonical_format)
        .def_property_readonly("is_symmetric", &cs::CSCMatrix::is_symmetric)
        .def_property_readonly("is_triangular", &cs::CSCMatrix::is_triangular)
        //
        .def("__call__", py::overload_cast<cs::csint, cs::csint>(&cs::CSCMatrix::operator(), py::const_))
        .def("__getitem__",
            [](cs::CSCMatrix& self, py::tuple t) -> cs::CSCMatrix::ItemProxy {
                if (t.size() != 2) {
                    throw py::index_error("Index must be a tuple of length 2.");
                }
                cs::csint i = t[0].cast<cs::csint>();
                cs::csint j = t[1].cast<cs::csint>();
                return self(i, j);
            },
            py::return_value_policy::reference_internal  // keep proxy alive
        )
        .def("__setitem__",
            [](cs::CSCMatrix& self, py::tuple t, double v) {
                if (t.size() != 2) {
                    throw py::index_error("Index must be a tuple of length 2.");
                }
                cs::csint i = t[0].cast<cs::csint>();
                cs::csint j = t[1].cast<cs::csint>();
                self(i, j) = v;
            }
        )
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
        //
        .def("tocoo", &cs::CSCMatrix::tocoo)
        .def("to_dense_vector", &cs::CSCMatrix::to_dense_vector, py::arg("order")='F')
        .def("toarray", &sparse_to_ndarray<cs::CSCMatrix>, py::arg("order")='C')
        //
        .def("transpose", &cs::CSCMatrix::transpose, py::arg("values")=true)
        .def_property_readonly("T", &cs::CSCMatrix::T)
        //
        .def("sort", &cs::CSCMatrix::sort)
        .def("tsort", &cs::CSCMatrix::tsort)
        .def("qsort", &cs::CSCMatrix::qsort)
        //
        .def("band", py::overload_cast<cs::csint, cs::csint>
                        (&cs::CSCMatrix::band, py::const_))
        //
        .def("scale", &cs::CSCMatrix::scale)
        //
        .def("__mul__", py::overload_cast<const double>(&cs::CSCMatrix::dot, py::const_), py::is_operator())
        .def("__rmul__", py::overload_cast<const double>(&cs::CSCMatrix::dot, py::const_), py::is_operator())
        .def("dot", py::overload_cast<const double>(&cs::CSCMatrix::dot, py::const_))
        .def("dot", py::overload_cast<std::span<const double>>(&cs::CSCMatrix::dot, py::const_))
        .def("dot", py::overload_cast<const cs::CSCMatrix&>(&cs::CSCMatrix::dot, py::const_))
        .def("__matmul__", py::overload_cast<const double>(&cs::CSCMatrix::dot, py::const_))
        .def("__matmul__", py::overload_cast<std::span<const double>>(&cs::CSCMatrix::dot, py::const_))
        .def("__matmul__", py::overload_cast<const cs::CSCMatrix&>(&cs::CSCMatrix::dot, py::const_))
        //
        .def("add", &cs::CSCMatrix::add)
        .def("__add__", &cs::CSCMatrix::add)
        //
        // Convert these "p_inv" arguments to "p" for python interface to
        // match the CSparse MATLAB interface.
        .def("permute", 
            [](const cs::CSCMatrix& self,
               const std::vector<cs::csint> p,
               const std::vector<cs::csint> q,
               bool values=true) 
            {
                return self.permute(cs::inv_permute(p), q, values);
            },
            py::arg("p"), py::arg("q"), py::arg("values")=true
        )
        .def("symperm", 
            [](const cs::CSCMatrix& self,
               const std::vector<cs::csint> p,
               bool values=true)
            {
                return self.symperm(cs::inv_permute(p), values);
            },
            py::arg("p"), py::arg("values")=true
        )
        .def("permute_transpose",
            [](const cs::CSCMatrix& self, 
               const std::vector<cs::csint> p,
               const std::vector<cs::csint> q,
               bool values=true)
            {
                return self.permute_transpose(cs::inv_permute(p), q, values);
            },
            py::arg("p"), py::arg("q"), py::arg("values")=true
        )
        .def("permute_rows", 
            [](const cs::CSCMatrix& self,
               const std::vector<cs::csint> p,
               bool values=true)
            {
                return self.permute_rows(cs::inv_permute(p), values);
            },
            py::arg("p"), py::arg("values")=true
        )
        .def("permute_cols", &cs::CSCMatrix::permute_cols,
             py::arg("q"), py::arg("values")=true)
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
        .def("sum_cols", &cs::CSCMatrix::sum_cols)
        //
        .def("__repr__", [](const cs::CSCMatrix& A) {
            return A.to_string(false);  // don't print all elements
        })
        .def("__str__", &cs::CSCMatrix::to_string,
            py::arg("verbose")=true,
            py::arg("threshold")=1000
        );

    // -------------------------------------------------------------------------
    //         Example Matrices
    // -------------------------------------------------------------------------
    m.def("davis_example_small", []() { return scipy_from_coo(cs::davis_example_small()); });
    m.def("davis_example_chol", []() { return scipy_from_csc(cs::davis_example_chol()); });
    m.def("davis_example_qr",
        [](double add_diag=0.0) {
            return scipy_from_csc(cs::davis_example_qr(add_diag));
        },
        py::arg("add_diag")=0.0
    );
    m.def("davis_example_amd", []() { return scipy_from_csc(cs::davis_example_amd()); });

    // -------------------------------------------------------------------------
    //         General Functions
    // -------------------------------------------------------------------------
    m.def("gaxpy",         wrap_vector_func(&cs::gaxpy));
    m.def("gatxpy",        wrap_vector_func(&cs::gatxpy));
    m.def("sym_gaxpy",     wrap_vector_func(&cs::sym_gaxpy));
    m.def("gaxpy_row",     wrap_gaxpy_mat(&cs::gaxpy_row));
    m.def("gaxpy_col",     wrap_gaxpy_mat(&cs::gaxpy_col));
    m.def("gaxpy_block",   wrap_gaxpy_mat(&cs::gaxpy_block));
    m.def("gatxpy_row",    wrap_gaxpy_mat(&cs::gatxpy_row));
    m.def("gatxpy_col",    wrap_gaxpy_mat(&cs::gatxpy_col));
    m.def("gatxpy_block",  wrap_gaxpy_mat(&cs::gatxpy_block));

    //--------------------------------------------------------------------------
    //        Utility Functions
    //--------------------------------------------------------------------------
    // Define the pvec/ipvec function pointer types
    using pvec_func_double_t = std::vector<double> (*)(const std::vector<cs::csint>&, const std::vector<double>&);
    using pvec_func_csint_t = std::vector<cs::csint> (*)(const std::vector<cs::csint>&, const std::vector<cs::csint>&);

    m.def("pvec", 
        [](const std::vector<cs::csint>& p, const py::object& b_obj) {
            return dispatch_pvec_ipvec(
                p,
                b_obj,
                static_cast<pvec_func_double_t>(&cs::pvec<double>),
                static_cast<pvec_func_csint_t>(&cs::pvec<cs::csint>)
            );
        },
        py::arg("p"), py::arg("b")
    );
    m.def("ipvec", 
        [](const std::vector<cs::csint>& p, const py::object& b_obj) {
            return dispatch_pvec_ipvec(
                p,
                b_obj,
                static_cast<pvec_func_double_t>(&cs::ipvec<double>),
                static_cast<pvec_func_csint_t>(&cs::ipvec<cs::csint>)
            );
        },
        py::arg("p"), py::arg("b")
    );
    m.def("inv_permute", &cs::inv_permute);
    m.def("scipy_from_coo", &scipy_from_coo);
    m.def("scipy_from_csc", &scipy_from_csc);
    m.def("csc_from_scipy", &csc_from_scipy);
    // m.def("coo_from_scipy", &coo_from_scipy);  // TODO
    m.def("residual_norm",
        [](const py::object& A_scipy,
           const std::vector<double>& x,
           const std::vector<double>& b
        ) {
            std::vector<double> resid;
            cs::CSCMatrix A = csc_from_scipy(A_scipy);
            return cs::residual_norm(A, x, b, resid);
        },
        py::arg("A"), py::arg("x"), py::arg("b")
    );

    //--------------------------------------------------------------------------
    //        Decomposition Functions
    //--------------------------------------------------------------------------
    // ---------- Cholesky decomposition
    m.def("etree",
        [] (const py::object& A_scipy, bool ata=false) {
            cs::CSCMatrix A = csc_from_scipy(A_scipy);
            return cs::etree(A, ata);
        },
        py::arg("A"),
        py::arg("ata")=false
    );

    m.def("post", &cs::post);

    m.def("rowcnt",
        [] (
            const py::object& A_scipy,
            const std::vector<cs::csint>& parent,
            const std::vector<cs::csint>& post
        ) {
            cs::CSCMatrix A = csc_from_scipy(A_scipy);
            return cs::rowcnt(A, parent, post);
        },
        py::arg("A"), py::arg("parent"), py::arg("post")
    );

    m.def("chol",
        [] (
            const py::object& A_scipy,
            const std::string& order="Natural",
            bool use_postorder=false
        ) {
            cs::CSCMatrix A = csc_from_scipy(A_scipy);
            cs::AMDOrder order_enum = string_to_amdorder(order);
            cs::SymbolicChol S = cs::schol(A, order_enum, use_postorder);
            return cs::chol(A, S);
        },
        py::arg("A"),
        py::arg("order")="Natural",
        py::arg("use_postorder")=false
    );

    m.def("symbolic_cholesky",
        [](
            const py::object& A_scipy,
            const std::string& order="Natural",
            bool use_postorder=false
        ) {
            cs::CSCMatrix A = csc_from_scipy(A_scipy);
            cs::AMDOrder order_enum = string_to_amdorder(order);
            cs::SymbolicChol S = cs::schol(A, order_enum, use_postorder);
            // TODO Fill the values with 1.0 for the symbolic factorization?
            // cs::CSCMatrix L = cs::symbolic_cholesky(A, S);
            // std::fill(L.v_.begin(), L.v_.end(), 1.0);
            // return L;
            return cs::symbolic_cholesky(A, S);
        },
        py::arg("A"),
        py::arg("order")="Natural",
        py::arg("use_postorder")=false
    );

    m.def("leftchol",
        [] (
            const py::object& A_scipy,
            const std::string& order="Natural",
            bool use_postorder=false
        ) {
            cs::CSCMatrix A = csc_from_scipy(A_scipy);
            cs::AMDOrder order_enum = string_to_amdorder(order);
            cs::SymbolicChol S = cs::schol(A, order_enum, use_postorder);
            cs::CholResult res = cs::symbolic_cholesky(A, S);
            res.L = cs::leftchol(A, S, res.L);
            return res;
        },
        py::arg("A"),
        py::arg("order")="Natural",
        py::arg("use_postorder")=false
    );

    m.def("rechol",
        [] (
            const py::object& A_scipy,
            const std::string& order="Natural",
            bool use_postorder=false
        ) {
            cs::CSCMatrix A = csc_from_scipy(A_scipy);
            cs::AMDOrder order_enum = string_to_amdorder(order);
            cs::SymbolicChol S = cs::schol(A, order_enum, use_postorder);
            cs::CholResult res = cs::symbolic_cholesky(A, S);
            res.L = cs::rechol(A, S, res.L);
            return res;
        },
        py::arg("A"),
        py::arg("order")="Natural",
        py::arg("use_postorder")=false
    );

    m.def("chol_update_",
        [] (const py::object& L_scipy,
            bool update,
            const py::object& C_scipy,
            const std::vector<cs::csint>& parent
        ) {
            cs::CSCMatrix L = csc_from_scipy(L_scipy);
            const cs::CSCMatrix C = csc_from_scipy(C_scipy);
            return scipy_from_csc(cs::chol_update(L, update, C, parent));
        },
        py::arg("L"), py::arg("update"), py::arg("C"), py::arg("parent")
    );

    // ---------- QR decomposition
    // Define the python qr function here, and call the C++ sqr function.
    m.def("qr",
        [](
            const py::object& A_scipy,
            const std::string& order="Natural",
            bool use_postorder=false
        ) {
            cs::CSCMatrix A = csc_from_scipy(A_scipy);
            cs::AMDOrder order_enum = string_to_amdorder(order);
            cs::SymbolicQR S = cs::sqr(A, order_enum, use_postorder);
            return cs::qr(A, S);
        },
        py::arg("A"),
        py::arg("order")="Natural",
        py::arg("use_postorder")=false
    );

    // ---------- LU decomposition
    m.def("slu",
        [](
            const py::object& A_scipy,
            const std::string& order="Natural",
            bool qr_bound=false,
            double alpha=1.0
        ) {
            cs::CSCMatrix A = csc_from_scipy(A_scipy);
            cs::AMDOrder order_enum = string_to_amdorder(order);
            cs::SymbolicLU S = cs::slu(A, order_enum, qr_bound, alpha);
            return py::make_tuple(S.lnz, S.unz, py::cast(S.q));
        },
        py::arg("A"),
        py::arg("order")="Natural",
        py::arg("qr_bound")=false,
        py::arg("alpha")=1.0
    );

    // Define the python lu function here, and call the C++ slu function.
    m.def("lu",
        [](
            const py::object& A_scipy,
            const std::string& order="Natural",
            double tol=1.0
        ) {
            cs::CSCMatrix A = csc_from_scipy(A_scipy);
            cs::AMDOrder order_enum = string_to_amdorder(order);
            cs::SymbolicLU S = cs::slu(A, order_enum);
            return cs::lu(A, S, tol);
        },
        py::arg("A"),
        py::arg("order")="Natural",
        py::arg("tol")=1.0
    );

    // ---------- Fill-reducing orderings
    m.def("amd",
        [](
            const py::object& A_scipy,
            const std::string& order="Natural"
        ) {
            cs::CSCMatrix A = csc_from_scipy(A_scipy);
            cs::AMDOrder order_enum = string_to_amdorder(order);
            return cs::amd(A, order_enum);
        },
        py::arg("A"),
        py::arg("order")="APlusAT"
    );

    m.def("maxtrans_r",
        [] (const py::object& A_scipy, cs::csint seed=0) {
            const cs::CSCMatrix A = csc_from_scipy(A_scipy);
            return cs::detail::maxtrans_r(A, seed);
        },
        py::arg("A"),
        py::arg("seed")=0
    );

    m.def("maxtrans",
        [] (const py::object& A_scipy, cs::csint seed=0) {
            const cs::CSCMatrix A = csc_from_scipy(A_scipy);
            return cs::maxtrans(A, seed);
        },
        py::arg("A"),
        py::arg("seed")=0
    );

    m.def("dmperm",
        [] (const py::object& A_scipy, cs::csint seed=0) {
            cs::CSCMatrix A = csc_from_scipy(A_scipy);
            return cs::dmperm(A, seed);
        },
        py::arg("A"),
        py::arg("seed")=0
    );

    m.def("scc",
        [] (const py::object& A_scipy) {
            cs::CSCMatrix A = csc_from_scipy(A_scipy);
            return cs::scc(A);
        },
        py::arg("A")
    );

    //--------------------------------------------------------------------------
    //      Solve functions
    //--------------------------------------------------------------------------
    m.def("reach", 
        [](const py::object& A_scipy, const py::object& b_scipy) {
            const cs::CSCMatrix A = csc_from_scipy(A_scipy);
            const cs::CSCMatrix b = csc_from_scipy(b_scipy);

            if (b.shape()[1] != 1) {
                throw std::invalid_argument(
                    "b must be a column vector (shape (N, 1))."
                );
            }

            if (A.shape()[0] != b.shape()[0]) {
                throw std::invalid_argument(
                    "Matrix A and vector b must have compatible shapes."
                );
            }

            return cs::reach(A, b, 0);
        },
        py::arg("A"), py::arg("b")
    );
    
    m.def("reach_r", 
        [](const py::object& A_scipy, const py::object& b_scipy) {
            const cs::CSCMatrix A = csc_from_scipy(A_scipy);
            const cs::CSCMatrix b = csc_from_scipy(b_scipy);

            if (b.shape()[1] != 1) {
                throw std::invalid_argument(
                    "b must be a column vector (shape (N, 1))."
                );
            }

            if (A.shape()[0] != b.shape()[0]) {
                throw std::invalid_argument(
                    "Matrix A and vector b must have compatible shapes."
                );
            }

            return cs::detail::reach_r(A, b);
        },
        py::arg("A"), py::arg("b")
    );

    m.def("lsolve", make_trisolver(&cs::lsolve), py::arg("L"), py::arg("b"));
    m.def("usolve", make_trisolver(&cs::usolve), py::arg("U"), py::arg("b"));
    m.def("ltsolve", make_trisolver(&cs::ltsolve), py::arg("L"), py::arg("b"));
    m.def("utsolve", make_trisolver(&cs::utsolve), py::arg("U"), py::arg("b"));
    m.def("lsolve_opt", make_trisolver(&cs::lsolve_opt), py::arg("L"), py::arg("b"));
    m.def("usolve_opt", make_trisolver(&cs::usolve_opt), py::arg("U"), py::arg("b"));

    m.def("chol_solve",
        wrap_solve(&cs::chol_solve),
        py::arg("A"),
        py::arg("b"),
        py::arg("order")="Natural"  // CSparse default is "ATANoDenseRows"
    );

    m.def("qr_solve",
        // wrap_solve(&cs::qr_solve), // TODO?
        [](
            const py::object& A_scipy,
            const std::vector<double>& b,
            const std::string& order="Natural"
        ) {
            const cs::CSCMatrix A = csc_from_scipy(A_scipy);
            cs::AMDOrder order_enum = string_to_amdorder(order);
            std::vector<double> x = cs::qr_solve(A, b, order_enum).x;
            return py::cast(x);
        },
        py::arg("A"),
        py::arg("b"),
        py::arg("order")="Natural"  // CSparse default is "ATA"
    );

    m.def("lu_solve",
        wrap_solve(&cs::lu_solve),
        py::arg("A"),
        py::arg("b"),
        py::arg("order")="Natural",  // CSparse default is "ATANoDenseRows"
        py::arg("tol")=1.0,
        py::arg("ir_steps")=0
    );
}

/*==============================================================================
 *============================================================================*/
