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
#include <pybind11/numpy.h>

#include "csparse.h"

namespace py = pybind11;


/** Template function to convert a matrix to a NumPy array.
 *
 * @param self  the matrix to convert
 * @param order the order of the NumPy array ('C' or 'F')
 *
 * @return a NumPy array with the same data as the matrix
 */
template <typename T>
auto matrix_to_ndarray(const T& self, const char order)
{
    // Get the matrix in dense column-major order
    std::vector<double> v = self.to_dense_vector('C');
    auto [N_rows, N_cols] = self.shape();

    // Create a NumPy array with specified dimensions
    py::array_t<double> result({N_rows, N_cols});

    // Get a pointer to the underlying data of the NumPy array.
    auto buffer_info = result.request();
    double* ptr = static_cast<double*>(buffer_info.ptr);

    // Calculate strides based on order
    std::vector<ssize_t> strides;
    if (order == 'C') { // C-style (row-major)
        strides = {
            static_cast<ssize_t>(N_cols * sizeof(double)),
            sizeof(double)
        };
    } else if (order == 'F') { // Fortran-style (column-major)
        strides = {
            sizeof(double),
            static_cast<ssize_t>(N_rows * sizeof(double))
        };
    } else {
        throw std::runtime_error("Invalid order specified. Use 'C' or 'F'.");
    }

    // Assign strides to the buffer info. This is crucial!
    buffer_info.strides = strides;

    // Copy the data from the vector to the NumPy array.  This is the most
    // straightforward way.
    std::copy(v.begin(), v.end(), ptr);

    return result;
};



PYBIND11_MODULE(csparse, m) {
    m.doc() = "CSparse module for sparse matrix operations.";

    //--------------------------------------------------------------------------
    //        Enums and Structs
    //--------------------------------------------------------------------------
    // Register the enum class 'AMDOrder'
    py::enum_<cs::AMDOrder>(m, "AMDOrder")
        .value("Natural", cs::AMDOrder::Natural)
        .value("APlusAT", cs::AMDOrder::APlusAT)
        .value("ATANoDenseRows", cs::AMDOrder::ATANoDenseRows)
        .value("ATA", cs::AMDOrder::ATA)
        .export_values();

    // Bind the Symbolic structs
    py::class_<cs::SymbolicChol>(m, "SymbolicChol")
        // Expose the members of the struct as attributes in Python
        .def(py::init<>())  // Default constructor
        .def_readwrite("p_inv", &cs::SymbolicChol::p_inv)
        .def_readwrite("parent", &cs::SymbolicChol::parent)
        .def_readwrite("cp", &cs::SymbolicChol::cp)
        .def_readwrite("lnz", &cs::SymbolicChol::lnz);

    // Bind the Symbolic structs
    py::class_<cs::SymbolicQR>(m, "SymbolicQR")
        // Expose the members of the struct as attributes in Python
        .def(py::init<>())  // Default constructor
        .def_readwrite("p_inv", &cs::SymbolicQR::p_inv)
        .def_readwrite("q", &cs::SymbolicQR::q)
        .def_readwrite("parent", &cs::SymbolicQR::parent)
        .def_readwrite("leftmost", &cs::SymbolicQR::leftmost)
        .def_readwrite("m2", &cs::SymbolicQR::m2)
        .def_readwrite("vnz", &cs::SymbolicQR::vnz)
        .def_readwrite("rnz", &cs::SymbolicQR::rnz);

    // Bind the QRResult struct
    py::class_<cs::QRResult>(m, "QRResult")
        .def_readwrite("V", &cs::QRResult::V)
        .def_readwrite("beta", &cs::QRResult::beta)
        .def_readwrite("R", &cs::QRResult::R)
        .def_readwrite("p_inv", &cs::QRResult::p_inv)
        .def_readwrite("q", &cs::QRResult::q);

    //--------------------------------------------------------------------------
    //        COOMatrix class
    //--------------------------------------------------------------------------
    py::class_<cs::COOMatrix>(m, "COOMatrix")
        .def(py::init<>())
        .def(py::init<
            const std::vector<double>&,
            const std::vector<cs::csint>&,
            const std::vector<cs::csint>&,
            const cs::Shape>()
        )
        .def(py::init<const cs::Shape&, cs::csint>())
        // TODO how to handle a file pointer -> overload with std::string
        // .def(py::init<std::istream&>())
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
        .def_property_readonly("row", &cs::COOMatrix::row)
        .def_property_readonly("column", &cs::COOMatrix::column)
        .def_property_readonly("data", &cs::COOMatrix::data)
        //
        .def("assign", py::overload_cast
                        <cs::csint, cs::csint, double>(&cs::COOMatrix::assign))
        .def("__setitem__",
            [](cs::COOMatrix& A, py::tuple t, double v) {
                cs::csint i = t[0].cast<cs::csint>();
                cs::csint j = t[1].cast<cs::csint>();
                A.assign(i, j, v);
            }
        )
        //
        .def("compress", &cs::COOMatrix::compress)
        .def("tocsc", &cs::COOMatrix::tocsc)
        .def("to_dense_vector", &cs::COOMatrix::to_dense_vector, py::arg("order")='F')
        .def("toarray", &matrix_to_ndarray<cs::COOMatrix>, py::arg("order")='C')
        //
        .def("transpose", &cs::COOMatrix::transpose)
        .def_property_readonly("T", &cs::COOMatrix::T)
        //
        .def("dot", &cs::COOMatrix::dot)
        .def("__mul__", &cs::COOMatrix::dot)
        //
        .def("__repr__", [](const cs::COOMatrix& A) {
            return A.to_string(false);  // don't print all elements
        })
        .def("__str__", &cs::COOMatrix::to_string,
            py::arg("verbose")=true,
            py::arg("threshold")=1000
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
        .def("to_canonical", &cs::CSCMatrix::to_canonical)
        .def_property_readonly("has_sorted_indices", &cs::CSCMatrix::has_sorted_indices)
        .def_property_readonly("has_canonical_format", &cs::CSCMatrix::has_canonical_format)
        .def_property_readonly("is_symmetric", &cs::CSCMatrix::is_symmetric)
        //
        .def("__call__", py::overload_cast<cs::csint, cs::csint>(&cs::CSCMatrix::operator(), py::const_))
        .def("__getitem__",
            [](cs::CSCMatrix& A, py::tuple t) {
                cs::csint i = t[0].cast<cs::csint>();
                cs::csint j = t[1].cast<cs::csint>();
                return A(i, j);
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
        .def("__setitem__",
            [](cs::CSCMatrix& A, py::tuple t, double v) {
                cs::csint i = t[0].cast<cs::csint>();
                cs::csint j = t[1].cast<cs::csint>();
                A.assign(i, j, v);
            }
        )
        //
        .def("tocoo", &cs::CSCMatrix::tocoo)
        .def("to_dense_vector", &cs::CSCMatrix::to_dense_vector, py::arg("order")='F')
        .def("toarray", &matrix_to_ndarray<cs::CSCMatrix>, py::arg("order")='C')
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
        .def("sum_cols", &cs::CSCMatrix::sum_cols)
        //
        .def("__repr__", [](const cs::CSCMatrix& A) {
            return A.to_string(false);  // don't print all elements
        })
        .def("__str__", &cs::CSCMatrix::to_string,
            py::arg("verbose")=true,
            py::arg("threshold")=1000
        );

    //--------------------------------------------------------------------------
    //        Utility Functions
    //--------------------------------------------------------------------------
    m.def("inv_permute", &cs::inv_permute);

    //--------------------------------------------------------------------------
    //        Decomposition Functions
    //--------------------------------------------------------------------------
    // TODO update these interfaces so we don't need to expose the symbolic
    // structures (or possibly even the output structures?) in python
    // We should just be able to call "Q, R = qr(A)" like in scipy.

    // Cholesky decomposition
    m.def("etree", &cs::etree, py::arg("A"), py::arg("ata")=false);
    m.def("post", &cs::post);
    m.def("schol",
        &cs::schol,
        py::arg("A"),
        py::arg("ordering")=cs::AMDOrder::Natural,
        py::arg("use_postorder")=false
    );
    m.def("symbolic_cholesky", &cs::symbolic_cholesky);
    m.def("chol",
        &cs::chol,
        py::arg("A"),
        py::arg("S"),
        py::arg("drop_tol")=0.0
    );
    m.def("leftchol", &cs::leftchol);
    m.def("rechol", &cs::rechol);

    // QR decomposition
    m.def("sqr", 
        &cs::sqr,
        py::arg("A"),
        py::arg("order")=cs::AMDOrder::Natural,
        py::arg("use_postorder")=false
    );
    // Could name this _qr and then have a python function qr() that calls sqr,
    // and converts from V, beta to Q, R, (and p, q), or just include p_inv and
    // q in the QRResult struct. The python function could then just return
    // a tuple of numpy/sparse arrays, and we don't have to expose the
    // SymbolicQR struct.
    m.def("qr", &cs::qr);
    m.def("qr_pivoting", &cs::qr_pivoting,
        py::arg("A"),
        py::arg("S"),
        py::arg("tol")=0.0
    );

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
