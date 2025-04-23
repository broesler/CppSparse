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


/** Convert a vector to a NumPy array.
 *
 * @param self  the vector to convert
 *
 * @return a NumPy array with the same data as the vector
 */
template <typename T>
auto vector_to_numpy(const std::vector<T>& vec)
{
    auto result = py::array_t<T>(vec.size());
    py::buffer_info buf = result.request();
    T* ptr = static_cast<T*>(buf.ptr);
    std::memcpy(ptr, vec.data(), vec.size() * sizeof(T));
    return result;
};


/** Convert an array to a NumPy array.
 *
 * @param self  the array to convert
 *
 * @return a NumPy array with the same data as the array
 */
template <typename T, std::size_t N>
auto array_to_numpy(const std::array<T, N>& arr)
{
    auto result = py::array_t<T>(N);
    py::buffer_info buf = result.request();
    T* ptr = static_cast<T*>(buf.ptr);
    std::memcpy(ptr, arr.data(), N * sizeof(T));
    return result;
};



/** Convert a matrix to a NumPy array.
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


/** Convert a CSCMatrix to a SciPy CSC matrix.
 *
 * @param matrix  the CSCMatrix to convert
 *
 * @return a SciPy CSC matrix
 */
py::object csc_matrix_to_scipy_csc(const cs::CSCMatrix& A, py::module_& m) {
    py::module_ np = py::module_::import("numpy");
    py::module_ sparse = py::module_::import("scipy.sparse");

    // Convert indptr, indices, and data to NumPy arrays
    auto indptr_array = vector_to_numpy(A.indptr());
    auto indices_array = vector_to_numpy(A.indices());
    auto data_array = vector_to_numpy(A.data());

    // Create the SciPy CSC A
    auto [M, N] = A.shape();

    return sparse.attr("csc_array")(
        py::make_tuple(data_array, indices_array, indptr_array),
        py::arg("shape")=py::make_tuple(M, N)
    );
}


/** Convert a string to an AMDOrder enum.
 *
 * @param order  the string to convert
 *
 * @return the AMDOrder enum
 */
cs::AMDOrder string_to_amdorder(const std::string& order)
{
    if (order == "Natural") { return cs::AMDOrder::Natural; }
    if (order == "APlusAT") { return cs::AMDOrder::APlusAT; }
    if (order == "ATANoDenseRows") { return cs::AMDOrder::ATANoDenseRows; }
    if (order == "ATA") { return cs::AMDOrder::ATA; }
    throw std::runtime_error("Invalid AMDOrder specified.");
}


PYBIND11_MODULE(csparse, m) {
    m.doc() = "CSparse module for sparse matrix operations.";

    //--------------------------------------------------------------------------
    //        Enums and Structs
    //--------------------------------------------------------------------------
    // Bind the QRResult struct
    py::class_<cs::QRResult>(m, "QRResult")
        .def_property_readonly("V", [&m](const cs::QRResult& qr) {
            return csc_matrix_to_scipy_csc(qr.V, m);
        })
        .def_property_readonly("beta", [](const cs::QRResult& qr) {
            return vector_to_numpy(qr.beta);
        })
        .def_property_readonly("R", [&m](const cs::QRResult& qr) {
            return csc_matrix_to_scipy_csc(qr.R, m);
        })
        .def_property_readonly("p_inv", [](const cs::QRResult& qr) {
            return vector_to_numpy(qr.p_inv);
        })
        .def_property_readonly("q", [](const cs::QRResult& qr) {
            return vector_to_numpy(qr.q);
        });

    // Bind the LUResult struct
    py::class_<cs::LUResult>(m, "LUResult")
        .def_property_readonly("L", [&m](const cs::LUResult& lu) {
            return csc_matrix_to_scipy_csc(lu.L, m);
        })
        .def_property_readonly("U", [&m](const cs::LUResult& lu) {
            return csc_matrix_to_scipy_csc(lu.U, m);
        })
        .def_property_readonly("p_inv", [](const cs::LUResult& lu) {
            return vector_to_numpy(lu.p_inv);
        })
        .def_property_readonly("q", [](const cs::LUResult& lu) {
            return vector_to_numpy(lu.q);
        });

    // Bind the DMPermResult struct
    py::class_<cs::DMPermResult>(m, "DMPermResult")
        .def_property_readonly("p", [](const cs::DMPermResult& res) {
            return vector_to_numpy(res.p);
        })
        .def_property_readonly("q", [](const cs::DMPermResult& res) {
            return vector_to_numpy(res.q);
        })
        .def_property_readonly("r", [](const cs::DMPermResult& res) {
            return vector_to_numpy(res.r);
        })
        .def_property_readonly("s", [](const cs::DMPermResult& res) {
            return vector_to_numpy(res.s);
        })
        .def_property_readonly("Nb", [](const cs::DMPermResult& res) {
            return res.Nb;
        })
        .def_property_readonly("cc", [](const cs::DMPermResult& res) {
            return array_to_numpy(res.cc);
        })
        .def_property_readonly("rr", [](const cs::DMPermResult& res) {
            return array_to_numpy(res.rr);
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
        // Define the copy constructor explicitly so pybind11 knows how to do
        // the type conversion
        .def(py::init<const cs::CSCMatrix&>(), "Copy constructor")
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
        .def("permute", &cs::CSCMatrix::permute,
             py::arg("p_inv"), py::arg("q"), py::arg("values")=true)
        .def("symperm", &cs::CSCMatrix::symperm,
             py::arg("p_inv"), py::arg("values")=true)
        .def("permute_transpose", &cs::CSCMatrix::permute_transpose,
             py::arg("p_inv"), py::arg("q"), py::arg("values")=true)
        .def("permute_rows", &cs::CSCMatrix::permute_rows,
             py::arg("p_inv"), py::arg("values")=true)
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
    m.def("_davis_example_small", &cs::davis_example_small);
    m.def("_davis_example_chol", &cs::davis_example_chol);
    m.def("_davis_example_qr", &cs::davis_example_qr);
    m.def("_davis_example_amd", &cs::davis_example_amd);

    //--------------------------------------------------------------------------
    //        Utility Functions
    //--------------------------------------------------------------------------
    m.def("inv_permute", &cs::inv_permute);

    //--------------------------------------------------------------------------
    //        Decomposition Functions
    //--------------------------------------------------------------------------
    // ---------- Cholesky decomposition
    m.def("etree", &cs::etree, py::arg("A"), py::arg("ata")=false);
    m.def("post", &cs::post);

    m.def("chol",
        [] (
            const cs::CSCMatrix& A,
            const std::string& order="Natural",
            bool use_postorder=false
        ) {
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
            const cs::CSCMatrix& A,
            const std::string& order="Natural",
            bool use_postorder=false
        ) {
            cs::AMDOrder order_enum = string_to_amdorder(order);
            cs::SymbolicChol S = cs::schol(A, order_enum, use_postorder);
            // TODO Fill the values with 1.0 for the symbolic factorization?
            // cs::CSCMatrix L = cs::symbolic_cholesky(A, S);
            // std::fill(L.v_.begin(), L.v_.end(), 1.0);
            // return L;
            return cs::symbolic_cholesky(A, S);;
        },
        py::arg("A"),
        py::arg("order")="Natural",
        py::arg("use_postorder")=false
    );

    m.def("leftchol",
        [] (
            const cs::CSCMatrix& A,
            const std::string& order="Natural",
            bool use_postorder=false
        ) {
            cs::AMDOrder order_enum = string_to_amdorder(order);
            cs::SymbolicChol S = cs::schol(A, order_enum, use_postorder);
            cs::CSCMatrix L = cs::symbolic_cholesky(A, S);
            return cs::leftchol(A, S, L);
        },
        py::arg("A"),
        py::arg("order")="Natural",
        py::arg("use_postorder")=false
    );

    m.def("rechol",
        [] (
            const cs::CSCMatrix& A,
            const std::string& order="Natural",
            bool use_postorder=false
        ) {
            cs::AMDOrder order_enum = string_to_amdorder(order);
            cs::SymbolicChol S = cs::schol(A, order_enum, use_postorder);
            cs::CSCMatrix L = cs::symbolic_cholesky(A, S);
            return cs::rechol(A, S, L);
        },
        py::arg("A"),
        py::arg("order")="Natural",
        py::arg("use_postorder")=false
    );

    // TODO implement ichol

    // ---------- QR decomposition
    // Define the python qr function here, and call the C++ sqr function.
    m.def("qr",
        [](
            const cs::CSCMatrix& A,
            const std::string& order="Natural",
            bool use_postorder=false
        ) {
            cs::AMDOrder order_enum = string_to_amdorder(order);
            cs::SymbolicQR S = cs::sqr(A, order_enum, use_postorder);
            return cs::qr(A, S);
        },
        py::arg("A"),
        py::arg("order")="Natural",
        py::arg("use_postorder")=false
    );

    // ---------- LU decomposition
    // Define the python lu function here, and call the C++ slu function.
    m.def("lu",
        [](
            const cs::CSCMatrix& A,
            const std::string& order="Natural",
            double tol=1.0
        ) {
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
            const cs::CSCMatrix& A,
            const std::string& order="Natural"
        ) {
            cs::AMDOrder order_enum = string_to_amdorder(order);
            return cs::amd(A, order_enum);
        },
        py::arg("A"),
        py::arg("order")="APlusAT"
    );

    m.def("dmperm", &cs::dmperm, py::arg("A"), py::arg("seed")=0);


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
