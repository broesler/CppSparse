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
#include <vector>

#include "csparse.h"

namespace py = pybind11;

// --- Custom Type Caster for std::vector<T> to py::array_t<T> ---
// This caster will only apply to return values (output)

namespace pybind11 { namespace detail {
    template <typename T>
    struct type_caster<std::vector<T>> : public type_caster_base<std::vector<T>> {
        using base = type_caster_base<std::vector<T>>;
        using base::value;

        static constexpr auto name = _("numpy.ndarray");

        // C++ (std::vector<T>) to Python (py::array_t<T>)
        // This is the "output" conversion (when a C++ function returns std::vector)
        static handle cast(
            const std::vector<T>& src,
            return_value_policy policy,
            handle parent
        ) {
            py::array_t<T> arr(src.size(), src.data());
            return arr.release();
        }

        // Python (py::array_t<T> or list) to C++ (std::vector<T>)
        bool load(handle src, bool convert) {
            return base::load(src, convert);
        }
    };
}} // namespace pybind11::detail


/** Convert a vector to a NumPy array.
 *
 * @param self  the vector to convert
 *
 * @return a NumPy array with the same data as the vector
 */
template <typename T>
inline py::array_t<T> vector_to_numpy(const std::vector<T>& vec)
{
    return py::array_t<T>(vec.size(), vec.data());
};


/** Convert an array to a NumPy array.
 *
 * @param self  the array to convert
 *
 * @return a NumPy array with the same data as the array
 */
template <typename T, std::size_t N>
inline py::array_t<T> array_to_numpy(const std::array<T, N>& arr)
{
    return py::array_t<T>(arr.size(), arr.data());
};


/** Convert a dense matrix to a NumPy array.
 *
 * @param self  the dense matrix to convert
 * @param order the order of the NumPy array ('C' or 'F')
 *
 * @return a NumPy array with the same data as the matrix
 */
template <typename T>
auto matrix_to_ndarray(const T& self, const char order)
{
    // Get the matrix in dense column-major order
    std::vector<double> v = self.to_dense_vector('C');
    auto [M, N] = self.shape();

    // Create a NumPy array with specified dimensions
    py::array_t<double> result({M, N});

    // Get a pointer to the underlying data of the NumPy array.
    auto buffer_info = result.request();
    double* ptr = static_cast<double*>(buffer_info.ptr);

    // Calculate strides based on order
    std::vector<ssize_t> strides;
    if (order == 'C') { // C-style (row-major)
        strides = {
            static_cast<ssize_t>(N * sizeof(double)),
            sizeof(double)
        };
    } else if (order == 'F') { // Fortran-style (column-major)
        strides = {
            sizeof(double),
            static_cast<ssize_t>(M * sizeof(double))
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
py::object csc_matrix_to_scipy_csc(const cs::CSCMatrix& A) {
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


/** Convert a Scipy sparse matrix to a CSCMatrix.
 *
 * @param matrix  the SciPy sparse matrix to convert
 *
 * @return a CSCMatrix
 */
cs::CSCMatrix scipy_sparse_to_csparse(const py::object& obj)
{
    // Check that we have a scipy sparse matrix
    if (!py::hasattr(obj, "tocsc")) {
        throw py::type_error("Input is not convertible to a SciPy CSC matrix.");
    }

    const py::object A = obj.attr("tocsc")();

    if (!py::hasattr(A, "data") ||
        !py::hasattr(A, "indices") ||
        !py::hasattr(A, "indptr")) {
        throw py::type_error("Input is not a SciPy CSC matrix.");
    }

    // Get the data, indices, and indptr from the SciPy CSC matrix
    auto data = A.attr("data").cast<py::array_t<double>>();
    auto indices = A.attr("indices").cast<py::array_t<cs::csint>>();
    auto indptr = A.attr("indptr").cast<py::array_t<cs::csint>>();

    // Convert to std::vector
    std::vector<double> data_vec(data.data(), data.data() + data.size());
    std::vector<cs::csint> indices_vec(indices.data(), indices.data() + indices.size());
    std::vector<cs::csint> indptr_vec(indptr.data(), indptr.data() + indptr.size());

    // Get the shape of the A
    auto shape = A.attr("shape").cast<std::tuple<cs::csint, cs::csint>>();
    cs::Shape A_shape = {std::get<0>(shape), std::get<1>(shape)};

    return cs::CSCMatrix(data_vec, indices_vec, indptr_vec, A_shape);
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
        .def_property_readonly("V", [](const cs::QRResult& qr) {
            return csc_matrix_to_scipy_csc(qr.V);
        })
        .def_property_readonly("beta", [](const cs::QRResult& qr) {
            return vector_to_numpy(qr.beta);
        })
        .def_property_readonly("R", [](const cs::QRResult& qr) {
            return csc_matrix_to_scipy_csc(qr.R);
        })
        .def_property_readonly("p_inv", [](const cs::QRResult& qr) {
            return vector_to_numpy(qr.p_inv);
        })
        .def_property_readonly("q", [](const cs::QRResult& qr) {
            return vector_to_numpy(qr.q);
        })
        // Add the __iter__ method to make it unpackable
        .def("__iter__", [](const cs::QRResult& qr) {
            // This is a generator function in C++ that yields the elements.
            // The order here determines the unpacking order in Python.
            // Define a local variable because make_iterator needs an lvalue.
            py::object result = py::make_tuple(
                csc_matrix_to_scipy_csc(qr.V),
                vector_to_numpy(qr.beta),
                csc_matrix_to_scipy_csc(qr.R),
                vector_to_numpy(qr.p_inv),
                vector_to_numpy(qr.q)
            );
            return py::make_iterator(result);
        });

    // Bind the LUResult struct
    py::class_<cs::LUResult>(m, "LUResult")
        .def_property_readonly("L", [](const cs::LUResult& lu) {
            return csc_matrix_to_scipy_csc(lu.L);
        })
        .def_property_readonly("U", [](const cs::LUResult& lu) {
            return csc_matrix_to_scipy_csc(lu.U);
        })
        .def_property_readonly("p_inv", [](const cs::LUResult& lu) {
            return vector_to_numpy(lu.p_inv);
        })
        .def_property_readonly("q", [](const cs::LUResult& lu) {
            return vector_to_numpy(lu.q);
        })
        .def("__iter__", [](const cs::LUResult& lu) {
            py::object result = py::make_tuple(
                csc_matrix_to_scipy_csc(lu.L),
                csc_matrix_to_scipy_csc(lu.U),
                vector_to_numpy(lu.p_inv),
                vector_to_numpy(lu.q)
            );
            return py::make_iterator(result);
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
        .def_property_readonly("cc", [](const cs::DMPermResult& res) {
            return array_to_numpy(res.cc);
        })
        .def_property_readonly("rr", [](const cs::DMPermResult& res) {
            return array_to_numpy(res.rr);
        })
        .def_property_readonly("Nb", [](const cs::DMPermResult& res) {
            return res.Nb;
        })
        .def("__iter__", [](const cs::DMPermResult& res) {
            py::object result = py::make_tuple(
                vector_to_numpy(res.p),
                vector_to_numpy(res.q),
                vector_to_numpy(res.r),
                vector_to_numpy(res.s),
                array_to_numpy(res.cc),
                array_to_numpy(res.rr),
                res.Nb
            );
            return py::make_iterator(result);
        });

    // Bind the SCCResult struct
    py::class_<cs::SCCResult>(m, "SCCResult")
        .def_property_readonly("p", [](const cs::SCCResult& res) {
            return vector_to_numpy(res.p);
        })
        .def_property_readonly("r", [](const cs::SCCResult& res) {
            return vector_to_numpy(res.r);
        })
        .def_property_readonly("Nb", [](const cs::SCCResult& res) {
            return res.Nb;
        })
        .def("__iter__", [](const cs::SCCResult& res) {
            py::object result = py::make_tuple(
                vector_to_numpy(res.p),
                vector_to_numpy(res.r),
                res.Nb
            );
            return py::make_iterator(result);
        });  // keep the tuple as long as the iterator

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
            py::arg("rows"),
            py::arg("columns"),
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
        .def_property_readonly("row", &cs::COOMatrix::row)
        .def_property_readonly("col", &cs::COOMatrix::col)
        .def_property_readonly("data", &cs::COOMatrix::data)
        //
        .def("insert", py::overload_cast
                        <cs::csint, cs::csint, double>(&cs::COOMatrix::insert))
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
        .def("to_dense_vector", &cs::COOMatrix::to_dense_vector, py::arg("order")='F')
        .def("toarray", &matrix_to_ndarray<cs::COOMatrix>, py::arg("order")='C')
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
        .def("__iadd__",
            [](cs::CSCMatrix::ItemProxy& self, double v) {
                return self += v;
            }, py::is_operator()
        )
        .def("__isub__",
            [](cs::CSCMatrix::ItemProxy& self, double v) {
                return self -= v;
            }, py::is_operator()
        )
        .def("__imul__",
            [](cs::CSCMatrix::ItemProxy& self, double v) {
                return self *= v;
            }, py::is_operator()
        )
        .def("__idiv__",
            [](cs::CSCMatrix::ItemProxy& self, double v) {
                return self /= v;
            }, py::is_operator()
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
        .def("toarray", &matrix_to_ndarray<cs::CSCMatrix>, py::arg("order")='C')
        //
        .def("transpose", &cs::CSCMatrix::transpose, py::arg("values")=true)
        .def_property_readonly("T", &cs::CSCMatrix::T)
        //
        .def("band", py::overload_cast<cs::csint, cs::csint>
                        (&cs::CSCMatrix::band, py::const_))
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
    m.def("_davis_example_qr", &cs::davis_example_qr, py::arg("add_diag")=0.0);
    m.def("_davis_example_amd", &cs::davis_example_amd);

    // -------------------------------------------------------------------------
    //         General Functions
    // -------------------------------------------------------------------------
    m.def("gaxpy", &cs::gaxpy);
    m.def("gaxpy_row", &cs::gaxpy_row);
    m.def("gaxpy_col", &cs::gaxpy_col);
    m.def("gaxpy_block", &cs::gaxpy_block);
    m.def("gatxpy", &cs::gatxpy);
    m.def("gatxpy_row", &cs::gatxpy_row);
    m.def("gatxpy_col", &cs::gatxpy_col);
    m.def("gatxpy_block", &cs::gatxpy_block);
    m.def("sym_gaxpy", &cs::sym_gaxpy);

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
            const py::object& A_scipy,
            const std::string& order="Natural",
            bool use_postorder=false
        ) {
            cs::CSCMatrix A = scipy_sparse_to_csparse(A_scipy);
            cs::AMDOrder order_enum = string_to_amdorder(order);
            cs::SymbolicChol S = cs::schol(A, order_enum, use_postorder);
            // TODO make CholResult struct with named members?
            cs::CSCMatrix L = cs::chol(A, S);
            return py::make_tuple(
                csc_matrix_to_scipy_csc(L),
                cs::inv_permute(S.p_inv)
            );
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

    m.def("chol_update_", &cs::chol_update);

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
            const py::object& A_scipy,
            const std::string& order="Natural",
            double tol=1.0
        ) {
            cs::CSCMatrix A = scipy_sparse_to_csparse(A_scipy);
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
    m.def("scc", &cs::scc, py::arg("A"));

    //--------------------------------------------------------------------------
    //      Solve functions
    //--------------------------------------------------------------------------
    m.def("lsolve", &cs::lsolve);
    m.def("usolve", &cs::usolve);
    m.def("ltsolve", &cs::ltsolve);
    m.def("utsolve", &cs::utsolve);
    m.def("lsolve_opt", &cs::lsolve_opt);
    m.def("usolve_opt", &cs::usolve_opt);

    m.def("chol_solve",
        [](
            const cs::CSCMatrix& A,
            const std::vector<double>& b,
            const std::string& order
        ) {
            cs::AMDOrder order_enum = string_to_amdorder(order);
            return cs::chol_solve(A, b, order_enum);
        },
        py::arg("A"),
        py::arg("b"),
        py::arg("order")="Natural"  // CSparse default is "ATANoDenseRows"
    );

    m.def("qr_solve",
        [](
            const cs::CSCMatrix& A,
            const std::vector<double>& b,
            const std::string& order
        ) {
            cs::AMDOrder order_enum = string_to_amdorder(order);
            return cs::qr_solve(A, b, order_enum);
        },
        py::arg("A"),
        py::arg("b"),
        py::arg("order")="Natural"  // CSparse default is "ATA"
    );

    m.def("lu_solve",
        [](
            const cs::CSCMatrix& A,
            const std::vector<double>& b,
            const std::string& order
        ) {
            cs::AMDOrder order_enum = string_to_amdorder(order);
            return cs::lu_solve(A, b, order_enum);
        },
        py::arg("A"),
        py::arg("b"),
        py::arg("order")="Natural"  // CSparse default is "ATANoDenseRows"
    );


}

/*==============================================================================
 *============================================================================*/
