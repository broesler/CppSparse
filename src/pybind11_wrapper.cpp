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


PYBIND11_MODULE(csparse, m)
{
    m.doc() = "C++Sparse module for sparse matrix operations.";

    //--------------------------------------------------------------------------
    //        Structs
    //--------------------------------------------------------------------------
    py::class_<cs::Problem>(m, "Problem",
        R"pbdoc(
        A data structure to represent a sparse linear system *Ax = b*.

        Attributes
        ----------
        A : (M, N) CSCMatrix
            The original matrix A.
        C : (M, M) CSCMatrix
            The symmetric version of the original matrix.
        is_sym : int
            -1 if A is lower triangular, 1 if upper triangular, 0 otherwise.
        b : (M,) np.ndarray
            The right-hand side vector b.
        x : (N,) np.ndarray
            The solution vector x.
        resid : (N,) np.ndarray
            The residual vector *b - Ax*.
        )pbdoc"
    )
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
            py::kw_only(),
            py::arg("droptol")=0,
            R"pbdoc(
            Create a Problem instance from a given matrix T.

            Parameters
            ----------
            T : (M, N) CSCMatrix
                The input matrix.
            droptol : float, optional
                Drop tolerance for zeroing small entries. Default is 0 (no dropping).

            Returns
            -------
            Problem
                An instance of the Problem class.
            )pbdoc"
        );

    py::class_<cs::CholResult>(m, "CholResult",
        R"pbdoc(
        A data structure to represent the result of a Cholesky factorization.

        Attributes
        ----------
        L : (N, N) CSCMatrix
            The lower-triangular Cholesky factor.
        p : (N,) np.ndarray of int
            The fill-reducing permutation vector.
        )pbdoc"
    )
        .def_property_readonly("L", [](const cs::CholResult& res) { return scipy_from_csc(res.L); })
        .def_property_readonly("p", [](const cs::CholResult& res) { return cs::inv_permute(res.p_inv); })
        .def("__iter__", [](const cs::CholResult& res) {
            py::object result = py::make_tuple(
                scipy_from_csc(res.L),
                cs::inv_permute(res.p_inv)
            );
            return py::make_iterator(result);
        })
        .def("solve",
            [](const cs::CholResult& self, const std::vector<double>& b) {
                std::vector<double> x(b);  // copy b to x
                self.solve(x);
                return x;
            },
            py::arg("b"),
            R"pbdoc(
            Solve the linear system Ax = b using the Cholesky factorization.

            Parameters
            ----------
            b : (N,) np.ndarray
                The right-hand side vector.

            Returns
            -------
            x : (N,) np.ndarray
                The solution vector.
            )pbdoc"
        )
        .def("lsolve",
            [](
                const cs::CholResult& self,
                const py::object& b_obj,
                const std::vector<cs::csint>& parent
            ) {
                cs::CSCMatrix b = csc_from_scipy(b_obj);
                auto [xi, x] = self.lsolve(b, parent);
                return x;
            },
            py::arg("b"),
            py::arg("parent")=std::vector<cs::csint>{},
            R"pbdoc(
            Solve `Lx = b` with sparse RHS `b`, where `L` is a lower-triangular
            Cholesky factor.

            Parameters
            ----------
            b : (N, 1) CSCMatrix
                A sparse RHS vector, stored as a CSCMatrix.
            parent : array_like of int, optional
                The parent vector of the elimination tree of `L`. If not given,
                the function will compute it from `L`.

            Returns
            -------
            xi : list of int
                The row indices of the non-zero entries in x.
            x : (N,) np.ndarray
                The solution vector, stored as a dense vector.
            )pbdoc"
        )
        .def("ltsolve",
            [](
                const cs::CholResult& self,
                const py::object& b_obj,
                const std::vector<cs::csint>& parent
            ) {
                cs::CSCMatrix b = csc_from_scipy(b_obj);
                auto [xi, x] = self.ltsolve(b, parent);
                return x;
            },
            py::arg("b"),
            py::arg("parent")=std::vector<cs::csint>{},
            R"pbdoc(
            Solve `L^T x = b` with sparse RHS `b`, where `L` is
            a lower-triangular Cholesky factor.

            Parameters
            ----------
            b : (N, 1) CSCMatrix
                A sparse RHS vector, stored as a CSCMatrix.
            parent : array_like of int, optional
                The parent vector of the elimination tree of `L`. If not given,
                the function will compute it from `L`.

            Returns
            -------
            xi : list of int
                The row indices of the non-zero entries in x.
            x : (N,) np.ndarray
                The solution vector, stored as a dense vector.
            )pbdoc"
        );

    py::class_<cs::QRResult>(m, "QRResult",
        R"pbdoc(
        A data structure to represent the result of a QR factorization.

        Attributes
        ----------
        V : (M, K) CSCMatrix
            The Householder vectors.
        beta : (K,) np.ndarray of float
            The scaling factors for the Householder reflections.
        R : (M, N) CSCMatrix
            The upper-triangular matrix R.
        p : (M,) np.ndarray of int
            The row permutation vector.
        q : (N,) np.ndarray of int
            The column permutation vector.
        )pbdoc"
    )
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
        })
        .def("solve",
            [](const cs::QRResult& self, const std::vector<double>& b) {
                cs::csint N = self.R.shape()[1];
                std::vector<double> x(N);  // create output vector
                self.solve(b, x);
                return x;
            },
            py::arg("b"),
            R"pbdoc(
            Solve the linear system `Ax = b` using QR factorization.

            Parameters
            ----------
            b : (M,) np.ndarray
                The right-hand side vector.

            Returns
            -------
            x : (N,) np.ndarray
                The solution vector.
            )pbdoc"
        )
        .def("tsolve",
            [](const cs::QRResult& self, const std::vector<double>& b) {
                cs::csint M2 = self.V.shape()[0];
                std::vector<double> x(M2);  // create output vector
                self.tsolve(b, x);
                return x;
            },
            py::arg("b"),
            R"pbdoc(
            Solve the linear system :math:`A^{\top} x = b` using QR factorization.

            Parameters
            ----------
            b : (N,) np.ndarray
                The right-hand side vector.

            Returns
            -------
            x : (M,) np.ndarray
                The solution vector.
            )pbdoc"
        );

    py::class_<cs::LUResult>(m, "LUResult",
        R"pbdoc(
        A data structure to represent the result of an LU factorization.

        Attributes
        ----------
        L : (M, M) CSCMatrix
            The lower-triangular matrix L.
        U : (M, N) CSCMatrix
            The upper-triangular matrix U.
        p : (M,) np.ndarray of int
            The row permutation vector.
        q : (N,) np.ndarray of int
            The column permutation vector.
        )pbdoc"
    )
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
        })
        .def("solve",
            [](const cs::LUResult& self, const std::vector<double>& b) {
                std::vector<double> x(b);  // copy b to x
                self.solve(x);
                return x;
            },
            py::arg("b"),
            R"pbdoc(
            Solve the linear system `Ax = b` using the LU factorization.

            Parameters
            ----------
            b : (M,) np.ndarray
                The right-hand side vector.

            Returns
            -------
            x : (N,) np.ndarray
                The solution vector.
            )pbdoc"
        )
        .def("tsolve",
            [](const cs::LUResult& self, const std::vector<double>& b) {
                std::vector<double> x(b);  // copy b to x
                self.tsolve(x);
                return x;
            },
            py::arg("b"),
            R"pbdoc(
            Solve the linear system :math:`A^{\top} x = b` using the LU
            factorization.

            Parameters
            ----------
            b : (N,) np.ndarray
                The right-hand side vector.

            Returns
            -------
            x : (M,) np.ndarray
                The solution vector.
            )pbdoc"
        );

    // Bind the MaxMatch struct
    py::class_<cs::MaxMatch>(m, "MaxMatch",
        R"pbdoc(
        A data structure to represent the result of a maximum matching.

        Attributes
        ----------
        jmatch : (M,) np.ndarray of int
            The column matches for each row.
        imatch : (N,) np.ndarray of int
            The row matches for each column.
        )pbdoc"
    )
        .def_readonly("jmatch", &cs::MaxMatch::jmatch)
        .def_readonly("imatch", &cs::MaxMatch::imatch)
        .def("__iter__", [](const cs::MaxMatch& res) {
            py::object result = py::make_tuple(res.jmatch, res.imatch);
            return py::make_iterator(result);
        });

    // Bind the DMPermResult struct
    py::class_<cs::DMPermResult>(m, "DMPermResult",
        R"pbdoc(
        A data structure to represent the result of a Dulmage-Mendelsohn permutation.

        Attributes
        ----------
        p : (M,) np.ndarray of int
            The row permutation vector.
        q : (N,) np.ndarray of int
            The column permutation vector.
        r : (Nb+1,) np.ndarray of int
            The row block boundaries of the permuted matrix.
        s : (Nb+1,) np.ndarray of int
            The column block boundaries of the permuted matrix.
        cc : (5,) np.ndarray of int
            The coarse column decomposition.
        rr : (5,) np.ndarray of int
            The coarse row decomposition.
        Nb : int
            The number of blocks in the fine Dulmage-Mendelsohn decomposition.
        )pbdoc"
    )
        .def_readonly("p", &cs::DMPermResult::p)
        .def_readonly("q", &cs::DMPermResult::q)
        .def_readonly("r", &cs::DMPermResult::r)
        .def_readonly("s", &cs::DMPermResult::s)
        .def_property_readonly("cc",
            [](const cs::DMPermResult& self) { return array_to_numpy(self.cc); }
        )
        .def_property_readonly("rr",
            [](const cs::DMPermResult& self) { return array_to_numpy(self.rr); }
        )
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
    py::class_<cs::SCCResult>(m, "SCCResult",
        R"pbdoc(
        A data structure representing the strongly connected components of a matrix.

        Attributes
        ----------
        p : (M,) np.ndarray of int
            The row permutation vector.
        r : (Nb+1,) np.ndarray of int
            The block boundaries of the permuted matrix.
        Nb : int
            The number of blocks in the decomposition.
        )pbdoc"
    )
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
    py::class_<cs::COOMatrix>(m, "COOMatrix",
        R"pbdoc(
        A class to represent a sparse matrix in Coordinate (COO) format.

        Attributes
        ----------
        data : (nnz,) np.array of float
            The non-zero values of the matrix.
        row : (nnz,) np.array of int
            The row indices of the non-zero values.
        col : (nnz,) np.array of int
            The column indices of the non-zero values.
        nnz : int
            The number of non-zero elements in the matrix.
        nzmax : int
            The maximum number of non-zero elements allocated.
        shape : 2-tuple
            The shape of the matrix.
        )pbdoc"
    )
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
        .def(py::init<const cs::Shape&, cs::csint>(),
            py::arg("shape"),
            py::arg("nzmax")=0
        )
        //
        .def_static("from_file", &cs::COOMatrix::from_file, py::arg("filename"))
        .def_static("random",
            &cs::COOMatrix::random,
            py::arg("M"),
            py::arg("N"),
            py::arg("density")=0.1,
            py::kw_only(),
            py::arg("seed")=0,
            R"pbdoc(
            Create a random sparse matrix in COO format.

            Parameters
            ----------
            M : int
                The number of rows.
            N : int
                The number of columns.
            density : float, optional
                The fraction of non-zero elements. Default is 0.1.
            seed : int, optional
                The random seed. Default is 0.

            Returns
            -------
            COOMatrix
                A random sparse matrix in COO format.
            )pbdoc"
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
        .def("insert",
            py::overload_cast<cs::csint, cs::csint, double>(&cs::COOMatrix::insert),
            R"pbdoc(
            Insert a single value into the matrix at position (i, j).

            Parameters
            ----------
            i : int
                The row index.
            j : int
                The column index.
            v : float
                The value to insert.
            )pbdoc"
        )
        .def("__setitem__",
            [](cs::COOMatrix& A, py::tuple t, double v) {
                cs::csint i = t[0].cast<cs::csint>();
                cs::csint j = t[1].cast<cs::csint>();
                A.insert(i, j, v);
            }
        )
        //
        .def("compress", &cs::COOMatrix::compress,
            R"pbdoc(
            Convert the COO matrix to a compressed sparse column (CSC) format.

            Returns
            -------
            CSCMatrix
                The equivalent matrix in CSC format. The columns are not
                guaranteed to be sorted, and duplicates are allowed.
            )pbdoc"
        )
        .def("tocsc", &cs::COOMatrix::tocsc,
            R"pbdoc(
            Convert the COO matrix to a compressed sparse column (CSC) format.

            Returns
            -------
            CSCMatrix
                The equivalent matrix in canonical CSC format. The columns are
                guaranteed to be sorted, and duplicate entries are summed.
            )pbdoc"
        )
        .def("toscipy",
            [](const cs::COOMatrix& self) {
                return scipy_from_coo(self);
            },
            R"pbdoc(
            Convert the COO matrix to a SciPy sparse matrix.

            Returns
            -------
            scipy.sparse.coo_matrix
                The equivalent SciPy COO sparse matrix.
            )pbdoc"
        )
        .def("to_dense_vector",
            [](const cs::CSCMatrix& self, const char order_) {
                auto order = denseorder_from_char(order_);
                return self.to_dense_vector(order);
            },
            py::arg("order")='F',
            R"pbdoc(
            Convert the COO matrix to a dense vector.

            Parameters
            ----------
            order : {'C', 'F'}, optional
                The order of the array, either 'C' for row-major or 'F' for
                column-major order. Default is 'F'.

            Returns
            -------
            np.ndarray
                A dense array representation of the matrix.
            )pbdoc"
        )
        .def("toarray", &sparse_to_ndarray<cs::COOMatrix>, py::arg("order")='C',
            R"pbdoc(
            Convert the COO matrix to a dense vector.

            Parameters
            ----------
            order : {'C', 'F'}, optional
                The order of the array, either 'C' for row-major or 'F' for
                column-major order. Default is 'F'.

            Returns
            -------
            np.ndarray
                A dense array representation of the matrix.
            )pbdoc"
        )
        //
        .def("transpose", &cs::COOMatrix::transpose)
        .def_property_readonly("T", &cs::COOMatrix::T)
        //
        .def("dot", &cs::COOMatrix::dot,
            R"pbdoc(
            Multiply the COO matrix by a dense vector.

            Parameters
            ----------
            x : np.ndarray
                The dense vector by which to multiply.

            Returns
            -------
            np.ndarray
                The result of the matrix-vector multiplication.
            )pbdoc"
        )
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
    py::class_<cs::CSCMatrix>(m, "CSCMatrix",
        R"pbdoc(
        A class to represent a sparse matrix in Compressed Sparse Column (CSC) format.

        Attributes
        ----------
        data : (nnz,) np.array of float
            The non-zero values of the matrix.
        indices : (nnz,) np.array of int
            The row indices of the non-zero values.
        indptr : (N+1,) np.array of int
            The column pointer indices.
        nnz : int
            The number of non-zero elements in the matrix.
        nzmax : int
            The maximum number of non-zero elements allocated.
        shape : 2-tuple
            The shape of the matrix.
        has_sorted_indices : bool
            True if the row indices within each column are sorted.
        has_canonical_format : bool
            True if the matrix is in canonical format (sorted row indices, no
            duplicate entries, no explicit zeros).
        is_symmetric : bool
            True if the matrix is numerically symmetric.
        is_triangular : int
            -1 if the matrix is lower triangular, 1 if upper triangular, 0 otherwise.
        )pbdoc"
    )
        .def(py::init<>())
        .def(py::init<
            const std::vector<double>&,
            const std::vector<cs::csint>&,
            const std::vector<cs::csint>&,
            const cs::Shape&>(),
            py::arg("data"),
            py::arg("indices"),
            py::arg("indptr"),
            py::arg("shape")=cs::Shape{0, 0}
        )
        .def(py::init<const cs::Shape&, cs::csint, bool>(),
            py::arg("shape"),
            py::kw_only(),
            py::arg("nzmax")=0,
            py::arg("values")=true
        )
        .def(py::init<const cs::COOMatrix&>())
        .def(py::init(
                [](
                    const std::vector<double>& A,
                    const cs::Shape& shape,
                    const char order_
                ) {
                    auto order = denseorder_from_char(order_);
                    return cs::CSCMatrix(A, shape, order);
                }
            ),
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
        .def("dropzeros", &cs::CSCMatrix::dropzeros,
            "Remove explicit zero entries from the matrix."
        )
        .def("droptol", &cs::CSCMatrix::droptol, py::arg("tol")=1e-15,
            R"pbdoc(
            Remove entries with absolute value within the specified tolerance of zero.

            Parameters
            ----------
            tol : float, optional
                The tolerance against which to compare the absolute value of the
                matrix entries.
            )pbdoc"
        )
        .def("to_canonical", &cs::CSCMatrix::to_canonical,
            R"pbdoc(
            Convert the matrix to canonical format in-place.

            The row indices are guaranteed to be sorted, no duplicates are
            allowed, and no numerically zero entries are allowed.

            This function takes O(M + N + nnz) time.
            )pbdoc"
        )
        .def("toscipy",
            [](const cs::CSCMatrix& self) {
                return scipy_from_csc(self);
            },
            R"pbdoc(
            Convert the CSC matrix to a SciPy sparse matrix.

            Returns
            -------
            scipy.sparse.csc_matrix
                The equivalent SciPy CSC sparse matrix.
            )pbdoc"
        )
        //
        .def_property_readonly("has_sorted_indices", &cs::CSCMatrix::has_sorted_indices)
        .def_property_readonly("has_canonical_format", &cs::CSCMatrix::has_canonical_format)
        .def_property_readonly("is_symmetric", &cs::CSCMatrix::is_symmetric)
        .def_property_readonly("is_triangular", &cs::CSCMatrix::is_triangular)
        .def_property_readonly("structural_symmetry", &cs::CSCMatrix::structural_symmetry,
            R"pbdoc(
            Compute the structural symmetry of the matrix.

            The structural symmetry is defined as:

            .. math::
                \operatorname{sym}(S) = \frac{|(S \land S^{\top})|}{|S|},

            where :math:`S = A - \operatorname{diag}(A)` (off-diagonal elements
            only).

            In Scipy:

            .. code-block:: python

              S = A - sparse.diags_array(A.diagonal())
              sym = (S * S.T).nnz / S.nnz

            Returns
            -------
            float in [0, 1]
                The structural symmetry of the matrix.
            )pbdoc"
        )
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
        .def("assign",
            py::overload_cast<cs::csint, cs::csint, double>(&cs::CSCMatrix::assign),
            R"pbdoc(
            Assign a value to a specific element in the matrix.

            This function takes O(log M) time if the columns are sorted, and O(M) time
            otherwise.

            Parameters
            ----------
            i, j : int
                The row and column indices of the element to access.
            v : float
                The value to be assigned.
            )pbdoc"
        )
        .def("assign",
            py::overload_cast<
                const std::vector<cs::csint>&,
                const std::vector<cs::csint>&,
                const std::vector<double>&
            >(&cs::CSCMatrix::assign),
            R"pbdoc(
            Assign a dense matrix to the CSCMatrix at the specified locations.

            Parameters
            ----------
            rows, cols : (N,) array_like of int
                The row and column indices of the elements to access.
            C : (N,) array_like of float
                The dense matrix to be assigned, in column-major order.
            )pbdoc"
        )
        .def("assign",
            py::overload_cast<
                const std::vector<cs::csint>&,
                const std::vector<cs::csint>&,
                const cs::CSCMatrix&
            >(&cs::CSCMatrix::assign),
            R"pbdoc(
            Assign a sparse matrix to the CSCMatrix at the specified locations.

            Parameters
            ----------
            rows, cols : (N,) array_like of int
                The row and column indices of the elements to access.
            C : CSCMatrix
                The sparse matrix to be assigned.
            )pbdoc"
        )
        //
        .def("tocoo", &cs::CSCMatrix::tocoo,
            R"pbdoc(
            Convert the CSC matrix to a coordinate (triplet) format matrix.

            Returns
            -------
            COOMatrix
                A copy of the `CSCMatrix` in COO (triplet) format.
            )pbdoc"
        )
        .def("to_dense_vector",
            [](const cs::CSCMatrix& self, const char order_) {
                auto order = denseorder_from_char(order_);
                return self.to_dense_vector(order);
            },
            py::arg("order")='F',
            R"pbdoc(
            Convert the CSC matrix to a dense vector.

            Parameters
            ----------
            order : {'C', 'F'}, optional
                The order of the array, either 'C' for row-major or 'F' for
                column-major order.

            Returns
            -------
            np.ndarray
                A dense array representation of the matrix.

            See Also
            --------
            toarray : Convert the CSC matrix to a dense array.
            )pbdoc"
        )
        .def("toarray", &sparse_to_ndarray<cs::CSCMatrix>, py::arg("order")='C',
            R"pbdoc(
            Convert the CSC matrix to a dense vector.

            Parameters
            ----------
            order : {'C', 'F'}, optional
                The order of the array, either 'C' for row-major or 'F' for
                column-major order.

            Returns
            -------
            np.ndarray
                A dense array representation of the matrix.

            See Also
            --------
            toarray : Convert the CSC matrix to a dense array.
            )pbdoc"
        )
        //
        .def("transpose", &cs::CSCMatrix::transpose,
            py::kw_only(),
            py::arg("values")=true,
            R"pbdoc(
            Transpose the CSC matrix.

            Parameters
            ----------
            values : bool, optional
                If False, do not copy the numerical values, only the row indices.
            )pbdoc"
        )
        .def_property_readonly("T", &cs::CSCMatrix::T, "Alias of `transpose()`.")
        //
        .def("sort", &cs::CSCMatrix::sort,
            R"pbdoc(
            Sort rows and columns in-place using two transposes, but more
            efficiently than calling `transpose` twice.
            )pbdoc"
        )
        .def("tsort", &cs::CSCMatrix::tsort,
            "Sort rows and columns in a copy via two transposes."
        )
        .def("qsort", &cs::CSCMatrix::qsort,
            "Sort rows and columns in place using std::sort (quicksort)."
        )
        //
        .def("band",
            py::overload_cast<cs::csint, cs::csint>(&cs::CSCMatrix::band, py::const_),
            R"pbdoc(
            Keep any entries within the specified band, in-place.

            Parameters
            ----------
            kl, ku  : int
                The lower and upper diagonals within which to keep entries. The
                main diagonal is 0, with sub-diagonals < 0, and super-diagonals
                > 0.
            )pbdoc"
        )
        //
        .def("scale", &cs::CSCMatrix::scale,
            "Scale the rows and/or columns of the matrix in-place."
        )
        //
        .def("__mul__", py::overload_cast<const double>(&cs::CSCMatrix::dot, py::const_), py::is_operator())
        .def("__rmul__", py::overload_cast<const double>(&cs::CSCMatrix::dot, py::const_), py::is_operator())
        .def("dot",
            py::overload_cast<const double>(&cs::CSCMatrix::dot, py::const_),
            py::arg("other"),
            R"pbdoc(
            Compute the dot product of the CSC matrix with a scalar, dense vector,
            or another CSC matrix.

            Parameters
            ----------
            other : float, np.ndarray, or CSCMatrix
                The scalar, dense vector, or CSC matrix to multiply.

            Returns
            -------
            float, np.ndarray or CSCMatrix
                The result of the dot product. Type matches the input type.
            )pbdoc"
        )
        .def("dot",
            py::overload_cast<std::span<const double>>(&cs::CSCMatrix::dot, py::const_),
            py::arg("other")
        )
        .def("dot",
            py::overload_cast<const cs::CSCMatrix&>(&cs::CSCMatrix::dot, py::const_),
            py::arg("other")
        )
        .def("__matmul__", py::overload_cast<const double>(&cs::CSCMatrix::dot, py::const_))
        .def("__matmul__", py::overload_cast<std::span<const double>>(&cs::CSCMatrix::dot, py::const_))
        .def("__matmul__", py::overload_cast<const cs::CSCMatrix&>(&cs::CSCMatrix::dot, py::const_))
        //
        .def("add", &cs::CSCMatrix::add,
            py::arg("B"),
            R"pbdoc(
            Add another CSC matrix to this matrix.

            Parameters
            ----------
            B : CSCMatrix
                The matrix to add.

            Returns
            -------
            CSCMatrix
                A new CSCMatrix representing the sum of the two matrices.
            )pbdoc"
        )
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
            py::arg("p"),
            py::arg("q"),
            py::kw_only(),
            py::arg("values")=true,
            R"pbdoc(
            Permute the rows and columns of a copy of the matrix.

            Equivalent to calling ``A[p[:, np.newaxis], q]`` in NumPy notation.

            Parameters
            ----------
            p : (M,) array_like of int
                The row permutation vector.
            q : (N,) array_like of int
                The column permutation vector.
            values : bool, optional
                If False, do not copy the numerical values, only the row indices.
            )pbdoc"
        )
        .def("symperm",
            [](const cs::CSCMatrix& self,
               const std::vector<cs::csint> p,
               bool values=true)
            {
                return self.symperm(cs::inv_permute(p), values);
            },
            py::arg("p"),
            py::kw_only(),
            py::arg("values")=true,
            R"pbdoc(
            Symmetrically permute the rows and columns of a copy of the matrix.

            Equivalent to calling ``A.permute(p, p)``,
            or ``A[p[:, np.newaxis], p]`` in NumPy notation.

            Parameters
            ----------
            p : (M,) array_like of int
                The symmetric permutation vector.
            values : bool, optional
                If False, do not copy the numerical values, only the row indices.
            )pbdoc"
        )
        .def("permute_transpose",
            [](const cs::CSCMatrix& self,
               const std::vector<cs::csint> p,
               const std::vector<cs::csint> q,
               bool values=true)
            {
                return self.permute_transpose(cs::inv_permute(p), q, values);
            },
            py::arg("p"),
            py::arg("q"),
            py::kw_only(),
            py::arg("values")=true,
            R"pbdoc(
            Permute the rows and columns of the transpose of the matrix.

            Equivalent to calling ``A.T[p[:, np.newaxis], q]`` in NumPy notation.

            Parameters
            ----------
            p : (N,) array_like of int
                The row permutation vector.
            q : (M,) array_like of int
                The column permutation vector.
            values : bool, optional
                If False, do not copy the numerical values, only the row indices.
            )pbdoc"
        )
        .def("permute_rows",
            [](const cs::CSCMatrix& self,
               const std::vector<cs::csint> p,
               bool values=true)
            {
                return self.permute_rows(cs::inv_permute(p), values);
            },
            py::arg("p"),
            py::kw_only(),
            py::arg("values")=true,
            R"pbdoc(
            Permute the rows of a copy of the matrix.

            Equivalent to calling ``A[p[:, np.newaxis], :]`` in NumPy notation.

            Parameters
            ----------
            p : (M,) array_like of int
                The row permutation vector.
            values : bool, optional
                If False, do not copy the numerical values, only the row indices.
            )pbdoc"
        )
        .def("permute_cols", &cs::CSCMatrix::permute_cols,
             py::arg("q"),
             py::kw_only(),
             py::arg("values")=true,
             R"pbdoc(
            Permute the columns of a copy of the matrix.

            Equivalent to calling ``A[:, q]`` in NumPy notation.

            Parameters
            ----------
            q : (N,) array_like of int
                The column permutation vector.
            values : bool, optional
                If False, do not copy the numerical values, only the row indices.
            )pbdoc"
        )
        //
        .def("norm", &cs::CSCMatrix::norm,
            R"pbdoc(
            Compute the 1-norm of the matrix (maximum column sum).

            The 1-norm is defined as:

            .. math::
                \|A\|_1 = \max_j \sum_{i=1}^{m} |a_{ij}|.

            Returns
            -------
            float
                The 1-norm of the matrix.
            )pbdoc"
        )
        .def("fronorm", &cs::CSCMatrix::fronorm,
            R"pbdoc(
            Compute the Frobenius norm of the matrix.

            The Frobenius norm is defined as:

            .. math::
                \|A\|_F =
                \( \sum_{i=1}^{m} \sum_{j=1}^{n} |a_{ij}|^2 \)^{\frac{1}{2}}.

            Returns
            -------
            float
                The Frobenius norm of the matrix.
            )pbdoc"
        )
        //
        .def("slice", &cs::CSCMatrix::slice,
            py::arg("i_start"),
            py::arg("i_end"),
            py::arg("j_start"),
            py::arg("j_end"),
            R"pbdoc(
            Extract a submatrix given row and column index ranges.

            Equivalent to calling ``A[i_start:i_end, j_start:j_end]`` in NumPy
            notation.

            Parameters
            ----------
            i_start, i_end : int
                The starting (inclusive) and ending (exclusive) row indices.
            j_start, j_end : int
                The starting (inclusive) and ending (exclusive) column indices.

            Returns
            -------
            res : (i_end - i_start, j_end - j_start) CSCMatrix
                The extracted submatrix.
            )pbdoc"
        )
        .def("index", &cs::CSCMatrix::index,
            py::arg("rows"),
            py::arg("cols"),
            R"pbdoc(
            Extract a submatrix given row and column index arrays.

            Equivalent to calling ``A[rows[:, np.newaxis], cols]`` in NumPy
            notation.

            Parameters
            ----------
            rows : (Mr,) array_like of int
                The row indices.
            cols : (Nc,) array_like of int
                The column indices.

            Returns
            -------
            res : (Mr, Nc) CSCMatrix
                The extracted submatrix.
            )pbdoc"
        )
        .def("scatter",
            [](const cs::CSCMatrix& self, const cs::csint k) {
                std::vector<double> x(self.shape()[0]);  // (M,)
                if (k < 0 || k >= self.shape()[1]) {
                    throw py::index_error("Column index out of bounds.");
                }
                self.scatter(k, x);
                return x;
            },
            py::arg("k"),
            "Scatter the k-th column of the matrix into a dense vector."
        )
        .def("add_empty_top", &cs::CSCMatrix::add_empty_top,
            "Add empty rows to the top of the matrix."
        )
        .def("add_empty_bottom", &cs::CSCMatrix::add_empty_bottom,
            "Add empty rows to the bottom of the matrix."
        )
        .def("add_empty_left", &cs::CSCMatrix::add_empty_left,
            "Add empty columns to the left of the matrix."
        )
        .def("add_empty_right", &cs::CSCMatrix::add_empty_right,
            "Add empty columns to the right of the matrix."
        )
        //
        .def("sum_rows", &cs::CSCMatrix::sum_rows,
            R"pbdoc(
            Compute the sum of each row in the matrix.

            Returns
            -------
            (M,) np.ndarray
                A dense array containing the sum of each row.
            )pbdoc"
        )
        .def("sum_cols", &cs::CSCMatrix::sum_cols,
            R"pbdoc(
            Compute the sum of each column in the matrix.

            Returns
            -------
            (N,) np.ndarray
                A dense array containing the sum of each column.
            )pbdoc"
        )
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
    m.def("davis_example_small",
        []() { return scipy_from_coo(cs::davis_example_small()); },
        R"pbdoc(
        Define the 4x4 matrix from Davis Equation (2.1) [p 7--8].

        .. code-block:: python

            A = [[4.5,   0, 3.2,   0],
                 [3.1, 2.9,   0, 0.9],
                 [  0, 1.7,   3,   0],
                 [3.5, 0.4,   0,   1]]

        See: Davis, Timothy A. "Direct Methods for Sparse Linear Systems",
             Eqn (2.1), p. 7--8.

        Returns
        -------
        coo_array
            A 4x4 matrix in COO format.
        )pbdoc"
    );
    m.def("davis_example_chol",
        []() { return scipy_from_csc(cs::davis_example_chol()); },
        R"pbdoc(
        Define the 11x11 matrix in Davis, Figure 4.2, p 39.

        This matrix is sparse and symmetric positive definite. We arbitrarily assign
        the diagonal to the 0-based index values + 10, and the off-diagonals to 1.

        .. code-block:: python

            A = [[10,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0],
                 [ 0, 11,  1,  0,  0,  0,  0,  1,  0,  0,  0],
                 [ 0,  1, 12,  0,  0,  0,  0,  0,  0,  1,  1],
                 [ 0,  0,  0, 13,  0,  1,  0,  0,  0,  1,  0],
                 [ 0,  0,  0,  0, 14,  0,  0,  1,  0,  0,  1],
                 [ 1,  0,  0,  1,  0, 15,  0,  0,  1,  1,  0],
                 [ 1,  0,  0,  0,  0,  0, 16,  0,  0,  0,  1],
                 [ 0,  1,  0,  0,  1,  0,  0, 17,  0,  1,  1],
                 [ 0,  0,  0,  0,  0,  1,  0,  0, 18,  0,  0],
                 [ 0,  0,  1,  1,  0,  1,  0,  1,  0, 19,  1],
                 [ 0,  0,  1,  0,  1,  0,  1,  1,  0,  1, 20]]

        See: Davis, Timothy A. "Direct Methods for Sparse Linear Systems",
             Figure 4.2, p 39.

        Returns
        -------
        csc_array
            An 11x11 matrix in CSC format.
        )pbdoc"
    );
    m.def("davis_example_qr",
        [](double add_diag=0.0, bool random_vals=false) {
            return scipy_from_csc(cs::davis_example_qr(add_diag, random_vals));
        },
        py::arg("add_diag")=0.0,
        py::arg("random_vals")=false,
        R"pbdoc(
        Define the 8x8 matrix in Davis, Figure 5.1, p 74.

        This matrix is sparse, unsymmetric positive definite. We arbitrarily assign
        the diagonal to the 1-based index values (except 8), and off-diagonals to 1.

        .. code-block:: python

            A = [[1., 0., 0., 1., 0., 0., 1., 0.],
                 [0., 2., 1., 0., 0., 0., 1., 0.],
                 [0., 0., 3., 1., 0., 0., 0., 0.],
                 [1., 0., 0., 4., 0., 0., 1., 0.],
                 [0., 0., 0., 0., 5., 1., 0., 0.],
                 [0., 0., 0., 0., 1., 6., 0., 1.],
                 [0., 1., 1., 0., 0., 0., 7., 1.],
                 [0., 0., 0., 0., 1., 1., 1., 0.]]

        See: Davis, Timothy A. "Direct Methods for Sparse Linear Systems",
             Figure 5.1, p 74.

        Parameters
        ----------
        add_diag : float
            If non-zero, add this value to the diagonal of the matrix. Can be
            used to make the matrix positive definite.
        random_vals : bool
            If True, randomize the non-zero values in the matrix, before adding
            to the diagonal.

        Returns
        -------
        csc_array
            An 8x8 matrix in CSC format.
        )pbdoc"
    );
    m.def("davis_example_amd",
        []() { return scipy_from_csc(cs::davis_example_amd()); },
        R"pbdoc(
        Build the 10 x 10 symmetric, positive definite AMD example matrix.

        .. code-block:: python

            A = [[10.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.],
                 [ 0., 11.,  0.,  0.,  1.,  1.,  0.,  0.,  1.,  0.],
                 [ 0.,  0., 12.,  0.,  1.,  1.,  1.,  0.,  0.,  0.],
                 [ 1.,  0.,  0., 13.,  0.,  0.,  1.,  1.,  0.,  0.],
                 [ 0.,  1.,  1.,  0., 14.,  0.,  1.,  0.,  1.,  0.],
                 [ 1.,  1.,  1.,  0.,  0., 15.,  0.,  0.,  0.,  0.],
                 [ 0.,  0.,  1.,  1.,  1.,  0., 16.,  1.,  1.,  1.],
                 [ 0.,  0.,  0.,  1.,  0.,  0.,  1., 17.,  1.,  1.],
                 [ 0.,  1.,  0.,  0.,  1.,  0.,  1.,  1., 18.,  1.],
                 [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1., 19.]]

        See: Davis, Figure 7.1, p 101.

        Returns
        -------
        csc_array
            The 10x10 matrix in CSC format.
        )pbdoc"
    );

    // -------------------------------------------------------------------------
    //         General Functions
    // -------------------------------------------------------------------------
    m.def("gaxpy", make_vector_func(&cs::gaxpy),
        "Perform the sparse matrix-vector operation `z = Ax + y`."
    );
    m.def("gatxpy", make_vector_func(&cs::gatxpy),
        R"pbdoc(Perform the sparse matrix-vector operation :math:`z = A^{\top} x + y`.)pbdoc"
    );
    m.def("sym_gaxpy", make_vector_func(&cs::sym_gaxpy),
        "Perform the sparse matrix-vector operation `z = Ax + y`, assuming `A` is symmetric."
    );
    m.def("gaxpy_row", make_gaxpy_matrix_func<false>(&cs::gaxpy_row),
        R"pbdoc(Perform the sparse matrix-matrix operation `Z = AX + Y`, where
        `X` and `Y` are dense matrices in row-major format.)pbdoc"
    );
    m.def("gaxpy_col", make_gaxpy_matrix_func(&cs::gaxpy_col),
        R"pbdoc(Perform the sparse matrix-matrix operation `Z = AX + Y`, where
        `X` and `Y` are dense matrices in column-major format.)pbdoc"
    );
    m.def("gaxpy_block", make_gaxpy_matrix_func(&cs::gaxpy_block),
        R"pbdoc(Perform the sparse matrix-matrix operation `Z = AX + Y`, where
        `X` and `Y` are dense matrices in column-major format.)pbdoc"
    );
    m.def("gatxpy_row", make_gaxpy_matrix_func<false>(&cs::gatxpy_row),
        R"pbdoc(Perform the sparse matrix-matrix operation
        :math:`Z = A^{\top} X + Y`, where `X` and `Y` are dense matrices in
        row-major format.)pbdoc"
    );
    m.def("gatxpy_col", make_gaxpy_matrix_func(&cs::gatxpy_col),
        R"pbdoc(Perform the sparse matrix-matrix operation
        :math:`Z = A^{\top} X + Y`, where `X` and `Y` are dense matrices in
        column-major format.)pbdoc"
    );
    m.def("gatxpy_block", make_gaxpy_matrix_func(&cs::gatxpy_block),
        R"pbdoc(Perform the sparse matrix-matrix operation
        :math:`Z = A^{\top} X + Y`, where `X` and `Y` are dense matrices in
        column-major format.)pbdoc"
    );

    //--------------------------------------------------------------------------
    //        Utility Functions
    //--------------------------------------------------------------------------
    m.def("pvec",
        make_pvec_wrapper(
            [](const std::vector<cs::csint>& p, const std::vector<double>& b) {
                return cs::pvec<double>(p, b);
            },
            [](const std::vector<cs::csint>& p, const std::vector<cs::csint>& b) {
                return cs::pvec<cs::csint>(p, b);
            }
        ),
        py::arg("p"),
        py::arg("b"),
        R"pbdoc(
        Permute a vector according to the permutation vector `p`.

        Equivalent to `b[p]` in NumPy notation.

        Parameters
        ----------
        p : (N,) array_like of int
            The permutation vector.
        b : (N,) array_like
            The vector to be permuted.

        Returns
        -------
        np.ndarray
            The permuted vector.
        )pbdoc"
    );
    m.def("ipvec",
        make_pvec_wrapper(
            [](const std::vector<cs::csint>& p, const std::vector<double>& b) {
                return cs::ipvec<double>(p, b);
            },
            [](const std::vector<cs::csint>& p, const std::vector<cs::csint>& b) {
                return cs::ipvec<cs::csint>(p, b);
            }
        ),
        py::arg("p"),
        py::arg("b"),
        R"pbdoc(
        Inversely permute a vector according to the permutation vector `p`.

        Equivalent to `b[np.argsort(p)]` or `b[p] = b` in NumPy notation.

        Parameters
        ----------
        p : (N,) array_like of int
            The permutation vector.
        b : (N,) array_like
            The vector to be inversely permuted.

        Returns
        -------
        np.ndarray
            The inversely permuted vector.
        )pbdoc"
    );
    m.def("inv_permute", &cs::inv_permute,
        R"pbdoc(Invert a permutation vector.

        Equivalent to `np.argsort(p)` in NumPy notation.

        Parameters
        ----------
        p : (N,) array_like of int
            The permutation vector.

        Returns
        -------
        np.ndarray
            The inverted permutation vector.
        )pbdoc"
    );

    m.def("scipy_from_coo", &scipy_from_coo, "Convert a COOMatrix to a SciPy `csc_array`.");
    m.def("scipy_from_csc", &scipy_from_csc, "Convert a CSCMatrix to a SciPy `coo_array`.");
    m.def("csc_from_scipy", &csc_from_scipy, "Convert a SciPy sparse matrix to a CSCMatrix.");
    m.def("coo_from_scipy", &coo_from_scipy, "Convert a SciPy sparse matrix to a COOMatrix.");

    m.def("residual_norm",
        [](const py::object& A_scipy,
           const std::vector<double>& x,
           const std::vector<double>& b
        ) {
            std::vector<double> resid;
            cs::CSCMatrix A = csc_from_scipy(A_scipy);
            return cs::residual_norm(A, x, b, resid);
        },
        py::arg("A"),
        py::arg("x"),
        py::arg("b"),
        R"pbdoc(
        Compute the residual norm :math:`\|Ax - b\|_2`.

        Parameters
        ----------
        A : (M, N) CSCMatrix
            The sparse matrix.
        x : (N,) array_like of float
            The solution vector.
        b : (M,) array_like of float
            The right-hand side vector.

        Returns
        -------
        float
            The residual 2-norm, :math:`\|Ax - b\|_2`.
        )pbdoc"
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
        py::arg("ATA")=false,
        R"pbdoc(
        Compute the elimination tree of a matrix `A` or :math:`A^T A`.

        Parameters
        ----------
        A : (M, N) CSCMatrix
            The matrix to factorize.
        ATA : bool, optional
            If True, compute the elimination tree of :math:`A^T A`. Default is False.
        )pbdoc"
    );

    m.def("post", &cs::post, py::arg("parent"),
        R"pbdoc(
        Compute the postordering of an elimination tree.

        Parameters
        ----------
        parent : array_like of int
            The parent vector of the elimination tree.
        )pbdoc"
    );

    m.def("rowcnt",
        [] (
            const py::object& A_scipy,
            const std::vector<cs::csint>& parent,
            const std::vector<cs::csint>& post
        ) {
            cs::CSCMatrix A = csc_from_scipy(A_scipy);
            return cs::rowcnt(A, parent, post);
        },
        py::arg("A"), py::arg("parent"), py::arg("post"),
        R"pbdoc(
        Compute the row counts for Cholesky factorization.

        Parameters
        ----------
        A : (M, N) CSCMatrix
            The matrix to factorize.
        parent : array_like of int
            The parent vector of the elimination tree.
        post : array_like of int
            The postordering of the elimination tree.
        )pbdoc"
    );

    m.def("chol_colcounts",
        [] (const py::object& A_scipy, bool ata=false) {
            const cs::CSCMatrix A = csc_from_scipy(A_scipy);
            return cs::chol_colcounts(A, ata);
        },
        py::arg("A"), py::arg("ATA")=false,
        R"pbdoc(
        Compute the column counts for Cholesky factorization.

        Parameters
        ----------
        A : (M, N) CSCMatrix
            The matrix to factorize.
        ATA : bool, optional
            If True, compute the column counts for :math:`A^T A`. Default is False.
        )pbdoc"
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
        py::arg("A"), py::arg("order")="Natural", py::arg("use_postorder")=false,
        R"pbdoc(
        Perform Cholesky factorization of a sparse matrix.

        Parameters
        ----------
        A : (M, N) CSCMatrix
            The matrix to factorize.
        order : str, optional
            The ordering strategy to use. Options are "Natural", "AMD", "COLAMD".
            Default is "Natural".
        use_postorder : bool, optional
            Whether to use postordering in the factorization. Default is False.
        )pbdoc"
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
        py::arg("use_postorder")=false,
        R"pbdoc(
        Perform symbolic Cholesky factorization of a sparse matrix.

        Parameters
        ----------
        A : (M, N) CSCMatrix
            The matrix to factorize.
        order : str, optional
            The ordering strategy to use. Options are "Natural", "AMD", "COLAMD".
            Default is "Natural".
        use_postorder : bool, optional
            Whether to use postordering in the factorization. Default is False.
        )pbdoc"
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
        py::arg("use_postorder")=false,
        R"pbdoc(
        Perform left-looking Cholesky factorization of a sparse matrix,
        given the non-zero pattern.

        Parameters
        ----------
        A : (M, N) CSCMatrix
            The matrix to factorize.
        order : str, optional
            The ordering strategy to use. Options are "Natural", "AMD", "COLAMD".
            Default is "Natural".
        use_postorder : bool, optional
            Whether to use postordering in the factorization. Default is False.
        )pbdoc"
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
        py::arg("use_postorder")=false,
        R"pbdoc(
        Perform up-looking Cholesky factorization of a sparse matrix, given the
        non-zero pattern.

        Parameters
        ----------
        A : (M, N) CSCMatrix
            The matrix to factorize.
        order : str, optional
            The ordering strategy to use. Options are "Natural", "AMD", "COLAMD".
            Default is "Natural".
        use_postorder : bool, optional
            Whether to use postordering in the factorization. Default is False.
        )pbdoc"
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
        py::arg("L"),
        py::arg("update"),
        py::arg("C"),
        py::arg("parent"),
        R"pbdoc(
        Perform a rank-k update or downdate of a Cholesky factorization.

        Parameters
        ----------
        L : (N, N) CSCMatrix
            The Cholesky factor to be updated or downdated.
        update : bool
            If True, perform an update; if False, perform a downdate.
        C : (N, k) CSCMatrix
            The matrix used for the rank-k update/downdate.
        parent : array_like of int
            The parent vector of the elimination tree.
        )pbdoc"
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
        py::arg("use_postorder")=false,
        R"pbdoc(
        Perform QR factorization of a sparse matrix.

        Parameters
        ----------
        A : (M, N) CSCMatrix
            The matrix to factorize.
        order : str, optional
            The ordering strategy to use. Options are "Natural", "AMD", "COLAMD".
            Default is "Natural".
        use_postorder : bool, optional
            Whether to use postordering in the factorization. Default is False.
        )pbdoc"
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
        py::arg("alpha")=1.0,
        R"pbdoc(
        Perform symbolic LU factorization of a sparse matrix.

        Parameters
        ----------
        A : (M, N) CSCMatrix
            The matrix to factorize.
        order : str, optional
            The ordering strategy to use. Options are "Natural", "AMD", "COLAMD".
            Default is "Natural".
        qr_bound : bool, optional
            Whether to use a QR-based bound for allocating memory. Default is False.
        alpha : float, optional
            The multiple of the memory estimate, :math:`\alpha(4 |A| + N)`, to
            allocate. Default is 1.0.
        )pbdoc"
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
        py::arg("tol")=1.0,
        R"pbdoc(
        Perform LU factorization of a sparse matrix.

        Parameters
        ----------
        A : (M, N) CSCMatrix
            The matrix to factorize.
        order : str, optional
            The ordering strategy to use. Options are "Natural", "AMD", "COLAMD".
            Default is "Natural".
        tol : float, optional
            The pivot tolerance. Default is 1.0 (partial pivoting). Smaller
            values increasingly rely on the existing diagonal elements, which
            decreases fill-in but may lead to numerical instability.
        )pbdoc"
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
        py::arg("order")="APlusAT",
        R"pbdoc(
        Compute the approximate minimum degree (AMD) ordering of a sparse matrix.

        Parameters
        ----------
        A : (M, N) CSCMatrix
            The matrix for which to compute the ordering.
        order : str, optional
            The form of the matrix to use for ordering. Options are:

            * `Natural`: natural ordering (no permutation)
            * `APlusAT`: AMD ordering of :math:`A + A^T`. This option is
              appropriate for Cholesky factorization, or LU factorization with
              substantial entries on the diagonal and a roughly symmetric
              nonzero pattern. If `lu` is used, `tol < 1.0` should be used to
              prefer the diagonal entries for partial pivoting.
            * `ATANoDenseRows`: AMD ordering of :math:`A^T A`, with "dense"
              rows removed from `A`. This option is appropriate for LU
              factorization of unsymmetric matrices and produces a similar
              ordering to that of `COLAMD`.
            * `ATA`: AMD ordering of :math:`A^T A`. This option is appropriate
              for QR factorization, or for LU factorization if `A` has no
              "dense" rows. A "dense" row is defined as a row with more than
              :math:`10 \sqrt{N}` nonzeros, where `N` is the number of columns
              in the matrix.

        Returns
        -------
        p : (N,) np.ndarray of int
            The AMD ordering permutation vector.
        )pbdoc"
    );

    m.def("maxtrans_r",
        [] (const py::object& A_scipy, cs::csint seed=0) {
            const cs::CSCMatrix A = csc_from_scipy(A_scipy);
            return cs::detail::maxtrans_r(A, seed);
        },
        py::arg("A"),
        py::arg("seed")=0,
        R"pbdoc(
        Compute a maximum transversal of a sparse matrix using a recursive
        algorithm.

        Parameters
        ----------
        A : (M, N) CSCMatrix
            The input matrix.

        Returns
        -------
        p : (N,) np.ndarray of int
            The column indices of the maximum transversal. If column ``j`` is
            not matched, then ``p[j] = -1``.
        )pbdoc"
    );

    m.def("maxtrans",
        [] (const py::object& A_scipy, cs::csint seed=0) {
            const cs::CSCMatrix A = csc_from_scipy(A_scipy);
            return cs::maxtrans(A, seed);
        },
        py::arg("A"),
        py::arg("seed")=0,
        R"pbdoc(
        Compute a maximum transversal of a sparse matrix.

        Parameters
        ----------
        A : (M, N) CSCMatrix
            The input matrix.
        seed : int, optional
            Seed for the random number generator. Default is 0. If `seed` is 0,
            no permutation is applied. If `seed` is -1, the permutation is the
            reverse of the identity. Otherwise, a random permutation is
            generated.

        )pbdoc"
    );

    m.def("dmperm",
        [] (const py::object& A_scipy, cs::csint seed=0) {
            cs::CSCMatrix A = csc_from_scipy(A_scipy);
            return cs::dmperm(A, seed);
        },
        py::arg("A"),
        py::arg("seed")=0,
        R"pbdoc(
        Compute the Dulmage-Mendelsohn decomposition of a sparse matrix.

        Parameters
        ----------
        A : (M, N) CSCMatrix
            The input matrix.
        seed : int, optional
            Seed for the random number generator. Default is 0. If `seed` is 0,
            no permutation is applied. If `seed` is -1, the permutation is the
            reverse of the identity. Otherwise, a random permutation is
            generated.

        Returns
        -------
        DMPermResult
            A data structure containing the row and column permutations, as well
            as the block boundaries of the decomposition.
        )pbdoc"
    );

    m.def("scc",
        [] (const py::object& A_scipy) {
            cs::CSCMatrix A = csc_from_scipy(A_scipy);
            return cs::scc(A);
        },
        py::arg("A"),
        R"pbdoc(
        Compute the strongly connected components of a matrix.

        Parameters
        ----------
        A : (M, N) CSCMatrix
            The input matrix.

        Returns
        -------
        SCCResult
            A data structure containing the component labels and the number of
            components.
        )pbdoc"
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

            std::vector<cs::csint> xi;
            xi.reserve(A.shape()[1]);
            cs::reach(A, b, 0, xi);
            return xi;
        },
        py::arg("A"),
        py::arg("b"),
        R"pbdoc(
        Compute the reachability set of a sparse matrix and a right-hand side vector.

        Parameters
        ----------
        A : (M, N) CSCMatrix
            The input matrix.
        b : (M, 1) CSCMatrix
            The right-hand side vector.

        Returns
        -------
        xi : np.ndarray of int
            The row indices that are reachable.
        )pbdoc"
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
        py::arg("A"),
        py::arg("b"),
        R"pbdoc(
        Compute the reachability set of a sparse matrix and a right-hand side
        vector, using a recursive algorithm.

        Parameters
        ----------
        A : (M, N) CSCMatrix
            The input matrix.
        b : (M, 1) CSCMatrix
            The right-hand side vector.

        Returns
        -------
        xi : np.ndarray of int
            The row indices that are reachable.
        )pbdoc"
    );

    //--------------------------------------------------------------------------
    //      Triangular Solvers
    //--------------------------------------------------------------------------
    m.def(
        "lsolve",
        make_simple_solver(
            [](const cs::CSCMatrix& L, const std::vector<double>& B) {
                return cs::lsolve(L, B);
            },
            [](const cs::CSCMatrix& L, const cs::CSCMatrix& B) {
                return cs::lsolve(L, B);
            }
        ),
        py::arg("L"),
        py::arg("B"),
        R"pbdoc(
        Solve a lower-triangular system of equations `L x = B`.

        Parameters
        ----------
        L : (N, N) CSCMatrix
            The lower-triangular matrix.
        B : (N,) or (N, K) array_like or sparse matrix
            The right-hand side vector or matrix.

        Returns
        -------
        x : (N,) or (N, K) np.ndarray or sparse matrix
            The solution vector or matrix. The type matches that of `B`.
        )pbdoc"
    );

    m.def(
        "usolve",
        make_simple_solver(
            [](const cs::CSCMatrix& U, const std::vector<double>& B) {
                return cs::usolve(U, B);
            },
            [](const cs::CSCMatrix& U, const cs::CSCMatrix& B) {
                return cs::usolve(U, B);
            }
        ),
        py::arg("U"),
        py::arg("B"),
        R"pbdoc(
        Solve a upper-triangular system of equations `U x = B`.

        Parameters
        ----------
        U : (N, N) CSCMatrix
            The upper-triangular matrix.
        B : (N,) or (N, K) array_like or sparse matrix
            The right-hand side vector or matrix.

        Returns
        -------
        x : (N,) or (N, K) np.ndarray or sparse matrix
            The solution vector or matrix. The type matches that of `B`.
        )pbdoc"
    );

    m.def("ltsolve", make_dense_solver(&cs::ltsolve), py::arg("L"), py::arg("B"),
        R"pbdoc(
        Solve a transposed lower-triangular system of equations `L^T x = B`.

        Parameters
        ----------
        L : (N, N) CSCMatrix
            The lower-triangular matrix.
        B : (N,) or (N, K) array_like or sparse matrix
            The right-hand side vector or matrix.

        Returns
        -------
        x : (N,) or (N, K) np.ndarray or sparse matrix
            The solution vector or matrix. The type matches that of `B`.
        )pbdoc"
    );

    m.def("utsolve", make_dense_solver(&cs::utsolve), py::arg("U"), py::arg("B"),
        R"pbdoc(
        Solve a transposed upper-triangular system of equations `U^T x = B`.

        Parameters
        ----------
        U : (N, N) CSCMatrix
            The upper-triangular matrix.
        B : (N,) or (N, K) array_like or sparse matrix
            The right-hand side vector or matrix.

        Returns
        -------
        x : (N,) or (N, K) np.ndarray or sparse matrix
            The solution vector or matrix. The type matches that of `B`.
        )pbdoc"
    );

    m.def("lsolve_opt", make_dense_solver(&cs::lsolve_opt), py::arg("L"), py::arg("B"),
        R"pbdoc(
        Solve a lower-triangular system of equations `L x = B`, with an
        optimized algorithm.

        Parameters
        ----------
        L : (N, N) CSCMatrix
            The lower-triangular matrix.
        B : (N,) or (N, K) array_like or sparse matrix
            The right-hand side vector or matrix.

        Returns
        -------
        x : (N,) or (N, K) np.ndarray or sparse matrix
            The solution vector or matrix. The type matches that of `B`.
        )pbdoc"
    );

    m.def("usolve_opt", make_dense_solver(&cs::usolve_opt), py::arg("U"), py::arg("B"),
        R"pbdoc(
        Solve an upper-triangular system of equations `U x = B`, with an
        optimized algorithm.

        Parameters
        ----------
        U : (N, N) CSCMatrix
            The upper-triangular matrix.
        B : (N,) or (N, K) array_like or sparse matrix
            The right-hand side vector or matrix.

        Returns
        -------
        x : (N,) or (N, K) np.ndarray or sparse matrix
            The solution vector or matrix. The type matches that of `B`.
        )pbdoc"
    );

    //--------------------------------------------------------------------------
    //      Full Matrix Solvers
    //--------------------------------------------------------------------------
    m.def("chol_solve",
        make_chol_solver(
            // dense solver
            [](
                const cs::CSCMatrix& A,
                const std::vector<double>& B,
                cs::AMDOrder order
            ) {
                return cs::chol_solve(A, B, order);
            },
            // sparse solver
            [](
                const cs::CSCMatrix& A,
                const cs::CSCMatrix& B,
                cs::AMDOrder order
            ) {
                return cs::chol_solve(A, B, order);
            }
        ),
        py::arg("A"),
        py::arg("B"),
        py::arg("order")="APlusAT",  // CSparse default is "APlusAT"
        R"pbdoc(
        Solve a system of equations `A x = B` using Cholesky factorization.

        Parameters
        ----------
        A : (N, N) CSCMatrix
            The symmetric positive definite matrix.
        B : (N,) or (N, K) array_like or sparse matrix
            The right-hand side vector or matrix.
        order : str, optional
            The ordering strategy to use. Options are "Natural", "AMD", "COLAMD",
            "APlusAT". Default is "APlusAT".

        Returns
        -------
        x : (N,) or (N, K) np.ndarray or sparse matrix
            The solution vector or matrix. The type matches that of `B`.
        )pbdoc"
    );

    m.def("qr_solve",
        make_chol_solver(
            // dense solver
            [](
                const cs::CSCMatrix& A,
                const std::vector<double>& B,
                cs::AMDOrder order
            ) {
                return cs::qr_solve(A, B, order).x;
            },
            // sparse solver
            [](
                const cs::CSCMatrix& A,
                const cs::CSCMatrix& B,
                cs::AMDOrder order
            ) {
                return cs::qr_solve(A, B, order).x;
            }
        ),
        py::arg("A"),
        py::arg("B"),
        py::arg("order")="ATA",  // CSparse default is "ATA"
        R"pbdoc(
        Solve a system of equations `A x = B` using QR factorization.

        Parameters
        ----------
        A : (M, N) CSCMatrix
            The matrix.
        B : (M,) or (M, K) array_like or sparse matrix
            The right-hand side vector or matrix.
        order : str, optional
            The ordering strategy to use. Options are "Natural", "AMD", "COLAMD",
            "ATA". Default is "ATA".

        Returns
        -------
        x : (N,) or (N, K) np.ndarray or sparse matrix
            The solution vector or matrix. The type matches that of `B`.

        Notes
        -----
        If `A` has full column rank, this function returns th least-squares
        solution to the system of equations. Otherwise, it returns
        a minimum-norm solution, which may not be unique.
        )pbdoc"
    );

    m.def("lu_solve",
        make_lu_solver(
            // dense solver
            [](
                const cs::CSCMatrix& A,
                const std::vector<double>& B,
                cs::AMDOrder order,
                double tol,
                cs::csint ir_steps
            ) {
                return cs::lu_solve(A, B, order, tol, ir_steps);
            },
            // sparse solver
            [](
                const cs::CSCMatrix& A,
                const cs::CSCMatrix& B,
                cs::AMDOrder order,
                double tol,
                cs::csint ir_steps
            ) {
                return cs::lu_solve(A, B, order, tol, ir_steps);
            }
        ),
        py::arg("A"),
        py::arg("B"),
        py::arg("order")="ATANoDenseRows",  // CSparse default is "ATANoDenseRows"
        py::arg("tol")=1.0,
        py::arg("ir_steps")=0,
        R"pbdoc(
        Solve a system of equations `A x = B` using LU factorization.

        Parameters
        ----------
        A : (M, N) CSCMatrix
            The matrix.
        B : (M,) or (M, K) array_like or sparse matrix
            The right-hand side vector or matrix.
        order : str, optional
            The ordering strategy to use. Options are "Natural", "AMD", "COLAMD",
            "ATANoDenseRows". Default is "ATANoDenseRows".
        tol : float, optional
            The pivot tolerance. Default is 1.0 (partial pivoting). Smaller
            values increasingly rely on the diagonal entries for partial
            pivoting, which decreases fill-in but may lead to numerical
            instability.
        ir_steps : int, optional
            The number of iterative refinement steps to perform. Default is 0.

        Returns
        -------
        x : (N,) or (N, K) np.ndarray or sparse matrix
            The solution vector or matrix. The type matches that of `B`.
        )pbdoc"
    );

    m.def("spsolve",
        make_simple_solver(
            [](const cs::CSCMatrix& A, const std::vector<double>& B) {
                return cs::spsolve(A, B);
            },
            [](const cs::CSCMatrix& A, const cs::CSCMatrix& B) {
                return cs::spsolve(A, B);
            }
        ),
        py::arg("A"),
        py::arg("B"),
        R"pbdoc(
        Solve a system of equations `A x = B` using sparse direct methods.

        Parameters
        ----------
        A : (M, N) CSCMatrix
            The matrix.
        B : (M,) or (M, K) array_like or sparse matrix
            The right-hand side vector or matrix.

        Returns
        -------
        x : (N,) or (N, K) np.ndarray or sparse matrix
            The solution vector or matrix. The type matches that of `B`.

        See Also
        --------
        lsolve : Solve a lower-triangular system.
        usolve : Solve an upper-triangular system.
        chol_solve : Solve a system using Cholesky factorization.
        qr_solve : Solve a system using QR factorization.
        lu_solve : Solve a system using LU factorization.
        )pbdoc"
    );

}  // PYBIND11_MODULE


/*==============================================================================
 *============================================================================*/
