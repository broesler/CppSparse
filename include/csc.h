//==============================================================================
//    File: csc.h
// Created: 2024-10-09 20:57
//  Author: Bernie Roesler
//
//  Description: Implements the compressed sparse column matrix class.
//
//==============================================================================

#ifndef _CSPARSE_CSC_H_
#define _CSPARSE_CSC_H_

#include <cassert>
#include <functional>
#include <iostream>
#include <string>
#include <string_view>
#include <sstream>
#include <vector>

#include "types.h"

namespace cs {

class CSCMatrix
{
    // Private members
    static constexpr std::string_view format_desc_ = "Compressed Sparse Column";
    std::vector<double> v_;  // numerical values, size nzmax
    std::vector<csint> i_;   // row indices, size nzmax
    std::vector<csint> p_;   // column pointers (CSC size N_);
    csint M_ = 0;            // number of rows
    csint N_ = 0;            // number of columns
    bool has_sorted_indices_ = false;
    bool has_canonical_format_ = false;

    /** Print elements of the matrix between `start` and `end`.
     *
     * @param ss          the output string stream
     * @param start, end  print the all elements where `p ∈ [start, end]`, counting
     *        column-wise.
     */
    void write_elems_(std::stringstream& ss, csint start, csint end) const;

    public:
        friend class COOMatrix;

        /** 
         * @typedef KeepFunc
         * @brief A boolean function pointer type that acts on an element of
         * a matrix.
         *
         * This type is used by the function `CSCMatrix::fkeep`. If `fk` returns
         * `true` for `A(i, j)`, that element will be kept in the matrix.
         *
         * @param i, j  the row and column indices of the element
         * @param Aij  the value of the element `A(i, j)`
         *
         * @return keep  a boolean that is true if the element `A(i, j)` should
         *         be kept in the matrix.
         */
        using KeepFunc = std::function<bool(csint i, csint j, double Aij)>;

        //----------------------------------------------------------------------
        //        Constructors
        //----------------------------------------------------------------------
        CSCMatrix();

        /** Construct a CSCMatrix from arrays of values and coordinates.
         *
         * The entries are *not* sorted in any order, and duplicates are allowed. Any
         * duplicates will be summed.
         *
         * The matrix shape `(M, N)` will be inferred from the maximum indices given.
         *
         * @param data the values of the entries in the matrix
         * @param indices row indices of each element.
         * @param indptr array indices of the start of each column in `indices`. The
         *        first `indptr` element is always 0.
         *
         * @return a new CSCMatrix object
         */
        CSCMatrix(
            const std::vector<double>& data,
            const std::vector<csint>& indices,
            const std::vector<csint>& indptr,
            const Shape& shape
        );

        /** Allocate a CSCMatrix for a given shape and number of non-zeros.
         *
         * @param M, N  integer dimensions of the rows and columns
         * @param nzmax integer capacity of space to reserve for non-zeros
         * @param values if `true`, allocate space for the values array
         */
        CSCMatrix(csint M, csint N, csint nzmax=0, bool values=true);

        /** Convert a coordinate format matrix to a compressed sparse column matrix in
         * canonical format.
         *
         * The columns are guaranteed to be sorted, no duplicates are allowed, and no
         * numerically zero entries are allowed.
         *
         * This function takes O(M + N + nnz) time.
         *
         * See: Davis, Exercises 2.2, 2.9.
         *
         * @return a copy of the `COOMatrix` in canonical CSC format.
         */
        CSCMatrix(const COOMatrix& A);  // Exercise 2.2

        /** Create a sparse copy of a dense matrix in column-major fomr.
         *
         * See: Davis, Exercise 2.16.
         *
         * @param A a dense matrix in column-major form
         * @param M, N the size of the matrix
         *
         * @return C a compressed sparse column version of the matrix
         */
        CSCMatrix(const std::vector<double>& A, csint M, csint N);

        /** Reallocate a CSCMatrix to a new number of non-zeros.
         *
         * @param A      matrix to be resized
         * @param nzmax  maximum number of non-zeros. If `nzmax <= A.nzmax()`,
         *        then `nzmax` will be set to `A.nnz()`.
         *
         * @return A     a reference to the input object for method chaining.
         */
        CSCMatrix& realloc(csint nzmax=0);

        //----------------------------------------------------------------------
        //        Accessors
        //----------------------------------------------------------------------
        csint nnz() const;                   // number of non-zeros
        csint nzmax() const;                 // maximum number of non-zeros
        Shape shape() const;  // the dimensions of the matrix

        const std::vector<csint>& indices() const;     // indices and data
        const std::vector<csint>& indptr() const;
        const std::vector<double>& data() const;

        /** Convert a CSCMatrix to canonical format in-place.
         *
         * The columns are guaranteed to be sorted, no duplicates are allowed, and no
         * numerically zero entries are allowed.
         *
         * This function takes O(M + N + nnz) time.
         *
         * See: Davis, Exercise 2.9.
         *
         * @return a reference to itself for method chaining.
         */
        CSCMatrix& to_canonical();

        bool has_sorted_indices() const;
        bool has_canonical_format() const;

        /** Returns true if `A(i, j) == A(i, j)` for all `i, j`.
        *
        * See: Davis, Exercise 2.13.
        *
        * @return true if the matrix is symmetric.
        */
        bool is_symmetric() const;

        /** Return the value of the requested element.
         *
         * This function takes O(log M) time if the columns are sorted, and O(M) time
         * if they are not.
         *
         * @param i, j the row and column indices of the element to access.
         *
         * @return the value of the element at `(i, j)`.
         */
        const double operator()(csint i, csint j) const;

        /** Return a reference to the value of the requested element for use in
         * assignment, e.g. `A(i, j) = 56.0`.
         *
         * This function takes O(log M) time if the columns are sorted, and O(M)
         * time if they are not.
         *
         * @param i, j the row and column indices of the element to access.
         *
         * @return a reference to the value of the element at `(i, j)`.
         */
        double& operator()(csint i, csint j);

        /** Assign a value to a specific element in the matrix.
         *
         * This function takes O(log M) time if the columns are sorted, and O(M) time
         * if they are not.
         *
         * See: Davis, Exercise 2.25 assign by index.
         *
         * @param i, j the row and column indices of the element to access.
         * @param v the value to be assigned.
         *
         * @return a reference to itself for method chaining.
         */
        CSCMatrix& assign(csint i, csint j, double v);

        /** Assign a dense matrix to the CSCMatrix at the specified locations.
         *
         * See: Davis, Exercise 2.25.
         *
         * @param rows, cols the row and column indices of the elements to access.
         * @param C the dense matrix to be assigned.
         *
         * @return a reference to itself for method chaining.
         */
        CSCMatrix& assign(
            const std::vector<csint>& i,
            const std::vector<csint>& j,
            const std::vector<double>& C  // dense column-major
        );

        /** Assign a sparse matrix to the CSCMatrix at the specified locations.
         *
         * See: Davis, Exercise 2.25.
         *
         * @param rows, cols the row and column indices of the elements to access.
         * @param C the sparse matrix to be assigned.
         *
         * @return a reference to itself for method chaining.
         */
        CSCMatrix& assign(
            const std::vector<csint>& rows,
            const std::vector<csint>& cols,
            const CSCMatrix& C
        );

        /** Insert a single element at a specified location.
         *
         * This function is a helper for assign and operator().
         *
         * @param i, j the row and column indices of the element to access.
         * @param v the value to be assigned.
         * @param p the pointer to the column in the matrix.
         *
         * @return a reference to the inserted value.
         */
        double& insert(csint i, csint j, double v, csint p);

        //----------------------------------------------------------------------
        //        Format Conversions
        //----------------------------------------------------------------------
        /** Convert a compressed sparse column matrix to a coordinate (triplet) format
         * matrix.
         *
         * See: Davis, Exercise 2.2, Matlab `find`.
         *
         * @return a copy of the `CSCMatrix` in COO (triplet) format.
         */
        COOMatrix tocoo() const;

        /** Convert a CSCMatrix to a dense column-major array.
         *
         * See: Davis, Exercise 2.16 (inverse)
         *
         * @param order the order of the array, either 'F' for Fortran (column-major)
         *       or 'C' for C (row-major).
         *
         * @return a copy of the matrix as a dense column-major array.
         */
        std::vector<double> to_dense_vector(const char order='F') const;

        /** Convert a CSCMatrix to a double if it is a 1x1 matrix.
         *
         * @return the value of the matrix if it is a 1x1 matrix.
         */
        operator const double() const {
            assert((M_ == 1) && (N_ == 1));
            return v_[0];
        }

        /** Transpose the matrix as a copy.
        *
        * This operation can be viewed as converting a Compressed Sparse Column matrix
        * into a Compressed Sparse Row matrix.
        *
        * This function takes
        *   - O(N) extra space for the workspace
        *   - O(M + N + nnz) time
        *       == nnz column counts + N columns * M potential non-zeros per column
        *
        * @param values if `true`, allocate space for the values array.
        *
        * @return new CSCMatrix object with transposed rows and columns.
        */
        CSCMatrix transpose(bool values=true) const;
        CSCMatrix T() const;  // transpose a copy (alias)

        /** Sort rows and columns in a copy via two transposes.
         *
         * See: Davis, Exercise 2.7.
         *
         * @return C  a copy of the matrix with sorted columns.
         */
        CSCMatrix tsort() const;

        /** Sort rows and columns in place using std::sort.
         *
         * See: Davis, Exercise 2.8.
         *
         * This function takes
         *   - O(3*M) extra space ==
         *       2 workspaces for row indices and values + vector of sorted indices
         *   - O(N * M log M + nnz) time ==
         *       sort a length M vector for each of N columns
         *
         * @return a reference to the object for method chaining
         */
        CSCMatrix& qsort();

        /** Sort rows and columns in place two transposes, but more efficiently than
         * calling `transpose` twice.
         *
         * See: Davis, Exercise 2.11.
         *
         * This function takes O(M) extra space and O(M * N + nnz) time.
         *
         * @return A  a reference to the matrix, now with sorted columns.
         */
        CSCMatrix& sort();

        /** Sum duplicate entries in place.
         *
         * This function takes
         *   - O(N) extra space for the workspace
         *   - O(nnz) time
         *
         * @return a reference to the object for method chaining
         */
        CSCMatrix& sum_duplicates();

        /** Keep matrix entries for which `fkeep` returns true, remove others.
        *
        * @param fk  a boolean function that acts on each element. If `fk`
        *        returns `true`, that element will be kept in the matrix. 
        *
        * @return a reference to the object for method chaining.
        */
        CSCMatrix& fkeep(KeepFunc fk);

        // Overload for copies
        /** Keep matrix entries for which `fkeep` returns true, remove others.
        *
        * @param fk  a boolean function that acts on each element. If `fk`
        *        returns `true`, that element will be kept in the matrix. 
        *
        * @return a copy of the matrix with entries removed.
        */
        CSCMatrix fkeep(KeepFunc fk) const;

        /** Drop any exactly zero entries from the matrix.
         *
         * This function takes O(nnz) time.
         *
         * @return a reference to the object for method chaining
         */
        CSCMatrix& dropzeros();

        /** Drop any entries within `tol` of zero.
         *
         * This function takes O(nnz) time.
         *
         * @param tol the tolerance against which to compare the absolute value of the
         *        matrix entries.
         *
         * @return a reference to the object for method chaining
         */
        CSCMatrix& droptol(double tol=1e-15);

        /** Keep any entries within the specified band, in-place.
         *
         * See: Davis, Exercise 2.15.
         *
         * @param kl, ku  the lower and upper diagonals within which to keep entries.
         *        The main diagonal is 0, with sub-diagonals < 0, and
         *        super-diagonals > 0.
         *
         * @return a reference to the matrix with entries removed.
         */
        CSCMatrix& band(csint kl, csint ku);

        /** Keep any entries within the specified band.
        *
        * @param kl, ku  the lower and upper diagonals within which to keep entries.
        *        The main diagonal is 0, with sub-diagonals < 0, and
        *        super-diagonals > 0.
        *
        * @return a copy of the matrix with entries removed.
        */
        CSCMatrix band(csint kl, csint ku) const;

        //----------------------------------------------------------------------
        //        Math Operations
        //----------------------------------------------------------------------
        /** Matrix-vector multiply `y = Ax + y`.
         *
         * @param x  a dense multiplying vector
         * @param y  a dense adding vector which will be used for the output
         *
         * @return y a copy of the updated vector
         */
        std::vector<double> gaxpy(
            const std::vector<double>& x,
            const std::vector<double>& y
        ) const;

        /** Matrix transpose-vector multiply `y = A.T x + y`.
         *
         * See: Davis, Exercise 2.1. Compute \f$ A^T x + y \f$ without explicitly
         * computing the transpose.
         *
         * @param x  a dense multiplying vector
         * @param y[in,out]  a dense adding vector which will be used for the output
         *
         * @return y a copy of the updated vector
         */
        std::vector<double> gatxpy(
            const std::vector<double>& x,
            const std::vector<double>& y
        ) const;

        /** Matrix-vector multiply `y = Ax + y` symmetric A (\f$ A = A^T \f$).
         *
         * See: Davis, Exercise 2.3.
         *
         * @param x  a dense multiplying vector
         * @param y  a dense adding vector which will be used for the output
         *
         * @return y a copy of the updated vector
         */
        std::vector<double> sym_gaxpy(
            const std::vector<double>& x,
            const std::vector<double>& y
        ) const;

        /** Matrix multiply `Y = AX + Y` column-major dense matrices `X` and `Y`.
         *
         * See: Davis, Exercise 2.27(a).
         *
         * @param X  a dense multiplying matrix in column-major order
         * @param[in,out] Y  a dense adding matrix which will be used for the output
         *
         * @return Y a copy of the updated matrix
         */
        std::vector<double> gaxpy_col(
            const std::vector<double>& X,
            const std::vector<double>& Y
        ) const;

        /** Matrix multiply `Y = AX + Y` for row-major dense matrices `X` and `Y`.
        *
        * See: Davis, Exercise 2.27(b).
        *
        * @param X  a dense multiplying matrix in row-major order
        * @param[in,out] Y  a dense adding matrix which will be used for the output
        *
        * @return Y a copy of the updated matrix
        */
        std::vector<double> gaxpy_row(
            const std::vector<double>& X,
            const std::vector<double>& Y
        ) const;

        /** Matrix multiply `Y = AX + Y` column-major dense matrices `X` and
         * `Y`, but operate on blocks of columns.
         *
         * See: Davis, Exercise 2.27(c).
         *
         * @param X  a dense multiplying matrix in column-major order
         * @param[in,out] Y  a dense adding matrix which will be used for the output
         *
         * @return Y a copy of the updated matrix
         */
        std::vector<double> gaxpy_block(
            const std::vector<double>& X,
            const std::vector<double>& Y
        ) const;

        /** Matrix multiply `Y = A.T X + Y` column-major dense matrices `X` and `Y`.
         *
         * See: Davis, Exercise 2.28(a).
         *
         * @param X  a dense multiplying matrix in column-major order
         * @param[in,out] Y  a dense adding matrix which will be used for the output
         *
         * @return Y a copy of the updated matrix
         */
        std::vector<double> gatxpy_col(
            const std::vector<double>& X,
            const std::vector<double>& Y
        ) const;

        /** Matrix multiply `Y = A.T X + Y` for row-major dense matrices `X` and `Y`.
         *
         * See: Davis, Exercise 2.27(b).
         *
         * @param X  a dense multiplying matrix in row-major order
         * @param[in,out] Y  a dense adding matrix which will be used for the output
         *
         * @return Y a copy of the updated matrix
         */
        std::vector<double> gatxpy_row(
            const std::vector<double>& X,
            const std::vector<double>& Y
        ) const;

        /** Matrix multiply `Y = A.T X + Y` column-major dense matrices `X` and
         * `Y`, but operate on blocks of columns.
         *
         * See: Davis, Exercise 2.28(c).
         *
         * @param X  a dense multiplying matrix in column-major order
         * @param[in,out] Y  a dense adding matrix which will be used for the output
         *
         * @return Y a copy of the updated matrix
         */
        std::vector<double> gatxpy_block(
            const std::vector<double>& X,
            const std::vector<double>& Y
        ) const;

        /** Scale the rows and columns of a matrix by \f$ A = RAC \f$, where *R* and *C*
         * are diagonal matrices.
         *
         * See: Davis, Exercise 2.4.
         *
         * @param r, c  vectors of length M and N, respectively, representing the
         * diagonals of R and C, where A is size M-by-N.
         *
         * @return RAC the scaled matrix
         */
        CSCMatrix scale(const std::vector<double>& r, const std::vector<double> c) const;

        /** Matrix-vector right-multiply (see cs_multiply) */
        std::vector<double> dot(const std::vector<double>& x) const;

        /** Scale a matrix by a scalar */
        CSCMatrix dot(const double c) const;

        /** Matrix-matrix multiplication
         *
         * @note This function may *not* return a matrix with sorted columns!
         *
         * @param A, B  the CSC-format matrices to multiply.
         *        A is size M x K, B is size K x N.
         *
         * @return C    a CSC-format matrix of size M x N.
         *         C.nnz() <= A.nnz() + B.nnz().
         */
        CSCMatrix dot(const CSCMatrix& B) const;

        /** Matrix-matrix multiplication with two passes
         *
         * See: Davis, Exercise 2.20.
         *
         * @note This function may *not* return a matrix with sorted columns!
         *
         * @param A, B  the CSC-format matrices to multiply.
         *        A is size M x K, B is size K x N.
         *
         * @return C    a CSC-format matrix of size M x N.
         *         C.nnz() <= A.nnz() + B.nnz().
         */
        CSCMatrix dot_2x(const CSCMatrix& B) const;  // Exercise 2.20

        /** Multiply two sparse column vectors \f$ c = x^T y \f$.
         *
         * See: Davis, Exercise 2.18, `cs_dot`.
         *
         * @param x, y two column vectors stored as a CSCMatrix. The number of columns
         *        in each argument must be 1.
         *
         * @return c  the dot product `x.T() * y`, but computed more efficiently than
         *         the complete matrix dot product.
         */
        double vecdot(const CSCMatrix& y) const;

        /** Matrix-matrix add and subtract */
        CSCMatrix add(const CSCMatrix& B) const;
        CSCMatrix subtract(const CSCMatrix& B) const;

        /** Add two matrices (and optionally scale them) `C = alpha * A + beta * B`.
         *
         * @note This function may *not* return a matrix with sorted columns!
         *
         * @param A, B  the CSC matrices
         * @param alpha, beta  scalar multipliers
         *
         * @return out a CSC matrix
         */
        friend CSCMatrix add_scaled(
            const CSCMatrix& A,
            const CSCMatrix& B,
            double alpha,
            double beta
        );

        /** Add two sparse column vectors \f$ z = x + y \f$.
         *
         * See: Davis, Exercise 2.21
         *
         * @param x, y two column vectors stored as a CSCMatrix. The number of columns
         *        in each argument must be 1.
         *
         * @return z  the sum of the two vectors, but computed more efficiently than
         *         the complete matrix addition.
         */
        friend std::vector<csint> saxpy(
            const CSCMatrix& a,
            const CSCMatrix& b,
            std::vector<csint>& w,
            std::vector<double>& x
        );

        /** Compute `x += beta * A(:, j)`.
         *
         * This function also updates `w`, sets the sparsity pattern in `C._i`,
         * and returns updated `nz`. The values corresponding to `C._i` are
         * accumulated in `x`, and then gathered in the calling function, so
         * that we can account for any duplicate entries.
         *
         * @param j     column index of `A`
         * @param beta  scalar value by which to multiply `A`
         * @param[in,out] w, x  workspace vectors of row indices and values, respectively
         * @param mark  separator index for `w`. All `w[i] < mark`are row indices that
         *              are not yet in `Cj`.
         * @param[in,out] C    CSC matrix where output non-zero pattern is stored
         * @param[in,out] nz   current number of non-zeros in `C`.
         * @param fs    first call to scatter. Default is false to skip the
         *        optimization.
         * @param values if true, copy values from the original matrix,
         *
         * @return nz  updated number of non-zeros in `C`.
         */
        csint scatter(
            csint j,
            double beta,
            std::vector<csint>& w,
            std::vector<double>& x,
            csint mark,
            CSCMatrix& C,
            csint nz,
            bool fs=false,    // Exercise 2.19
            bool values=true  // needed in qr
        ) const;

        //----------------------------------------------------------------------
        //        Permutations
        //----------------------------------------------------------------------
        /** Permute a matrix \f$ C = PAQ \f$.
         *
         * @note In Matlab, this call is `C = A(p, q)`.
         *
         * @param p_inv, q  *inverse* row and (non-inverse) column permutation
         *        vectors. `p_inv` is length `M` and `q` is length `N`,
         *        where `A` is `M`-by-`N`.
         * @param values  if true, copy values from the original matrix,
         *        otherwise, only the structure is copied.
         *
         * @return C  permuted matrix
         */
        CSCMatrix permute(
            const std::vector<csint> p_inv,
            const std::vector<csint> q,
            bool values=true
        ) const;

        /** Permute a symmetric matrix with only the upper triangular part stored.
         *
         * @param p_inv  *inverse* permutation vector. Both rows and columns are
         *        permuted with this vector to retain symmetry.
         * @param values  if true, copy values from the original matrix,
         *        otherwise, only the structure is copied.
         *
         * @return C  permuted matrix
         */
        CSCMatrix symperm(const std::vector<csint> p_inv, bool values=true) const;

        /** Permute and transpose a matrix \f$ C = PA^TQ \f$.
         *
         * See: Davis, Exercise 2.26.
         *
         * @note In Matlab, this call is `C = A(p, q)'`.
         *
         * @param p_inv, q_inv  *inverse* row and column permutation vectors.
         *        `p_inv` is length `M` and `q` is length `N`,
         *        where `A` is `M`-by-`N`.
         * @param values  if true, copy values from the original matrix,
         *        otherwise, only the structure is copied.
         *
         * @return C  permuted and transposed matrix
         */
        CSCMatrix permute_transpose(
            const std::vector<csint>& p_inv,
            const std::vector<csint>& q_inv,
            bool values=true
        ) const;

        /** Permute the rows of a matrix.
         *
         * @note In Matlab, this call is `C = A(p, :)`.
         *
         * @param p_inv  *inverse* row permutation vector. `p_inv` is length `M`.
         * @param values if true, copy values from the original matrix,
         *        otherwise, only the structure is copied.
         *
         * @return C  permuted matrix
         */
        CSCMatrix permute_rows(const std::vector<csint> p_inv, bool values=true) const;

        /** Permute the columns of a matrix.
         *
         * @note In Matlab, this call is `C = A(:, q)`.
         *
         * @param q  column permutation vector. `q` is length `N`.
         * @param values if true, copy values from the original matrix,
         *        otherwise, only the structure is copied.
         *
         * @return C  permuted matrix
         */
        CSCMatrix permute_cols(const std::vector<csint> q, bool values=true) const;

        /** Compute the 1-norm of the matrix (maximum column sum).
         *
         * The 1-norm is defined as \f$ \|A\|_1 = \max_j \sum_{i=1}^{m} |a_{ij}| \f$.
         */
        double norm() const;

        /** Compute the Frobenius norm of the matrix.
         *
         * The Frobenius norm is defined as
         * $$
         *      \|A\|_F = 
         *      \( \sum_{i=1}^{m} \sum_{j=1}^{n} |a_{ij}|^2 \)^{\frac{1}{2}}
         * $$.
         */
        double fronorm() const;

        /** Check a matrix for valid compressed sparse column format.
         *
         * See: Davis, Exercise 2.12 "cs_ok"
         *
         * @param sorted  if true, check if columns are sorted.
         * @param values  if true, check if values exist and are all non-zero.
         *
         * @return true if matrix is valid compressed sparse column format.
         */
        bool is_valid(const bool sorted=false, const bool values=false) const;

        /** Concatenate two matrices horizontally.
         *
         * See: Davis, Exercise 2.22 `cs_hcat`.
         *
         * @note This function may *not* return a matrix with sorted columns!
         *
         * @param A, B  the CSC matrices to concatenate. They must have the same number
         *        of rows.
         *
         * @return C  the concatenated matrix.
         */
        friend CSCMatrix hstack(const CSCMatrix& A, const CSCMatrix& B);

        /** Concatenate two matrices vertically.
         *
         * See: Davis, Exercise 2.22 `cs_hcat`.
         *
         * @note This function may *not* return a matrix with sorted columns!
         *
         * @param A, B  the CSC matrices to concatenate. They must have the same number
         *        of columns.
         *
         * @return C  the concatenated matrix.
         */
        friend CSCMatrix vstack(const CSCMatrix& A, const CSCMatrix& B);

        /** Slice a matrix by row and column with contiguous indices.
         *
         * See: Davis, Exercise 2.23.
         *
         * @param i_start, i_end  the row indices to keep, where `i ∈ [i_start, i_end)`.
         * @param j_start, j_end  the column indices to keep, where `j ∈ [j_start,
         *        j_end)`.
         *
         * @return C  the submatrix A(i_start:i_end, j_start:j_end).
         */
        CSCMatrix slice(
            const csint i_start,
            const csint i_end,
            const csint j_start,
            const csint j_end
        ) const;

        /** Select a submatrix by arbitrary row and column indices.
         *
         * See: Davis, Exercise 2.24.
         *
         * This function takes O(|rows| + |cols|) + O(log M) time if the columns are
         * sorted, and + O(M) time if they are not.
         *
         * @param i, j vectors of the row and column indices to keep. The indices need
         *        not be consecutive, or sorted. Duplicates are allowed.
         *
         * @return C  the submatrix of A of dimension `length(i)`-by-`length(j)`.
         */
        CSCMatrix index(
            const std::vector<csint>& rows,
            const std::vector<csint>& cols
        ) const;

        /** Add empty rows to the top of the matrix.
        *
        * See: Davis, Exercise 2.29.
        *
        * @param k  the number of rows to add.
        *
        * @return C  the matrix with `k` empty rows added to the top.
        */
        CSCMatrix add_empty_top(const csint k) const;

        /** Add empty rows to the bottom of the matrix.
         *
         * See: Davis, Exercise 2.29.
         *
         * @param k  the number of rows to add.
         *
         * @return C  the matrix with `k` empty rows added to the bottom.
         */
        CSCMatrix add_empty_bottom(const csint k) const;

        /** Add empty columns to the left of the matrix.
         *
         * See: Davis, Exercise 2.29.
         *
         * @param k  the number of columns to add.
         *
         * @return C  the matrix with `k` empty columns added to the left.
         */
        CSCMatrix add_empty_left(const csint k) const;

        /** Add empty columns to the right of the matrix.
        *
        * See: Davis, Exercise 2.29.
        *
        * @param k  the number of columns to add.
        *
        * @return C  the matrix with `k` empty columns added to the right.
        */
        CSCMatrix add_empty_right(const csint k) const;

        /** Sum the rows of a matrix.
         *
         * @return out  a vector of length `M` containing the sum of each row.
         */
        std::vector<double> sum_rows() const;

        /** Sum the columns of a matrix.
         *
         * @return out  a vector of length `N` containing the sum of each column.
         */
        std::vector<double> sum_cols() const;

        // double sum() const;
        // std::vector<double> sum(const int axis) const;

        //----------------------------------------------------------------------
        //        Triangular Matrix Solutions
        //----------------------------------------------------------------------
        friend std::vector<double> lsolve(const CSCMatrix& A, const std::vector<double>& b);
        friend std::vector<double> ltsolve(const CSCMatrix& A, const std::vector<double>& b);
        friend std::vector<double> usolve(const CSCMatrix& A, const std::vector<double>& b);
        friend std::vector<double> utsolve(const CSCMatrix& A, const std::vector<double>& b);

        friend std::vector<double> lsolve_opt(const CSCMatrix& A, const std::vector<double>& b);
        friend std::vector<double> usolve_opt(const CSCMatrix& A, const std::vector<double>& b);

        friend std::vector<double> lsolve_rows(const CSCMatrix& A, const std::vector<double>& b);
        friend std::vector<double> usolve_rows(const CSCMatrix& A, const std::vector<double>& b);
        friend std::vector<double> lsolve_cols(const CSCMatrix& A, const std::vector<double>& b);
        friend std::vector<double> usolve_cols(const CSCMatrix& A, const std::vector<double>& b);

        friend std::vector<csint> find_lower_diagonals(const CSCMatrix& A);
        friend std::vector<csint> find_upper_diagonals(const CSCMatrix& A);

        friend std::vector<double> tri_solve_perm(
            const CSCMatrix& A,
            const std::vector<double>& b,
            bool is_upper
        );

        friend TriPerm find_tri_permutation(const CSCMatrix& A);

        friend SparseSolution spsolve(
            const CSCMatrix& A,
            const CSCMatrix& B,
            csint k,
            bool lo
        );

        friend std::vector<csint> reach(const CSCMatrix& A, const CSCMatrix& B, csint k);
        friend std::vector<csint>& dfs(
            const CSCMatrix& A,
            csint j,
            std::vector<bool>& marked,
            std::vector<csint>& xi
        );

        //----------------------------------------------------------------------
        //        Cholesky Decomposition
        //----------------------------------------------------------------------
        friend std::vector<csint> etree(const CSCMatrix& A, bool ata);

        friend std::vector<csint> ereach(
            const CSCMatrix& A,
            csint k,
            const std::vector<csint>& parent
        );

        friend std::vector<csint> ereach_post(
            const CSCMatrix& A,
            csint k,
            const std::vector<csint>& parent
        );

        friend std::vector<csint> ereach_queue(
            const CSCMatrix& A,
            csint k,
            const std::vector<csint>& parent
        );

        friend std::vector<csint> rowcnt(
            const CSCMatrix& A,
            const std::vector<csint>& parent,
            const std::vector<csint>& postorder
        );

        friend void init_ata(
            const CSCMatrix& AT,
            const std::vector<csint>& post,
            std::vector<csint>& head,
            std::vector<csint>& next
        );

        friend std::vector<csint> counts(
            const CSCMatrix& A,
            const std::vector<csint>& parent,
            const std::vector<csint>& postorder,
            bool ata
        );

        friend std::vector<csint> chol_rowcounts(const CSCMatrix& A);
        friend std::vector<csint> chol_colcounts(const CSCMatrix& A, bool ata);

        friend CSCMatrix symbolic_cholesky(const CSCMatrix& A, const Symbolic& S);

        friend CSCMatrix chol(const CSCMatrix& A, const Symbolic& S, double drop_tol);
        friend CSCMatrix& leftchol(const CSCMatrix& A, const Symbolic& S, CSCMatrix& L);
        friend CSCMatrix& rechol(const CSCMatrix& A, const Symbolic& S, CSCMatrix& L);

        friend CSCMatrix ichol(
            const CSCMatrix& A,
            ICholMethod method,
            double drop_tol
        );

        friend CSCMatrix& chol_update(
            CSCMatrix& L,
	        int sigma,
	        const CSCMatrix& w,
	        const std::vector<csint>& parent
        );

        friend CholCounts chol_etree_counts(const CSCMatrix& A);

        friend SparseSolution chol_lsolve(
            const CSCMatrix& L,
            const CSCMatrix& b,
            std::vector<csint> parent
        );

        friend SparseSolution chol_ltsolve(
            const CSCMatrix& L,
            const CSCMatrix& b,
            std::vector<csint> parent
        );

        friend std::vector<csint> topological_order(
            const CSCMatrix& b,
            const std::vector<csint>& parent,
            bool forward
        );

        //----------------------------------------------------------------------
        //        QR Decomposition
        //----------------------------------------------------------------------
        friend std::vector<double> happly(
            const CSCMatrix& V,
            csint j,
            double beta,
            const std::vector<double>& x
        );

        friend void vcount(const CSCMatrix& A, Symbolic& S);

        friend QRResult qr(const CSCMatrix& A, const Symbolic& S);

        //----------------------------------------------------------------------
        //        Printing
        //----------------------------------------------------------------------
        /** Print the matrix in dense format.
         *
         * @param os  a reference to the output stream.
         *
         * @return os  a reference to the output stream.
         */
        void print_dense(std::ostream& os=std::cout) const;

        /** Convert the matrix to a string.
         *
         * @param verbose     if True, print all non-zeros and their coordinates
         * @param threshold   if `nnz > threshold`, print only the first and last
         *        3 entries in the matrix. Otherwise, print all entries.
         */
        std::string to_string(
            bool verbose=false,
            csint threshold=1000
        ) const;

        /** Print the matrix.
         *
         * @param os          the output stream, defaults to std::cout
         * @param verbose     if True, print all non-zeros and their coordinates
         * @param threshold   if `nz > threshold`, print only the first and last
         *        3 entries in the matrix. Otherwise, print all entries.
         */
        void print(
            std::ostream& os=std::cout,
            bool verbose=false,
            csint threshold=1000
        ) const;

};  // class CSCMatrix


/*------------------------------------------------------------------------------
 *          Free Functions
 *----------------------------------------------------------------------------*/
CSCMatrix operator+(const CSCMatrix& A, const CSCMatrix& B);
CSCMatrix operator-(const CSCMatrix& A, const CSCMatrix& B);

std::vector<double> operator*(const CSCMatrix& A, const std::vector<double>& B);
CSCMatrix operator*(const CSCMatrix& A, const CSCMatrix& B);
CSCMatrix operator*(const CSCMatrix& A, const double c);
CSCMatrix operator*(const double c, const CSCMatrix& A);

std::ostream& operator<<(std::ostream& os, const CSCMatrix& A);


}  // namespace cs

#endif  // _CSC_H_

//==============================================================================
//==============================================================================
