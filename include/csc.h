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
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <sstream>
#include <span>
#include <vector>

#include "types.h"
#include "sparse_matrix.h"

namespace cs {

class CSCMatrix : public SparseMatrix
{
public:
    friend class COOMatrix;
    friend class TestCSCMatrix;  // dummy class for testing
    friend struct CholResult;    // for use with CholResult::lsolve

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

    //--------------------------------------------------------------------------
    //         Default Constructor, Copy/Move, Destructor
    //--------------------------------------------------------------------------
    CSCMatrix() = default;

    // Copy/move constructors and assignment operators
    CSCMatrix(const CSCMatrix& other) = default;                 // copy constructor
    CSCMatrix& operator=(const CSCMatrix& other) = default;      // copy assignment

    CSCMatrix(CSCMatrix&& other) noexcept = default;             // move constructor
    CSCMatrix& operator=(CSCMatrix&& other) noexcept = default;  // move assignment

    // Destructor
    virtual ~CSCMatrix() noexcept = default;

    //--------------------------------------------------------------------------
    //        Constructors
    //--------------------------------------------------------------------------
    /** Construct a CSCMatrix from arrays of values and coordinates.
     *
     * The entries are *not* sorted in any order, and duplicates are allowed. Any
     * duplicates will be summed.
     *
     * The matrix shape `(M, N)` will be inferred from the maximum indices given.
     *
     * @param data the values of the entries in the matrix. This vector may
     *        be empty to create a symbolic matrix.
     * @param indices row indices of each element. The length of this array
     *        must be equal to the length of `data`, if `data` is not empty.
     * @param indptr array indices of the start of each column in `indices`.
     *        The first `indptr` element is always 0, and the last element
     *        is always `nnz()`, which is the length of `indices`.
     * @param shape the dimensions of the matrix
     *
     * @return a new CSCMatrix object
     */
    CSCMatrix(
        std::span<const double> data,
        std::span<const csint> indices,
        std::span<const csint> indptr,
        const Shape& shape
    );

    /** Allocate a CSCMatrix for a given shape and number of non-zeros.
     *
     * @param shape  the dimensions of the matrix
     * @param nzmax integer capacity of space to reserve for non-zeros
     * @param values if `true`, allocate space for the values array
     */
    explicit CSCMatrix(const Shape& shape, csint nzmax=0, bool values=true);

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
    explicit CSCMatrix(const COOMatrix& A);  // Exercise 2.2

    /** Create a sparse copy of a dense matrix in column-major fomr.
     *
     * See: Davis, Exercise 2.16.
     *
     * @param A  a dense matrix in column-major form
     * @param shape  the dimensions of the matrix
     * @param order  the order of the dense matrix, either DenseOrder::RowMajor
     *        or DenseOrder::ColMajor
     *
     * @return C a compressed sparse column version of the matrix
     */
    CSCMatrix(
        std::span<const double> A,
        const Shape& shape,
        const DenseOrder order = DenseOrder::ColMajor
    );

    /** Reallocate a CSCMatrix to a new number of non-zeros.
     *
     * @param A      matrix to be resized
     * @param nzmax  maximum number of non-zeros. If `nzmax <= A.nzmax()`,
     *        then `nzmax` will be set to `A.nnz()`.
     */
    virtual void realloc(csint nzmax=0);  // virtual for override in testing

    // --------------------------------------------------------------------------
    //          Accessors
    // --------------------------------------------------------------------------
    virtual csint nnz() const override;    // number of non-zeros
    virtual csint nzmax() const override;  // maximum number of non-zeros
    virtual Shape shape() const override;  // the dimensions of the matrix

    const std::vector<csint>& indices() const;
    const std::vector<csint>& indptr() const;
    virtual const std::vector<double>& data() const override;

    /** Return the number of non-zeros in column j. */
    csint col_length(csint j) const
    {
        return p_[j+1] - p_[j];
    }

    /** Return a span over the row indices of column j. */
    auto row_indices(csint j) const
    {
        return std::span(i_).subspan(p_[j], col_length(j));
    }

    /** Return a span over the values of column j. */
    auto col_values(csint j) const
    {
        return std::span(v_).subspan(p_[j], col_length(j));
    }

    /** Return an iterator over the index "pointers" of column j. */
    auto indptr_range(csint j) const
    {
        return std::views::iota(p_[j], p_[j+1]);
    }

    /** Return an iterator over the indices and values of column j. */
    auto column(csint j) const
    {
        return indptr_range(j) | std::views::transform(
            [this](csint p) {
                return std::pair{i_[p], !v_.empty() ? v_[p] : 0.0};
            }
        );
    }

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

    /** Return -1 if lower triangular, 1 if upper triangular, 0 otherwise.
     *
     * See: Davis, Exercise 2.13.
     *
     * @return -1 if square and lower triangular, 1 if square and upper
     *         triangular, 0 otherwise.
     */
    csint is_triangular() const;

    /** Check if the columns have sorted indices by iteration.
     *
     * This function actually iterates through the columns and checks if the
     * indices are sorted, as opposed to just checking the property
     * `has_sorted_indices_`. This is a brute-force method, and is only
     * intended for internal testing purposes.
     *
     * @return true if the columns are sorted, false otherwise.
     */
    bool _test_sorted() const;

    // -------------------------------------------------------------------------
    //          Item Assignment
    // -------------------------------------------------------------------------
    // Create a proxy class for the matrix item that allows for assignment or
    // lookup on a non-const matrix.
    class ItemProxy {
    private:
        CSCMatrix& A_;
        csint i_;
        csint j_;

        // Allow CSCMatrix to create proxies
        friend class CSCMatrix;

        // Constructor is private so only CSCMatrix can create it.
        ItemProxy(CSCMatrix& A, csint i, csint j) : A_(A), i_(i), j_(j) {}

        // Apply a compound assignment operator: `A(i, j) += v`.
        template <typename BinaryOp>
        ItemProxy& apply_binary_op_(double other, BinaryOp op) {
            GetItemResult res = A_.get_item_(i_, j_);
            A_.set_item_with_op_(i_, j_, other, res.found, res.index, op);
            return *this;
        }

    public:
        // Type conversion: `double v = A(i, j);`
        operator double() const {
            return A_.get_item_(i_, j_).value;
        }

        // Copy constructor (explicitly defined for "+=" etc. operators)
        ItemProxy (const ItemProxy& other)
            : A_(other.A_), i_(other.i_), j_(other.j_) {}

        // Assignment operator: `A(i, j) = v;`
        // Returning a reference to the object allows for method chaining
        // and assignment like `A(i, j) = A(k, l) = v;`
        ItemProxy& operator=(double v) {
            A_.set_item_(i_, j_, v);
            return *this;
        }

        // Copy assignment operator: `A(i, j) = B(i, j);`
        ItemProxy& operator=(const ItemProxy& other) {
            A_.set_item_(i_, j_, other.A_.get_item_(other.i_, other.j_).value);
            return *this;
        }

        // Comparison operators
        auto operator<=>(const ItemProxy& other) const {
            return static_cast<double>(*this) <=> static_cast<double>(other);
        }

        auto operator<=>(double v) const {
            return static_cast<double>(*this) <=> v;
        }

        // Compound assignment operators: `A(i, j) += v;` etc.
        ItemProxy& operator+=(double v) { return apply_binary_op_(v, std::plus<double>()); }
        ItemProxy& operator-=(double v) { return apply_binary_op_(v, std::minus<double>()); }
        ItemProxy& operator*=(double v) { return apply_binary_op_(v, std::multiplies<double>()); }
        ItemProxy& operator/=(double v) {
            if (v == 0.0) {
                throw std::runtime_error("Division by zero");
            }
            return apply_binary_op_(v, std::divides<double>());
        }

        // Increment/decrement operators
        // Pre-in/decrement: `++A(i, j);`
        ItemProxy& operator++() { return apply_binary_op_(1.0, std::plus<double>()); }
        ItemProxy& operator--() { return apply_binary_op_(1.0, std::minus<double>()); }

        // Post-in/decrement: `A(i, j)++;`
        double operator++(int) {
            GetItemResult res = A_.get_item_(i_, j_);
            A_.set_item_with_op_(i_, j_, 1.0, res.found, res.index, std::plus<double>());
            return res.value;
        }

        double operator--(int) {
            GetItemResult res = A_.get_item_(i_, j_);
            A_.set_item_with_op_(i_, j_, 1.0, res.found, res.index, std::minus<double>());
            return res.value;
        }
    };

    /** Return the value of the requested element.
     *
     * This function takes O(log M) time if the columns are sorted, and O(M) time
     * if they are not.
     *
     * @param i, j the row and column indices of the element to access.
     *
     * @return the value of the element at `(i, j)`.
     */
    double operator()(csint i, csint j) const {
        return get_item_(i, j).value;
    }

    /** Return a proxy item for the value of the requested element for use in
     * assignment, e.g. `A(i, j) = 56.0`.
     *
     * The proxy item allows for either item lookup via `double v = A(i, j)`, or
     * item assignment `A(i, j) = v` on a non-const matrix, in a way that only
     * changes the matrix on item assignment.
     *
     * This function takes O(log M) time if the columns are sorted, and O(M)
     * time if they are not.
     *
     * @param i, j the row and column indices of the element to access.
     *
     * @return a proxy reference to the value of the element at `(i, j)`.
     */
    ItemProxy operator()(csint i, csint j) {
        return ItemProxy(*this, i, j);
    }

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
        std::span<const csint> i,
        std::span<const csint> j,
        std::span<const double> C  // dense column-major
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
        std::span<const csint> rows,
        std::span<const csint> cols,
        const CSCMatrix& C
    );

    //--------------------------------------------------------------------------
    //        Format Conversions
    //--------------------------------------------------------------------------
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
    virtual std::vector<double> to_dense_vector(
        const DenseOrder order = DenseOrder::ColMajor
    ) const override;

    /** Convert a CSCMatrix to a double if it is a 1x1 matrix.
     *
     * @return the value of the matrix if it is a 1x1 matrix.
     */
    operator double() const {
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

    /** Return the kth diagonal of the matrix as a dense vector.
     *
     * @param k  the diagonal to extract. The main diagonal is 0, with
     *        sub-diagonals < 0, and super-diagonals > 0.
     *
     * @return  a dense vector containing the values of the kth diagonal.
     */
    std::vector<double> diagonal(csint k=0) const;

    /** Compute the structural symmetry of the matrix.
     *
     * See: Davis, Exercise 8.1.
     *
     * The structural symmetry is defined as:
     * \f$ sym(S) = \frac{nnz(S \land S^T)}{nnz(S)} \f$,
     * where \f$ S = A - \diag(A) \f$ (off-diagonal elements only).
     *
     * In Scipy:
     *   S = A - sparse.diags_array(A.diagonal())
     *   sym = (S * S.T).nnz / S.nnz
     *
     * @return sym  the structural symmetry of the matrix.
     */
    double structural_symmetry() const;

    //--------------------------------------------------------------------------
    //        Math Operations
    //--------------------------------------------------------------------------
    /** Scale the rows and columns of a matrix by \f$ A = RAC \f$, where *R* and *C*
     * are diagonal matrices.
     *
     * See: Davis, Exercise 2.4.
     *
     * @param r, c  vectors of length M and N, respectively, representing the
     *        diagonals of R and C, where A is size M-by-N.
     *
     * @return RAC the scaled matrix
     */
    CSCMatrix scale(std::span<const double> r, std::span<const double> c) const;

    /** Matrix-dense matrix right-multiply (see cs_multiply)
     *
     * @param X  the dense matrix to multiply. X is size N x K, stored in
     *        column-major order.
     *
     * @return Y  the dense matrix result. Y is size M x K, stored in
     *         column-major order.
     */
    virtual std::vector<double> dot(std::span<const double> X) const override;

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
     * See: Davis, Section 2.9, `cs_add`, and Exercise 2.21 `cs_saxpy`.
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
        std::span<csint> w,
        std::span<double> x,
        csint mark,
        CSCMatrix& C,
        csint nz,
        bool fs=false    // Exercise 2.19
    ) const;

    /** Scatter a column of the matrix into a dense vector.
     *
     * @param k  the column index to scatter
     * @param x  the dense vector to scatter into. Must be length M.
     */
    void scatter(csint k, std::span<double> x) const;

    //--------------------------------------------------------------------------
    //        Permutations
    //--------------------------------------------------------------------------
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
        std::span<const csint> p_inv,
        std::span<const csint> q,
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
    CSCMatrix symperm(std::span<const csint> p_inv, bool values=true) const;

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
        std::span<const csint> p_inv,
        std::span<const csint> q_inv,
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
    CSCMatrix permute_rows(std::span<const csint> p_inv, bool values=true) const;

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
    CSCMatrix permute_cols(std::span<const csint> q, bool values=true) const;

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
    bool is_valid(const bool sorted=true, const bool values=true) const;

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
        std::span<const csint> rows,
        std::span<const csint> cols
    ) const;

    /** Add empty rows to the top of the matrix.
     *
     * See: Davis, Exercise 2.29.
     *
     * @param k  the number of rows to add
     *
     * @return  a reference to the modified matrix
     */
    CSCMatrix& add_empty_top(const csint k);

    /** Add empty rows to the bottom of the matrix.
     *
     * See: Davis, Exercise 2.29.
     *
     * @param k  the number of rows to add
     *
     * @return  a reference to the modified matrix
     */
    CSCMatrix& add_empty_bottom(const csint k);

    /** Add empty columns to the left of the matrix.
     *
     * See: Davis, Exercise 2.29.
     *
     * @param k  the number of columns to add
     *
     * @return  a reference to the modified matrix
     */
    CSCMatrix& add_empty_left(const csint k);

    /** Add empty columns to the right of the matrix.
    *
    * See: Davis, Exercise 2.29.
    *
    * @param k  the number of columns to add
    *
    * @return  a reference to the modified matrix
    */
    CSCMatrix& add_empty_right(const csint k);

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

    //--------------------------------------------------------------------------
    //        Triangular Matrix Solutions
    //--------------------------------------------------------------------------
    friend void lsolve_inplace(const CSCMatrix& A, std::span<double> x);
    friend void ltsolve_inplace(const CSCMatrix& A, std::span<double> x);
    friend void usolve_inplace(const CSCMatrix& A, std::span<double> x);
    friend void utsolve_inplace(const CSCMatrix& A, std::span<double> x);

    friend void lsolve_inplace_opt(const CSCMatrix& A, std::span<double> x);
    friend void usolve_inplace_opt(const CSCMatrix& A, std::span<double> x);

    friend std::vector<double> lsolve_rows(const CSCMatrix& A, std::span<const double> b);
    friend std::vector<double> usolve_rows(const CSCMatrix& A, std::span<const double> b);
    friend std::vector<double> lsolve_cols(const CSCMatrix& A, std::span<const double> b);
    friend std::vector<double> usolve_cols(const CSCMatrix& A, std::span<const double> b);

    friend std::vector<csint> find_lower_diagonals(const CSCMatrix& A);
    friend std::vector<csint> find_upper_diagonals(const CSCMatrix& A);

    friend TriPerm find_tri_permutation(const CSCMatrix& A);

    friend void tri_solve_perm_inplace(
        const CSCMatrix& A,
        const TriPerm& tri_perm,
        std::span<double> b,
        std::span<double> x
    );

    friend void spsolve(
        const CSCMatrix& A,
        const CSCMatrix& B,
        csint k,
        SparseSolution& sol,
        std::span<const csint> p_inv,
        bool lower
    );

    friend void reach(
        const CSCMatrix& A,
        const CSCMatrix& B,
        csint k,
        std::vector<csint>& xi,
        std::span<const csint> p_inv
    );

    friend void dfs(
        const CSCMatrix& A,
        csint j,
        std::span<char> marked,
        std::vector<csint>& xi,
        std::vector<csint>& pstack,
        std::vector<csint>& rstack,
        std::span<const csint> p_inv
    );

    friend std::vector<csint> detail::reach_r(
        const CSCMatrix& A,
        const CSCMatrix& B
    );

    friend void detail::dfs_r(
        const CSCMatrix& A,
        csint j,
        std::span<char> marked,
        std::vector<csint>& xi
    );

    //--------------------------------------------------------------------------
    //        Cholesky Decomposition
    //--------------------------------------------------------------------------
    friend CholResult symbolic_cholesky(const CSCMatrix& A, const SymbolicChol& S);

    friend CholResult chol(const CSCMatrix& A, const SymbolicChol& S);
    friend CSCMatrix& leftchol(const CSCMatrix& A, const SymbolicChol& S, CSCMatrix& L);
    friend CSCMatrix& rechol(const CSCMatrix& A, const SymbolicChol& S, CSCMatrix& L);

    friend CholResult ichol_nofill(const CSCMatrix& A, const SymbolicChol& S);
    friend CholResult icholt(
        const CSCMatrix& A,
        const SymbolicChol& S,
        double drop_tol
    );

    friend CSCMatrix& chol_update(
        CSCMatrix& L,
        bool update,
        const CSCMatrix& w,
        std::span<const csint> parent
    );

    //--------------------------------------------------------------------------
    //        QR Decomposition
    //--------------------------------------------------------------------------
    friend void happly(
        const CSCMatrix& V,
        csint j,
        double beta,
        std::span<double> x
    );

    friend std::vector<csint> find_leftmost(const CSCMatrix& A);
    friend void vcount(const CSCMatrix& A, SymbolicQR& S);

    friend QRResult symbolic_qr(const CSCMatrix& A, const SymbolicQR& S);
    friend QRResult qr(const CSCMatrix& A, const SymbolicQR& S);
    friend void reqr(const CSCMatrix& A, const SymbolicQR& S, QRResult& res);

    friend CSCMatrix apply_qtleft(
        const CSCMatrix& V,
        std::span<const double> beta,
        std::span<const csint> p_inv,
        const CSCMatrix& Y
    );

    // -------------------------------------------------------------------------
    //         LU Decomposition
    // -------------------------------------------------------------------------
    friend LUResult lu_original(const CSCMatrix& A, const SymbolicLU& S, double tol);
    friend LUResult lu(
        const CSCMatrix& A,
        const SymbolicLU& S,
        double tol,
        double col_tol
    );
    friend LUResult relu(const CSCMatrix& A, const LUResult& R, const SymbolicLU& S);
    friend LUResult lu_crout(const CSCMatrix& A, const SymbolicLU& S);
    friend LUResult ilutp(
        const CSCMatrix& A,
        const SymbolicLU& S,
        double drop_tol,
        double tol
    );
    friend LUResult ilu_nofill(const CSCMatrix& A, const SymbolicLU& S);
    friend std::vector<double> lu_solve(
        const CSCMatrix& A,
        const CSCMatrix& B,
        AMDOrder order,
        double tol,
        csint ir_steps
    );

    // -------------------------------------------------------------------------
    //         Fill-reducing Orderings
    // -------------------------------------------------------------------------
    friend CSCMatrix build_graph(const CSCMatrix& A, const AMDOrder order, csint dense);
    friend std::vector<csint> amd(const CSCMatrix& A, const AMDOrder order);

    friend MaxMatch detail::maxtrans_r(const CSCMatrix& A, csint seed);
    friend bool detail::augment_r(
        csint k,
        const CSCMatrix& A,
        std::span<csint> jmatch,
        std::span<csint> cheap,
        std::span<csint> w,
        csint j
    );

    friend MaxMatch maxtrans(const CSCMatrix& A, csint seed);
    friend void augment(
        csint k,
        const CSCMatrix& A,
        std::span<csint> jmatch,
        std::span<csint> cheap,
        std::span<csint> w,
        std::span<csint> js,
        std::span<csint> is,
        std::span<csint> ps
    );

    friend SCCResult scc(const CSCMatrix& A);
    friend DMPermResult dmperm(const CSCMatrix& A, csint seed);
    friend void bfs(
        const CSCMatrix& A,
        csint N,
        std::span<csint> wi,
        std::span<csint> wj,
        std::span<csint> queue,
        std::span<const csint> imatch,
        std::span<const csint> jmatch,
        csint mark
    );

    // Unary minus operator
    friend CSCMatrix operator-(const CSCMatrix& A);

protected:
    /** Return the format description of the matrix. */
    virtual std::string_view get_format_desc_() const override
    {
        return format_desc_;
    }

    /** Print elements of the matrix between `start` and `end`.
     *
     * @param ss          the output string stream
     * @param start, end  print the all elements where `p ∈ [start, end]`, counting
     *        column-wise.
     */
    virtual void write_elems_(std::stringstream& ss, csint start, csint end) const override;


private:
    static constexpr std::string_view format_desc_ = "C++Sparse Compressed Sparse Column";
    std::vector<double> v_;  // numerical values, size nzmax
    std::vector<csint> i_;   // row indices, size nzmax
    std::vector<csint> p_;   // column pointers (CSC size N_);
    csint M_ = 0;            // number of rows
    csint N_ = 0;            // number of columns
    bool has_sorted_indices_ = false;
    bool has_canonical_format_ = false;

    // Internal struct for get_item_ use
    struct GetItemResult {
        double value;
        bool found;
        csint index;
    };

    /** Search a sorted column for the item at index (i, j).
     *
     * This function is used by get/set_item_ to find the index of the item when
     * the matrix has canonical format with sorted row indices and no
     * duplicates.
     *
     * @param i, j  the row and column indices of the element to access.
     *
     * @return  a tuple containing a boolean indicating if the item was found,
     *          and the index of the item in the `i_` and `v_` arrays.
     */
    std::pair<bool, csint> binary_search_(csint i, csint j) const;

    /** Return the value of A(i, j).
     *
     * @param i, j  the row and column indices of the element to access.
     *
     * @return a tuple containing the value of the element, a boolean indicating
     *         if the item was found, and the index of the item in the `i_` and
     *         `v_` arrays.
     * */
    GetItemResult get_item_(csint i, csint j) const;

    /** Set the value of A(i, j).
     *
     * This function replaces the existing value at A(i, j), i.e.,
     * any duplicate entries with the same row and column indices will be
     * set to 0, but not removed.
     */
    void set_item_(csint i, csint j, double v);

    /** Set the value of `A(i, j)` with a binary operation like `A(i, j) += v`.
     *
     * This function is used by the `ItemProxy` class to define the compound
     * assignment operators like `+=`, `-=`, etc. without repeating the
     * search for the item.
     *
     * @param i, j  the row and column indices of the element to access.
     * @param v  the value on which `A(i, j)` will be operated.
     * @param found  a boolean indicating if the item was found.
     * @param k  the index of the item in the `i_` and `v_` arrays.
     * @param op  the binary operation to be performed.
     */
    template <typename BinaryOp>
    void set_item_with_op_(csint i, csint j, double v, bool found, csint k, BinaryOp op)
    {
        if (found) {
            // Update the value
            v_[k] = op(v_[k], v);

            // Duplicates may exist, so zero them out
            if (has_sorted_indices_ && !has_canonical_format_) {
                // Duplicates are in order, so don't need to search entire column
                csint i_size = static_cast<csint>(i_.size());
                for (csint p = k + 1; p < i_size && i_[p] == i; p++) {
                    v_[p] = 0.0;
                }
            } else {
                // Linear search through entire rest of column
                for (csint p = k + 1; p < p_[j+1]; p++) {
                    if (i_[p] == i) {
                        v_[p] = 0.0;
                    }
                }
            }
        } else {
            // Value does not exist, so insert it
            insert_(i, j, op(0.0, v), k);
        }
    }

    /** Insert a single element at a specified location.
     *
     * This function is a helper for set_item_.
     *
     * @param i, j  the row and column indices of the element to access.
     * @param v  the value to be assigned.
     * @param p  the pointer to the column in the matrix.
     *
     * @return a reference to the inserted value.
     */
    double& insert_(csint i, csint j, double v, csint p);


};  // class CSCMatrix


/*------------------------------------------------------------------------------
 *          Free Functions
 *----------------------------------------------------------------------------*/
CSCMatrix operator+(const CSCMatrix& A, const CSCMatrix& B);
CSCMatrix operator-(const CSCMatrix& A, const CSCMatrix& B);
CSCMatrix operator-(const CSCMatrix& A);

CSCMatrix operator*(const CSCMatrix& A, const CSCMatrix& B);
CSCMatrix operator*(const CSCMatrix& A, const double c);
CSCMatrix operator*(const double c, const CSCMatrix& A);


/** Matrix-vector multiply `y = Ax + y`.
 *
 * @param A  a CSC matrix
 * @param x  a dense multiplying vector
 * @param y  a dense adding vector which will be used for the output
 *
 * @return y a copy of the updated vector
 */
std::vector<double> gaxpy(
    const CSCMatrix& A,
    std::span<const double> x,
    std::span<const double> y
);


/** Matrix transpose-vector multiply `y = A.T x + y`.
 *
 * See: Davis, Exercise 2.1. Compute \f$ A^T x + y \f$ without explicitly
 * computing the transpose.
 *
 * @param A  a CSC matrix
 * @param x  a dense multiplying vector
 * @param y[in,out]  a dense adding vector which will be used for the output
 *
 * @return y a copy of the updated vector
 */
std::vector<double> gatxpy(
    const CSCMatrix& A,
    std::span<const double> x,
    std::span<const double> y
);


/** Matrix-vector multiply `y = Ax + y` symmetric A (\f$ A = A^T \f$).
 *
 * See: Davis, Exercise 2.3.
 *
 * @param A  a CSC matrix
 * @param x  a dense multiplying vector
 * @param y  a dense adding vector which will be used for the output
 *
 * @return y a copy of the updated vector
 */
std::vector<double> sym_gaxpy(
    const CSCMatrix& A,
    std::span<const double> x,
    std::span<const double> y
);


/** Matrix multiply `Y = AX + Y` column-major dense matrices `X` and `Y`.
 *
 * See: Davis, Exercise 2.27(a).
 *
 * @param A  a CSC matrix
 * @param X  a dense multiplying matrix in column-major order
 * @param[in,out] Y  a dense adding matrix which will be used for the output
 *
 * @return Y a copy of the updated matrix
 */
std::vector<double> gaxpy_col(
    const CSCMatrix& A,
    std::span<const double> X,
    std::span<const double> Y
);


/** Matrix multiply `Y = AX + Y` for row-major dense matrices `X` and `Y`.
 *
 * See: Davis, Exercise 2.27(b).
 *
 * @param A  a CSC matrix
 * @param X  a dense multiplying matrix in row-major order
 * @param[in,out] Y  a dense adding matrix which will be used for the output
 *
 * @return Y a copy of the updated matrix
 */
std::vector<double> gaxpy_row(
    const CSCMatrix& A,
    std::span<const double> X,
    std::span<const double> Y
);


/** Matrix multiply `Y = AX + Y` column-major dense matrices `X` and
 * `Y`, but operate on blocks of columns.
 *
 * See: Davis, Exercise 2.27(c).
 *
 * @param A  a CSC matrix
 * @param X  a dense multiplying matrix in column-major order
 * @param[in,out] Y  a dense adding matrix which will be used for the output
 *
 * @return Y a copy of the updated matrix
 */
std::vector<double> gaxpy_block(
    const CSCMatrix& A,
    std::span<const double> X,
    std::span<const double> Y
);


/** Matrix multiply `Y = A.T X + Y` column-major dense matrices `X` and `Y`.
 *
 * See: Davis, Exercise 2.28(a).
 *
 * @param A  a CSC matrix
 * @param X  a dense multiplying matrix in column-major order
 * @param[in,out] Y  a dense adding matrix which will be used for the output
 *
 * @return Y a copy of the updated matrix
 */
std::vector<double> gatxpy_col(
    const CSCMatrix& A,
    std::span<const double> X,
    std::span<const double> Y
);


/** Matrix multiply `Y = A.T X + Y` for row-major dense matrices `X` and `Y`.
 *
 * See: Davis, Exercise 2.27(b).
 *
 * @param A  a CSC matrix
 * @param X  a dense multiplying matrix in row-major order
 * @param[in,out] Y  a dense adding matrix which will be used for the output
 *
 * @return Y a copy of the updated matrix
 */
std::vector<double> gatxpy_row(
    const CSCMatrix& A,
    std::span<const double> X,
    std::span<const double> Y
);


/** Matrix multiply `Y = A.T X + Y` column-major dense matrices `X` and
 * `Y`, but operate on blocks of columns.
 *
 * See: Davis, Exercise 2.28(c).
 *
 * @param A  a CSC matrix
 * @param X  a dense multiplying matrix in column-major order
 * @param[in,out] Y  a dense adding matrix which will be used for the output
 *
 * @return Y a copy of the updated matrix
 */
std::vector<double> gatxpy_block(
    const CSCMatrix& A,
    std::span<const double> X,
    std::span<const double> Y
);


/** Add two sparse column vectors \f$ x = a + b \f$.
 *
 * See: Davis, Exercise 2.21
 *
 * @param a, b two column vectors stored as a CSCMatrix. The number of columns
 *        in each argument must be 1.
 * @param w[out]  pre-allocated workspace vector of length M. On output,
 *        contains non-zeros where `x` was updated.
 * @param x[out]  pre-allocated dense vector of length M to accumulate the
 *        result.
 */
void saxpy(
    const CSCMatrix& a,
    const CSCMatrix& b,
    std::span<char> w,
    std::span<double> x
);


}  // namespace cs

#endif  // _CSC_H_

//==============================================================================
//==============================================================================
