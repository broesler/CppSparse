/*==============================================================================
 *     File: csc.cpp
 *  Created: 2024-10-09 20:58
 *   Author: Bernie Roesler
 *
 *  Description: Implements the compressed sparse column matrix class
 *
 *============================================================================*/

#include <algorithm>  // for std::lower_bound
#include <cmath>      // for std::fabs
#include <ranges>     // for std::views::reverse

#include "csparse.h"

namespace cs {

/*------------------------------------------------------------------------------
 *     Constructors
 *----------------------------------------------------------------------------*/
CSCMatrix::CSCMatrix() {};

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
 * @return a new CSCMatrix object
 */
CSCMatrix::CSCMatrix(
    const std::vector<double>& data,
    const std::vector<csint>& indices,
    const std::vector<csint>& indptr,
    const Shape& shape
    )
    : v_(data),
      i_(indices),
      p_(indptr),
      M_(shape[0]),
      N_(shape[1])
{}


/** Allocate a CSCMatrix for a given shape and number of non-zeros.
 *
 * @param M, N  integer dimensions of the rows and columns
 * @param nzmax integer capacity of space to reserve for non-zeros
 */
CSCMatrix::CSCMatrix(csint M, csint N, csint nzmax)
    : v_(nzmax),
      i_(nzmax),
      p_(N + 1),
      M_(M),
      N_(N)
{}


/** Convert a coordinate format matrix to a compressed sparse column matrix in
 * canonical format.
 *
 * The columns are guaranteed to be sorted, no duplicates are allowed, and no
 * numerically zero entries are allowed.
 *
 * This function takes O(M + N + nnz) time.
 *
 * See: Davis, Exercise 2.9.
 *
 * @return a copy of the `COOMatrix` in canonical CSC format.
 */
CSCMatrix::CSCMatrix(const COOMatrix& A) : CSCMatrix(A.compress())
{
    sum_duplicates();  // O(N) space, O(nnz) time
    dropzeros();       // O(nnz) time
    sort();            // O(M) space, O(M + N + nnz) time
    has_canonical_format_ = true;
}


/** Create a sparse copy of a dense matrix in column-major fomr.
 *
 * See: Davis, Exercise 2.16.
 *
 * @param A a dense matrix in column-major form
 * @param M, N the size of the matrix
 *
 * @return C a compressed sparse column version of the matrix
 */
CSCMatrix::CSCMatrix(const std::vector<double>& A, csint M, csint N)
    : M_(M),
      N_(N)
{
    assert(A.size() == (M * N));  // ensure input is valid

    // Allocate memory
    v_.reserve(A.size());
    i_.reserve(A.size());
    p_.reserve(N);

    csint nz = 0;  // count number of non-zeros

    for (csint j = 0; j < N; j++) {
        p_.push_back(nz);

        for (csint i = 0; i < M; i++) {
            double val = A[i + j * N];  // linear index for column-major order

            // Only store non-zeros
            if (val != 0.0) {
                i_.push_back(i);
                v_.push_back(val);
                nz++;
            }
        }
    }

    // Finalize and free unused space
    p_.push_back(nz);
    realloc();

    has_sorted_indices_ = true;    // guaranteed by algorithm
    has_canonical_format_ = true;  // guaranteed by input format
}


/** Reallocate a CSCMatrix to a new number of non-zeros.
 *
 * @param A      matrix to be resized
 * @param nzmax  maximum number of non-zeros. If `nzmax <= A.nzmax()`, then
 *        `nzmax` will be set to `A.nnz()`.
 *
 * @return A     a reference to the input object for method chaining.
 */
CSCMatrix& CSCMatrix::realloc(csint nzmax)
{
    csint Z = (nzmax <= 0) ? p_[N_] : nzmax;

    p_.resize(N_ + 1);  // always contains N_ columns + nz
    i_.resize(Z);
    v_.resize(Z);

    p_.shrink_to_fit();  // deallocate memory
    i_.shrink_to_fit();
    v_.shrink_to_fit();

    return *this;
}

/*------------------------------------------------------------------------------
 *         Accessors
 *----------------------------------------------------------------------------*/
csint CSCMatrix::nnz() const { return v_.size(); }
csint CSCMatrix::nzmax() const { return v_.capacity(); }

Shape CSCMatrix::shape() const
{
    return Shape {M_, N_};
}

const std::vector<csint>& CSCMatrix::indices() const { return i_; }
const std::vector<csint>& CSCMatrix::indptr() const { return p_; }
const std::vector<double>& CSCMatrix::data() const { return v_; }


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
CSCMatrix& CSCMatrix::to_canonical()
{
    sum_duplicates();
    dropzeros();
    sort();
    has_canonical_format_ = true;
    return *this;
}


bool CSCMatrix::has_sorted_indices() const { return has_sorted_indices_; }
bool CSCMatrix::has_canonical_format() const { return has_canonical_format_; }


/** Return the value of the requested element.
 *
 * This function takes O(log M) time if the columns are sorted, and O(M) time
 * if they are not.
 *
 * @param i, j the row and column indices of the element to access.
 *
 * @return the value of the element at `(i, j)`.
 */
const double CSCMatrix::operator()(csint i, csint j) const
{
    // Assert indices are in-bounds
    assert(i >= 0 && i < M_);
    assert(j >= 0 && j < N_);

    if (has_canonical_format_) {
        // Binary search for t <= i
        auto start = i_.begin() + p_[j];
        auto end = i_.begin() + p_[j+1];

        auto t = std::lower_bound(start, end, i);

        // Check that we actually found the index t == i
        if (t != end && *t == i) {
            return v_[std::distance(i_.begin(), t)];
        } else {
            return 0.0;
        }

    } else {
        // NOTE this code assumes that columns are *not* sorted, and that
        // duplicate entries may exist, so it will search through *every*
        // element in a column.
        double out = 0.0;

        for (csint p = p_[j]; p < p_[j+1]; p++) {
            if (i_[p] == i) {
                out += v_[p];  // sum duplicate entries
            }
        }

        return out;
    }
}



/** Return a reference to the value of the requested element for use in
 * assignment, e.g. `A(i, j) = 56.0`.
 *
 * This function takes O(log M) time if the columns are sorted, and O(M) time
 * if they are not.
 *
 * @param i, j the row and column indices of the element to access.
 *
 * @return a reference to the value of the element at `(i, j)`.
 */
double& CSCMatrix::operator()(const csint i, const csint j)
{
    // Assert indices are in-bounds
    assert(i >= 0 && i < M_);
    assert(j >= 0 && j < N_);

    if (has_canonical_format_) {
        // Binary search for t <= i
        auto start = i_.begin() + p_[j];
        auto end = i_.begin() + p_[j+1];

        auto t = std::lower_bound(start, end, i);

        auto k = std::distance(i_.begin(), t);

        // Check that we actually found the index t == i
        if (t != end && *t == i) {
            return v_[k];
        } else {
            // Value does not exist, so add a place-holder here.
            return this->insert(i, j, 0.0, k);
        }

    } else {
        // Linear search for the element
        csint k;
        bool found = false;

        for (csint p = p_[j]; p < p_[j+1]; p++) {
            if (i_[p] == i) {
                if (!found) {
                    k = p;  // store the index of the element
                    found = true;
                } else {
                    v_[k] += v_[p];  // accumulate duplicates here
                    v_[p] = 0;       // zero out duplicates
                }
            }
        }

        if (found) {
            return v_[k];
        } else {
            // Columns are not sorted, so we can just insert the element at the
            // beginning of the column.
            return this->insert(i, j, 0.0, p_[j]);
        }
    }
}


/** Assign a value to a specific element in the matrix.
 *
 * This function takes O(log M) time if the columns are sorted, and O(M) time
 * if they are not.
 *
 * @param i, j the row and column indices of the element to access.
 * @param v the value to be assigned.
 *
 * @return a reference to itself for method chaining.
 */
CSCMatrix& CSCMatrix::assign(csint i, csint j, double v)
{
    (*this)(i, j) = v;
    return *this;
}


/** Assign a dense matrix to the CSCMatrix at the specified locations.
 *
 * See: Davis, Exercise 2.25.
 *
 * @param rows, cols the row and column indices of the elements to access.
 * @param C the dense matrix to be assigned.
 *
 * @return a reference to itself for method chaining.
 */
CSCMatrix& CSCMatrix::assign(
    const std::vector<csint>& rows,
    const std::vector<csint>& cols,
    const std::vector<double>& C
    )
{
    assert(C.size() == rows.size() * cols.size());

    for (csint i = 0; i < rows.size(); i++) {
        for (csint j = 0; j < cols.size(); j++) {
            (*this)(rows[i], cols[j]) = C[i + j * rows.size()];
        }
    }

    return *this;
}


/** Assign a sparse matrix to the CSCMatrix at the specified locations.
 *
 * See: Davis, Exercise 2.25.
 *
 * @param rows, cols the row and column indices of the elements to access.
 * @param C the sparse matrix to be assigned.
 *
 * @return a reference to itself for method chaining.
 */
CSCMatrix& CSCMatrix::assign(
    const std::vector<csint>& rows,
    const std::vector<csint>& cols,
    const CSCMatrix& C
    )
{
    assert(C.M_ == rows.size());
    assert(C.N_ == cols.size());

    for (csint j = 0; j < cols.size(); j++) {
        for (csint p = C.p_[j]; p < C.p_[j+1]; p++) {
            csint i = C.i_[p];
            (*this)(rows[i], cols[j]) = C.v_[p];
        }
    }

    return *this;
}


/** Insert a single element at a specified location.
 *
 * @param i, j the row and column indices of the element to access.
 * @param v the value to be assigned.
 * @param p the pointer to the column in the matrix.
 *
 * @return a reference to the inserted value.
 */
double& CSCMatrix::insert(csint i, csint j, double v, csint p)
{
    i_.insert(i_.begin() + p, i);
    v_.insert(v_.begin() + p, v);

    // Increment all subsequent pointers
    for (csint k = j + 1; k < p_.size(); k++) {
        p_[k]++;
    }

    return v_[p];
}



/** Returns true if `A(i, j) == A(i, j)` for all `i, j`.
 *
 * See: Davis, Exercise 2.13.
 *
 * @return true if the matrix is symmetric.
 */
bool CSCMatrix::is_symmetric() const
{
    assert(has_canonical_format_);

    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            csint i = i_[p];

            if (i == j)
                continue;  // skip diagonal

            if (v_[p] != (*this)(j, i))
                return false;
        }
    }

    return true;
}


/*------------------------------------------------------------------------------
       Format Operations
----------------------------------------------------------------------------*/
/** Convert a compressed sparse column matrix to a coordinate (triplet) format
 * matrix.
 *
 * See: Davis, Exercise 2.2.
 *
 * @return a copy of the `CSCMatrix` in COO (triplet) format.
 */
COOMatrix CSCMatrix::tocoo() const { return COOMatrix(*this); }


/** Convert a CSCMatrix to a dense column-major array.
 *
 * See: Davis, Exercise 2.16.
 *
 * @param order the order of the array, either 'F' for Fortran (column-major)
 *       or 'C' for C (row-major).
 *
 * @return a copy of the matrix as a dense column-major array.
 */
std::vector<double> CSCMatrix::toarray(const char order) const
{
    std::vector<double> A(M_ * N_, 0.0);
    csint idx;

    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            // Column- vs row-major order
            if (order == 'F') {
                idx = i_[p] + j * M_;
            } else if (order == 'C') {
                idx = j + i_[p] * N_;
            } else {
                throw std::invalid_argument("Invalid order argument. Use 'F' or 'C'.");
            }

            if (has_canonical_format_) {
                A[idx] = v_[p];
            } else {
                A[idx] += v_[p];  // account for duplicates
            }
        }
    }

    return A;
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
 * @return new CSCMatrix object with transposed rows and columns.
 */
CSCMatrix CSCMatrix::transpose() const
{
    std::vector<csint> w(M_);   // workspace
    CSCMatrix C(N_, M_, nnz());  // output

    // Compute number of elements in each row
    for (csint p = 0; p < nnz(); p++)
        w[i_[p]]++;

    // Row pointers are the cumulative sum of the counts, starting with 0.
    C.p_ = cumsum(w);
    w = C.p_;  // copy back into workspace

    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            // place A(i, j) as C(j, i)
            csint q = w[i_[p]]++;
            C.i_[q] = j;
            C.v_[q] = v_[p];
        }
    }

    return C;
}


// Alias for transpose
CSCMatrix CSCMatrix::T() const { return this->transpose(); }


/** Sort rows and columns in a copy via two transposes.
 *
 * See: Davis, Exercise 2.7.
 *
 * @return C  a copy of the matrix with sorted columns.
 */
CSCMatrix CSCMatrix::tsort() const
{
    CSCMatrix C = this->transpose().transpose();
    C.has_sorted_indices_ = true;
    return C;
}


/** Sort rows and columns in place using std::sort.
 *
 * This function takes
 *   - O(3*M) extra space ==
 *       2 workspaces for row indices and values + vector of sorted indices
 *   - O(N * M log M + nnz) time ==
 *       sort a length M vector for each of N columns
 *
 * @return a reference to the object for method chaining
 */
CSCMatrix& CSCMatrix::qsort()
{
    // Allocate workspaces
    std::vector<csint> w(nnz());
    std::vector<double> x(nnz());

    for (csint j = 0; j < N_; j++) {
        // Pointers to the rows
        csint p = p_[j],
              pn = p_[j+1];

        // clear workspaces
        w.clear();
        x.clear();

        // Copy the row indices and values into the workspace
        std::copy(i_.begin() + p, i_.begin() + pn, std::back_inserter(w));
        std::copy(v_.begin() + p, v_.begin() + pn, std::back_inserter(x));

        // argsort the rows to get indices
        std::vector<csint> idx = argsort(w);

        // Re-assign the values
        for (csint i = 0; i < idx.size(); i++) {
            i_[p + i] = w[idx[i]];
            v_[p + i] = x[idx[i]];
        }
    }

    has_sorted_indices_ = true;

    return *this;
}


/** Sort rows and columns in place two transposes, but more efficiently than
 * calling `transpose` twice.
 *
 * This function takes O(M) extra space and O(M * N + nnz) time.
 *
 * See: Davis, Exercise 2.11.
 *
 * @return A  a reference to the matrix, now with sorted columns.
 */
CSCMatrix& CSCMatrix::sort()
{
    // ----- first transpose
    std::vector<csint> w(M_);   // workspace
    CSCMatrix C(N_, M_, nnz());  // intermediate transpose

    // Compute number of elements in each row
    for (csint p = 0; p < nnz(); p++)
        w[i_[p]]++;

    // Row pointers are the cumulative sum of the counts, starting with 0.
    C.p_ = cumsum(w);
    w = C.p_;  // copy back into workspace

    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            // place A(i, j) as C(j, i)
            csint q = w[i_[p]]++;
            C.i_[q] = j;
            C.v_[q] = v_[p];
        }
    }

    // ----- second transpose
    // Copy column counts to avoid repeat work
    w = p_;

    for (csint j = 0; j < C.N_; j++) {
        for (csint p = C.p_[j]; p < C.p_[j+1]; p++) {
            // place C(i, j) as A(j, i)
            csint q = w[C.i_[p]]++;
            i_[q] = j;
            v_[q] = C.v_[p];
        }
    }

    has_sorted_indices_ = true;

    return *this;
}


/** Sum duplicate entries in place.
 *
 * This function takes
 *   - O(N) extra space for the workspace
 *   - O(nnz) time
 *
 * @return a reference to the object for method chaining
 */
CSCMatrix& CSCMatrix::sum_duplicates()
{
    csint nz = 0;  // count actual number of non-zeros (excluding dups)
    std::vector<csint> w(M_, -1);                      // row i not yet seen

    for (csint j = 0; j < N_; j++) {
        csint q = nz;                                  // column j will start at q
        for (csint p = p_[j]; p < p_[j + 1]; p++) {
            csint i = i_[p];                          // A(i, j) is nonzero
            if (w[i] >= q) {
                v_[w[i]] += v_[p];                   // A(i, j) is a duplicate
            } else {
                w[i] = nz;                          // record where row i occurs
                i_[nz] = i;                          // keep A(i, j)
                v_[nz++] = v_[p];
            }
        }
        p_[j] = q;                                    // record start of column j
    }

    p_[N_] = nz;                                     // finalize A
    realloc();

    return *this;
}


/** Keep matrix entries for which `fkeep` returns true.
 *
 * @param fk a boolean function that acts on each element. If `fk` returns
 *        `true`, that element will be kept in the matrix. The function `fk` has
 *        four parameters:
 *        @param i, j integer indices of the element
 *        @param v the value of the element
 *        @param other a void pointer for any additional argument (*i.e.*
 *               a non-zero tolerance against which to compare)
 *        @return keep a boolean that is true if the element `A(i, j)` should be
 *                kept in the matrix.
 * @param other a pointer to the additional argument in `fk`.
 */
CSCMatrix& CSCMatrix::fkeep(
    bool (*fk) (csint, csint, double, void *),
    void *other
)
{
    csint nz = 0;  // count actual number of non-zeros

    for (csint j = 0; j < N_; j++) {
        csint p = p_[j];  // get current location of column j
        p_[j] = nz;       // record new location of column j
        for (; p < p_[j+1]; p++) {
            if (fk(i_[p], j, v_[p], other)) {
                v_[nz] = v_[p];  // keep A(i, j)
                i_[nz++] = i_[p];
            }
        }
    }

    p_[N_] = nz;    // finalize A
    realloc();

    return *this;
};


/** Return true if A(i, j) is non-zero */
bool CSCMatrix::nonzero(csint i, csint j, double Aij, void *other)
{
    return (Aij != 0);
}


/** Drop any exactly zero entries from the matrix.
 *
 * This function takes O(nnz) time.
 *
 * @return a reference to the object for method chaining
 */
CSCMatrix& CSCMatrix::dropzeros()
{
    return fkeep(&nonzero, nullptr);
}


/** Return true if abs(A(i j)) > tol */
bool CSCMatrix::abs_gt_tol(csint i, csint j, double Aij, void *tol)
{
    return (std::fabs(Aij) > *((double *) tol));
}


/** Drop any entries within `tol` of zero.
 *
 * This function takes O(nnz) time.
 *
 * @param tol the tolerance against which to compare the absolute value of the
 *        matrix entries.
 *
 * @return a reference to the object for method chaining
 */
CSCMatrix& CSCMatrix::droptol(double tol)
{
    return fkeep(&abs_gt_tol, &tol);
}


/** Return true if `A(i, j)` is within the diagonals `limits = {lower, upper}`. */
bool CSCMatrix::in_band(csint i, csint j, double Aij, void *limits)
{
    auto [kl, ku] = *((Shape *) limits);
    return ((i <= (j - kl)) && (i >= (j - ku)));
};


/** Keep any entries within the specified band.
 *
 * @param kl, ku  the lower and upper diagonals within which to keep entries.
 * The main diagonal is 0, with sub-diagonals < 0, and super-diagonals > 0.
 *
 * @return a copy of the matrix with entries removed.
 */
CSCMatrix CSCMatrix::band(const csint kl, const csint ku)
{
    assert(kl <= ku);
    Shape limits {kl, ku};

    return fkeep(&in_band, &limits);
}


/*------------------------------------------------------------------------------
       Math Operations
----------------------------------------------------------------------------*/
/** Matrix-vector multiply `y = Ax + y`.
 *
 * @param x  a dense multiplying vector
 * @param y  a dense adding vector which will be used for the output
 *
 * @return y a copy of the updated vector
 */
std::vector<double> CSCMatrix::gaxpy(
    const std::vector<double>& x,
    const std::vector<double>& y
    ) const
{
    assert(M_ == y.size());  // addition
    assert(N_ == x.size());  // multiplication

    std::vector<double> out = y;  // copy the input vector

    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            out[i_[p]] += v_[p] * x[j];
        }
    }

    return out;
};


/** Matrix transpose-vector multiply `y = T x + y`.
 *
 * See: Davis, Exercise 2.1. Compute \f$ A^T x + y \f$ without explicitly
 * computing the transpose.
 *
 * @param x  a dense multiplying vector
 * @param y[in,out]  a dense adding vector which will be used for the output
 *
 * @return y a copy of the updated vector
 */
std::vector<double> CSCMatrix::gatxpy(
    const std::vector<double>& x,
    const std::vector<double>& y
    ) const
{
    assert(M_ == x.size());  // multiplication
    assert(N_ == y.size());  // addition

    std::vector<double> out = y;  // copy the input vector

    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            out[j] += v_[p] * x[i_[p]];
        }
    }

    return out;
};


/** Matrix-vector multiply `y = Ax + y` for symmetric A (\f$ A = A^T \f$).
 *
 * See: Davis, Exercise 2.3.
 *
 * @param x  a dense multiplying vector
 * @param y  a dense adding vector which will be used for the output
 *
 * @return y a copy of the updated vector
 */
std::vector<double> CSCMatrix::sym_gaxpy(
    const std::vector<double>& x,
    const std::vector<double>& y
    ) const
{
    assert(M_ == N_);  // matrix must be square to be symmetric
    assert(N_ == x.size());
    assert(x.size() == y.size());

    std::vector<double> out = y;  // copy the input vector

    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            csint i = i_[p];

            if (i > j)
                continue;  // skip lower triangular

            // Add the upper triangular elements
            out[i] += v_[p] * x[j];

            // If off-diagonal, also add the symmetric element
            if (i < j)
                out[j] += v_[p] * x[i];
        }
    }

    return out;
};


/** Matrix multiply `Y = AX + Y` for column-major dense matrices `X` and `Y`.
 *
 * See: Davis, Exercise 2.27(a).
 *
 * @param X  a dense multiplying matrix in column-major order
 * @param[in,out] Y  a dense adding matrix which will be used for the output
 *
 * @return Y a copy of the updated matrix
 */
std::vector<double> CSCMatrix::gaxpy_col(
    const std::vector<double>& X,
    const std::vector<double>& Y
    ) const
{
    assert(X.size() % N_ == 0);  // check that X.size() is a multiple of N_
    assert(Y.size() == M_ * (X.size() / N_));

    std::vector<double> out = Y;  // copy the input matrix

    csint K = X.size() / N_;  // number of columns in X

    // For each column of X
    for (csint k = 0; k < K; k++) {
        // Compute one column of Y (see gaxpy)
        for (csint j = 0; j < N_; j++) {
            double x_val = X[j + k * N_];  // cache value

            // Only compute if x_val is non-zero
            if (x_val != 0.0) {
                for (csint p = p_[j]; p < p_[j+1]; p++) {
                    // Indexing in column-major order
                    out[i_[p] + k * M_] += v_[p] * x_val;
                }
            }
        }
    }

    return out;
}


/** Matrix multiply `Y = AX + Y` for column-major dense matrices `X` and `Y`,
 * but operate on blocks of columns.
 *
 * See: Davis, Exercise 2.27(c).
 *
 * @param X  a dense multiplying matrix in column-major order
 * @param[in,out] Y  a dense adding matrix which will be used for the output
 *
 * @return Y a copy of the updated matrix
 */
std::vector<double> CSCMatrix::gaxpy_block(
    const std::vector<double>& X,
    const std::vector<double>& Y
    ) const
{
    assert(X.size() % N_ == 0);  // check that X.size() is a multiple of N_
    assert(Y.size() == M_ * (X.size() / N_));

    std::vector<double> out = Y;  // copy the input matrix

    csint K = X.size() / N_;  // number of columns in X

    const csint BLOCK_SIZE = 32;  // block size for column operations

    // For each column of X
    for (csint k = 0; k < K; k++) {
        // Take a block of columns
        for (csint j_start = 0; j_start < N_; j_start += BLOCK_SIZE) {
            csint j_end = std::min(j_start + BLOCK_SIZE, N_);
            // Compute one column of Y (see gaxpy)
            for (csint j = j_start; j < j_end; j++) {
                double x_val = X[j + k * N_];  // cache value

                // Only compute if x_val is non-zero
                if (x_val != 0.0) {
                    for (csint p = p_[j]; p < p_[j+1]; p++) {
                        // Indexing in column-major order
                        out[i_[p] + k * M_] += v_[p] * x_val;
                    }
                }
            }
        }
    }

    return out;
}


/** Matrix multiply `Y = AX + Y` for row-major dense matrices `X` and `Y`.
 *
 * See: Davis, Exercise 2.27(b).
 *
 * @param X  a dense multiplying matrix in row-major order
 * @param[in,out] Y  a dense adding matrix which will be used for the output
 *
 * @return Y a copy of the updated matrix
 */
std::vector<double> CSCMatrix::gaxpy_row(
    const std::vector<double>& X,
    const std::vector<double>& Y
    ) const
{
    assert(X.size() % N_ == 0);  // check that X.size() is a multiple of N_
    assert(Y.size() == M_ * (X.size() / N_));

    std::vector<double> out = Y;  // copy the input matrix

    csint K = X.size() / N_;  // number of columns in X

    // For each column of X
    for (csint k = 0; k < K; k++) {
        // Compute one column of Y (see gaxpy)
        for (csint j = 0; j < N_; j++) {
            double x_val = X[k + j * K];  // cache value (row-major indexing)

            // Only compute if x_val is non-zero
            if (x_val != 0.0) {
                for (csint p = p_[j]; p < p_[j+1]; p++) {
                    // Indexing in row-major order
                    out[k + i_[p] * K] += v_[p] * x_val;
                }
            }
        }
    }

    return out;
}


/** Matrix multiply `Y = T X + Y` for column-major dense matrices `X` and `Y`.
 *
 * See: Davis, Exercise 2.28(a).
 *
 * @param X  a dense multiplying matrix in column-major order
 * @param[in,out] Y  a dense adding matrix which will be used for the output
 *
 * @return Y a copy of the updated matrix
 */
std::vector<double> CSCMatrix::gatxpy_col(
    const std::vector<double>& X,
    const std::vector<double>& Y
    ) const
{
    assert(X.size() % M_ == 0);  // check that X.size() is a multiple of M_
    assert(Y.size() == N_ * (X.size() / M_));

    std::vector<double> out = Y;  // copy the input matrix

    csint K = X.size() / M_;  // number of columns in X

    // For each column of X
    for (csint k = 0; k < K; k++) {
        // Compute one column of Y (see gaxpy)
        for (csint j = 0; j < N_; j++) {
            for (csint p = p_[j]; p < p_[j+1]; p++) {
                // Indexing in column-major order
                out[j + k * N_] += v_[p] * X[i_[p] + k * M_];
            }
        }
    }

    return out;
}


/** Matrix multiply `Y = T X + Y` for column-major dense matrices `X` and `Y`,
 * but operate on blocks of columns.
 *
 * See: Davis, Exercise 2.28(c).
 *
 * @param X  a dense multiplying matrix in column-major order
 * @param[in,out] Y  a dense adding matrix which will be used for the output
 *
 * @return Y a copy of the updated matrix
 */
std::vector<double> CSCMatrix::gatxpy_block(
    const std::vector<double>& X,
    const std::vector<double>& Y
    ) const
{
    assert(X.size() % M_ == 0);  // check that X.size() is a multiple of N_
    assert(Y.size() == N_ * (X.size() / M_));

    std::vector<double> out = Y;  // copy the input matrix

    csint K = X.size() / M_;  // number of columns in X

    const csint BLOCK_SIZE = 32;  // block size for column operations

    // For each column of X
    for (csint k = 0; k < K; k++) {
        // Take a block of columns
        for (csint j_start = 0; j_start < N_; j_start += BLOCK_SIZE) {
            csint j_end = std::min(j_start + BLOCK_SIZE, N_);
            // Compute one column of Y (see gaxpy)
            for (csint j = j_start; j < j_end; j++) {
                for (csint p = p_[j]; p < p_[j+1]; p++) {
                    // Indexing in column-major order
                    out[j + k * N_] += v_[p] * X[i_[p] + k * M_];
                }
            }
        }
    }

    return out;
}


/** Matrix multiply `Y = T X + Y` for row-major dense matrices `X` and `Y`.
 *
 * See: Davis, Exercise 2.27(b).
 *
 * @param X  a dense multiplying matrix in row-major order
 * @param[in,out] Y  a dense adding matrix which will be used for the output
 *
 * @return Y a copy of the updated matrix
 */
std::vector<double> CSCMatrix::gatxpy_row(
    const std::vector<double>& X,
    const std::vector<double>& Y
    ) const
{
    assert(X.size() % M_ == 0);  // check that X.size() is a multiple of N_
    assert(Y.size() == N_ * (X.size() / M_));

    std::vector<double> out = Y;  // copy the input matrix

    csint K = X.size() / M_;  // number of columns in X

    // For each column of X
    for (csint k = 0; k < K; k++) {
        // Compute one column of Y (see gaxpy)
        for (csint j = 0; j < N_; j++) {
            for (csint p = p_[j]; p < p_[j+1]; p++) {
                // Indexing in row-major order
                out[k + j * K] += v_[p] * X[k + i_[p] * K];
            }
        }
    }

    return out;
}


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
CSCMatrix CSCMatrix::scale(const std::vector<double>& r, const std::vector<double> c) const
{
    assert(r.size() == M_);
    assert(c.size() == N_);

    CSCMatrix out(*this);

    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            out.v_[p] *= r[i_[p]] * c[j];
        }
    }

    return out;
}


/** Matrix-vector right-multiply. */
std::vector<double> CSCMatrix::dot(const std::vector<double>& x) const
{
    assert(N_ == x.size());

    std::vector<double> out(M_);

    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            out[i_[p]] += v_[p] * x[j];
        }
    }

    return out;
}


/** Scale a matrix by a scalar */
CSCMatrix CSCMatrix::dot(const double c) const
{
    CSCMatrix out(v_, i_, p_, shape());
    out.v_ *= c;
    return out;
}


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
CSCMatrix CSCMatrix::dot(const CSCMatrix& B) const
{
    auto [M, Ka] = shape();
    auto [Kb, N] = B.shape();
    assert(Ka == Kb);

    // NOTE See Problem 2.20 for how to compute nnz(A*B)
    CSCMatrix C(M, N, nnz() + B.nnz());  // output

    // Allocate workspaces
    std::vector<csint> w(M);
    std::vector<double> x(M);

    csint nz = 0;  // track total number of non-zeros in C

    bool fs = true;  // Exercise 2.19 -- first call to scatter

    for (csint j = 0; j < N; j++) {
        C.p_[j] = nz;  // column j of C starts here

        // Compute x = A @ B[:, j]
        for (csint p = B.p_[j]; p < B.p_[j+1]; p++) {
            // Compute x += A[:, B.i_[p]] * B.v_[p]
            nz = scatter(B.i_[p], B.v_[p], w, x, j+1, C, nz, fs);
            fs = false;
        }

        // Gather values into the correct locations in C
        for (csint p = C.p_[j]; p < nz; p++) {
            C.v_[p] = x[C.i_[p]];
        }
    }

    // Finalize and deallocate unused memory
    C.p_[N] = nz;
    C.realloc();

    return C;
}


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
CSCMatrix CSCMatrix::dot_2x(const CSCMatrix& B) const
{
    auto [M, Ka] = shape();
    auto [Kb, N] = B.shape();
    assert(Ka == Kb);

    // Allocate workspace
    std::vector<csint> w(M);

    // Compute nnz(A*B) by counting non-zeros in each column of C
    csint nz_C = 0;

    for (csint j = 0; j < N; j++) {
        csint mark = j + 1;
        for (csint p = B.p_[j]; p < B.p_[j+1]; p++) {
            // Scatter, but without x or C
            csint k = B.i_[p];  // B(k, j) is non-zero
            for (csint pa = p_[k]; pa < p_[k+1]; pa++) {
                csint i = i_[pa];     // A(i, k) is non-zero
                if (w[i] < mark) {
                    w[i] = mark;     // i is new entry in column k
                    nz_C++;         // count non-zeros in C, but don't compute
                }
            }
        }
    }

    // Allocate the correct size output matrix
    CSCMatrix C(M, N, nz_C);

    // Compute the actual multiplication
    std::fill(w.begin(), w.end(), 0);  // reset workspace
    std::vector<double> x(M);

    csint nz = 0;  // track total number of non-zeros in C
    bool fs = true;  // first call to scatter

    for (csint j = 0; j < N; j++) {
        C.p_[j] = nz;  // column j of C starts here

        // Compute x = A @ B[:, j]
        for (csint p = B.p_[j]; p < B.p_[j+1]; p++) {
            // Compute x += A[:, B.i_[p]] * B.v_[p]
            nz = scatter(B.i_[p], B.v_[p], w, x, j+1, C, nz, fs);
            fs = false;
        }

        // Gather values into the correct locations in C
        for (csint p = C.p_[j]; p < nz; p++) {
            C.v_[p] = x[C.i_[p]];
        }
    }

    // Finalize and deallocate unused memory
    C.p_[N] = nz;
    C.realloc();

    return C;
}

/** Multiply two sparse column vectors \f$ c = x^T y \f$.
 *
 * See: Davis, Exercise 2.18
 *
 * @param x, y two column vectors stored as a CSCMatrix. The number of columns
 *        in each argument must be 1.
 *
 * @return c  the dot product `x.T() * y`, but computed more efficiently than
 *         the complete matrix dot product.
 */
double CSCMatrix::vecdot(const CSCMatrix& y) const
{
    assert((N_ == 1) && (y.N_ == 1));  // both must be column vectors
    assert(M_ == y.M_);
    double z = 0.0;

    if (has_sorted_indices_ && y.has_sorted_indices_) {
        csint p = 0, q = 0;  // pointer to row index of each vector

        while ((p < nnz()) && (q < y.nnz())) {
            csint i = i_[p];    // row index of each vector
            csint j = y.i_[q];

            if (i == j) {
                z += v_[p++] * y.v_[q++];
            } else if (i < j) {
                p++;
            } else {  // (j < i)
                q++;
            }
        }

    } else {  // unsorted indices
        std::vector<double> w(M_);  // workspace

        // Expand this vector
        for (csint p = 0; p < nnz(); p++) {
            w[i_[p]] = v_[p];
        }

        // Multiply by non-zero entries in y and sum
        for (csint q = 0; q < y.nnz(); q++) {
            csint i = y.i_[q];
            if (w[i] != 0) {
                z += w[i] * y.v_[q];
            }
        }
    }

    return z;
}

// Operators
std::vector<double> operator*(const CSCMatrix& A, const std::vector<double>& x) { return A.dot(x); }
CSCMatrix operator*(const CSCMatrix& A, const double c) { return A.dot(c); }
CSCMatrix operator*(const double c, const CSCMatrix& A) { return A.dot(c); }
CSCMatrix operator*(const CSCMatrix& A, const CSCMatrix& B) { return A.dot(B); }


/** Add two matrices (and optionally scale them) `C = alpha * A + beta * B`.
 *
 * @note This function may *not* return a matrix with sorted columns!
 *
 * @param A, B  the CSC matrices
 * @param alpha, beta  scalar multipliers
 *
 * @return out a CSC matrix
 */
CSCMatrix add_scaled(
    const CSCMatrix& A,
    const CSCMatrix& B,
    double alpha=1.0,
    double beta=1.0
    )
{
    assert(A.shape() == B.shape());
    auto [M, N] = A.shape();

    CSCMatrix C(M, N, A.nnz() + B.nnz());  // output

    // Allocate workspaces
    std::vector<csint> w(M);
    std::vector<double> x(M);

    csint nz = 0;    // track total number of non-zeros in C
    bool fs = true;  // Exercise 2.19 -- first call to scatter

    for (csint j = 0; j < N; j++) {
        C.p_[j] = nz;  // column j of C starts here
        nz = A.scatter(j, alpha, w, x, j+1, C, nz, fs);  // alpha * A(:, j)
        fs = false;
        nz = B.scatter(j,  beta, w, x, j+1, C, nz, fs);  //  beta * B(:, j)

        // Gather results into the correct column of C
        for (csint p = C.p_[j]; p < nz; p++) {
            C.v_[p] = x[C.i_[p]];
        }
    }

    // Finalize and deallocate unused memory
    C.p_[N] = nz;
    C.realloc();

    return C;
}


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
std::vector<csint> saxpy(
    const CSCMatrix& a,
    const CSCMatrix& b,
    std::vector<csint>& w,
    std::vector<double>& x
    )
{
    assert(a.shape() == b.shape());
    assert((a.N_ == 1) && (b.N_ == 1));  // both must be column vectors

    for (csint p = 0; p < a.nnz(); p++) {
        csint i = a.i_[p];
        w[i] = 1;        // mark as non-zero
        x[i] = a.v_[p];  // copy x into w
    }

    for (csint p = 0; p < b.nnz(); p++) {
        csint i = b.i_[p];
        if (w[i] == 0) {
            w[i] = 1;         // mark as non-zero
            x[i] = b.v_[p];   // copy b into w
        } else {
            x[i] += b.v_[p];  // add b to x
        }
    }

    return w;
}


/** Add a matrix B. */
CSCMatrix CSCMatrix::add(const CSCMatrix& B) const
{
    return add_scaled(*this, B, 1.0, 1.0);
}


CSCMatrix operator+(const CSCMatrix& A, const CSCMatrix& B) { return A.add(B); }


/** Compute ``x += beta * A(:, j)``.
 *
 * This function also updates ``w``, sets the sparsity pattern in ``C._i``, and
 * returns updated ``nz``. The values corresponding to ``C._i`` are accumulated
 * in ``x``, and then gathered in the calling function, so that we can account
 * for any duplicate entries.
 *
 * @param A     CSC matrix by which to multiply
 * @param j     column index of `A`
 * @param beta  scalar value by which to multiply `A`
 * @param[in,out] w, x  workspace vectors of row indices and values, respectively
 * @param mark  separator index for `w`. All `w[i] < mark`are row indices that
 *              are not yet in `Cj`.
 * @param[in,out] C    CSC matrix where output non-zero pattern is stored
 * @param[in,out] nz   current number of non-zeros in `C`.
 * @param fs    first call to scatter
 *
 * @return nz  updated number of non-zeros in `C`.
 */
csint CSCMatrix::scatter(
    csint j,
    double beta,
    std::vector<csint>& w,
    std::vector<double>& x,
    csint mark,
    CSCMatrix& C,
    csint nz,
    bool fs
    ) const
{
    if (fs) {
        // If it's the first call, we can just copy the (scaled) column
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            csint i = i_[p];       // A(i, j) is non-zero
            w[i] = mark;             // i is new entry in column j
            C.i_[nz++] = i;          // add i to sparsity pattern of C(:, j)
            x[i] = beta * v_[p];   // x = beta * A(i, j)
        }
    } else {
        // Otherwise, we need to accumulate the values
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            csint i = i_[p];           // A(i, j) is non-zero
            if (w[i] < mark) {
                w[i] = mark;             // i is new entry in column j
                C.i_[nz++] = i;          // add i to pattern of C(:, j)
                x[i] = beta * v_[p];   // x = beta * A(i, j)
            } else {
                x[i] += beta * v_[p];  // i exists in C(:, j) already
            }
        }
    }

    return nz;
}


/** Permute a matrix \f$ C = PAQ \f$.
 *
 * @note In Matlab, this call is `C = A(p, q)`.
 *
 * @param p_inv, q  *inverse* row and (non-inverse) column permutation vectors.
 *        `p_inv` is length `M` and `q` is length `N`, where `A` is `M`-by-`N`.
 *
 * @return C  permuted matrix
 */
CSCMatrix CSCMatrix::permute(
    const std::vector<csint> p_inv,
    const std::vector<csint> q
    ) const
{
    CSCMatrix C(M_, N_, nnz());
    csint nz = 0;

    for (csint k = 0; k < N_; k++) {
        C.p_[k] = nz;                   // column k of C is column q[k] of A
        csint j = q[k];

        for (csint t = p_[j]; t < p_[j+1]; t++) {
            C.v_[nz] = v_[t];           // row i of A is row p_inv[i] of C
            C.i_[nz++] = p_inv[i_[t]];
        }
    }

    C.p_[N_] = nz;

    return C;
}


/** Permute the rows of a matrix.
 *
 * @note In Matlab, this call is `C = A(p, :)`.
 *
 * @param p_inv  *inverse* row permutation vector. `p_inv` is length `M`.
 *
 * @return C  permuted matrix
 */
CSCMatrix CSCMatrix::permute_rows(const std::vector<csint> p_inv) const
{
    std::vector<csint> q(N_);
    std::iota(q.begin(), q.end(), 0);  // identity permutation
    return permute(p_inv, q);
}


/** Permute the columns of a matrix.
 *
 * @note In Matlab, this call is `C = A(:, q)`.
 *
 * @param q  column permutation vector. `q` is length `N`.
 *
 * @return C  permuted matrix
 */
CSCMatrix CSCMatrix::permute_cols(const std::vector<csint> q) const
{
    std::vector<csint> p_inv(M_);
    std::iota(p_inv.begin(), p_inv.end(), 0);  // identity permutation
    return permute(p_inv, q);
}


/** Permute a symmetric matrix with only the upper triangular part stored.
 *
 * @param p_inv  *inverse* permutation vector. Both rows and columns are
 *        permuted with this vector to retain symmetry.
 *
 * @return C  permuted matrix
 */
CSCMatrix CSCMatrix::symperm(const std::vector<csint> p_inv) const
{
    assert(M_ == N_);  // matrix must be square. Symmetry not checked.

    CSCMatrix C(N_, N_, nnz());
    std::vector<csint> w(N_);  // workspace for column counts

    // Count entries in each column of C
    for (csint j = 0; j < N_; j++) {
        csint j2 = p_inv[j];  // column j of A is column j2 of C

        for (csint p = p_[j]; p < p_[j+1]; p++) {
            csint i = i_[p];

            if (i > j)
                continue;   // skip lower triangular part of A

            csint i2 = p_inv[i];    // row i of A is row i2 of C
            w[std::max(i2, j2)]++;  // column count of C
        }
    }

    // Row pointers are the cumulative sum of the counts, starting with 0.
    C.p_ = cumsum(w);
    w = C.p_;  // copy back into workspace

    for (csint j = 0; j < N_; j++) {
        csint j2 = p_inv[j];  // column j of A is column j2 of C

        for (csint p = p_[j]; p < p_[j+1]; p++) {
            csint i = i_[p];

            if (i > j)
                continue;   // skip lower triangular part of A

            csint i2 = p_inv[i];  // row i of A is row i2 of C
            csint q = w[std::max(i2, j2)]++;
            C.i_[q] = std::min(i2, j2);
            C.v_[q] = v_[p];
        }
    }

    return C;
}


/** Permute and transpose a matrix \f$ C = PA^TQ \f$.
 *
 * See: Davis, Exercise 2.26.
 *
 * @note In Matlab, this call is `C = A(p, q)'`.
 *
 * @param p_inv, q_inv  *inverse* row and column permutation vectors.
 *        `p_inv` is length `M` and `q` is length `N`, where `A` is `M`-by-`N`.
 *
 * @return C  permuted and transposed matrix
 */
CSCMatrix CSCMatrix::permute_transpose(
    const std::vector<csint>& p_inv,
    const std::vector<csint>& q_inv
    ) const
{
    std::vector<csint> w(M_);    // workspace
    CSCMatrix C(N_, M_, nnz());  // output

    // Compute number of elements in each permuted row (aka column of C)
    for (csint p = 0; p < nnz(); p++)
        w[p_inv[i_[p]]]++;

    C.p_ = cumsum(w);
    w = C.p_;  // copy back into workspace

    // place A(i, j) as C(j, i) (permuted)
    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            csint t = w[p_inv[i_[p]]]++;
            C.i_[t] = q_inv[j];
            C.v_[t] = v_[p];
        }
    }

    return C;
}


/** Compute the 1-norm of the matrix.
 *
 * The 1-norm is defined as \f$ \|A\|_1 = \max_j \sum_{i=1}^{m} |a_{ij}| \f$.
 */
double CSCMatrix::norm() const
{
    double the_norm = 0;

    for (csint j = 0; j < N_; j++) {
        double s = 0;

        for (csint p = p_[j]; p < p_[j+1]; p++) {
            s += std::fabs(v_[p]);
        }

        the_norm = std::max(the_norm, s);
    }

    return the_norm;
}


/** Check a matrix for valid compressed sparse column format.
 *
 * See: Davis, Exercise 2.12 "cs_ok"
 *
 * @param sorted  if true, check if columns are sorted.
 * @param values  if true, check if values exist and are all non-zero.
 *
 * @return true if matrix is valid compressed sparse column format.
 */
bool CSCMatrix::is_valid(const bool sorted, const bool values) const
{
    // Check number of columns
    if (p_.size() != (N_ + 1)) {
        // std::cout << "Columns inconsistent!" << std::endl;
        return false;
    }

    // TODO not sure how we're supposed to use O(M) space? Column counts can't
    // be independently checked.

    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            csint i = i_[p];

            if (i > M_) {
                // std::cout << "Invalid row index!" << std::endl;
                return false;  // invalid row index
            }

            if (sorted && (p < (p_[j+1] - 1)) && (i > i_[p+1])) {
                // std::cout << "Columns not sorted!" << std::endl;
                return false;  // not sorted in ascending order
            }

            if (values && v_[p] == 0.0) {
                // std::cout << "Explicit zeros!" << std::endl;
                return false;  // no zeros allowed
            }
        }
    }

    return true;
}


/** Concatenate two matrices horizontally.
 *
 * @note This function may *not* return a matrix with sorted columns!
 *
 * @param A, B  the CSC matrices to concatenate. They must have the same number
 *        of rows.
 *
 * @return C  the concatenated matrix.
 */
CSCMatrix hstack(const CSCMatrix& A, const CSCMatrix& B)
{
    assert(A.M_ == B.M_);

    // Copy the first matrix
    CSCMatrix C = A;
    C.N_ += B.N_;
    C.realloc(A.nnz() + B.nnz());

    // Copy the second matrix
    for (csint j = 0; j < B.N_; j++) {
        C.p_[A.N_ + j] = B.p_[j] + A.nnz();
    }

    std::copy(B.i_.begin(), B.i_.end(), C.i_.begin() + A.nnz());
    std::copy(B.v_.begin(), B.v_.end(), C.v_.begin() + A.nnz());

    C.p_[C.N_] = A.nnz() + B.nnz();

    if (!A.has_canonical_format_ || !B.has_canonical_format_) {
        C = C.to_canonical();
    }
    C.has_canonical_format_ = true;

    return C;
}

/** Concatenate two matrices vertically.
 *
 * @note This function may *not* return a matrix with sorted columns!
 *
 * @param A, B  the CSC matrices to concatenate. They must have the same number
 *        of columns.
 *
 * @return C  the concatenated matrix.
 */
CSCMatrix vstack(const CSCMatrix& A, const CSCMatrix& B)
{
    assert(A.N_ == B.N_);

    CSCMatrix C(A.M_ + B.M_, A.N_, A.nnz() + B.nnz());

    csint nz = 0;

    for (csint j = 0; j < C.N_; j++) {
        C.p_[j] = nz;  // column j of C starts here

        // Copy column j from the first matrix
        for (csint p = A.p_[j]; p < A.p_[j+1]; p++) {
            C.i_[nz] = A.i_[p];
            C.v_[nz] = A.v_[p];
            nz++;
        }

        // Copy column j from the second matrix
        for (csint p = B.p_[j]; p < B.p_[j+1]; p++) {
            C.i_[A.p_[j+1] + p] = A.M_ + B.i_[p];
            C.v_[A.p_[j+1] + p] = B.v_[p];
            nz++;
        }
    }

    C.p_[C.N_] = nz;

    return C.to_canonical();
}


/** Slice a matrix by row and column.
 *
 * @param i_start, i_end  the row indices to keep, where `i ∈ [i_start, i_end)`.
 * @param j_start, j_end  the column indices to keep, where `j ∈ [j_start,
 *        j_end)`.
 *
 * @return C  the submatrix A(i_start:i_end, j_start:j_end).
 */
CSCMatrix CSCMatrix::slice(
    const csint i_start,
    const csint i_end,
    const csint j_start,
    const csint j_end
    ) const
{
    assert((i_start >= 0) && (i_end <= M_) && (i_start < i_end));
    assert((j_start >= 0) && (j_end <= N_) && (j_start < j_end));

    CSCMatrix C(i_end - i_start, j_end - j_start, nnz());

    csint nz = 0;

    for (csint j = j_start; j < j_end; j++) {
        C.p_[j - j_start] = nz;  // column j of C starts here

        for (csint p = p_[j]; p < p_[j+1]; p++) {
            csint i = i_[p];

            if ((i >= i_start) && (i < i_end)) {
                C.i_[nz] = i - i_start;
                C.v_[nz] = v_[p];
                nz++;
            }
        }
    }

    C.p_[C.N_] = nz;
    C.realloc();

    if (!has_canonical_format_) {
        C = C.to_canonical();
    }
    C.has_canonical_format_ = true;

    return C;
}


/** Select a submatrix by row and column indices.
 *
 * This function takes O(|rows| + |cols|) + O(log M) time if the columns are
 * sorted, and + O(M) time if they are not.
 *
 * @param i, j vectors of the row and column indices to keep. The indices need
 *        not be consecutive, or sorted. Duplicates are allowed.
 *
 * @return C  the submatrix of A of dimension `length(i)`-by-`length(j)`.
 */
CSCMatrix CSCMatrix::index(
    const std::vector<csint>& rows,
    const std::vector<csint>& cols
    ) const
{
    CSCMatrix C(rows.size(), cols.size(), nnz());

    csint nz = 0;

    for (csint j = 0; j < cols.size(); j++) {
        C.p_[j] = nz;  // column j of C starts here

        // Iterate over `rows` and find the corresponding indices in `i_`.
        for (csint k = 0; k < rows.size(); k++) {
            double val = (*this)(rows[k], cols[j]);
            if (val != 0) {
                C.i_[nz] = k;
                C.v_[nz] = val;
                nz++;
            }
        }
    }

    C.p_[C.N_] = nz;
    C.realloc();
    
    // Canonical format guaranteed by construction
    C.has_canonical_format_ = true;

    return C;
}


/** Add empty rows to the top of the matrix.
 *
 * See: Davis, Exercise 2.29.
 *
 * @param k  the number of rows to add.
 *
 * @return C  the matrix with `k` empty rows added to the top.
 */
CSCMatrix CSCMatrix::add_empty_top(const csint k) const
{
    CSCMatrix C = *this;  // copy the matrix
    C.M_ += k;

    // Increate all row indices by k
    for (auto& i : C.i_) {
        i += k;
    }

    return C;
}


/** Add empty rows to the bottom of the matrix.
 *
 * See: Davis, Exercise 2.29.
 *
 * @param k  the number of rows to add.
 *
 * @return C  the matrix with `k` empty rows added to the bottom.
 */
CSCMatrix CSCMatrix::add_empty_bottom(const csint k) const
{
    CSCMatrix C = *this;  // copy the matrix
    C.M_ += k;
    return C;
}


/** Add empty rows to the left of the matrix.
 *
 * See: Davis, Exercise 2.29.
 *
 * @param k  the number of rows to add.
 *
 * @return C  the matrix with `k` empty rows added to the left.
 */
CSCMatrix CSCMatrix::add_empty_left(const csint k) const
{
    CSCMatrix C = *this;  // copy the matrix
    C.N_ += k;
    C.p_.insert(C.p_.begin(), k, 0);  // insert k zeros at the beginning
    return C;
}


/** Add empty rows to the right of the matrix.
 *
 * See: Davis, Exercise 2.29.
 *
 * @param k  the number of rows to add.
 *
 * @return C  the matrix with `k` empty rows added to the right.
 */
CSCMatrix CSCMatrix::add_empty_right(const csint k) const
{
    CSCMatrix C = *this;  // copy the matrix
    C.N_ += k;
    C.p_.insert(C.p_.end(), k, nnz());  // insert k nnz() at the end
    return C;
}


/** Sum the rows of a matrix.
 *
 * @return out  a vector of length `M` containing the sum of each row.
 */
std::vector<double> CSCMatrix::sum_rows() const
{
    std::vector<double> out(M_, 0.0);

    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            out[i_[p]] += v_[p];
        }
    }

    return out;
}


/** Sum the columns of a matrix.
 *
 * @return out  a vector of length `N` containing the sum of each column.
 */
std::vector<double> CSCMatrix::sum_cols() const
{
    std::vector<double> out(N_, 0.0);

    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            out[j] += v_[p];
        }
    }

    return out;
}


/*------------------------------------------------------------------------------
 *      Matrix Solutions 
 *----------------------------------------------------------------------------*/
/** Forward solve a lower-triangular system \f$ Lx = b \f$.
 *
 * @note This function assumes that the diagonal entry of `L` is always present
 * and is the first entry in each column. Otherwise, the row indices in each
 * column of `L` may appear in any order.
 *
 * @param L  a lower-triangular matrix
 * @param b  a dense vector
 *
 * @return x  the solution vector
 */
std::vector<double> CSCMatrix::lsolve(const std::vector<double>& b) const
{
    assert(M_ == N_);
    assert(M_ == b.size());

    std::vector<double> x = b;

    for (csint j = 0; j < N_; j++) {
        x[j] /= v_[p_[j]];
        for (csint p = p_[j] + 1; p < p_[j+1]; p++) {
            x[i_[p]] -= v_[p] * x[j];
        }
    }

    return x;
}


/** Backsolve a lower-triangular system \f$ L^Tx = b \f$.
 *
 * @note This function assumes that the diagonal entry of `L` is always present
 * and is the first entry in each column. Otherwise, the row indices in each
 * column of `L` may appear in any order.
 *
 * @param L  a lower-triangular matrix
 * @param b  a dense vector
 *
 * @return x  the solution vector
 */
std::vector<double> CSCMatrix::ltsolve(const std::vector<double>& b) const
{
    assert(M_ == N_);
    assert(M_ == b.size());

    std::vector<double> x = b;

    for (csint j = N_ - 1; j >= 0; j--) {
        for (csint p = p_[j] + 1; p < p_[j+1]; p++) {
            x[j] -= v_[p] * x[i_[p]];
        }
        x[j] /= v_[p_[j]];
    }

    return x;
}


/** Backsolve an upper-triangular system \f$ Ux = b \f$.
 *
 * @note This function assumes that the diagonal entry of `U` is always present
 * and is the last entry in each column. Otherwise, the row indices in each
 * column of `U` may appear in any order.
 *
 * @param U  an upper-triangular matrix
 * @param b  a dense vector
 *
 * @return x  the solution vector
 */
std::vector<double> CSCMatrix::usolve(const std::vector<double>& b) const
{
    assert(M_ == N_);
    assert(M_ == b.size());

    std::vector<double> x = b;

    for (csint j = N_ - 1; j >= 0; j--) {
        x[j] /= v_[p_[j+1] - 1];  // diagonal entry
        for (csint p = p_[j]; p < p_[j+1] - 1; p++) {
            x[i_[p]] -= v_[p] * x[j];
        }
    }

    return x;
}


/** Forward solve an upper-triangular system \f$ U^T x = b \f$.
 *
 * @note This function assumes that the diagonal entry of `U` is always present
 * and is the last entry in each column. Otherwise, the row indices in each
 * column of `U` may appear in any order.
 *
 * @param U  an upper-triangular matrix
 * @param b  a dense vector
 *
 * @return x  the solution vector
 */
std::vector<double> CSCMatrix::utsolve(const std::vector<double>& b) const
{
    assert(M_ == N_);
    assert(M_ == b.size());

    std::vector<double> x = b;

    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j+1] - 1; p++) {
            x[j] -= v_[p] * x[i_[p]];
        }
        x[j] /= v_[p_[j+1] - 1];  // diagonal entry
    }

    return x;
}


/** Forward solve a lower-triangular system \f$ Lx = b \f$.
 *
 * See: Davis, Exercise 3.8
 *
 * @note This function assumes that the diagonal entry of `L` is always present
 * and is the first entry in each column. Otherwise, the row indices in each
 * column of `L` may appear in any order.
 *
 * @param L  a lower-triangular matrix
 * @param b  a dense vector
 *
 * @return x  the solution vector
 */
std::vector<double> CSCMatrix::lsolve_opt(const std::vector<double>& b) const
{
    assert(M_ == N_);
    assert(M_ == b.size());

    std::vector<double> x = b;

    for (csint j = 0; j < N_; j++) {
        x[j] /= v_[p_[j]];

        // Exercise 3.8: improve performance by checking for zeros
        double x_val = x[j];  // cache value
        if (x_val != 0) {
            for (csint p = p_[j] + 1; p < p_[j+1]; p++) {
                x[i_[p]] -= v_[p] * x_val;
            }
        }
    }

    return x;
}


/** Backsolve an upper-triangular system \f$ Ux = b \f$.
 *
 * @note This function assumes that the diagonal entry of `U` is always present
 * and is the last entry in each column. Otherwise, the row indices in each
 * column of `U` may appear in any order.
 *
 * @param U  an upper-triangular matrix
 * @param b  a dense vector
 *
 * @return x  the solution vector
 */
std::vector<double> CSCMatrix::usolve_opt(const std::vector<double>& b) const
{
    assert(M_ == N_);
    assert(M_ == b.size());

    std::vector<double> x = b;

    for (csint j = N_ - 1; j >= 0; j--) {
        x[j] /= v_[p_[j+1] - 1];  // diagonal entry

        double x_val = x[j];  // cache value
        if (x_val != 0) {
            for (csint p = p_[j]; p < p_[j+1] - 1; p++) {
                x[i_[p]] -= v_[p] * x_val;
            }
        }
    }

    return x;
}


/** Find the diagonal indices of a row-permuted lower triangular matrix.
 *
 * See: Davis, Exercise 3.3
 *
 * @return p_diags  a vector of pointers to the indices of the diagonal entries.
 */
std::vector<csint> CSCMatrix::find_lower_diagonals() const
{
    assert(M_ == N_);

    std::vector<bool> is_marked(N_, false);  // workspace
    std::vector<csint> p_diags(N_);  // diagonal indicies (inverse permutation)

    for (csint j = N_ - 1; j >= 0; j--) {
        csint N_unmarked = 0;

        for (csint p = p_[j]; p < p_[j+1]; p++) {
            csint i = i_[p];
            // Mark the rows viewed so far
            if (!is_marked[i]) {
                is_marked[i] = true;
                p_diags[j] = p;
                N_unmarked++;
            }
        }

        // If 0 or > 1 "diagonal" entries found, the matrix is not permuted.
        if (N_unmarked != 1) {
            throw std::runtime_error("Matrix is not a permuted lower triangular matrix!");
        }
    }

    return p_diags;
}


/** Solve Lx = b with a row-permuted L. The permutation is unknown.
 *
 * See: Davis, Exercise 3.3
 *
 * @param b  a dense RHS vector, *not* permuted.
 *
 * @return x  the dense solution vector, also *not* permuted.
 */
std::vector<double> CSCMatrix::lsolve_rows(const std::vector<double>& b) const
{
    assert(M_ == N_);
    assert(M_ == b.size());

    // First (backward) pass to find diagonal entries
    // p_diags is a vector of pointers to the diagonal entries
    std::vector<csint> p_diags = find_lower_diagonals();

    // Compute the row permutation vector
    std::vector<csint> permuted_rows(N_);
    for (csint i = 0; i < N_; i++) {
        permuted_rows[i_[p_diags[i]]] = i;
    }

    // Second (forward) pass to solve the system
    std::vector<double> x = b;

    // Perform the permuted forward solve
    for (csint j = 0; j < N_; j++) {
        csint d = p_diags[j];  // pointer to the diagonal entry
        double& x_val = x[j];  // cache diagonal value

        if (x_val != 0) {
            x_val /= v_[d];    // solve for x[d]
            for (csint p = p_[j]; p < p_[j+1]; p++) {
                csint i = permuted_rows[i_[p]];
                if (p != d) {
                    x[i] -= v_[p] * x_val;  // update the off-diagonals
                }
            }
        }
    }

    return x;
}


/** Solve Lx = b with a column-permuted L. The permutation is unknown.
 *
 * See: Davis, Exercise 3.5
 *
 * @param b  a dense RHS vector, *not* permuted.
 *
 * @return x  the dense solution vector, also *not* permuted.
 */
std::vector<double> CSCMatrix::lsolve_cols(const std::vector<double>& b) const
{
    assert(M_ == N_);
    assert(M_ == b.size());

    // First O(N) pass to find the diagonal entries
    // Assume that the first entry in each column has the smallest row index
    std::vector<csint> p_diags(N_, -1);
    for (csint j = 0; j < N_; j++) {
        if (p_diags[j] == -1) {
            p_diags[j] = p_[j];  // pointer to the diagonal entry
        } else {
            // We have seen this column index before
            throw std::runtime_error("Matrix is not a permuted lower triangular matrix!");
        }
    }

    // Compute the column permutation vector
    std::vector<csint> permuted_cols(N_);
    for (csint i = 0; i < N_; i++) {
        permuted_cols[i_[p_diags[i]]] = i;
    }

    // Second (forward) pass to solve the system
    std::vector<double> x = b;

    // Perform the permuted forward solve
    for (const auto& j : permuted_cols) {
        csint d = p_diags[j];      // pointer to the diagonal entry
        double& x_val = x[i_[d]];  // cache diagonal value

        if (x_val != 0) {
            x_val /= v_[d];  // solve for x[i_[d]]
            for (csint p = p_[j]+1; p < p_[j+1]; p++) {
                x[i_[p]] -= v_[p] * x_val;  // update the off-diagonals
            }
        }
    }

    return x;
}


/** Find the diagonal indices of a row-permuted upper triangular matrix.
 *
 * See: Davis, Exercise 3.4
 *
 * @return p_diags  a vector of pointers to the indices of the diagonal entries.
 */
std::vector<csint> CSCMatrix::find_upper_diagonals() const
{
    assert(M_ == N_);

    std::vector<bool> is_marked(N_, false);  // workspace
    std::vector<csint> p_diags(N_);  // diagonal indicies (inverse permutation)

    for (csint j = 0; j < N_; j++) {
        csint N_unmarked = 0;

        for (csint p = p_[j]; p < p_[j+1]; p++) {
            csint i = i_[p];
            // Mark the rows viewed so far
            if (!is_marked[i]) {
                is_marked[i] = true;
                p_diags[j] = p;
                N_unmarked++;
            }
        }

        // If 0 or > 1 "diagonal" entries found, the matrix is not permuted.
        if (N_unmarked != 1) {
            throw std::runtime_error("Matrix is not a permuted upper triangular matrix!");
        }
    }

    return p_diags;
}


/** Solve Ux = b with a row-permuted U. The permutation is unknown.
 *
 * See: Davis, Exercise 3.4
 *
 * @param b  a dense RHS vector, *not* permuted.
 *
 * @return x  the dense solution vector, also *not* permuted.
 */
std::vector<double> CSCMatrix::usolve_rows(const std::vector<double>& b) const
{
    assert(M_ == N_);
    assert(M_ == b.size());

    // First (backward) pass to find diagonal entries
    // p_diags is a vector of pointers to the diagonal entries
    std::vector<csint> p_diags = find_upper_diagonals();

    // Compute the row permutation vector
    std::vector<csint> permuted_rows(N_);
    for (csint i = 0; i < N_; i++) {
        permuted_rows[i_[p_diags[i]]] = i;
    }

    // Second (forward) pass to solve the system
    std::vector<double> x = b;

    // Perform the permuted forward solve
    for (csint j = N_ - 1; j >= 0; j--) {
        csint d = p_diags[j];  // pointer to the diagonal entry
        double& x_val = x[j];  // cache diagonal value

        if (x_val != 0) {
            x_val /= v_[d];    // solve for x[d]
            for (csint p = p_[j]; p < p_[j+1]; p++) {
                csint i = permuted_rows[i_[p]];
                if (p != d) {
                    x[i] -= v_[p] * x_val;  // update the off-diagonals
                }
            }
        }
    }

    return x;
}


/** Solve Ux = b with a column-permuted U. The permutation is unknown.
 *
 * See: Davis, Exercise 3.6
 *
 * @param b  a dense RHS vector, *not* permuted.
 *
 * @return x  the dense solution vector, also *not* permuted.
 */
std::vector<double> CSCMatrix::usolve_cols(const std::vector<double>& b) const
{
    assert(M_ == N_);
    assert(M_ == b.size());

    // First O(N) pass to find the diagonal entries
    // Assume that the last entry in each column has the largest row index
    std::vector<csint> p_diags(N_, -1);
    for (csint j = 0; j < N_; j++) {
        if (p_diags[j] == -1) {
            p_diags[j] = p_[j+1] - 1;  // pointer to the diagonal entry
        } else {
            // We have seen this column index before
            throw std::runtime_error("Matrix is not a permuted lower triangular matrix!");
        }
    }

    // Compute the column permutation vector
    std::vector<csint> permuted_cols(N_);
    for (csint i = 0; i < N_; i++) {
        permuted_cols[i_[p_diags[i]]] = i;
    }

    // Second (forward) pass to solve the system
    std::vector<double> x = b;

    // Perform the permuted forward solve
    for (const auto& j : std::views::reverse(permuted_cols)) {
        csint d = p_diags[j];      // pointer to the diagonal entry
        double& x_val = x[i_[d]];  // cache diagonal value

        if (x_val != 0) {
            x_val /= v_[d];  // solve for x[i_[d]]
            for (csint p = p_[j]; p < p_[j+1] - 1; p++) {
                x[i_[p]] -= v_[p] * x_val;  // update the off-diagonals
            }
        }
    }

    return x;
}


/** Find the permutation vectors of a permuted triangular matrix.
 *
 * See: Davis, Exercise 3.7
 *
 * @return p_inv, q_inv  the inverse row and column permutation vectors.
 */
std::pair<std::vector<csint>, std::vector<csint>> CSCMatrix::find_tri_permutation() const
{
    assert(M_ == N_);

    // Create a vector of row counts and corresponding set vector
    std::vector<csint> r(N_, 0);
    std::vector<csint> z(N_, 0);  // z[i] is XORed with each column j in row i

    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            r[i_[p]]++;
            z[i_[p]] ^= j;
        }
    }

    // Create a list of singleton row indices
    std::vector<csint> singles;
    singles.reserve(N_);

    for (csint i = 0; i < N_; i++) {
        if (r[i] == 1) {
            singles.push_back(i);
        }
    }

    // Iterate through the columns to get the permutation vectors
    std::vector<csint> p_inv(N_, -1);
    std::vector<csint> q_inv(N_, -1);

    for (csint k = 0; k < N_; k++) {
        // Take a singleton row
        if (singles.empty()) {
            throw std::runtime_error("Matrix is not a permuted triangular matrix!");
        }

        csint i = singles.back();
        singles.pop_back();
        csint j = z[i];  // column index

        // Update the permutations
        p_inv[k] = i;
        q_inv[k] = j;

        // Decrement each row count, and update the set vector
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            csint t = i_[p];
            if (--r[t] == 1) {
                singles.push_back(t);
            }
            z[t] ^= j;  // removes j from the set
        }
    }

    return std::make_pair(p_inv, q_inv);
}


/** Solve a permuted triangular system \f$ PAQx = b \f$.
 *
 * See: Davis, Exercise 3.7
 *
 * @param b  a dense RHS vector, *not* permuted.
 * @param p_inv  the inverse row permutation vector.
 * @param q  the column permutation vector.
 *
 * @return x  the dense solution vector, also *not* permuted.
 */
std::vector<double> CSCMatrix::tri_solve_perm(
    const std::vector<double>& b,
    const std::vector<csint>& p_inv,
    const std::vector<csint>& q) const
{
    assert(M_ == N_);
    assert(M_ == b.size());
    assert(N_ == p_inv.size());
    assert(N_ == q.size());

    // Copy the RHS vector
    std::vector<double> x = b;

    // NOTE ASSUME LOWER TRIANGULAR FOR NOW
    for (csint k = 0; k < N_; k++) {
        csint j = q[k];    // permuted column
        double& x_val = x[k];  // un-permuted row of x
        if (x_val != 0) {
            x_val /= v_[p_[j]];  // diagonal entry
            for (csint p = p_[j] + 1; p < p_[j+1]; p++) {
                x[p_inv[i_[p]]] -= v_[p] * x_val;  // off-diagonals
            }
        }
    }

    return x;
}


/** Solve a row- and column-permuted triangular system P A Q x = b, for unknown
 * P and Q.
 *
 * See: Davis, Exercise 3.7
 *
 * @param b  a dense RHS vector, *not* permuted.
 *
 * @return x  the dense solution vector, also *not* permuted.
 */
std::vector<double> CSCMatrix::tri_solve_perm(const std::vector<double>& b) const
{
    assert(M_ == N_);
    assert(M_ == b.size());

    // Get the permutation vectors
    auto [p_inv, q_inv] = find_tri_permutation();

    // return tri_solve_perm(b, p_inv, inv_permute(q_inv));
    return permute(p_inv, inv_permute(q_inv)).lsolve(b);
}


/** Solve a triangular system \f$ Lx = b_k \f$ for column `k` of `B`.
 *
 * @note If `lo` is non-zero, this function assumes that the diagonal entry of
 * `L` is always present and is the first entry in each column. Otherwise, the
 * row indices in each column of `L` may appear in any order.
 * If `lo` is zero, the function assumes that the diagonal entry of `U` is
 * always present and is the last entry in each column.
 *
 * @param B  a dense matrix
 * @param k  the column index of `B` to solve
 * @param xi[out]  the row indices of the non-zero entries in `x`. This is
 *        a vector of length `2*G.N_` that is also used as a workspace. The
 *        first `G.N_` entries hold the output stack and the recursion stack for
 *        `j`. The second `G.N_` entries hold the stack for `p` in `dfs`.
 *        The row indices of the non-zero entries in `x` are stored in
 *        `xi[top:G.N_-1]` on output.
 * @param x[out]  the numerical values of the solution vector
 * @param lo  the lower bound of the diagonal entries of `G`. If `lo` is
 *        non-zero, the function solves \f$ Lx = b_k`, otherwise it solves
 *        \f$ Ux = b_k \f$.
 *
 * @return top  the index of `xi` where the non-zero entries of `x` begin. They
 *         are located from `top` through `G.N_ - 1`.
 */
std::pair<std::vector<csint>, std::vector<double>> CSCMatrix::spsolve(
    const CSCMatrix& B,
    csint k,
    bool lo
    ) const
{
    // Populate xi with the non-zero indices of x
    std::vector<csint> xi = reach(B, k);
    std::vector<double> x(N_);  // dense output vector

    // Clear non-zeros of x
    for (auto& i : xi) {
        x[i] = 0.0;
    }

    // scatter B(:, k) into x
    for (csint p = B.p_[k]; p < B.p_[k+1]; p++) {
        x[B.i_[p]] = B.v_[p];
    }

    // Solve Lx = b_k or Ux = b_k
    for (auto& j : xi) {  // x(j) is nonzero
        csint J = j;  // j maps to col J of G (NOTE ignore for now)
        if (J < 0) {
            continue;                                // x(j) is not in the pattern of G
        }
        x[j] /= v_[lo ? p_[J] : p_[J+1] - 1];  // x(j) /= G(j, j)
        csint p = lo ? p_[J] + 1 : p_[J];        // lo: L(j,j) 1st entry
        csint q = lo ? p_[J+1]   : p_[J+1] - 1;  // up: U(j,j) last entry
        for (; p < q; p++) {
            x[i_[p]] -= v_[p] * x[j];            // x[i] -= G(i, j) * x[j]
        }
    }

    return std::make_pair(xi, x);
}


/** Compute the reachability indices of a column `k` in a sparse matrix `B`,
 * given a sparse matrix `G`.
 *
 * @param B  a sparse matrix containing the RHS in column `k`
 * @param k  the column index of `B` containing the RHS
 * 
 * @return xi  the row indices of the non-zero entries in `x`, in topological
 *      order of the graph of 
 */
std::vector<csint> CSCMatrix::reach(const CSCMatrix& B, csint k) const
{
    std::vector<bool> is_marked(N_, false);
    std::vector<csint> xi;  // do not initialize for dfs call!
    xi.reserve(N_);

    for (csint p = B.p_[k]; p < B.p_[k+1]; p++) {
        csint j = B.i_[p];  // consider nonzero B(j, k)
        if (!is_marked[j]) {
            xi = dfs(j, is_marked, xi);
        }
    }

    // xi is returned from dfs in reverse order, since it is a stack
    xi.shrink_to_fit();                  // free unused memory
    std::reverse(xi.begin(), xi.end());  // O(N)
    return xi;
}


/** Perform depth-first search on a graph.
 *
 * @param j  the starting node
 * @param is_marked  a boolean vector of length `N_` that marks visited nodes
 * @param[in,out] xi  the row indices of the non-zero entries in `x`. This
 *      vector is used as a stack to store the output. It should not be
 *      initialized, other than by a previous call to `dfs`.
 *
 * @return xi  a reference to the row indices of the non-zero entries in `x`.
 */
std::vector<csint>& CSCMatrix::dfs(
    csint j,
    std::vector<bool>& is_marked,
    std::vector<csint>& xi
    ) const
{
    std::vector<csint> rstack, pstack;  // recursion and pause stacks
    rstack.reserve(N_);
    pstack.reserve(N_);

    rstack.push_back(j);       // initialize the recursion stack

    bool done = false;  // true if no unvisited neighbors

    while (!rstack.empty()) {
        j = rstack.back();  // get j from the top of the recursion stack
        csint jnew = j;  // j maps to col jnew of G (NOTE ignore p_inv for now)

        if (!is_marked[j]) {
            is_marked[j] = true;  // mark node j as visited
            pstack.push_back((jnew < 0) ? 0 : p_[jnew]);
        }

        done = true;  // node j done if no unvisited neighbors
        csint q = (jnew < 0) ? 0 : p_[jnew+1];

        // examine all neighbors of j
        for (csint p = pstack.back(); p < q; p++) {
            csint i = i_[p];        // consider neighbor node i
            if (!is_marked[i]) {
                pstack.back() = p;    // pause dfs of node j
                rstack.push_back(i);  // start dfs at node i
                done = false;         // node j has unvisited neighbors
                break;
            }
        }

        if (done) {
            pstack.pop_back();
            rstack.pop_back();  // node j is done; pop it from the stack
            xi.push_back(j);    // node j is the next on the output stack
        }
    }

    return xi;
}


/*------------------------------------------------------------------------------
 *         Printing
 *----------------------------------------------------------------------------*/
/** Print the matrix in dense format.
 *
 * @param os  a reference to the output stream.
 *
 * @return os  a reference to the output stream.
 */
void CSCMatrix::print_dense(std::ostream& os) const
{
    print_dense_vec(toarray(), M_, N_, os, 'C');
}


/** Print elements of the matrix between `start` and `end`.
 *
 * @param os          the output stream, defaults to std::cout
 * @param start, end  print the all elements where `p ∈ [start, end]`, counting
 *        column-wise.
 */
void CSCMatrix::print_elems_(std::ostream& os, const csint start, const csint end) const
{
    csint n = 0;  // number of elements printed
    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j + 1]; p++) {
            if ((n >= start) && (n < end)) {
                os << "(" << i_[p] << ", " << j << "): " << v_[p] << std::endl;
            }
            n++;
        }
    }
}


/** Print the matrix
 *
 * @param os          the output stream, defaults to std::cout
 * @param verbose     if True, print all non-zeros and their coordinates
 * @param threshold   if `nz > threshold`, print only the first and last
 *        3 entries in the matrix. Otherwise, print all entries.
 */
void CSCMatrix::print(std::ostream& os, const bool verbose, const csint threshold) const
{
    csint nnz_ = nnz();
    os << "<" << format_desc_ << " matrix" << std::endl;
    os << "        with " << nnz_ << " stored elements "
        << "and shape (" << M_ << ", " << N_ << ")>" << std::endl;

    if (verbose) {
        if (nnz_ < threshold) {
            // Print all elements
            print_elems_(os, 0, nnz_);  // FIXME memory leak?
        } else {
            // Print just the first and last 3 non-zero elements
            print_elems_(os, 0, 3);
            os << "..." << std::endl;
            print_elems_(os, nnz_ - 3, nnz_);
        }
    }
}


std::ostream& operator<<(std::ostream& os, const CSCMatrix& A)
{
    A.print(os, true);  // verbose printing assumed
    return os;
}


}  // namespace cs

/*==============================================================================
 *============================================================================*/
