/*==============================================================================
 *     File: csc.cpp
 *  Created: 2024-10-09 20:58
 *   Author: Bernie Roesler
 *
 *  Description: Implements the compressed sparse column matrix class
 *
 *============================================================================*/

#include <algorithm>
#include <cmath>
#include <span>

#include "csparse.h"


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
    const std::array<csint, 2>& shape
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
CSCMatrix::CSCMatrix(const COOMatrix& A)
{
    CSCMatrix C = A.compress();
    p_ = C.p_;
    i_ = C.i_;
    v_ = C.v_;
    M_ = C.M_;
    N_ = C.N_;

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

    // TODO review uses and potentially use "reserve" when Z > nnz()
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

std::array<csint, 2> CSCMatrix::shape() const
{
    return std::array<csint, 2> {M_, N_};
}

const std::vector<csint>& CSCMatrix::indices() const { return i_; }
const std::vector<csint>& CSCMatrix::indptr() const { return p_; }
const std::vector<double>& CSCMatrix::data() const { return v_; }

CSCMatrix CSCMatrix::to_canonical() const
{
    CSCMatrix C = *this;
    C.sum_duplicates();
    C.dropzeros();
    C.sort();
    C.has_canonical_format_ = true;
    return C;
}
bool CSCMatrix::has_sorted_indices() const { return has_sorted_indices_; }
bool CSCMatrix::has_canonical_format() const { return has_canonical_format_; }


/** Return the value of the requested element */
const double CSCMatrix::operator()(csint i, csint j) const
{
    // Assert indices are in-bounds
    assert(i >= 0 && i < M_);
    assert(j >= 0 && j < N_);

    if (has_canonical_format_) {
        csint p = p_[j];  // pointer to the row indices of column j
        std::span rows{i_.begin() + p, p_[j+1] - p};  // view of row indices

        // Binary search for t <= i
        auto t = std::lower_bound(rows.begin(), rows.end(), i);

        // Check that we actually found the index t == i
        if (t != rows.end() && *t == i) {
            auto idx = std::distance(rows.begin(), t);
            return v_[p + idx];
        } else {
            return 0.0;
        }

    } else {
        // NOTE this code assumes that columns are *not* sorted, and that
        // duplicate entries may exist, so it will search through *every*
        // element in a column.
        double out = 0;

        for (csint p = p_[j]; p < p_[j+1]; p++) {
            if (i_[p] == i) {
                out += v_[p];  // sum duplicate entries
            }
        }

        return out;
    }
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
    // Also copy the cumulative sum back into the workspace for iteration
    C.p_ = cumsum(w);

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
    // Also copy the cumulative sum back into the workspace for iteration
    C.p_ = cumsum(w);

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
    std::vector<int> w(M_, -1);                      // row i not yet seen

    for (csint j = 0; j < N_; j++) {
        int q = nz;                                  // column j will start at q
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
    int ii = (int) i,  // cast to signed int to preserve signs
        jj = (int) j;
    int kl, ku;        // extract lower/upper diagonals
    std::tie(kl, ku) = *((std::array<int, 2> *) limits);
    return ((ii <= (jj - kl)) && (ii >= (jj - ku)));
};


/** Keep any entries within the specified band.
 *
 * @param kl, ku  the lower and upper diagonals within which to keep entries.
 * The main diagonal is 0, with sub-diagonals < 0, and super-diagonals > 0.
 *
 * @return a copy of the matrix with entries removed.
 */
CSCMatrix CSCMatrix::band(const int kl, const int ku)
{
    assert(kl <= ku);
    std::array<int, 2> limits {kl, ku};

    return fkeep(&in_band, &limits);
}


/*------------------------------------------------------------------------------
       Math Operations
----------------------------------------------------------------------------*/
/** Matrix-vector multiply `y = Ax + y`.
 *
 * @param A  a sparse matrix
 * @param x  a dense multiplying vector
 * @param y  a dense adding vector which will be used for the output
 *
 * @return y a copy of the updated vector
 */
std::vector<double> gaxpy(
    const CSCMatrix& A,
    const std::vector<double>& x,
    std::vector<double> y
    )
{
    assert(A.M_ == y.size());  // addition
    assert(A.N_ == x.size());  // multiplication
    for (csint j = 0; j < A.N_; j++) {
        for (csint p = A.p_[j]; p < A.p_[j+1]; p++) {
            y[A.i_[p]] += A.v_[p] * x[j];
        }
    }
    return y;
};


/** Matrix transpose-vector multiply `y = A.T x + y`.
 *
 * See: Davis, Exercise 2.1. Compute \f$ A^T x + y \f$ without explicitly
 * computing the transpose.
 *
 * @param A  a sparse matrix
 * @param x  a dense multiplying vector
 * @param y[in,out]  a dense adding vector which will be used for the output
 *
 * @return y a copy of the updated vector
 */
std::vector<double> gatxpy(
    const CSCMatrix& A,
    const std::vector<double>& x,
    std::vector<double> y
    )
{
    assert(A.M_ == x.size());  // multiplication
    assert(A.N_ == y.size());  // addition
    for (csint j = 0; j < A.N_; j++) {
        for (csint p = A.p_[j]; p < A.p_[j+1]; p++) {
            y[j] += A.v_[p] * x[A.i_[p]];
        }
    }
    return y;
};


/** Matrix-vector multiply `y = Ax + y` for symmetric A (\f$ A = A^T \f$).
 *
 * See: Davis, Exercise 2.3.
 *
 * @param A  a sparse symmetric matrix, only `A(i, j)` where `j >= i` is stored.
 * @param x  a dense multiplying vector
 * @param y  a dense adding vector which will be used for the output
 *
 * @return y a copy of the updated vector
 */
std::vector<double> sym_gaxpy(
    const CSCMatrix& A,
    const std::vector<double>& x,
    std::vector<double> y
    )
{
    assert(A.M_ == A.N_);  // matrix must be square to be symmetric
    assert(A.N_ == x.size());
    assert(x.size() == y.size());
    for (csint j = 0; j < A.N_; j++) {
        for (csint p = A.p_[j]; p < A.p_[j+1]; p++) {
            csint i = A.i_[p];

            if (i > j)
                continue;  // skip lower triangular

            // Add the upper triangular elements
            y[i] += A.v_[p] * x[j];

            // If off-diagonal, also add the symmetric element
            if (i < j)
                y[j] += A.v_[p] * x[i];
        }
    }
    return y;
};


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


// TODO test left-multiply
// std::vector<double> operator*(const std::vector<double>& x, const CSCMatrix& A)
// {
//     return (A.T() * x).T
// }

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
    csint M, Ka, Kb, N;
    std::tie(M, Ka) = shape();
    std::tie(Kb, N) = B.shape();
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
            nz = scatter(*this, B.i_[p], B.v_[p], w, x, j+1, C, nz, fs);
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

    // TODO put into canonical format and test
    // C = C.dropzeros().sort();
    // C.has_canonical_format_ = true;

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
    csint M, Ka, Kb, N;
    std::tie(M, Ka) = shape();
    std::tie(Kb, N) = B.shape();
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
            nz = scatter(*this, B.i_[p], B.v_[p], w, x, j+1, C, nz, fs);
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

    // TODO put into canonical format and test
    // C = C.dropzeros().sort();
    // C.has_canonical_format_ = true;

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
    csint M, N;
    std::tie(M, N) = A.shape();

    CSCMatrix C(M, N, A.nnz() + B.nnz());  // output

    // Allocate workspaces
    std::vector<csint> w(M);
    std::vector<double> x(M);

    csint nz = 0;    // track total number of non-zeros in C
    bool fs = true;  // Exercise 2.19 -- first call to scatter

    for (csint j = 0; j < N; j++) {
        C.p_[j] = nz;  // column j of C starts here
        nz = scatter(A, j, alpha, w, x, j+1, C, nz, fs);  // alpha * A(:, j)
        fs = false;
        nz = scatter(B, j,  beta, w, x, j+1, C, nz, fs);  //  beta * B(:, j)

        // Gather results into the correct column of C
        for (csint p = C.p_[j]; p < nz; p++) {
            C.v_[p] = x[C.i_[p]];
        }
    }

    // Finalize and deallocate unused memory
    C.p_[N] = nz;
    C.realloc();

    // TODO put into canonical format and test

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
csint scatter(
    const CSCMatrix& A,
    csint j,
    double beta,
    std::vector<csint>& w,
    std::vector<double>& x,
    csint mark,
    CSCMatrix& C,
    csint nz,
    bool fs
    )
{
    if (fs) {
        // If it's the first call, we can just copy the (scaled) column
        for (csint p = A.p_[j]; p < A.p_[j+1]; p++) {
            csint i = A.i_[p];       // A(i, j) is non-zero
            w[i] = mark;             // i is new entry in column j
            C.i_[nz++] = i;          // add i to sparsity pattern of C(:, j)
            x[i] = beta * A.v_[p];   // x = beta * A(i, j)
        }
    } else {
        // Otherwise, we need to accumulate the values
        for (csint p = A.p_[j]; p < A.p_[j+1]; p++) {
            csint i = A.i_[p];           // A(i, j) is non-zero
            if (w[i] < mark) {
                w[i] = mark;             // i is new entry in column j
                C.i_[nz++] = i;          // add i to pattern of C(:, j)
                x[i] = beta * A.v_[p];   // x = beta * A(i, j)
            } else {
                x[i] += beta * A.v_[p];  // i exists in C(:, j) already
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
CSCMatrix CSCMatrix::permute(const std::vector<csint> p_inv, const std::vector<csint> q) const
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
    // Also copy the cumulative sum back into the workspace for iteration
    C.p_ = cumsum(w);

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


// TODO raise appropriate errors with informative messages
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
    // TODO may be able to relax last condition and return empty matrix?
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


/** Select a submatrix by row and column indices, using COOMatrix construction.
 *
 * This function takes O(|rows| + |cols| + M + N) time (to sort the indices in
 * `tocsc`) + O(log M) time if the columns are sorted, and + O(M) time if they
 * are not.
 *
 * @param i, j vectors of the row and column indices to keep. The indices need
 *        not be consecutive, or sorted. Duplicates are allowed.
 *
 * @return C  the submatrix of A of dimension `length(i)`-by-`length(j)`.
 */
CSCMatrix CSCMatrix::index_lazy(
    const std::vector<csint>& rows,
    const std::vector<csint>& cols
    ) const
{
    COOMatrix C(rows.size(), cols.size(), nnz());

    for (csint i = 0; i < rows.size(); i++) {
        for (csint j = 0; j < cols.size(); j++) {
            C.assign(i, j, (*this)(rows[i], cols[j]));
        }
    }

    return C.tocsc();
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


/*------------------------------------------------------------------------------
 *         Printing
 *----------------------------------------------------------------------------*/
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


/*==============================================================================
 *============================================================================*/
