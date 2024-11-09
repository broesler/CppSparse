/*==============================================================================
 *     File: csc.cpp
 *  Created: 2024-10-09 20:58
 *   Author: Bernie Roesler
 *
 *  Description: Implements the compressed sparse column matrix class
 *
 *============================================================================*/

#include <cmath>

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


CSCMatrix::CSCMatrix(const COOMatrix& A)
    : v_(A.nnz()),
      i_(A.nnz()),
      p_(A.shape()[1] + 1)
{
    // Get the shape
    std::tie(M_, N_) = A.shape();

    csint nnz_ = A.nnz();
    std::vector<csint> ws(N_);  // workspace

    // Compute number of elements in each column
    for (csint k = 0; k < nnz_; k++)
        ws[A.j_[k]]++;

    // Column pointers are the cumulative sum
    p_ = cumsum(ws);

    for (csint k = 0; k < nnz_; k++) {
        // A(i, j) is the pth entry in the CSC matrix
        csint p = ws[A.j_[k]]++;  // "pointer" to the current element's column
        i_[p] = A.i_[k];
        v_[p] = A.v_[k];
    }
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

std::array<csint, 2> CSCMatrix::shape() const
{
    return std::array<csint, 2> {M_, N_};
}

const std::vector<csint>& CSCMatrix::indices() const { return i_; }
const std::vector<csint>& CSCMatrix::indptr() const { return p_; }
const std::vector<double>& CSCMatrix::data() const { return v_; }

bool CSCMatrix::has_sorted_indices() const { return has_sorted_indices_; }

// NOTE this code assumes that columns are *not* sorted, so it will search
// through *every* element in a column. If columns were sorted, and there were
// no duplicates allowed, we could also terminate and return 0 after i_[p] > i;
const double CSCMatrix::operator()(csint i, csint j) const
{
    double out = 0;

    for (csint p = p_[j]; p < p_[j+1]; p++) {
        if (i_[p] == i) {
            out += v_[p];  // sum duplicate entries
        }
    }

    return out;
}


/** Convert a compressed sparse column matrix to a coordinate (triplet) format
 * matrix.
 *
 * See: Davis, Exercise 2.2.
 *
 * @return a copy of the `CSCMatrix` in COO (triplet) format.
 */
COOMatrix CSCMatrix::tocoo() const { return COOMatrix(*this); }


/*------------------------------------------------------------------------------
       Format Operations
----------------------------------------------------------------------------*/
/** Transpose the matrix as a copy.
 *
 * This operation can be viewed as converting a Compressed Sparse Column matrix
 * into a Compressed Sparse Row matrix.
 *
 * This function takes 
 *   - O(N) extra space for the workspace
 *   - O(M * N + nnz) time
 *       == nnz column counts + N columns * M potential non-zeros per column
 *
 * @return new CSCMatrix object with transposed rows and columns.
 */
CSCMatrix CSCMatrix::transpose() const
{
    csint nnz_ = nnz();
    std::vector<double> data(nnz_);
    std::vector<csint> indices(nnz_), indptr(N_ + 1), ws(N_);

    // Compute number of elements in each row
    for (csint p = 0; p < nnz_; p++)
        ws[i_[p]]++;

    // Row pointers are the cumulative sum of the counts, starting with 0.
    // Also copy the cumulative sum back into the workspace for iteration
    indptr = cumsum(ws);

    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            // place A(i, j) as C(j, i)
            csint q = ws[i_[p]]++;
            indices[q] = j;
            data[q] = v_[p];
        }
    }

    return CSCMatrix {data, indices, indptr, {N_, M_}};
}


// Alias for transpose
CSCMatrix CSCMatrix::T() const { return this->transpose(); }

// TODO transpose in-place? Then we can sort in-place?
// Or just use the version of `sort` that uses quicksort directly.


/** Sort rows and columns in a copy. */
CSCMatrix CSCMatrix::sort() const
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
 *   - O(N * M log M) time ==
 *       sort a length M vector for each of N columns
 *
 * @return a reference to the object for method chaining
 */
CSCMatrix& CSCMatrix::sorted()
{
    for (csint j = 0; j < N_; j++) {
        // Pointers to the rows
        csint p = p_[j],
              pn = p_[j+1];

        assert(pn > p);
        csint Nc = pn - p;  // number of elements in the column

        // TODO allocate workspaces outside of loop (size M_) and have argsort
        // take a length argument to just sort the correct subset?

        // Allocate clean workspaces
        std::vector<csint> w(Nc);
        std::vector<double> x(Nc);

        // Copy the row indices and values into the workspace
        std::copy(i_.begin() + p, i_.begin() + pn, w.begin());
        std::copy(v_.begin() + p, v_.begin() + pn, x.begin());

        // argsort the rows to get indices
        std::vector<csint> idx = argsort(w);

        // Re-assign the values
        for (csint i = 0; i < Nc; i++) {
            i_[p + i] = w[idx[i]];
            v_[p + i] = x[idx[i]];
        }
    }

    has_sorted_indices_ = true;

    return *this;
}


/** Sum duplicate entries in place. */
CSCMatrix& CSCMatrix::sum_duplicates()
{
    csint nz = 0;  // count actual number of non-zeros (excluding dups)
    std::vector<int> ws(M_, -1);                      // row i not yet seen

    for (csint j = 0; j < N_; j++) {
        int q = nz;                                  // column j will start at q
        for (csint p = p_[j]; p < p_[j + 1]; p++) {
            csint i = i_[p];                          // A(i, j) is nonzero
            if (ws[i] >= q) {
                v_[ws[i]] += v_[p];                   // A(i, j) is a duplicate
            } else {
                ws[i] = nz;                          // record where row i occurs
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


/** Drop any exactly zero entries from the matrix. */
CSCMatrix& CSCMatrix::dropzeros()
{
    return fkeep(&nonzero, nullptr);
}


/** Return true if abs(A(i j)) > tol */
bool CSCMatrix::abs_gt_tol(csint i, csint j, double Aij, void *tol)
{
    return (std::fabs(Aij) > *((double *) tol));
}


/** Drop any entries within `tol` of zero. */
CSCMatrix& CSCMatrix::droptol(double tol)
{
    return fkeep(&abs_gt_tol, &tol);
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
    assert(A.N_ == x.size()); 
    assert(x.size() == y.size()); 
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
    assert(A.N_ == x.size()); 
    assert(x.size() == y.size()); 
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

    std::vector<double> out(x.size());

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
 * @note This function does *not* return a matrix with sorted columns!
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

    for (csint j = 0; j < N; j++) {
        C.p_[j] = nz;  // column j of C starts here 

        // Compute x += beta * A(:, j) for each non-zero row in B.
        for (csint p = B.p_[j]; p < B.p_[j+1]; p++) {
            nz = scatter(*this, B.i_[p], B.v_[p], w, x, j+1, C, nz);
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


// Operators
std::vector<double> operator*(const CSCMatrix& A, const std::vector<double>& x) 
{ return A.dot(x); }

CSCMatrix operator*(const CSCMatrix& A, const double c) { return A.dot(c); }
CSCMatrix operator*(const double c, const CSCMatrix& A) { return A.dot(c); }
CSCMatrix operator*(const CSCMatrix& A, const CSCMatrix& B) { return A.dot(B); }


/** Add two matrices (and optionally scale them) `C = alpha * A + beta * B`.
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

    // NOTE See Problem 2.20 for how to compute nnz(A*B)
    CSCMatrix C(M, N, A.nnz() + B.nnz());  // output

    // Allocate workspaces
    std::vector<csint> w(M);
    std::vector<double> x(M);

    csint nz = 0;  // track total number of non-zeros in C

    for (csint j = 0; j < N; j++) {
        C.p_[j] = nz;  // column j of C starts here
        nz = scatter(A, j, alpha, w, x, j+1, C, nz);  // alpha * A(:, j)
        nz = scatter(B, j,  beta, w, x, j+1, C, nz);  //  beta * B(:, j)

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


/** Add a matrix B. */
CSCMatrix CSCMatrix::add(const CSCMatrix& B) const
{
    assert(shape() == B.shape());
    csint M, N;
    std::tie(M, N) = shape();

    // NOTE See Problem 2.20 for how to compute nnz(A*B)
    CSCMatrix C(M, N, nnz() + B.nnz());  // output

    // Allocate workspaces
    std::vector<csint> w(M);
    std::vector<double> x(M);

    csint nz = 0;  // track total number of non-zeros in C

    for (csint j = 0; j < N; j++) {
        C.p_[j] = nz;  // column j of C starts here
        nz = scatter(*this, j, 1, w, x, j+1, C, nz);  // A(:, j)
        nz = scatter(    B, j, 1, w, x, j+1, C, nz);  // B(:, j)

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


CSCMatrix operator+(const CSCMatrix& A, const CSCMatrix& B) { return A.add(B); }


/** Compute x += beta * A(:, j).
 *
 * @param A     CSC matrix by which to multiply
 * @param j     column index of `A`
 * @param beta  scalar value by which to multiply `A`
 * @param w, x  workspace vectors of row indices and values, respectively
 * @param mark  separator index for `w`. All `w[i] < mark`are row indices that
 *              are not yet in `Cj`.
 * @param C     CSC matrix where output non-zero pattern is stored
 * @param nz   current number of non-zeros in `C`.
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
    csint nz
    )
{
    for (csint p = A.p_[j]; p < A.p_[j+1]; p++) {
        csint i = A.i_[p];           // A(i, j) is non-zero
        if (w[i] < mark) {
            w[i] = mark;             // i is new entry in column j
            C.i_[nz++] = i;         // add i to pattern of C(:, j)
            x[i] = beta * A.v_[p];   // x[i] = beta * A(i, j)
        } else {
            x[i] += beta * A.v_[p];  // i exists in C(:, j) already
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


/*------------------------------------------------------------------------------
 *         Printing
 *----------------------------------------------------------------------------*/
/** Print elements of the matrix between `start` and `end`.
 *
 * @param os          the output stream, defaults to std::cout
 * @param start, end  print the all elements where `p ∈ [start, end]`, counting
 *        column-wise.
 */
void CSCMatrix::print_elems_(std::ostream& os, csint start, csint end) const
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
void CSCMatrix::print(std::ostream& os, bool verbose, csint threshold) const
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
