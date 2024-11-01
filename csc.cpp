/*==============================================================================
 *     File: csc.cpp
 *  Created: 2024-10-09 20:58
 *   Author: Bernie Roesler
 *
 *  Description: Implements the compressed sparse column matrix class
 *
 *============================================================================*/

#include <cassert>
#include <sstream>
#include <numeric>

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


/*------------------------------------------------------------------------------
       Math Operations
----------------------------------------------------------------------------*/
/** Transpose the matrix as a copy.
 *
 * This operation can be viewed as converting a Compressed Sparse Column matrix
 * into a Compressed Sparse Row matrix.
 *
 * @return new CSCMatrix object with transposed rows and columns.
 */
CSCMatrix CSCMatrix::T() const
{
    csint nnz_ = nnz();
    std::vector<double> data(nnz_);
    std::vector<csint> indices(nnz_), indptr(N_ + 1), ws(N_);

    // Compute number of elements in each row
    for (csint p = 0; p < nnz_; p++)
        ws[i_[p]]++;

    // Row pointers are the cumulative sum of the counts, starting with 0
    std::partial_sum(ws.begin(), ws.end(), indptr.begin() + 1);

    // Also copy the cumulative sum back into the workspace for iteration
    ws = indptr;

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


/** Sum duplicate entries in place. */
CSCMatrix& CSCMatrix::sum_duplicates()
{
    csint nz = 0;  // count actual number of non-zeros (excluding dups)
    std::vector<int> ws(M_, -1);                   // row i not yet seen

    for (csint j = 0; j < N_; j++) {
        int q = nz;                                  // column j will start at q
        for (csint p = p_[j]; p < p_[j + 1]; p++) {
            csint i = i_[p];                         // A(i, j) is nonzero
            if (ws[i] >= q) {
                v_[ws[i]] += v_[p];                  // A(i, j) is a duplicate
            } else {
                ws[i] = nz;                          // record where row i occurs
                i_[nz] = i;                          // keep A(i, j)
                v_[nz++] = v_[p];
            }
        }
        p_[j] = q;                                   // record start of column j
    }

    p_[N_] = nz;                                     // finalize A
    p_.resize(N_ + 1);
    i_.resize(nz);
    v_.resize(nz);                                   // deallocate memory

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
    p_.resize(N_ + 1);
    i_.resize(nz);
    v_.resize(nz);  // deallocate memory TODO rewrite as `realloc`

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


/** Matrix-vector right-multiply. */
std::vector<double> operator*(const CSCMatrix& A, const std::vector<double>& x)
{
    assert(A.N_ == x.size()); 

    std::vector<double> out(x.size());

    for (csint j = 0; j < A.N_; j++) {
        for (csint p = A.p_[j]; p < A.p_[j+1]; p++) {
            out[A.i_[p]] = A.v_[p] * x[j];
        }
    }

    return out;
}


// TODO test left-multiply
// std::vector<double> operator*(const std::vector<double>& x, const CSCMatrix& A)
// {
//     return (A.T() * x).T
// }

/** Vector-vector addition */
std::vector<double> operator+(
    const std::vector<double>& a,
    const std::vector<double>& b
    )
{
    assert(a.size() == b.size());

    std::vector<double> out(a.size());

    for (csint i = 0; i < a.size(); i++) {
        out[i] = a[i] + b[i];
    }

    return out;
}

// TODO operator- for unary vector and vector-vector

/** Scale a vector by a scalar */
std::vector<double> operator*(const double c, const std::vector<double>& vec)
{
    std::vector<double> out(vec);
    for (auto& x : out) {
        x *= c;
    }
    return out;
}


std::vector<double> operator*(const std::vector<double>& vec, const double c)
{
    return c * vec;
}


std::vector<double>& operator*=(std::vector<double>& vec, const double c)
{
    for (auto& x : vec) {
        x *= c;
    }
    return vec;
}

/** Scale a matrix by a scalar */
CSCMatrix operator*(const double c, const CSCMatrix& A)
{
    CSCMatrix out(A.v_, A.i_, A.p_, A.shape());
    out.v_ *= c;
    return out;
}


CSCMatrix operator*(const CSCMatrix& A, const double c)
{
    return c * A;
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
CSCMatrix operator*(const CSCMatrix& A, const CSCMatrix& B)
{
    csint M, Ka, Kb, N;
    std::tie(M, Ka) = A.shape();
    std::tie(Kb, N) = B.shape();
    assert(Ka == Kb);

    // NOTE See Problem 2.20 for how to compute nnz(A*B)
    CSCMatrix C(M, N, A.nnz() + B.nnz());  // output

    // Allocate workspaces
    std::vector<csint> w(M);
    std::vector<double> x(M);

    csint nnz = 0;  // track total number of non-zeros in C

    for (csint j = 0; j < N; j++) {
        C.p_[j] = nnz;  // column j of C starts here 

        // Compute x += beta * A(:, j) for each non-zero row in B.
        for (csint p = B.p_[j]; p < B.p_[j+1]; p++) {
            nnz = scatter(A, B.i_[p], B.v_[p], w, x, j+1, C, nnz);
        }

        // Gather values into the correct locations in C
        for (csint p = C.p_[j]; p < nnz; p++) {
            C.v_[p] = x[C.i_[p]];
        }
    }

    // Finalize and deallocate unused memory
    C.p_[N] = nnz;
    C.p_.resize(N + 1);
    C.i_.resize(nnz);
    C.v_.resize(nnz);

    return C;
}


/** Add two matrices (and optionally scale them) `C = alpha * A + beta * B`.
 * 
 * @param A, B  the CSC matrices
 * @param alpha, beta  scalar multipliers
 *
 * @return out a CSC matrix
 */
CSCMatrix add(
    const CSCMatrix& A, const CSCMatrix& B,
    double alpha=1.0, double beta=1.0
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

    csint nnz = 0;  // track total number of non-zeros in C

    for (csint j = 0; j < N; j++) {
        C.p_[j] = nnz;  // column j of C starts here
        nnz = scatter(A, j, alpha, w, x, j+1, C, nnz);  // alpha * A(:, j)
        nnz = scatter(B, j,  beta, w, x, j+1, C, nnz);  //  beta * B(:, j)

        // Gather results into the correct column of C
        for (csint p = C.p_[j]; p < nnz; p++) {
            C.v_[p] = x[C.i_[p]];
        }
    }

    // Finalize and deallocate unused memory
    C.p_[N] = nnz;
    C.p_.resize(N + 1);
    C.i_.resize(nnz);
    C.v_.resize(nnz);

    return C;
}


CSCMatrix operator+(const CSCMatrix& A, const CSCMatrix& B)
{
    assert(A.shape() == B.shape());
    csint M, N;
    std::tie(M, N) = A.shape();

    // NOTE See Problem 2.20 for how to compute nnz(A*B)
    CSCMatrix C(M, N, A.nnz() + B.nnz());  // output

    // Allocate workspaces
    std::vector<csint> w(M);
    std::vector<double> x(M);

    csint nnz = 0;  // track total number of non-zeros in C

    for (csint j = 0; j < N; j++) {
        C.p_[j] = nnz;  // column j of C starts here
        nnz = scatter(A, j, 1, w, x, j+1, C, nnz);  // A(:, j)
        nnz = scatter(B, j, 1, w, x, j+1, C, nnz);  // B(:, j)

        // Gather results into the correct column of C
        for (csint p = C.p_[j]; p < nnz; p++) {
            C.v_[p] = x[C.i_[p]];
        }
    }

    // Finalize and deallocate unused memory
    C.p_[N] = nnz;
    C.p_.resize(N + 1);
    C.i_.resize(nnz);
    C.v_.resize(nnz);

    return C;
}

/** Compute x += beta * A(:, j).
 *
 * @param A     CSC matrix by which to multiply
 * @param j     column index of `A`
 * @param beta  scalar value by which to multiply `A`
 * @param w, x  workspace vectors of row indices and values, respectively
 * @param mark  separator index for `w`. All `w[i] < mark`are row indices that
 *              are not yet in `Cj`.
 * @param C     CSC matrix where output non-zero pattern is stored
 * @param nnz   current number of non-zeros in `C`.
 *
 * @return nnz  updated number of non-zeros in `C`.
 */
csint scatter(
    const CSCMatrix& A,
    csint j,
    double beta,
    std::vector<csint>& w,
    std::vector<double>& x,
    csint mark,
    CSCMatrix& C,
    csint nnz
    )
{
    for (csint p = A.p_[j]; p < A.p_[j+1]; p++) {
        csint i = A.i_[p];           // A(i, j) is non-zero
        if (w[i] < mark) {
            w[i] = mark;             // i is new entry in column j
            C.i_[nnz++] = i;         // add i to pattern of C(:, j)
            x[i] = beta * A.v_[p];   // x[i] = beta * A(i, j)
        } else {
            x[i] += beta * A.v_[p];  // i exists in C(:, j) already
        }
    }

    return nnz;
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
 * @param threshold   if `nnz > threshold`, print only the first and last
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
            print_elems_(os, 0, nnz_);
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
