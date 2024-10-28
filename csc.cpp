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
      p_(N),
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

const double CSCMatrix::operator()(csint i, csint j) const
{
    for (csint p = p_[j]; p < p_[j+1]; p++) {
        // NOTE this code assumes that columns are *not* sorted, so it will
        // search through *every* element in a column. If columns were sorted,
        // we could also terminate and return 0 after i_[p] > i;
        if (i_[p] == i) {
            return v_[p];
        }
    }

    return 0.0;
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
    v_.resize(nz);                                   // deallocate memory
    i_.resize(nz);
    p_.resize(nz);

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
    v_.resize(nz);  // deallocate memory TODO rewrite as `realloc`
    i_.resize(nz);
    p_.resize(nz);

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
    for (csint j = 0; j <= N_; j++) {
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
