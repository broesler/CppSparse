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
    : M_(M),
      N_(N)
{
    // TODO change these to just v_(nzmax) etc. in initialization list
    v_.reserve(nzmax);
    i_.reserve(nzmax);
    p_.reserve(N_);
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


/*------------------------------------------------------------------------------
       Math Operations 
----------------------------------------------------------------------------*/
/** Transpose the matrix as a copy.
 * 
 * @return new CSCMatrix object with transposed rows and columns.
 */
// CSCMatrix CSCMatrix::T() const
// {
//     return CSCMatrix(this->v_, this->p_, this->i_);
// }


/*------------------------------------------------------------------------------
 *         Printing
 *----------------------------------------------------------------------------*/
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
    os << "<Compressed Sparse Column matrix" << std::endl;
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
