/*==============================================================================
 *     File: csparse.cpp
 *  Created: 2024-10-01 21:07
 *   Author: Bernie Roesler
 *
 *  Description: Implements the sparse matrix classes
 *
 *============================================================================*/

#include <cassert>

#include "csparse.h"


/*------------------------------------------------------------------------------
 *     Constructors
 *----------------------------------------------------------------------------*/
COOMatrix::COOMatrix() {};

/** Construct a COOMatrix from arrays of values and coordinates
 *
 * The entries are *not* sorted in any order, and duplicates are allowed. Any
 * duplicates will be summed.
 */
COOMatrix::COOMatrix(
    const std::vector<double>& v,
    const std::vector<csint>& i,
    const std::vector<csint>& j
    )
    : v_(v),
      i_(i),
      j_(j),
      nnz_(v_.size()),
      M_(*std::max_element(i_.begin(), i_.end()) + 1),  // zero-based indexing
      N_(*std::max_element(j_.begin(), j_.end()) + 1),
      nzmax_(nnz_)  // minimum storage
{}


/*------------------------------------------------------------------------------
 *         Accessors
 *----------------------------------------------------------------------------*/
csint COOMatrix::nnz() const { return nnz_; }
csint COOMatrix::nzmax() const { return nzmax_; }

std::array<csint, 2> COOMatrix::shape() const
{
    return std::array<csint, 2> {M_, N_};
}

// Element accessors
// TODO somewhere we need to check for duplicate entries so that we can make
// assumptions about nzmax_ vs nnz_.

/** Assign a value to a set of indices.
 *
 * @param i, j  integer indices of the matrix
 * @param v     the value to be assigned
 * @return pointer to the values array where the element will be added
 *
 * @see cs_entry Davis p 12.
 */
void COOMatrix::assign(csint i, csint j, double v)
{
    // Since arrays are *not* sorted, need to find index of (i, j)... but linear
    // search leads to O(N^2) insertion! We should just return a reference to
    // the end of the array, doubling in size if needed.
    csint cap = v_.capacity();
    if (nnz_ >= cap) {
        nzmax_ = 2 * cap;
        v_.reserve(nzmax_);
        i_.reserve(nzmax_);
        j_.reserve(nzmax_);
    }

    csint p = nnz_;  // index of next element
    i_[p] = i;
    j_[p] = j;
    v_[p] = v;

    nnz_++;  // FIXME wrong for duplicate entries
    M_ = std::max(M_, i+1);
    N_ = std::max(N_, j+1);
}

/*------------------------------------------------------------------------------
 *         Printing
 *----------------------------------------------------------------------------*/
void COOMatrix::print(std::ostream& os, bool verbose, csint threshold) const
{
    if (nnz_ == 0) {
        os << "(null)" << std::endl;
        return;
    }

    os << "<COOrdinate sparse matrix" << std::endl;
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

std::ostream& operator<<(std::ostream& os, const COOMatrix& A)
{
    A.print(os, true);  // verbose printing assumed
    return os;
}


/*==============================================================================
 *============================================================================*/
