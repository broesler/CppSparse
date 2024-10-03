/*==============================================================================
 *     File: csparse.cpp
 *  Created: 2024-10-01 21:07
 *   Author: Bernie Roesler
 *
 *  Description: Implements the sparse matrix classes
 *
 *============================================================================*/

#include "csparse.h"


/*------------------------------------------------------------------------------
 *     Constructors
 *----------------------------------------------------------------------------*/
COOMatrix::COOMatrix() {};

// Construct a COOMatrix from arrays of values and coordinates
COOMatrix::COOMatrix(
    const std::vector<double>& v,
    const std::vector<csint>& i,
    const std::vector<csint>& j
    )
    : v_(v),
      i_(i),
      j_(j),
      nnz_(v_.size()),
      M_(*std::max_element(i_.begin(), i_.end()) + 1),
      N_(*std::max_element(j_.begin(), j_.end()) + 1),
      nzmax_(M_ * N_)
{}


/*------------------------------------------------------------------------------
 *         Accessors
 *----------------------------------------------------------------------------*/
csint COOMatrix::nnz() { return nnz_; }
csint COOMatrix::nzmax() { return nzmax_; }

std::array<csint, 2> COOMatrix::shape()
{
    return std::array<csint, 2> {M_, N_};
}

// Printing
void COOMatrix::print(bool verbose, std::ostream& os)
{
    if (nnz_ == 0) {
        os << "(null)" << std::endl;
        return;
    }

    os << "<COOrdinate sparse matrix" << std::endl;
    os << "        with " << nnz_ << " stored elements "
        << "and shape (" << M_ << ", " << N_ << ")>" << std::endl;
    
    // TODO if (verbose) print all elements
}

/*==============================================================================
 *============================================================================*/
