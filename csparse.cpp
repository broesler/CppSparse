/*==============================================================================
 *     File: csparse.cpp
 *  Created: 2024-10-01 21:07
 *   Author: Bernie Roesler
 *
 *  Description: Implements the sparse matrix classes
 *
 *============================================================================*/

#include "csparse.h"


// #define PRINT_ELEMS(a, b) {\
//     for (csint k = (a); k < (b); k++) \
//         os << "(" << i_[k] << ", " << j_[k] << "): " << v_[k] << std::endl;\
// }


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
            for (csint k = 0; k < nnz_; k++)
                os << "(" << i_[k] << ", " << j_[k] << "): " << v_[k] << std::endl;

        } else {
            // Print just the first and last 3 non-zero elements
            for (csint k = 0; k < 3; k++)
                os << "(" << i_[k] << ", " << j_[k] << "): " << v_[k] << std::endl;

            os << "..." << std::endl;

            for (csint k = nnz_ - 3; k < nnz_; k++)
                os << "(" << i_[k] << ", " << j_[k] << "): " << v_[k] << std::endl;
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
