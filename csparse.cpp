/*==============================================================================
 *     File: csparse.cpp
 *  Created: 2024-10-01 21:07
 *   Author: Bernie Roesler
 *
 *  Description: Implements the sparse matrix classes
 *
 *============================================================================*/

#include "csparse.h"


COOMatrix::COOMatrix() 
    : nzmax_(0),
      M_(0),
      N_(0),
      nz_(0)
{}

csint COOMatrix::nnz() { return nz_; }

std::array<csint, 2> COOMatrix::shape() 
{ 
    return std::array<csint, 2> {M_, N_};
}


/*==============================================================================
 *============================================================================*/
