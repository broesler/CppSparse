/*==============================================================================
 *     File: sparse_matrix.cpp
 *  Created: 2025-05-09 10:17
 *   Author: Bernie Roesler
 *
 *  Description: Implements the abstract SparseMatrix class.
 *
 *============================================================================*/

#include <iostream>

#include "sparse_matrix.h"


namespace cs {


// default destructor
SparseMatrix::~SparseMatrix() = default;


std::ostream& operator<<(std::ostream& os, const SparseMatrix& A)
{
    A.print(os, true);  // verbose printing assumed
    return os;
}


}  // namespace cs


/*==============================================================================
 *============================================================================*/
