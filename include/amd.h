//==============================================================================
//     File: amd.h
//  Created: 2025-04-18 08:53
//   Author: Bernie Roesler
//
//  Description: Declarations for AMD ordering.
//
//==============================================================================

#ifndef _CSPARSE_AMD_H_
#define _CSPARSE_AMD_H_

#include <vector>

#include "types.h"

namespace cs {


/** Compute the approximate minimum degree ordering of a matrix.
 *
 * This function computes the approximate minimum degree ordering of a matrix
 * using the AMD algorithm. The ordering is used to reduce the fill-in in the LU
 * decomposition of the matrix.
 *
 * @param A  the matrix to reorder
 * @param order  the ordering method to use:
 *       - `AMDOrder::Natural`: natural ordering (no permutation)
 *       - `AMDOrder::APlusAT`: AMD ordering of A + A^T. This option is appropriate for
 *         Cholesky factorization, or LU factorization with substantial entries
 *         on the diagonal and a roughly symmetric nonzero pattern. If `cs::lu`
 *         is used, `tol < 1.0` should be used to prefer the diagonal entries
 *         for partial pivoting.
 *       - `AMDOrder::ATANoDenseRows`: AMD ordering of A^T * A, with "dense"
 *         rows removed from `A`. This option is appropriate for LU
 *         factorization of unsymmetric matrices and produces a similar ordering
 *         to that of `COLAMD`.
 *       - `AMDOrder::ATA`: AMD ordering of A^T * A. This option is appropriate
 *         for QR factorization, or for LU factorization if `A` has no "dense"
 *         rows. A "dense" row is defined as a row with more than 
 *         \f$ 10 \sqrt{N} \f$ nonzeros, where \f$N\f$ is the number of columns
 *         in the matrix.
 *
 * @return the permutation vector
 */
std::vector<csint> amd(const CSCMatrix& A, const AMDOrder order=AMDOrder::Natural);


}  // namespace cs

#endif  // _CSPARSE_AMD_H_

//==============================================================================
//==============================================================================
