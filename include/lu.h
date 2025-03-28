//==============================================================================
//     File: lu.h
//  Created: 2025-03-19 12:32
//   Author: Bernie Roesler
//
//  Description: Declarations for LU decomposition.
//
//==============================================================================

#ifndef _CSPARSE_LU_H_
#define _CSPARSE_LU_H_

#include <vector>

#include "csc.h"
#include "types.h"


namespace cs {

/** Symbolic LU decomposition return struct (see: cs_symbolic aka css) */
struct SymbolicLU
{
    std::vector<csint> q;  ///< fill-reducting column permutation

    csint lnz,  ///< # entries in L
          unz;  ///< # entries in U
};


/** Numeric LU decomposition return struct (see: cs_numeric aka csn) */
struct LUResult
{
    CSCMatrix L,               ///< lower triangular matrix
              U;               ///< upper triangular matrix
    std::vector<csint> p_inv,  ///< row permutation of A
                       q;      ///< column permutation of A
};


/** Compute the symolic LU decomposition of A.
 *
 * This function computes the column permutation `q` and the column counts of
 * the matrices `L` and `U` in the LU decomposition of `A`.
 *
 * @param A  the matrix to factorize
 * @param order  the ordering method to use
 *
 * @return the symbolic factorization
 */
SymbolicLU slu(const CSCMatrix& A, AMDOrder order=AMDOrder::Natural);


/** Compute the numeric LU decomposition of A, such that \f$PAQ = LU\f$.
 *
 * This function computes the LU decomposition of `A` using the symbolic
 * factorization `S`.
 *
 * @param A  the matrix to factorize
 * @param S  the symbolic factorization
 * #param tol  the tolerance for pivoting. If `tol` is 1.0, partial pivoting is
 *        used. If `tol` is less than 1.0, diagonal pivoting is used.
 *
 * @return the LU decomposition, with 
 */
LUResult lu(const CSCMatrix& A, const SymbolicLU& S, double tol=1.0);


}  // namespace cs

#endif  // _CSPARSE_LU_H_

//==============================================================================
//==============================================================================
