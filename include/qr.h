//==============================================================================
//     File: qr.h
//  Created: 2025-02-11 12:41
//   Author: Bernie Roesler
//
//  Description: Declarations for QR decomposition using Householder
//      reflections and Givens rotations.
//
//==============================================================================

#ifndef _CSPARSE_QR_H_
#define _CSPARSE_QR_H_

#include <span>
#include <vector>

#include "csc.h"
#include "types.h"


namespace cs {


/** Householder reflection return struct. */
struct Householder {
    std::vector<double> v;  ///< the Householder vector
    double beta;            ///< the scaling factor
    double s;               ///< the first element of v
};


/** Numeric QR decomposition return struct. */
struct QRResult {
    CSCMatrix V;               ///< the Householder vectors
    std::vector<double> beta;  ///< the scaling factors
    CSCMatrix R;               ///< the upper triangular matrix
};


/** Compute the Householder reflection matrix for a given vector.
 *
 * The Householder reflection matrix is defined as
 * \f[
 *      H = I - \beta v v^T
 * \f]
 * where \f$ v = x - \beta e_1 \f$ and \f$ \beta = 2 / (v^T v) \f$.
 *
 * The result is defined such that applying the Householder reflection gives
 * \f$ Hx = s e_1 \f$, where \f$ s = \pm \|x\|_2 \f$.
 *
 * See: Algorithm 5.1.1, Golub & Van Loan, 3rd ed.
 *
 * @param x  the input vector, may be a subspan of a larger vector
 *
 * @return beta  the scaling factor
 * @return v  the Householder vector
 * @return s  the first element of v, which is guaranteed to be
 *         \f$ \pm \|x\|_2 \f$
 */
Householder house(std::span<const double> x);


/** Apply a Householder reflection to a dense vector `x` with a sparse `v`.
 *
 * The Householder reflection is applied as
 * \f[
 *     Hx = x - v \beta v^T x = s e_1
 * \f]
 * where \f$ s = \pm \|x\|_2 \f$, `v` is the Householder vector and `beta` is
 * the scaling factor.
 *
 * @param V  a CSCMatrix with a single column containing the Householder vector
 * @param j  the column index of the Householder vector in `V`
 * @param beta  the scaling factor
 * @param x  the dense vector to which to apply the reflection
 *
 * @return Hx  the result of applying the Householder reflection to `x`
 */
std::vector<double> happly(
    const CSCMatrix& V,
	csint j,
	double beta,
	const std::vector<double>& x
);


/** Compute the column counts of the matrix V containing Householder vectors.
 *
 * This function also computes a row permutation vector `S.p_inv`, so that the
 * diagonal entries of PA are all structurally non-zero.
 *
 * It also computes leftmost row index of each row in `A`, `S.leftmost`. 
 *
 * If `A` is structurally rank-deficient, then this function adds fictitious
 * rows to `A` to make it structurally full rank. The total row count including
 * these fictitious rows is stored in `S.m2`.
 *
 * @param A  the CSCMatrix that will be decomposed
 * @param[in,out] S  the symbolic QR decomposition of A. S.parent is expected to
 *        have been computed by cs::etree. Only p_inv, leftmost, lnz, and m2
 *        values are updated.
 */
void vcount(const CSCMatrix& A, Symbolic& S);


/** Perform symbolic analysis for the QR decomposition of a matrix.
 *
 * This function calls `vcount` to compute the column counts of the matrix `V`;
 * the total non-zeros in `V`, `S.lnz`; the row permutation vector `S.p_inv`;
 * the leftmost row index of each row in `A`, `S.leftmost`; and the total row
 * count (including fictitious rows), `S.m2`.
 *
 * It also computes the elimination tree of the matrix `A.T @ A`, `S.parent`;
 * a column permutation vector `S.q` per the desired `order`; and the column
 * counts of the Cholesky factor of `A.T @ A`, `S.cp`.
 *
 * @param A  the matrix to factorize
 * @param order  the ordering method to use:
 *       - 0: natural ordering
 *       - 1: amd(A + A.T)
 *       - 2: amd(A.T * A) with no dense rows
 *       - 3: amd(A.T * A)
 *
 * @return the Symbolic factorization
 */
Symbolic sqr(const CSCMatrix& A, AMDOrder order=AMDOrder::Natural);


/** Perform the numeric QR decomposition of a matrix.
 *
 * @param A  the matrix to factorize
 * @param S  the symbolic factorization of A
 *
 * @return the numeric factorization
 */
QRResult qr(const CSCMatrix& A, const Symbolic& S);


}  // namespace cs

#endif  // _CSPARSE_QR_H_

//==============================================================================
//==============================================================================
