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
 * @param x  the input vector
 *
 * @return beta  the scaling factor
 * @return v  the Householder vector
 * @return s  the first element of v, which is guaranteed to be
 *         \f$ \pm \|x\|_2 \f$
 */
Householder house(const std::vector<double>& x);


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


}  // namespace cs

#endif  // _CSPARSE_QR_H_

//==============================================================================
//==============================================================================
