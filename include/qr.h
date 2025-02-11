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


namespace cs {


/** Householder reflection return struct. */
struct Householder {
    std::vector<double> v;  ///< the Householder vector
    double beta,            ///< the scaling factor
           s;               ///< the first element of v
};


/** Compute the Householder reflection matrix for a given vector.
 *
 * The Householder reflection matrix is defined as
 * \f[
 *      H = I - \beta v v^T
 * \f]
 * where \f$ v = x - \beta e_1 \f$ and \f$ \beta = 2 / (v^T v) \f$.
 *
 * @param x  the input vector
 *
 * @return beta  the scaling factor
 * @return v  the Householder vector
 * @return s  the first element of v, which is guaranteed to be 
 *         \f$ \pm \|x\|_2 \f$
 */
Householder house(const std::vector<double>& x);


}  // namespace cs

#endif  // _CSPARSE_QR_H_

//==============================================================================
//==============================================================================
