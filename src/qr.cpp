/*==============================================================================
 *     File: qr.cpp
 *  Created: 2025-02-11 12:52
 *   Author: Bernie Roesler
 *
 *  Description: Implements QR decomposition using Householder reflections and
 *    Givens rotations.
 *
 *============================================================================*/

#include "qr.h"

namespace cs {

Householder house(const std::vector<double>& x)
{
    double beta, s, sigma = 0.0;
    std::vector<double> v(x);  // copy x into v

    // Compute the 2-norm of x
    for (const auto& vi : v) {
        sigma += vi * vi;
    }

    if (sigma == 0) {
        s = std::fabs(v[0]);   // s = |v(0)|
        beta = (v[0] <= 0) ? 2 : 0;
        v[0] = 1;
    } else {
        s = std::sqrt(sigma);  // s = norm(v)
        v[0] = (v[0] <= 0) ? (v[0] - s) : (-sigma / (v[0] + s));
        beta = -1 / (s * v[0]);
    }

    return {v, beta, s};
}



}  // namespace cs

/*==============================================================================
 *============================================================================*/
