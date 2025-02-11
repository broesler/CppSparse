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

    // sigma is the sum of squares of all elements *except* the first
    for (csint i = 1; i < v.size(); i++) {
        sigma += v[i] * v[i];
    }

    if (sigma == 0) {
        s = std::fabs(v[0]);   // s = |v(0)|
        beta = (v[0] <= 0) ? 2 : 0;
        v[0] = 1;
    } else {
        s = std::sqrt(v[0] * v[0] + sigma);  // s = norm(v)
        v[0] = (v[0] <= 0) ? (v[0] - s) : (-sigma / (v[0] + s));
        beta = -1 / (s * v[0]);
    }

    return {v, beta, s};
}


std::vector<double> happly(
    const CSCMatrix& V,
	csint j,
	double beta,
	const std::vector<double>& x
)
{
    std::vector<double> Hx(x);  // copy x into Hx
    double tau = 0.0;

    // tau = v^T x
    for (csint p = V.p_[j]; p < V.p_[j+1]; p++) {
        tau += V.v_[p] * x[V.i_[p]];
    }

    tau *= beta;  // tau = beta * v^T x

    // Hx = x - v*tau
    for (csint p = V.p_[j]; p < V.p_[j+1]; p++) {
        Hx[V.i_[p]] -= V.v_[p] * tau;
    }

    return Hx;
}


}  // namespace cs

/*==============================================================================
 *============================================================================*/
