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


inline auto sign(double x) { return std::copysign(1.0, x); }


Householder house(const std::vector<double>& x)
{
    double beta, s, sigma = 0.0;
    std::vector<double> v(x);  // copy x into v

    // sigma is the sum of squares of all elements *except* the first
    for (csint i = 1; i < v.size(); i++) {
        sigma += v[i] * v[i];
    }

    if (sigma == 0) {
        s = std::fabs(v[0]);   // s = |x(0)|
        beta = (v[0] <= 0) ? 2 : 0;  // make direction positive if x(0) < 0
        v[0] = 1;
    } else {
        s = std::sqrt(v[0] * v[0] + sigma);  // s = norm(x)

        //---------- LAPACK DLARFG algorithm
        // matches scipy.linalg.qr(mode='raw') and MATLAB
        double alpha = v[0];
        double b_ = -sign(alpha) * std::sqrt(alpha * alpha + sigma);
        beta = (b_ - alpha) / b_;

        v[0] = 1;
        for (csint i = 1; i < v.size(); i++) {
            v[i] /= (alpha - b_);
        }

        //---------- Davis book code (cs_house)
        // v[0] = (v[0] <= 0) ? (v[0] - s) : (-sigma / (v[0] + s));
        // beta = -1 / (s * v[0]);  // Davis book code

        // NOTE scale to be self-consistent with v[0] = 1, but does *not* match
        // the MATLAB or python v or beta.
        // double v0 = v[0];  // cache value before we change it to 1.0
        // beta *= v0 * v0;   // works with Davis book code + v[0] = 1 scaling

        //---------- Golub & Van Loan (Algorithm 5.1.1) (3 or 4ed)
        // Gives same result as the beta from Davis book, scaled by v0**2.
        // beta = 2 * (v0 * v0) / (v0 * v0 + sigma);

        // normalize to v[0] == 1
        // for (auto& vi : v) {
        //     vi /= v0;
        // }
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
