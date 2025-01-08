/*==============================================================================
 *     File: tri_solve.cpp
 *  Created: 2025-01-07 20:51
 *   Author: Bernie Roesler
 *
 *  Description: Implement triangular matrix solve functions.
 *
 *============================================================================*/

#include "csparse.h"


/** Forward solve a lower-triangular system \f$ Lx = b \f$.
 *
 * @param L  a lower-triangular matrix
 * @param b  a dense vector
 *
 * @return x  the solution vector
 */
std::vector<double> lsolve(const CSCMatrix& L, const std::vector<double>& b)
{
    assert(L.M_ == L.N_);
    assert(L.M_ == b.size());

    std::vector<double> x = b;

    for (csint j = 0; j < L.N_; j++) {
        x[j] /= L.v_[L.p_[j]];
        for (csint p = L.p_[j] + 1; p < L.p_[j+1]; p++) {
            x[L.i_[p]] -= L.v_[p] * x[j];
        }
    }

    return x;
}


/** Backsolve a lower-triangular system \f$ L^Tx = b \f$.
 *
 * @param L  a lower-triangular matrix
 * @param b  a dense vector
 *
 * @return x  the solution vector
 */
std::vector<double> ltsolve(const CSCMatrix& L, const std::vector<double>& b)
{
    assert(L.M_ == L.N_);
    assert(L.M_ == b.size());

    std::vector<double> x = b;

    for (csint j = L.N_ - 1; j >= 0; j--) {
        for (csint p = L.p_[j] + 1; p < L.p_[j+1]; p++) {
            x[j] -= L.v_[p] * x[L.i_[p]];
        }
        x[j] /= L.v_[L.p_[j]];
    }

    return x;
}

/*==============================================================================
 *============================================================================*/
