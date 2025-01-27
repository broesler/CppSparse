/*==============================================================================
 *     File: decomposition.cpp
 *  Created: 2025-01-27 13:14
 *   Author: Bernie Roesler
 *
 *  Description: Implements the symbolic factorization for a sparse matrix.
 *
 *============================================================================*/

#include <numeric>  // for std::iota

#include "csparse.h"

namespace cs {

/** Compute the symbolic Cholesky factorization of a sparse matrix.
 *
 * @note This function assumes that `A` is symmetric and positive definite.
 *
 * @param A the matrix to factorize
 * @param order the ordering method to use:
 *       - 0: natural ordering
 *       - 1: amd(A + A.T())
 *       - 2: amd(??)  FIXME
 *       - 3: amd(A^T A)
 *
 * @return the Symbolic factorization
 */
Symbolic symbolic_cholesky(const CSCMatrix& A, AMDOrder order)
{
    Symbolic S;

    if (order == AMDOrder::Natural) {
        // identity permutation
        S.p_inv = std::vector<csint>(A.shape()[1]);
        std::iota(S.p_inv.begin(), S.p_inv.end(), 0);
    } else {
        // TODO implement amd order
        // std::vector<csint> p = amd(order, A);  // P = amd(A + A.T()) or natural
        // S.p_inv = inv_permute(p);
        throw std::runtime_error("Ordering method not implemented!");
    }

    // Find pattern of Cholesky factor
    CSCMatrix C = A.symperm(S.p_inv, false);  // C = spones(triu(A(p, p)))
    S.parent = C.etree();                     // compute the elimination tree
    auto postorder = post(S.parent);          // postorder the elimination tree
    auto c = C.counts(S.parent, postorder);   // find column counts of L
    S.cp = cumsum(c);                         // find column pointers for L
    S.lnz = S.unz = S.cp.back();              // number of non-zeros in L and U

    return S;
}

} // namespace cs

/*==============================================================================
 *============================================================================*/
