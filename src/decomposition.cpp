/*==============================================================================
 *     File: decomposition.cpp
 *  Created: 2025-01-27 13:14
 *   Author: Bernie Roesler
 *
 *  Description: Implements the symbolic factorization for a sparse matrix.
 *
 *============================================================================*/

#include <cmath>    // std::sqrt
#include <numeric>  // std::iota

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
        // TODO set to empty vector?
        // NOTE if we allow an empty p_inv here, we should support an empty
        // p or p_inv argument to all permute functions: pvec, ipvec,
        // inv_permute, permute, and symperm... and all of the upper/lower
        // triangular permuted solvers!!
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


/** Compute the up-looking Cholesky factorization of a sparse matrix.
 *
 * @note This function assumes that `A` is symmetric and positive definite.
 *
 * @param A the matrix to factorize
 * @param S the Symbolic factorization of `A`
 *
 * @return the Cholesky factorization of `A`
 */
CSCMatrix chol(const CSCMatrix& A, const Symbolic& S)
{
    auto [M, N] = A.shape();
    CSCMatrix L(M, N, S.lnz);  // allocate result

    // Workspaces
    std::vector<csint> c(S.cp);  // column pointers for L
    std::vector<double> x(N);    // sparse accumulator

    // const CSCMatrix C = S.p_inv.empty() ? A : A.symperm(S.p_inv);
    const CSCMatrix C = A.symperm(S.p_inv);

    L.p_ = S.cp;  // column pointers for L

    // Compute L(:, k) for L*L' = C
    for (csint k = 0; k < N; k++) {
        //--- Nonzero pattern of L(:, k) ---------------------------------------
        const std::vector<csint> s = C.ereach(k, S.parent);  // pattern of L(k, :)
        x[k] = 0.0;  // x(0:k) is now zero

        // scatter into x = full(triu(C(:,k)))
        for (csint p = C.p_[k]; p < C.p_[k+1]; p++) {
            csint i = C.i_[p];
            if (i <= k) {
                x[i] = C.v_[p];
            }
        }

        double d = x[k];  // d = C(k, k)
        x[k] = 0.0;       // clear x for k + 1st iteration

        //--- Triangular Solve -------------------------------------------------
        // Solve L(0:k-1, 0:k-1) * x = C(:, k)
        for (const auto& i : s) {
            double lki = x[i] / L.v_[L.p_[i]];  // L(k, i) = x(i) / L(i, i)
            x[i] = 0.0;                         // clear x for k + 1st iteration

            for (csint p = L.p_[i] + 1; p < c[i]; p++) {
                x[L.i_[p]] -= L.v_[p] * lki;    // x -= L(i, :) * L(k, i)
            }

            d -= lki * lki;                     // d -= L(k, i) * L(k, i)

            csint p = c[i]++;
            L.i_[p] = k;                        // store L(k, i) in column i
            L.v_[p] = lki;
        }

        //--- Compute L(k, k) --------------------------------------------------
        if (d <= 0) {
            throw std::runtime_error("Matrix not positive definite!");
        }

        csint p = c[k]++;
        L.i_[p] = k;  // store L(k, k) = sqrt(d) in column k
        L.v_[p] = std::sqrt(d);
    }

    return L;
}


} // namespace cs

/*==============================================================================
 *============================================================================*/
