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


/** Update the Cholesky factor for \f$ A = A + σ w w^T \f$.
 *
 * @param L  the Cholesky factor of A
 * @param σ  +1 for an update, or -1 for a downdate
 * @param C  the update vector, as the first column in a CSCMatrix
 * @param parent  the elimination tree of A
 *
 * @return L  the updated Cholesky factor of A
 */
CSCMatrix& chol_update(
    CSCMatrix& L,
    int σ,  // TODO use a bool and set the ±1 in the function
    const CSCMatrix& C,
    const std::vector<csint>& parent
)
{
    assert(L.shape()[0] == C.shape()[0]);
    assert(C.shape()[1] == 1);  // C must be a column vector

    double α,
           β = 1.0,
           β2 = 1.0,
           δ,
           γ;

    std::vector<double> w(L.shape()[0]);  // sparse accumulator workspace

    // Find the minimum row index in the update vector
    csint p = C.p_[0];
    csint f = C.i_[p];
    for (; p < C.p_[1]; p++) {
        f = std::min(f, C.i_[p]);
        w[C.i_[p]] = C.v_[p];   // also scatter C into w
    }

    // Walk path f up to root
    for (csint j = f; j != -1; j = parent[j]) {
        p = L.p_[j];
        α = w[j] / L.v_[p];  // α = w(j) / L(j, j)
        β2 = β*β + σ * α*α;
        if (β2 <= 0) {
            throw std::runtime_error("Matrix not positive definite!");
        }
        β2 = std::sqrt(β2);
        δ = (σ > 0) ? (β / β2) : (β2 / β);
        γ = σ * α / (β2 * β);
        L.v_[p] = δ * L.v_[p] + ((σ > 0) ? (γ * w[j]) : 0.0);
        β = β2;
        for (p++; p < L.p_[j+1]; p++) {
            double w1 = w[L.i_[p]];
            double w2 = w1 - α * L.v_[p];
            w[L.i_[p]] = w2;
            L.v_[p] = δ * L.v_[p] + γ * ((σ > 0) ? w1 : w2);
        }
    }

    return L;
}


} // namespace cs

/*==============================================================================
 *============================================================================*/
