/*==============================================================================
 *     File: qr.cpp
 *  Created: 2025-02-11 12:52
 *   Author: Bernie Roesler
 *
 *  Description: Implements QR decomposition using Householder reflections and
 *    Givens rotations.
 *
 *============================================================================*/

#include <numeric>  // accumulate

#include "cholesky.h"  // Symbolic
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


void vcount(const CSCMatrix& A, Symbolic& S)
{
    auto [M, N] = A.shape();
    std::vector<csint> next(M),      // the next row index
                       head(N, -1),  // the first row index in each column
                       tail(N, -1),  // the last row index in each column
                       nque(N);      // the number of rows in each column

    S.p_inv.assign(std::max(M, N), 0);  // permutation vector
    S.leftmost.assign(N, -1);           // leftmost non-zero in each column

    for (csint k = N-1; k >= 0; k--) {
        for (csint p = A.p_[k]; p < A.p_[k+1]; p++) {
            S.leftmost[A.i_[p]] = k;  // leftmost[i] = min(find(A(i, :)))
        }
    }

    for (csint i = M-1; i >= 0; i--) {  // scan rows in reverse order
        S.p_inv[i] = -1;                // i is not yet in the permutation
        csint k = S.leftmost[i];
        if (k != -1) {                  // row i is not empty
            if (nque[k]++ == 0) {
                tail[k] = i;            // first row in queue k
            }
            next[i] = head[k];          // put i at head of queue k
            head[k] = i;
        }
    }

    S.lnz = 0;
    S.m2 = M;

    csint k; // declare outside loop for final row permutation
    for (k = 0; k < N; k++) {          // find row permutation and nnz(V)
        csint i = head[k];             // remove row i from queue k
        S.lnz++;                       // count V(k, k) as nonzero
        if (i < 0) {
            i = S.m2++;                // add a fictitious row
        }
        S.p_inv[i] = k;                // associate row i with V(:, k)
        if (--nque[k] <= 0) {          // skip if V(k+1:m, k) is empty
            continue;
        }
        S.lnz += nque[k];              // nque[k] is nnz(V(k+1:m, k))
        csint pa = S.parent[k];
        if (pa != -1) {                // move all rows to parent of k
            if (nque[pa] == 0) {
                tail[pa] = tail[k];
            }
            next[tail[k]] = head[pa];
            head[pa] = next[i];
            nque[pa] += nque[k];
        }
    }

    for (csint i = 0; i < M; i++) {    // finalize row permutation
        if (S.p_inv[i] < 0) {
            S.p_inv[i] = k++;
        }
    }
}


Symbolic sqr(const CSCMatrix& A, AMDOrder order)
{
    auto [M, N] = A.shape();
    Symbolic S;  // allocate result
    std::vector<csint> q(M);  // column permutation vector

    if (order == AMDOrder::Natural) {
        std::iota(q.begin(), q.end(), 0);  // identity permutation
    } else {
        // TODO implement amd order (see Chapter 7)
        // q = amd(order, A);  // P = amd(A + A.T()) or natural
        throw std::runtime_error("Ordering method not implemented!");
    }

    S.q = q;  // store the column permutation

    // Find pattern of Cholesky factor of A.T @ A
    CSCMatrix C = A.permute_cols(S.q, false);  // don't copy values
    bool CTC = true;
    S.parent = etree(C, CTC);  // etree of C.T @ C, C = A[:, q]
    S.cp = counts(C, S.parent, post(S.parent));  // TODO col counts chol(C.T @ C)
    vcount(C, S);  // compute p_inv, leftmost, lnz, m2
    S.unz = std::accumulate(S.cp.begin(), S.cp.end(), 0);
    assert(S.lnz >= 0 && S.unz >= 0);  // overflow guard

    return S;
}


}  // namespace cs

/*==============================================================================
 *============================================================================*/
