/*==============================================================================
 *     File: lu.cpp
 *  Created: 2025-03-19 12:32
 *   Author: Bernie Roesler
 *
 *  Description: 
 *
 *============================================================================*/

#include <cmath>    // fabs
#include <numeric>  // iota
#include <vector>
#include <stdexcept>

#include "types.h"
// #include "csc.h"
#include "lu.h"
#include "solve.h"

namespace cs {


SymbolicLU slu(const CSCMatrix& A, AMDOrder order)
{
    auto [M, N] = A.shape();
    SymbolicLU S;             // allocate result
    std::vector<csint> q(N);  // column permutation vector

    if (order == AMDOrder::Natural) {
        std::iota(q.begin(), q.end(), 0);  // identity permutation
    } else {
        // TODO implement amd order (see Chapter 7)
        // q = amd(order, A);  // P = amd(A + A.T()) or natural
        throw std::runtime_error("Ordering method not implemented!");
    }

    S.q = q;
    S.unz = 4 * A.nnz() + N;  // guess nnz(L) and nnz(U)
    S.lnz = S.unz;

    return S;
}


LUResult lu(const CSCMatrix& A, const SymbolicLU& S, double tol)
{
    auto [M, N] = A.shape();

    // Allocate result matrices
    CSCMatrix L({N, N}, S.lnz);  // lower triangular matrix
    CSCMatrix U({N, N}, S.unz);  // upper triangular matrix
    std::vector<csint> p_inv(N);  // row permutation vector

    // Allocate workspace
    // std::vector<csint> xi(2*N);
    // std::vector<double> x(N);

    csint lnz = 0,
          unz = 0;

    for (csint k = 0; k < N; k++) {  // Compute L[:, k] and U[:, k]
        // --- Triangular solve ------------------------------------------------
        L.p_[k] = lnz;  // L[:, k] starts here
        U.p_[k] = unz;  // U[:, k] starts here

        // Possibly reallocate L and U
        if (lnz + N > L.nzmax()) {
            L.realloc(2 * L.nzmax() + N);
        }

        if (lnz + N > U.nzmax()) {
            U.realloc(2 * U.nzmax() + N);
        }

        // Solve Lx = A[:, k]
        csint col = S.q[k];
        SparseSolution sol = spsolve(L, A, col, p_inv);  // x = L \ A[:, col]

        // --- Find pivot ------------------------------------------------------
        csint ipiv = -1;
        double a = -1;
        for (const auto& i : sol.xi) {
            if (p_inv[i] < 0) {  // row i is not yet pivotal
                double t = std::fabs(sol.x[i]);
                if (t > a) {
                    a = t;  // largest pivot candidate so far
                    ipiv = i;
                }
            } else {  // x(i) is the entry U(pinv[i], k)
                U.i_[unz] = p_inv[i];
                U.v_[unz++] = sol.x[i];
            } 
        }

        if (ipiv == -1 || a <= 0) {
            throw std::runtime_error("Matrix is singular!");
        }

        // tol = 1 for partial pivoting; tol < 1 gives preference to diagonal
        if (p_inv[col] < 0 && std::fabs(sol.x[col]) >= a * tol) {
            ipiv = col;
        }

        // --- Divide by pivot -------------------------------------------------
        double pivot = sol.x[ipiv];  // the chosen pivot
        U.i_[unz] = k;           // last entry in U[:, k] is U(k, k)
        U.v_[unz++] = pivot;
        p_inv[ipiv] = k;         // ipiv is the kth pivot row
        L.i_[lnz] = ipiv;        // first entry in L[:, k] is L(k, k) = 1
        L.v_[lnz++] = 1;
        for (const auto& i : sol.xi) {  // L(k+1:n, k) = x / pivot
            if (p_inv[i] < 0) {  // x(i) is an entry in L[:, k]
                L.i_[lnz] = i;  // save unpermuted row in L
                L.v_[lnz++] = sol.x[i] / pivot;  // scale pivot column
            }
        }
    }

    // --- Finalize L and U ---------------------------------------------------
    L.p_[N] = lnz;
    U.p_[N] = unz;
    // permute row indices of L for final p_inv
    for (csint p = 0; p < lnz; p++) {
        L.i_[p] = p_inv[L.i_[p]];
    }
    L.realloc();  // trim excess storage
    U.realloc();

    return {L, U, p_inv, S.q};
}



}  // namespace cs

/*==============================================================================
 *============================================================================*/
