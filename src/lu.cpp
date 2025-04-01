/*==============================================================================
 *     File: lu.cpp
 *  Created: 2025-03-19 12:32
 *   Author: Bernie Roesler
 *
 *  Description: Implementations for LU decomposition.
 *
 *============================================================================*/

#include <cmath>    // fabs
#include <numeric>  // iota
#include <ranges>   // views::reverse
#include <stdexcept>
#include <vector>

#include "types.h"
#include "lu.h"
#include "solve.h"
#include "utils.h"  // inv_permute

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

    // Exercise 6.6: modify to allow rectangular matrices
    const csint min_MN = std::min(M, N);

    // Allocate result matrices
    CSCMatrix L({M, min_MN}, S.lnz);  // lower triangular matrix
    CSCMatrix U({min_MN, N}, S.unz);  // upper triangular matrix
    std::vector<csint> p_inv(M, -1);  // row permutation vector

    csint lnz = 0,
          unz = 0;
    bool is_singular = false;

    for (csint k = 0; k < N; k++) {  // Compute L[:, k] and U[:, k]
        // --- Triangular solve ------------------------------------------------
        if (k < M) {
            L.p_[k] = lnz;  // L[:, k] starts here
        }

        if (k < N) {
            U.p_[k] = unz;  // U[:, k] starts here
        }

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
                if (i < U.M_) {  // if M > N, extra rows only in L
                    U.i_[unz] = p_inv[i];
                    U.v_[unz++] = sol.x[i];
                }
            }
        }

        // Exercise 6.6: modify to allow rectangular matrices
        // Only the rest of the columns of U need to be updated, L is done.
        if (k >= M) {
            continue;
        }

        // Exercise 6.5: modify to allow singular matrices
        // Two cases:
        //   1. ipiv == -1: occurs when we have a zero row.
        //   2. a <= 0: all entries in A[:, col] are zero, A[:, col] is
        //     structurally non-existent, or we have linearly dependent rows or
        //     columns of A.
        //   In either case, the column of L is just set to the identity, and
        //   the column of U is set to the non-zero entries of A[:, col].
        if ((ipiv == -1 || a <= 0) && !is_singular) {
            // throw std::runtime_error("Matrix is singular!");  // original
#ifdef DEBUG
            std::cerr << "[" << __FILE__ << ":" << __LINE__ 
                << "]: Warning: Matrix is singular!" << std::endl;
#endif
            is_singular = true;
        }

        // tol = 1 for partial pivoting; tol < 1 gives preference to diagonal
        if (p_inv[col] < 0 && std::fabs(sol.x[col]) >= a * tol) {
            ipiv = col;
        }

        // --- Divide by pivot -------------------------------------------------
        // Exercise 6.5: modify to allow singular matrices
        double pivot = 0;
        if (ipiv == -1) {
            // if all elements in a row are zero, then the row will never be
            // pivotal for any column, so ipiv stays as -1. Set it to col.
            ipiv = col;
        } else {
            pivot = sol.x[ipiv];  // the chosen pivot
            p_inv[ipiv] = k;      // ipiv is the kth pivot row
        }

        L.i_[lnz] = ipiv;  // first entry in L[:, k] is L(k, k) = 1
        L.v_[lnz++] = 1;

        // Exercise 6.5: modify to allow singular matrices
        if (pivot != 0) {
            U.i_[unz] = k;           // last entry in U[:, k] is U(k, k)
            U.v_[unz++] = pivot;

            for (const auto& i : sol.xi) {           // L(k+1:n, k) = x / pivot
                if (p_inv[i] < 0) {                  // x(i) is an entry in L[:, k]
                    L.i_[lnz] = i;                   // save unpermuted row in L
                    L.v_[lnz++] = sol.x[i] / pivot;  // scale pivot column
                }
            }
        }
    }

    // --- Finalize L and U ---------------------------------------------------
    L.p_[min_MN] = lnz;
    U.p_[N] = unz;

    // Exercise 6.5: modify to allow singular matrices
    // Exercise 6.6: modify to allow rectangular matrices
    // Assign indices to all missing rows that were pivoted to the end
    if (is_singular || M > N) {
        csint idx = M - 1;
        for (auto& i : p_inv | std::views::reverse) {
            if (i < 0) {
                i = idx--;
            }
        }
    }

    // permute row indices of L for final p_inv
    for (csint p = 0; p < lnz; p++) {
        L.i_[p] = p_inv[L.i_[p]];
    }

    L.realloc();  // trim excess storage
    U.realloc();

    return {L, U, p_inv, S.q};
}


// Exercise 6.4
LUResult relu(const CSCMatrix& A, const LUResult& R, const SymbolicLU& S)
{
    auto [M, N] = A.shape();

    // Copy result matrices without values
    CSCMatrix L {std::vector<double>(R.L.nnz()), R.L.i_, R.L.p_, R.L.shape()};
    CSCMatrix U {std::vector<double>(R.U.nnz()), R.U.i_, R.U.p_, R.U.shape()};

    // Initialize row permutation vector
    // NOTE we need this initialization because the -1 values are used in
    // spsolve() to determine which rows have been seen already, so a direct
    // copy of R.p_inv will fail.
    std::vector<csint> p_inv(N, -1);

    // TODO might be a way to avoid permuting/re-permuting the row indices?
    // The indices of L have already been permuted to the p_inv ordering by
    // the previous call to cs::lu(), so un-permute them here.
    const std::vector<csint> R_p = inv_permute(R.p_inv);
    for (csint p = 0; p < L.nnz(); p++) {
        L.i_[p] = R_p[L.i_[p]];
    }

    csint lnz = 0,
          unz = 0;

    for (csint k = 0; k < N; k++) {  // Compute L[:, k] and U[:, k]
        // --- Triangular solve ------------------------------------------------
        // Solve Lx = A[:, col], where col is the permuted column
        SparseSolution sol = spsolve(L, A, S.q[k], p_inv);  // x = L \ A[:, col]

        // --- Find pivot ------------------------------------------------------
        // Use the (un-permuted) pivot from the symbolic factorization
        csint ipiv = R_p[k];
        for (const auto& i : sol.xi) {
            if (p_inv[i] >= 0) {
                U.v_[unz++] = sol.x[i];  // x(i) is the entry U(p_inv[i], k)
            }
        }

        // --- Divide by pivot -------------------------------------------------
        double pivot = sol.x[ipiv];  // the chosen pivot
        p_inv[ipiv] = k;             // ipiv is the kth pivot row
        U.v_[unz++] = pivot;
        L.v_[lnz++] = 1;                         // L(k, k) = 1
        for (const auto& i : sol.xi) {           // L(k+1:n, k) = x / pivot
            if (p_inv[i] < 0) {                  // x(i) is an entry in L[:, k]
                L.v_[lnz++] = sol.x[i] / pivot;  // scale pivot column
            }
        }
    }

    // Permute the row indices of L back to the original
    for (csint p = 0; p < lnz; p++) {
        L.i_[p] = p_inv[L.i_[p]];
    }

    return {L, U, p_inv, S.q};
}

}  // namespace cs

/*==============================================================================
 *============================================================================*/
