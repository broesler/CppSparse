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
#include <span>
#include <stdexcept>
#include <vector>

#include "types.h"
#include "cholesky.h"
#include "lu.h"
#include "qr.h"
#include "solve.h"
#include "utils.h"  // inv_permute

namespace cs {


SymbolicLU slu(const CSCMatrix& A, AMDOrder order, bool qr_bound, double alpha)
{
    auto [M, N] = A.shape();
    SymbolicLU S;             // allocate result
    std::vector<csint> q;     // column permutation vector

    // Determine column ordering of A
    if (order == AMDOrder::Natural) {
        q.resize(N);
        std::iota(q.begin(), q.end(), 0);  // identity permutation
    } else {
        q = amd(A, order);  // order = ATANoDenseRows for LU
    }

    S.q = q;  // store the column permutation

    // Estimate non-zeros in L and U
    if ((order == AMDOrder::APlusAT) && (!qr_bound)) {
        // Exercise 6.10: symbolic Cholesky analysis
        SymbolicChol S_chol = schol(A, AMDOrder::APlusAT);
        S.lnz = S.unz = S_chol.lnz;
    } else if (qr_bound) {
        // Exercise 8.6: use QR upper bound
        bool use_postorder = true;
        SymbolicQR S_qr = sqr(A, order, use_postorder);
        S.lnz = S_qr.vnz;
        S.unz = S_qr.rnz;
    } else {
        // Optimistic LU factorization estimate (Davis, p. 85)
        S.lnz = S.unz = alpha * (4 * A.nnz() + N);  // guess nnz(L) and nnz(U)
    }

    return S;
}


LUResult lu_original(const CSCMatrix& A, const SymbolicLU& S, double tol)
{
    auto [M, N] = A.shape();

    if (M != N) {
        throw std::runtime_error("Matrix must be square!");
    }

    // Allocate result matrices
    CSCMatrix L({N, N}, S.lnz);       // lower triangular matrix
    CSCMatrix U({N, N}, S.unz);       // upper triangular matrix
    std::vector<csint> p_inv(N, -1);  // row permutation vector
    SparseSolution sol(N);            // workspace for triangular solves

    csint lnz = 0,
          unz = 0;

    for (auto k : L.column_range()) {  // Compute L[:, k] and U[:, k]
        // --- Triangular solve ------------------------------------------------
        L.p_[k] = lnz;  // L[:, k] starts here
        U.p_[k] = unz;  // U[:, k] starts here

        // Possibly reallocate L and U
        if (lnz + N > L.nzmax()) {
            L.realloc(2 * L.nzmax() + N);
        }

        if (unz + N > U.nzmax()) {
            U.realloc(2 * U.nzmax() + N);
        }

        // Solve Lx = A[:, k]
        csint col = S.q[k];
        spsolve(L, A, col, sol, p_inv);  // x = L \ A[:, col]

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
            throw std::runtime_error("Matrix is singular!");  // original
        }

        // tol = 1 for partial pivoting; tol < 1 gives preference to diagonal
        if (p_inv[col] < 0 && std::fabs(sol.x[col]) >= a * tol) {
            ipiv = col;
        }

        // --- Divide by pivot -------------------------------------------------
        double pivot = sol.x[ipiv];  // the chosen pivot
        p_inv[ipiv] = k;      // ipiv is the kth pivot row
        L.i_[lnz] = ipiv;  // first entry in L[:, k] is L(k, k) = 1
        L.v_[lnz++] = 1;
        U.i_[unz] = k;           // last entry in U[:, k] is U(k, k)
        U.v_[unz++] = pivot;

        for (const auto& i : sol.xi) {           // L(k+1:n, k) = x / pivot
            if (p_inv[i] < 0) {                  // x(i) is an entry in L[:, k]
                L.i_[lnz] = i;                   // save unpermuted row in L
                L.v_[lnz++] = sol.x[i] / pivot;  // scale pivot column
            }
        }
    }

    // --- Finalize L and U ---------------------------------------------------
    L.p_[N] = lnz;
    U.p_[N] = unz;

    // permute row indices of L for final p_inv
    for (csint p = 0; p < lnz; ++p) {
        L.i_[p] = p_inv[L.i_[p]];
    }

    L.realloc();  // trim excess storage
    U.realloc();

    return {L, U, p_inv, S.q};
}


/** Allocate more space for the next column.
 *
 * See: Davis, Exercise 6.11.
 *
 * @param R  the matrix to reallocate
 * @param k  the current column index
 * @param lower  if `true`, allocate space for the lower triangular matrix
 *       `L`, otherwise allocate space for the upper triangular matrix `U`.
 *
 * @throws std::bad_alloc if memory cannot be allocated.
 */
void lu_realloc(CSCMatrix& R, csint k, bool lower)
{
    auto [M, N] = R.shape();
    csint nzmax = 2 * R.nnz() + M;
    csint nzmin = lower ? (R.nnz() + M - k) : (R.nnz() + k + 1);

    // Try the nzmax size, then halve the distance to nzmin until it works
    csint size_req = nzmax;
    std::string err_msg;

    while (size_req > nzmin) {
        try {
            R.realloc(size_req);
            return;
        } catch (const std::bad_alloc& e) {
            err_msg = e.what();
            size_req = (size_req + nzmin) / 2;
        }
    }

    // if we get here, we failed to allocate memory
    if (!err_msg.empty()) {
        // Print error message with file and line number
        std::cerr << "Error in " << __FILE__ << ":" << __LINE__ << ": " 
            << err_msg << std::endl;
        std::cerr << "    Failed to allocate memory for LU factorization." << std::endl;
        throw std::bad_alloc();
    }
}


/** Assign missing values in the permutation vector.
 *
 * This function is used to assign missing values in the permutation vector
 * `p_inv` to valid indices. The missing values are indicated by "-1" entries
 * in the vector.
 *
 * @param p_inv  the permutation vector
 */
static void make_valid_permutation(std::span<csint> p_inv)
{
    csint M = p_inv.size();

    std::vector<char> marked(M, false);  // char faster for random access
    for (csint i : p_inv) {
        if (i >= 0) {
            marked[i] = true;
        }
    }

    // Find all missing values
    std::vector<csint> missing;
    for (csint i = 0; i < M; ++i) {
        if (!marked[i]) {
            missing.push_back(i);
        }
    }

    // Assign them to "empty" locations in reverse to preserve order
    csint idx = 0;
    for (csint& i : p_inv) {
        if (i < 0) {
            if (idx < static_cast<csint>(missing.size())) {
                i = missing[idx++];
            } else {
                throw std::runtime_error("More missing rows than values!");
            }
        }
    }
}


LUResult lu(
    const CSCMatrix& A,
	const SymbolicLU& S,
	double tol,
	double col_tol
)
{
    if (tol < 0 || tol > 1) {
        throw std::runtime_error("Pivot tolerance must be in [0, 1]!");
    }

    if (col_tol < 0) {
        throw std::runtime_error("Column pivot tolerance must be non-negative!");
    }

    auto [M, N] = A.shape();

    // Exercise 6.6: modify to allow rectangular matrices
    const csint min_MN = std::min(M, N);

    // Allocate result matrices
    CSCMatrix L({M, min_MN}, S.lnz);  // lower triangular matrix
    CSCMatrix U({min_MN, N}, S.unz);  // upper triangular matrix
    std::vector<csint> p_inv(M, -1);  // row permutation vector
    SparseSolution sol(M);            // workspace for triangular solves

    // Exercise 6.3: modify to allow column pivoting
    std::vector<csint> q = S.q;  // column permutation vector
    csint K = 0;  // count small pivots

    csint lnz = 0,
          unz = 0;
    bool is_singular = false;

    for (csint k = 0; k < N; ++k) {  // Compute L[:, k] and U[:, k]
        // --- Triangular solve ------------------------------------------------
        if (k < M) {
            L.p_[k] = lnz;  // L[:, k] starts here
        }

        if (k < N) {
            U.p_[k] = unz;  // U[:, k] starts here
        }

        // Possibly reallocate L and U
        if (lnz + N > L.nzmax()) {
            lu_realloc(L, k, true);
        }

        if (unz + N > U.nzmax()) {
            lu_realloc(U, k, false);
        }

        // Solve Lx = A[:, k]
        csint col = q[k];
        spsolve(L, A, col, sol, p_inv);  // x = L \ A[:, col]

        // --- Find pivot ------------------------------------------------------
        csint pre_unz = unz;  // Exercise 6.3: store in case we pivot the column
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
        if ((ipiv == -1 || a <= 0) && !is_singular) {
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
            // Found a pivot candidate
            pivot = sol.x[ipiv];  // the chosen pivot

            // Exercise 6.3: column pivoting
            if ((std::fabs(pivot) < col_tol) && k < N - K) {
                // pivot column to the end of the matrix, preserving the order
                csint v = std::move(q[k]);
                q.erase(q.begin() + k);
                q.push_back(v);
                ++K;            // count small pivots
                unz = pre_unz;  // reset unz to the last entry in U
                --k;            // decrement k to recompute this column of output
                continue;
            }

        }

        p_inv[ipiv] = k;   // ipiv is the kth pivot row
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
    if (is_singular) {
        // There will be remaining "-1" entries corresponding to empty rows:
        //     p_inv: [0, 1, -1, 7, 2, 3, 6, 4] -> needs a 5
        // Assign the remaining entries for a valid permutation.
        make_valid_permutation(p_inv);
    }

    // Exercise 6.6: modify to allow rectangular matrices
    // Assign indices to all missing rows that were pivoted to the end
    if (M > N) {
        csint idx = M - 1;
        for (auto& i : p_inv | std::views::reverse) {
            if (i < 0) {
                i = idx--;
            }
        }
    }

    // permute row indices of L for final p_inv
    for (csint p = 0; p < lnz; ++p) {
        L.i_[p] = p_inv[L.i_[p]];
    }

    L.realloc();  // trim excess storage
    U.realloc();

    return {L, U, p_inv, q};
}


// Exercise 6.4
LUResult relu(const CSCMatrix& A, const LUResult& R, const SymbolicLU& S)
{
    auto [M, N] = A.shape();

    // Copy result matrices without values
    CSCMatrix L{std::vector<double>(R.L.nnz()), R.L.i_, R.L.p_, R.L.shape()};
    CSCMatrix U{std::vector<double>(R.U.nnz()), R.U.i_, R.U.p_, R.U.shape()};

    // Initialize row permutation vector
    // NOTE we need this initialization because the -1 values are used in
    // spsolve() to determine which rows have been seen already, so a direct
    // copy of R.p_inv will fail.
    std::vector<csint> p_inv(N, -1);

    // NOTE might be a way to avoid permuting/re-permuting the row indices?
    // The indices of L have already been permuted to the p_inv ordering by
    // the previous call to cs::lu(), so un-permute them here.
    const std::vector<csint> R_p = inv_permute(R.p_inv);
    for (csint p = 0; p < L.nnz(); ++p) {
        L.i_[p] = R_p[L.i_[p]];
    }

    SparseSolution sol(M);  // workspace for triangular solves

    csint lnz = 0,
          unz = 0;

    for (auto k : A.column_range()) {  // Compute L[:, k] and U[:, k]
        // --- Triangular solve ------------------------------------------------
        // Solve Lx = A[:, col], where col is the permuted column
        spsolve(L, A, S.q[k], sol, p_inv);  // x = L \ A[:, col]

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
    for (csint p = 0; p < lnz; ++p) {
        L.i_[p] = p_inv[L.i_[p]];
    }

    return {L, U, p_inv, S.q};
}


// Exercise 6.7
LUResult lu_crout(const CSCMatrix& A, const SymbolicLU& S)
{
    auto [M, N] = A.shape();

    if (M != N) {
        throw std::runtime_error("Matrix must be square!");
    }

    // Allocate result matrices
    CSCMatrix L({N, N}, S.lnz);       // lower triangular matrix
    CSCMatrix UT({N, N}, S.unz);      // (transpose of) upper triangular matrix
    std::vector<csint> p_inv(N, -1);  // row permutation vector
    // TODO implement partial pivoting
    std::iota(p_inv.begin(), p_inv.end(), 0);  // identity permutation

    csint lnz = 0,
          unz = 0;

    for (auto k : A.column_range()) {  // Compute L[:, k] and U[k, :]
        L.p_[k] = lnz;   // L[:, k] starts here
        UT.p_[k] = unz;  // U[k, :] starts here

        // Possibly reallocate L and U
        if (lnz + N > L.nzmax()) {
            lu_realloc(L, k, true);
        }

        if (unz + N > UT.nzmax()) {
            lu_realloc(UT, k, false);
        }

        // ---------- Compute the row of U
        // U[k, k:] = A[k, k:] - L[k, :k] @ U[:k, k:]
        // NOTE how to do in a sparse way?
        //   * non-zeros will occur in row of U where either:
        //      * A[k, j] is non-zero, or
        //      * L[k, :k] @ U[:k, j] is non-zero.
        //   * The dot product will be non-zero where *both* 
        //      * L[k, :k] (row of L) and U[:k, j] (col of U) are non-zero.
        //
        //   * Set operation is J = A ∪ (L ∩ U).
        //   * Need to do slice of U for each j to compute intersection of L and
        //     U, so we *do* need to loop over all j values.
        //   * `vecdot` is already a sparse dot product.
        //   * `slice` is already a sparse operation (albeit a "slow" copy).
        //   * A(k, j) is already a sparse operation.
        //
        //   => looping over all j values is actually the most efficient way to
        //      do this operation.
        CSCMatrix L_col = L.slice(k, k+1, 0, k).T();  // == L[k, :k].T
        for (csint j = k; j < N; ++j) {
            double lu_dot = 0.0;
            if (k > 0) {
                CSCMatrix U_col = UT.slice(j, j+1, 0, k).T();  // == U[:k, j]
                lu_dot = L_col.vecdot(U_col);
            }

            double a = A(k, j) - lu_dot;

            if (std::fabs(a) > 0) {
                UT.i_[unz] = j;
                UT.v_[unz++] = a;
            }
        }

        // ---------- Compute the column of L
        // Place 1.0 on the diagonal
        L.i_[lnz] = k;
        L.v_[lnz++] = 1.0;

        // Compute the rest of the column
        // L[k+1:n, k] = (A[k+1:n, k] - L[k+1:n, :k] @ U[:k, k]) / U[k, k]
        CSCMatrix U_col = UT.slice(k, k+1, 0, k).T();  // == U[:k, k]
        for (csint i = k+1; i < N; ++i) {
            double lu_dot = 0.0;
            if (k > 0) {
                CSCMatrix L_col = L.slice(i, i+1, 0, k).T();  // == L[i, :k].T
                lu_dot = L_col.vecdot(U_col);
            }

            double a = A(i, k) - lu_dot;
            double pivot = UT.v_[UT.p_[k]];  // first element in col == UT(k, k);

            if (pivot == 0.0) {
                throw std::runtime_error("Matrix is singular!");
            }

            if (std::fabs(a) > 0) {
                L.i_[lnz] = i;
                L.v_[lnz++] = a / pivot;
            }
        }
    }

    // Finalize L and U
    L.p_[N] = lnz;
    UT.p_[N] = unz;

    L.realloc();  // trim excess storage
    UT.realloc();

    // By construction:
    //   * L and UT computed in increasing row index order
    //   * transpose of U on output also sorts
    //   * Numerically zero entries are excluded
    //   * A(i, k) sums duplicates
    L.has_canonical_format_ = true;
    UT.has_canonical_format_ = true;

    return {L, UT.T(), p_inv, S.q};
}


// Exercise 6.13
LUResult ilutp(
    const CSCMatrix& A,
    const SymbolicLU& S,
    double drop_tol,
    double tol
)
{
    auto [M, N] = A.shape();

    if (M != N) {
        throw std::runtime_error("Matrix must be square!");
    }

    if (drop_tol < 0) {
        throw std::runtime_error("Drop tolerance must be non-negative!");
    }

    // Allocate result matrices
    CSCMatrix L({N, N}, S.lnz);       // lower triangular matrix
    CSCMatrix U({N, N}, S.unz);       // upper triangular matrix
    std::vector<csint> p_inv(N, -1);  // row permutation vector
    SparseSolution sol(M);            // workspace for triangular solves

    csint lnz = 0,
          unz = 0;

    for (auto k : A.column_range()) {  // Compute L[:, k] and U[:, k]
        // --- Triangular solve ------------------------------------------------
        L.p_[k] = lnz;  // L[:, k] starts here
        U.p_[k] = unz;  // U[:, k] starts here

        // Possibly reallocate L and U
        if (lnz + N > L.nzmax()) {
            lu_realloc(L, k, true);
        }

        if (unz + N > U.nzmax()) {
            lu_realloc(U, k, false);
        }

        // Solve Lx = A[:, k]
        csint col = S.q[k];
        spsolve(L, A, col, sol, p_inv);  // x = L \ A[:, col]

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
                // NOTE to do relative tolerance, we need to separate the
                // assignment to U from the pivot search, so that we can compare
                // the U elements to the pivot.
                // Might be able to combine with the L update loop, as long as
                // we add the diagonal U(k, k) after the loop. The only issue is
                // the p_inv[ipiv] = k assignment, which needs to be done before
                // the L update loop, but after the U update loop.
                double x = sol.x[i];
                if (std::fabs(x) > drop_tol) {
                    U.i_[unz] = p_inv[i];
                    U.v_[unz++] = x;
                }
            }
        }

        if (ipiv == -1 || a <= 0) {
            throw std::runtime_error("Matrix is singular!");  // original
        }

        // tol = 1 for partial pivoting; tol < 1 gives preference to diagonal
        if (p_inv[col] < 0 && std::fabs(sol.x[col]) >= a * tol) {
            ipiv = col;
        }

        // --- Divide by pivot -------------------------------------------------
        // TODO modify to allow singular matrices (see Exercise 6.5)
        double pivot = sol.x[ipiv];  // the chosen pivot
        p_inv[ipiv] = k;             // ipiv is the kth pivot row
        L.i_[lnz] = ipiv;            // first entry in L[:, k] is L(k, k) = 1
        L.v_[lnz++] = 1;
        U.i_[unz] = k;               // last entry in U[:, k] is U(k, k)
        U.v_[unz++] = pivot;

        for (const auto& i : sol.xi) {          // L(k+1:n, k) = x / pivot
            if (p_inv[i] < 0) {                 // x(i) is an entry in L[:, k]
                double x = sol.x[i] / pivot;
                if (std::fabs(x) > drop_tol) {  // ensure entry is large enough
                    L.i_[lnz] = i;              // save unpermuted row in L
                    L.v_[lnz++] = x;            // scale pivot column
                }
            }
        }
    }

    // --- Finalize L and U ---------------------------------------------------
    L.p_[N] = lnz;
    U.p_[N] = unz;
    // permute row indices of L for final p_inv
    for (csint p = 0; p < lnz; ++p) {
        L.i_[p] = p_inv[L.i_[p]];
    }
    L.realloc();  // trim excess storage
    U.realloc();

    return {L, U, p_inv, S.q};
}


// Exercise 6.13
LUResult ilu_nofill(
    const CSCMatrix& A,
    const SymbolicLU& S
)
{
    auto [M, N] = A.shape();

    if (M != N) {
        throw std::runtime_error("Matrix must be square!");
    }

    // Allocate result matrices
    CSCMatrix L({N, N}, A.nnz());     // lower triangular matrix
    CSCMatrix U({N, N}, A.nnz());     // upper triangular matrix
    std::vector<csint> p_inv(N, -1);  // row permutation vector
    std::vector<csint> w(N, -1);      // workspace for values
    SparseSolution sol(M);            // workspace for triangular solves

    csint lnz = 0,
          unz = 0;

    for (auto k : A.column_range()) {  // Compute L[:, k] and U[:, k]
        // --- Triangular solve ------------------------------------------------
        L.p_[k] = lnz;  // L[:, k] starts here
        U.p_[k] = unz;  // U[:, k] starts here

        // NOTE no need for reallocation!

        // Solve Lx = A[:, k]
        csint col = S.q[k];
        spsolve(L, A, col, sol, p_inv);  // x = L \ A[:, col]

        // Scatter the pattern of A[:, col] into w
        for (auto i : A.row_indices(col)) {
            w[i] = k;  // mark the pattern of A[:, col]
        }

        // --- Find pivot ------------------------------------------------------
        // Use diagonal as pivot. MATLAB's ilu(type='nofill') errors when
        // there is a zero on the diagonal, and does not support pivoting.
        csint ipiv = k;
        for (const auto& i : sol.xi) {
            if (p_inv[i] >= 0 && w[i] == k) {
                // x(i) is the entry U(pinv[i], k) in the pattern of A[:, col]
                U.i_[unz] = p_inv[i];
                U.v_[unz++] = sol.x[i];
            }
        }

        // --- Divide by pivot -------------------------------------------------
        double pivot = sol.x[ipiv];  // the chosen pivot

        if (pivot == 0) {
            throw std::runtime_error("Matrix is singular!");
        }

        p_inv[ipiv] = k;             // ipiv is the kth pivot row
        L.i_[lnz] = ipiv;            // first entry in L[:, k] is L(k, k) = 1
        L.v_[lnz++] = 1;
        U.i_[unz] = k;               // last entry in U[:, k] is U(k, k)
        U.v_[unz++] = pivot;

        for (const auto& i : sol.xi) {           // L(k+1:n, k) = x / pivot
            if (p_inv[i] < 0 && w[i] == k) {     // x(i) is an entry in L[:, k]
                L.i_[lnz] = i;                   // save unpermuted row in L
                L.v_[lnz++] = sol.x[i] / pivot;  // scale pivot column
            }
        }
    }

    // --- Finalize L and U ---------------------------------------------------
    L.p_[N] = lnz;
    U.p_[N] = unz;
    L.realloc();  // trim excess storage
    U.realloc();

    return {L, U, p_inv, S.q};
}



}  // namespace cs

/*==============================================================================
 *============================================================================*/
