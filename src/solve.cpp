/*==============================================================================
 *     File: solve.cpp
 *  Created: 2025-01-30 13:52
 *   Author: Bernie Roesler
 *
 *  Description: Implementations of various matrix solvers.
 *
 *============================================================================*/

#include <algorithm>   // fill
#include <cassert>
#include <cmath>       // fabs
#include <functional>  // reference wrapper
#include <ranges>      // views::reverse
#include <span>
#include <vector>

#include "solve.h"
#include "csc.h"
#include "utils.h"
#include "cholesky.h"
#include "qr.h"
#include "lu.h"

namespace cs {

/*------------------------------------------------------------------------------
 *      Triangular Matrix Solutions
 *----------------------------------------------------------------------------*/
void lsolve_inplace(const CSCMatrix& L, std::span<double> x)
{
    for (csint j = 0; j < L.N_; j++) {
        x[j] /= L.v_[L.p_[j]];
        for (csint p = L.p_[j] + 1; p < L.p_[j+1]; p++) {
            x[L.i_[p]] -= L.v_[p] * x[j];
        }
    }
}


std::vector<double> lsolve(const CSCMatrix& L, const std::vector<double>& B)
{
    return trisolve_impl(L, B, lsolve_inplace);
}


std::vector<double> lsolve(const CSCMatrix& L, const CSCMatrix& b)
{
    return spsolve(L, b, 0, std::nullopt, true).x;
}


void ltsolve_inplace(const CSCMatrix& L, std::span<double> x)
{
    for (csint j = L.N_ - 1; j >= 0; j--) {
        for (csint p = L.p_[j] + 1; p < L.p_[j+1]; p++) {
            x[j] -= L.v_[p] * x[L.i_[p]];
        }
        x[j] /= L.v_[L.p_[j]];
    }
}


std::vector<double> ltsolve(const CSCMatrix& L, const std::vector<double>& B)
{
    return trisolve_impl(L, B, ltsolve_inplace);
}


void usolve_inplace(const CSCMatrix& U, std::span<double> x)
{
    for (csint j = U.N_ - 1; j >= 0; j--) {
        x[j] /= U.v_[U.p_[j+1] - 1];  // diagonal entry
        for (csint p = U.p_[j]; p < U.p_[j+1] - 1; p++) {
            x[U.i_[p]] -= U.v_[p] * x[j];
        }
    }
}


std::vector<double> usolve(const CSCMatrix& U, const std::vector<double>& B)
{
    return trisolve_impl(U, B, usolve_inplace);
}


std::vector<double> usolve(const CSCMatrix& U, const CSCMatrix b)
{
    return spsolve(U, b, 0, std::nullopt, false).x;
}


void utsolve_inplace(const CSCMatrix& U, std::span<double> x)
{
    for (csint j = 0; j < U.N_; j++) {
        for (csint p = U.p_[j]; p < U.p_[j+1] - 1; p++) {
            x[j] -= U.v_[p] * x[U.i_[p]];
        }
        x[j] /= U.v_[U.p_[j+1] - 1];  // diagonal entry
    }
}


std::vector<double> utsolve(const CSCMatrix& U, const std::vector<double>& B)
{
    return trisolve_impl(U, B, utsolve_inplace);
}


// Exercise 3.8
void lsolve_inplace_opt(const CSCMatrix& L, std::span<double> x)
{
    for (csint j = 0; j < L.N_; j++) {
        double& x_val = x[j];  // cache reference to value
        // Exercise 3.8: improve performance by checking for zeros
        if (x_val != 0) {
            x_val /= L.v_[L.p_[j]];
            for (csint p = L.p_[j] + 1; p < L.p_[j+1]; p++) {
                x[L.i_[p]] -= L.v_[p] * x_val;
            }
        }
    }
}


std::vector<double> lsolve_opt(const CSCMatrix& L, const std::vector<double>& b)
{
    return trisolve_impl(L, b, lsolve_inplace_opt);
}


// Exercise 3.8
void usolve_inplace_opt(const CSCMatrix& U, std::span<double> x)
{
    for (csint j = U.N_ - 1; j >= 0; j--) {
        double& x_val = x[j];  // cache reference to value
        if (x_val != 0) {
            x_val /= U.v_[U.p_[j+1] - 1];  // diagonal entry
            for (csint p = U.p_[j]; p < U.p_[j+1] - 1; p++) {
                x[U.i_[p]] -= U.v_[p] * x_val;
            }
        }
    }
}


std::vector<double> usolve_opt(const CSCMatrix& U, const std::vector<double>& b)
{
    return trisolve_impl(U, b, usolve_inplace_opt);
}


// Exercise 3.3
std::vector<csint> find_lower_diagonals(const CSCMatrix& A)
{
    assert(A.M_ == A.N_);

    std::vector<char> marked(A.N_, false);  // workspace
    std::vector<csint> p_diags(A.N_);       // diagonal indicies (inverse permutation)

    for (csint j = A.N_ - 1; j >= 0; j--) {
        csint N_unmarked = 0;

        for (csint p = A.p_[j]; p < A.p_[j+1]; p++) {
            csint i = A.i_[p];
            // Mark the rows viewed so far
            if (!marked[i]) {
                marked[i] = true;
                p_diags[j] = p;
                N_unmarked++;
            }
        }

        // If 0 or > 1 "diagonal" entries found, the matrix is not permuted.
        if (N_unmarked != 1) {
            throw PermutedTriangularMatrixError(
                "Matrix is not a permuted lower triangular matrix!"
            );
        }
    }

    return p_diags;
}


// Exercise 3.3
std::vector<double> lsolve_rows(const CSCMatrix& A, const std::vector<double>& b)
{
    assert(A.M_ == A.N_);
    assert(A.M_ == static_cast<csint>(b.size()));

    // First (backward) pass to find diagonal entries
    // p_diags is a vector of pointers to the diagonal entries
    std::vector<csint> p_diags = find_lower_diagonals(A);

    // Compute the row permutation vector
    std::vector<csint> p_inv(A.N_);
    for (csint j = 0; j < A.N_; j++) {
        p_inv[j] = A.i_[p_diags[j]];
    }

    // Second (forward) pass to solve the system PL x = b -> L x = P^T b
    std::vector<double> x(A.N_);
    std::vector<double> b_work = b;

    // Perform the permuted forward solve
    for (csint j = 0; j < A.N_; j++) {
        csint i = p_inv[j];        // permuted row index
        csint d = p_diags[j];      // pointer to the diagonal entry
        double x_val = b_work[i];  // cache diagonal value

        if (x_val != 0) {
            x_val /= A.v_[d];  // solve for x[d]
            x[j] = x_val;      // store solution in correct position
            // update the off-diagonals
            for (csint p = A.p_[j]; p < A.p_[j+1]; p++) {
                if (p != d) {
                    b_work[A.i_[p]] -= A.v_[p] * x_val;
                }
            }
        }
    }

    return x;
}


// Exercise 3.5
std::vector<double> lsolve_cols(const CSCMatrix& A, const std::vector<double>& b)
{
    assert(A.M_ == A.N_);
    assert(A.M_ == static_cast<csint>(b.size()));

    // First O(N) pass to find the diagonal entries
    // Assume that the first entry in each column has the smallest row index
    std::vector<csint> p_diags(A.N_, -1);
    for (csint j = 0; j < A.N_; j++) {
        if (p_diags[j] == -1) {
            p_diags[j] = A.p_[j];  // pointer to the diagonal entry
        } else {
            // We have seen this column index before
            throw PermutedTriangularMatrixError(
                "Matrix is not a permuted lower triangular matrix!"
            );
        }
    }

    // Compute the column permutation vector
    std::vector<csint> q_inv(A.N_);
    for (csint i = 0; i < A.N_; i++) {
        q_inv[A.i_[p_diags[i]]] = i;
    }

    // Second (forward) pass to solve the system LQ x = b
    std::vector<double> x(A.N_);
    std::vector<double> b_work = b;

    // Perform the permuted forward solve
    for (const auto& j : q_inv) {
        csint d = p_diags[j];      // pointer to the diagonal entry
        csint i = A.i_[d];         // permuted row index
        double x_val = b_work[i];  // cache diagonal value

        if (x_val != 0) {
            x_val /= A.v_[d];  // solve for x[A.i_[d]]
            x[j] = x_val;      // store solution in correct position
            for (csint p = A.p_[j]+1; p < A.p_[j+1]; p++) {
                b_work[A.i_[p]] -= A.v_[p] * x_val;  // update the off-diagonals
            }
        }
    }

    return x;
}


// Exercise 3.4
std::vector<csint> find_upper_diagonals(const CSCMatrix& U)
{
    assert(U.M_ == U.N_);

    std::vector<bool> marked(U.N_, false);  // workspace
    std::vector<csint> p_diags(U.N_);  // diagonal indicies (inverse permutation)

    for (csint j = 0; j < U.N_; j++) {
        csint N_unmarked = 0;

        for (csint p = U.p_[j]; p < U.p_[j+1]; p++) {
            csint i = U.i_[p];
            // Mark the rows viewed so far
            if (!marked[i]) {
                marked[i] = true;
                p_diags[j] = p;
                N_unmarked++;
            }
        }

        // If 0 or > 1 "diagonal" entries found, the matrix is not permuted.
        if (N_unmarked != 1) {
            throw PermutedTriangularMatrixError(
                "Matrix is not a permuted upper triangular matrix!"
            );
        }
    }

    return p_diags;
}


// Exercise 3.4
std::vector<double> usolve_rows(const CSCMatrix& A, const std::vector<double>& b)
{
    assert(A.M_ == A.N_);
    assert(A.M_ == static_cast<csint>(b.size()));

    // First (backward) pass to find diagonal entries
    // p_diags is a vector of pointers to the diagonal entries
    std::vector<csint> p_diags = find_upper_diagonals(A);

    // Compute the row permutation vector
    std::vector<csint> p_inv(A.N_);
    for (csint i = 0; i < A.N_; i++) {
        p_inv[i] = A.i_[p_diags[i]];
    }

    // Second (forward) pass to solve the system PU x = b -> U x = P^T b
    std::vector<double> x(A.N_);
    std::vector<double> b_work = b;

    // Perform the permuted backward solve
    for (csint j = A.N_ - 1; j >= 0; j--) {
        csint i = p_inv[j];        // permuted row index
        csint d = p_diags[j];      // pointer to the diagonal entry
        double x_val = b_work[i];  // cache diagonal value

        if (x_val != 0) {
            x_val /= A.v_[d];  // solve for x[d]
            x[j] = x_val;      // store solution in correct position
            for (csint p = A.p_[j]; p < A.p_[j+1]; p++) {
                if (p != d) {
                    b_work[A.i_[p]] -= A.v_[p] * x_val;  // update the off-diagonals
                }
            }
        }
    }

    return x;
}


std::vector<double> usolve_cols(const CSCMatrix& A, const std::vector<double>& b)
{
    assert(A.M_ == A.N_);
    assert(A.M_ == static_cast<csint>(b.size()));

    // First O(N) pass to find the diagonal entries
    // Assume that the last entry in each column has the largest row index
    std::vector<csint> p_diags(A.N_, -1);
    for (csint j = 0; j < A.N_; j++) {
        if (p_diags[j] == -1) {
            p_diags[j] = A.p_[j+1] - 1;  // pointer to the diagonal entry
        } else {
            // We have seen this column index before
            throw PermutedTriangularMatrixError(
                "Matrix is not a permuted upper triangular matrix!"
            );
        }
    }

    // Compute the column permutation vector
    std::vector<csint> q_inv(A.N_);
    for (csint i = 0; i < A.N_; i++) {
        q_inv[A.i_[p_diags[i]]] = i;
    }

    // Second (forward) pass to solve the system UQ x = b
    std::vector<double> x(A.N_);
    std::vector<double> b_work = b;

    // Perform the permuted backward solve
    for (const auto& j : std::views::reverse(q_inv)) {
        csint d = p_diags[j];      // pointer to the diagonal entry
        csint i = A.i_[d];         // permuted row index
        double x_val = b_work[i];  // cache diagonal value

        if (x_val != 0) {
            x_val /= A.v_[d];  // solve for x[A.i_[d]]
            x[j] = x_val;      // store solution in correct position
            for (csint p = A.p_[j]; p < A.p_[j+1] - 1; p++) {
                b_work[A.i_[p]] -= A.v_[p] * x_val;  // update the off-diagonals
            }
        }
    }

    return x;
}


// Exercise 3.7
TriPerm find_tri_permutation(const CSCMatrix& A)
{
    assert(A.M_ == A.N_);

    // Create a vector of row counts and corresponding set vector
    std::vector<csint> r(A.N_, 0);
    std::vector<csint> z(A.N_, 0);  // z[i] is XORed with each column j in row i

    for (csint j = 0; j < A.N_; j++) {
        for (csint p = A.p_[j]; p < A.p_[j+1]; p++) {
            r[A.i_[p]]++;
            z[A.i_[p]] ^= j;
        }
    }

    // Create a list of singleton row indices
    std::vector<csint> singles;
    singles.reserve(A.N_);

    for (csint i = 0; i < A.N_; i++) {
        if (r[i] == 1) {
            singles.push_back(i);
        }
    }

    // Iterate through the columns to get the permutation vectors
    std::vector<csint> p_inv(A.N_, -1);
    std::vector<csint> q_inv(A.N_, -1);
    std::vector<csint> p_diags(A.N_, -1);

    for (csint k = 0; k < A.N_; k++) {
        // Take a singleton row
        if (singles.empty()) {
            throw PermutedTriangularMatrixError(
                "Matrix is not a permuted triangular matrix!"
            );
        }

        csint i = singles.back();
        singles.pop_back();
        csint j = z[i];  // column index

        // Update the permutations
        p_inv[k] = i;
        q_inv[k] = j;

        // Decrement each row count, and update the set vector
        for (csint p = A.p_[j]; p < A.p_[j+1]; p++) {
            csint t = A.i_[p];
            if (--r[t] == 1) {
                singles.push_back(t);
            }
            z[t] ^= j;  // removes j from the set
            if (t == i) {
                p_diags[k] = p;  // store the pointers to the diagonal entries
            }
        }
    }

    return {p_inv, q_inv, p_diags};
}


// Exercise 3.7
void tri_solve_perm_inplace(
    const CSCMatrix& A,
    const TriPerm& tri_perm,
    std::span<double> b,
    std::span<double> x
)
{
    // Extract the permutation vectors
    auto [p_inv, q_inv, p_diags] = tri_perm;

    // Solve the system (PTQ) x = b => T (Q x) = (P^T b)
    for (csint k = 0; k < A.N_; k++) {
        csint i = p_inv[k];    // permuted row
        csint j = q_inv[k];    // permuted column
        csint d = p_diags[k];  // pointer to the diagonal entry

        // Solve for x[j]
        double x_val = b[i];
        if (x_val != 0) {
            x_val /= A.v_[d];  // diagonal entry
            x[j] = x_val;
            // Update off-diagonals
            for (csint p = A.p_[j]; p < A.p_[j+1]; p++) {
                if (p != d) {
                    b[A.i_[p]] -= A.v_[p] * x_val;
                }
            }
        }
    }
}


std::vector<double> tri_solve_perm(const CSCMatrix& A, const std::vector<double>& B)
{
    auto [M, N] = A.shape();
    csint MxK = static_cast<csint>(B.size());

    if (M != N) {
        throw std::runtime_error("Matrix must be square!");
    }

    if (MxK % M != 0) {
        throw std::runtime_error(
            std::format(
                "RHS vector size is not a multiple of matrix rows! {} % {} != 0",
                MxK, M
            )
        );
    }

    // Get the permutation vectors and check if A is permuted triangular
    const TriPerm tri_perm = find_tri_permutation(A);

    csint K = MxK / M;               // number of RHS columns
    std::vector<double> X(N * K);    // solution vector
    std::vector<double> B_work = B;  // copy the RHS vector

    std::span<double> X_span(X);
    std::span<double> B_work_span(B_work);

    // Solve each column of the system
    for (csint k = 0; k < K; k++) {
        auto B_work_k = B_work_span.subspan(k * M, M);
        auto X_k = X_span.subspan(k * N, N);
        tri_solve_perm_inplace(A, tri_perm, B_work_k, X_k);
    }

    return X;
}


SparseSolution spsolve(
    const CSCMatrix& A,
    const CSCMatrix& B,
    csint k,
    OptionalVectorRef<csint> p_inv_ref,
    bool lower
)
{
    // Populate xi with the non-zero indices of x
    std::vector<csint> xi = reach(A, B, k, p_inv_ref);
    std::vector<double> x(A.M_);  // dense output vector

    // scatter B(:, k) into x
    for (csint p = B.p_[k]; p < B.p_[k+1]; p++) {
        x[B.i_[p]] = B.v_[p];
    }

    // Solve Lx = b_k or Ux = b_k
    for (auto& j : xi) {  // x(j) is nonzero
        // j maps to col J of G
        csint J = p_inv_ref.has_value() ? p_inv_ref.value().get()[j] : j;
        if (J < 0) {
            continue;  // x(j) is not in the pattern of G
        }
        double& xj = x[j];                              // cache reference to value
        xj /= A.v_[lower ? A.p_[J] : A.p_[J+1] - 1];    // x(j) /= G(j, j)
        csint p = lower ? A.p_[J] + 1 : A.p_[J];        // lower: L(j,j) 1st entry
        csint q = lower ? A.p_[J+1]   : A.p_[J+1] - 1;  // up: U(j,j) last entry
        for (; p < q; p++) {
            x[A.i_[p]] -= A.v_[p] * xj;              // x[i] -= G(i, j) * x[j]
        }
    }

    return {xi, x};
}


std::vector<csint> reach(
    const CSCMatrix& A,
    const CSCMatrix& B,
    csint k,
    OptionalVectorRef<csint> p_inv_ref
)
{
    std::vector<char> marked(A.M_, false);
    std::vector<csint> xi,      // do not initialize for dfs call!
                       pstack,  // pause and recursion stacks
                       rstack;
    xi.reserve(A.N_);
    pstack.reserve(A.N_);
    rstack.reserve(A.N_);

    for (csint p = B.p_[k]; p < B.p_[k+1]; p++) {
        csint j = B.i_[p];  // consider nonzero B(j, k)
        if (!marked[j]) {
            xi = dfs(A, j, marked, xi, pstack, rstack, p_inv_ref);
        }
    }

    // xi is returned from dfs in reverse order, since it is a stack
    std::reverse(xi.begin(), xi.end());
    return xi;
}


std::vector<csint>& dfs(
    const CSCMatrix& A,
    csint j,
    std::vector<char>& marked,
    std::vector<csint>& xi,
    std::vector<csint>& pstack,
    std::vector<csint>& rstack,
    OptionalVectorRef<csint> p_inv_ref
)
{
    // Ensure the stacks are reserved and cleared
    if (static_cast<csint>(pstack.capacity()) < A.N_) { pstack.reserve(A.N_); }
    if (static_cast<csint>(rstack.capacity()) < A.N_) { rstack.reserve(A.N_); }
    pstack.clear();
    rstack.clear();

    rstack.push_back(j);  // initialize the recursion stack

    bool done = false;  // true if no unvisited neighbors

    while (!rstack.empty()) {
        j = rstack.back();  // get j from the top of the recursion stack
        // j maps to col jnew of G
        csint jnew = p_inv_ref.has_value() ? p_inv_ref.value().get()[j] : j;

        if (!marked[j]) {
            marked[j] = true;  // mark node j as visited
            pstack.push_back((jnew < 0) ? 0 : A.p_[jnew]);
        }

        done = true;  // node j done if no unvisited neighbors
        csint q = (jnew < 0) ? 0 : A.p_[jnew+1];

        // examine all neighbors of j
        for (csint p = pstack.back(); p < q; p++) {
            csint i = A.i_[p];        // consider neighbor node i
            if (!marked[i]) {
                pstack.back() = p;    // pause dfs of node j
                rstack.push_back(i);  // start dfs at node i
                done = false;         // node j has unvisited neighbors
                break;
            }
        }

        if (done) {
            pstack.pop_back();
            rstack.pop_back();  // node j is done; pop it from the stack
            xi.push_back(j);    // node j is the next on the output stack
        }
    }

    return xi;
}


namespace detail {

std::vector<csint> reach_r(const CSCMatrix& A, const CSCMatrix& B)
{
    std::vector<char> marked(A.M_, false);
    std::vector<csint> xi;
    xi.reserve(A.N_);

    for (csint p = B.p_[0]; p < B.p_[1]; p++) {
        csint j = B.i_[p];  // consider nonzero B(j, 0)
        if (!marked[j]) {
            dfs_r(A, j, marked, xi);
        }
    }

    // xi is returned from dfs in reverse order, since it is a stack
    std::reverse(xi.begin(), xi.end());
    return xi;
}


std::vector<csint>& dfs_r(
    const CSCMatrix& A,
    csint j,
    std::vector<char>& marked,
    std::vector<csint>& xi
)
{
    marked[j] = true;  // mark node j as visited

    for (csint p = A.p_[j]; p < A.p_[j+1]; p++) {
        csint i = A.i_[p];  // consider neighbor node i
        if (!marked[i]) {
            dfs_r(A, i, marked, xi);  // dfs recursively from i
        }
    }

    xi.push_back(j);  // push unvisited neighbor onto stack

    return xi;
}

}  // namespace detail

// -----------------------------------------------------------------------------
//         Cholesky Factorization Solvers
// -----------------------------------------------------------------------------
void CholResult::solve_inplace(std::span<double> b) const
{
    // Solve Ax = b ==> (P^T L L^T P) x = b
    std::vector<double> w(L.shape()[0]);  // workspace

    ipvec<double>(p_inv, b, w);  // permute b -> w = Pb
    lsolve_inplace(L, w);    // y = L \ b -> w = y
    ltsolve_inplace(L, w);   // P^T x = L^T \ y -> w = P^T x
    pvec<double>(p_inv, w, b);   // x = P P^T x
}


std::vector<csint> topological_order(
    const CSCMatrix& b,
    const std::vector<csint>& parent,
    bool forward
)
{
    assert(b.N_ == 1);
    csint N = b.M_;

    std::vector<char> marked(N, false);
    std::vector<csint> s, xi;
    s.reserve(N);
    xi.reserve(N);

    // Search up the tree for each non-zero in b
    for (csint p = b.p_[0]; p < b.p_[1]; p++) {
        csint i = b.i_[p];

        // Traverse up the elimination tree
        while (i != -1 && !marked[i]) {
            s.push_back(i);
            marked[i] = true;
            i = parent[i];
        }

        // Push pash onto output stack
        while (!s.empty()) {
            xi.push_back(s.back());
            s.pop_back();
        }
    }

    if (forward) {
        // Reverse the order of the stack to get the topological order
        std::reverse(xi.begin(), xi.end());
    }

    return xi;
}


void CholResult::lsolve_inplace_(
    std::span<const csint> xi,
    std::span<double> x
) const
{
    for (const auto& j : xi) {
        double& x_val = x[j];  // cache diagonal value
        x_val /= L.v_[L.p_[j]];
        for (csint p = L.p_[j] + 1; p < L.p_[j+1]; p++) {
            x[L.i_[p]] -= L.v_[p] * x_val;
        }
    }
}


void CholResult::ltsolve_inplace_(
    std::span<const csint> xi,
    std::span<double> x
) const
{
    for (const auto& j : xi) {
        double& x_val = x[j];  // cache diagonal value
        for (csint p = L.p_[j] + 1; p < L.p_[j+1]; p++) {
            x_val -= L.v_[p] * x[L.i_[p]];
        }
        x_val /= L.v_[L.p_[j]];
    }
}


void CholResult::solve_inplace(
    const CSCMatrix& B,
    csint k,
    const std::vector<csint> parent,
    std::span<double> x
) const
{
    // ----- Option 1: scatter into dense column and solve dense
    // // Solve Ax = b ==> (P^T L L^T P) x = b
    // std::vector<double> w(L.M_);  // workspace

    // // scatter permuted b -> w = Pb
    // assert(k < B.N_);
    // for (csint p = B.p_[k]; p < B.p_[k+1]; p++) {
    //     w[p_inv[B.i_[p]]] = B.v_[p];
    // }

    // lsolve_inplace(L, w);    // y = L \ b -> w = y
    // ltsolve_inplace(L, w);   // P^T x = L^T \ y -> w = P^T x
    // pvec<double>(p_inv, w, x);   // x = P P^T x

    // ----- Option 2: Solve as sparse column and scatter to dense at end
    CSCMatrix b = B.slice(0, B.M_, k, k+1);  // get single column

    // Permute the rows in-place
    for (auto& i : b.i_) {
        i = p_inv[i];
    }

    // Get the order of the nodes from the elimination tree
    std::vector<csint> xi = topological_order(b, parent);

    // Scatter into w
    std::vector<double> w(L.M_);  // workspace
    for (csint p = b.p_[0]; p < b.p_[1]; p++) {
        w[b.i_[p]] = b.v_[p];
    }

    lsolve_inplace_(xi, w);               // y = L \ b -> w = y
    std::reverse(xi.begin(), xi.end());  // reverse the order for L^T
    ltsolve_inplace_(xi, w);              // P^T x = L^T \ y -> w = P^T x
    pvec<double>(p_inv, w, x);           // x = P P^T x
}


// Exercise 4.3/4.4
template <bool IsTranspose>
SparseSolution CholResult::lsolve_impl_(
    const CSCMatrix& b,
    std::vector<csint> parent
) const
{
    csint N = L.N_;
    std::vector<double> x(N);

    // Scatter b into x
    assert(b.N_ == 1);
    for (csint p = b.p_[0]; p < b.p_[1]; p++) {
        x[b.i_[p]] = b.v_[p];
    }

    if (parent.empty()) {
        // Inspect L to get the parent vector, since it has sorted indices
        assert(L.has_sorted_indices());
        parent.assign(N, -1);
        for (csint j = 0; j < N-1; j++) {  // skip the last row (only diagonal)
            parent[j] = L.i_[L.p_[j]+1];   // first off-diagonal element
        }
    }

    std::vector<csint> xi;  // order of nodes from the elimination tree

    // Solve Lx = b or L^T x = b
    if constexpr (IsTranspose) {
        xi = topological_order(b, parent, false);
        ltsolve_inplace_(xi, x);
    } else {
        xi = topological_order(b, parent, true);
        lsolve_inplace_(xi, x);
    }

    return {xi, x};
}


// Exercise 4.3
SparseSolution CholResult::lsolve(
    const CSCMatrix& b,
    std::vector<csint> parent
) const
{
    return lsolve_impl_<false>(b, parent);
}


// Exercise 4.4
SparseSolution CholResult::ltsolve(
    const CSCMatrix& b,
    std::vector<csint> parent
) const
{
    return lsolve_impl_<true>(b, parent);
}


std::vector<double> chol_solve(
    const CSCMatrix& A,
    const std::vector<double>& B,
    AMDOrder order
)
{
    auto [M, N] = A.shape();
    csint MxK = static_cast<csint>(B.size());

    if (M != N) {
        throw std::runtime_error("Matrix must be square!");
    }

    if (MxK % M != 0) {
        throw std::runtime_error("RHS vector size is not a multiple of matrix rows!");
    }

    csint K = MxK / M;  // number of RHS columns

    // Factorize the matrix once
    SymbolicChol S = schol(A, order);
    CholResult res = chol(A, S);

    std::vector<double> X = B;  // solution matrix
    std::span<double> X_span(X);

    // Exercise 8.7/8.9: solve each column of the system
    for (csint k = 0; k < K; k++) {
        auto X_k = X_span.subspan(k * N, N);
        res.solve_inplace(X_k);
    }

    return X;
}


// Exercise 8.8/8.9
std::vector<double> chol_solve(
    const CSCMatrix& A,
    const CSCMatrix& B,
    AMDOrder order
)
{
    auto [M, N] = A.shape();
    auto [Mb, K] = B.shape();

    if (M != N) {
        throw std::runtime_error("Matrix must be square!");
    }

    if (M != Mb) {
        throw std::runtime_error(
            std::format("Matrix and RHS sizes do not match! Got {} and {}.", M, Mb)
        );
    }

    // Factorize the matrix once
    SymbolicChol S = schol(A, order);
    CholResult res = chol(A, S);

    std::vector<double> X(N * K);  // dense solution matrix
    std::span<double> X_span(X);

    // Solve each column of the system
    for (csint k = 0; k < K; k++) {
        auto X_k = X_span.subspan(k * N, N);
        res.solve_inplace(B, k, S.parent, X_k);
    }

    return X;
}


// -----------------------------------------------------------------------------
//         QR Factorization Solvers
// -----------------------------------------------------------------------------
void QRResult::solve_inplace(
    size_t M2,
    std::span<const double> b,
    std::span<double> x
) const
{
    // Solve P^T Q R E x = b
    std::vector<double> w(M2);
    ipvec<double>(p_inv, b, w);  // permute b -> E b -> w = Eb
    apply_qtleft(V, beta, w);    // y = Q^T E b -> w = y
    usolve_inplace(R, w);        // E x = R \ y -> w = E x
    ipvec<double>(q, w, x);      // x = E^T (E x)
}


void QRResult::tsolve_inplace(
    size_t M2,
    std::span<const double> b,
    std::span<double> x
) const
{
    // Solve P^T R^T Q^T E x = b
    std::vector<double> w(M2);
    pvec<double>(q, b, w);      // permute b -> E b -> w = Eb
    utsolve_inplace(R, w);      // y = R^T \ E b -> w = y
    apply_qleft(V, beta, w);    // P x = Q y -> w = P x
    pvec<double>(p_inv, w, x);  // x = P^T (P x)
}


QRSolveResult qr_solve(
    const CSCMatrix& A,
    const std::vector<double>& B,
    AMDOrder order
)
{
    auto [M, N] = A.shape();
    csint MxK = static_cast<csint>(B.size());

    if (MxK % M != 0) {
        throw std::runtime_error("RHS vector size is not a multiple of matrix rows!");
    }

    csint K = MxK / M;  // number of RHS columns

    // Factorize the matrix once
    SymbolicQR S;
    QRResult res;

    if (M >= N) {
        S = sqr(A, order);
        res = qr(A, S);
    } else {
        CSCMatrix AT = A.transpose();
        S = sqr(AT, order);
        res = qr(AT, S);
    }

    std::vector<double> X(N * K);  // solution matrix
    std::span<double> X_span(X);
    std::span<const double> B_span(B);

    for (csint k = 0; k < K; k++) {
        // Solve for each RHS column
        auto B_k = B_span.subspan(k * M, M);
        auto X_k = X_span.subspan(k * N, N);

        if (M >= N) {
            // Compute the least-squares solution
            res.solve_inplace(S.m2, B_k, X_k);
        } else {
            // Compute the minimum-norm solution
            res.tsolve_inplace(S.m2, B_k, X_k);
        }
    }

    // Compute the residual
    std::vector<double> R = B - A * X;

    return QRSolveResult{X, R, norm(R, 2)};
}


// -----------------------------------------------------------------------------
//         LU Factorization Solvers
// -----------------------------------------------------------------------------
// Exercise 6.1
std::vector<double> lu_solve(
    const CSCMatrix& A,
    const std::vector<double>& B,
    AMDOrder order,
    double tol,
    csint ir_steps
)
{
    auto [M, N] = A.shape();
    csint MxK = static_cast<csint>(B.size());

    if (M != N) {
        throw std::runtime_error("Matrix must be square!");
    }

    if (MxK % M != 0) {
        throw std::runtime_error("RHS vector size is not a multiple of matrix rows!");
    }

    csint K = MxK / M;  // number of RHS columns

    const SymbolicLU S = slu(A, order);
    const LUResult res = lu(A, S, tol);

    std::vector<double> X = B;  // solution matrix
    std::span<const double> B_span(B);
    std::span<double> X_span(X);

    std::vector<double> r;

    if (ir_steps > 0) {
        // Preallocate workspace for iterative refinement
        r.resize(M);
    }

    // Solve for each RHS column
    for (csint k = 0; k < K; k++) {
        // Create a view into the k-th columns of B and X
        auto B_k = B_span.subspan(k * M, M);
        auto X_k = X_span.subspan(k * N, N);

        // Solve Ax = B
        res.solve_inplace(X_k);

        // Exercise 8.5: Iterative refinement
        for (csint i = 0; i < ir_steps; i++) {
            r = B_k - A * X_k;     // r = b - Ax
            res.solve_inplace(r);  // solve Ad = r
            X_k += r;              // x += d
        }
    }

    return X;
}


// Exercise 8.8
std::vector<double> lu_solve(
    const CSCMatrix& A,
    const CSCMatrix& B,
    AMDOrder order,
    double tol,
    csint ir_steps
)
{
    auto [M, N] = A.shape();
    auto [Mb, K] = B.shape();

    if (M != N) {
        throw std::runtime_error("Matrix must be square!");
    }

    if (M != Mb) {
        throw std::runtime_error(
            std::format("Matrix and RHS sizes do not match! Got {} and {}.", M, Mb)
        );
    }

    // Factor the matrix once
    const SymbolicLU S = slu(A, order);
    const LUResult res = lu(A, S, tol);

    std::vector<double> X(N * K);  // dense solution matrix
    std::span<double> X_span(X);

    std::vector<double> r, B_k;

    if (ir_steps > 0) {
        r.resize(M);   // Preallocate workspaces for iterative refinement
        B_k.resize(M);
    }

    // Solve for each RHS column
    for (csint k = 0; k < K; k++) {
        // Create a view into the k-th column of X
        auto X_k = X_span.subspan(k * N, N);
        std::fill(B_k.begin(), B_k.end(), 0.0);

        // Scatter B[:, k] into X_k (and B_k if necessary)
        for (csint p = B.p_[k]; p < B.p_[k+1]; p++) {
            const csint i = B.i_[p];
            const double v = B.v_[p];

            X_k[i] = v;

            if (ir_steps > 0) {
                B_k[i] = v;
            }
        }

        // Solve Ax = B
        res.solve_inplace(X_k);

        // Exercise 8.5: Iterative refinement
        for (csint i = 0; i < ir_steps; i++) {
            r = B_k - A * X_k;     // r = b - Ax
            res.solve_inplace(r);  // solve Ad = r
            X_k += r;              // x += d
        }
    }

    return X;
}


// Exercise 6.1
// TODO support matrix RHS inputs in lu_tsolve
std::vector<double> lu_tsolve(
    const CSCMatrix& A,
    const std::vector<double>& b,
    AMDOrder order,
    double tol
)
{
    if (A.shape()[1] != static_cast<csint>(b.size())) {
        throw std::runtime_error("Matrix and RHS vector sizes do not match!");
    }

    // Compute the numeric factorization
    SymbolicLU S = slu(A, order);
    LUResult res = lu(A, S, tol);

    return res.tsolve(b);
}


/** Find the minimum index of all those where |x| == max(|x|).
 *
 * @param x  a vector of doubles
 *
 * @return j  the first index of the maximum absolute value
 */
static inline csint min_argmaxabs(const std::vector<double>& x)
{
    csint N = x.size();
    csint j = N;         // minimum index
    double max_val = 0;  // maximum absolute value

    for (csint i = N-1; i >= 0; i--) {
        double mval = std::fabs(x[i]);
        if (i < j && mval > max_val) {
            max_val = mval;
            j = i;
        }
    }

    return j;
}


// Exercise 6.15
double norm1est_inv(const LUResult& res)
{
    csint M = res.L.shape()[0];
    csint N = res.U.shape()[1];

    if (M != N) {
        throw std::runtime_error("Matrix must be square!");
    }

    double est = 0.0;
    std::vector<double> x(N, 1.0 / N);  // sum(x) == 1.0
    csint jold = -1;

    // Estimate the 1-norm
    for (csint k = 0; k < 5; k++) {
        if (k > 0) {
            // j is the first index where |x| == max(|x|) (infinity norm)
            csint j = min_argmaxabs(x);

            if (j == jold) {
                break;
            }

            // Set x to a unit vector in the j direction
            std::fill(x.begin(), x.end(), 0.0);
            x[j] = 1.0;
            jold = j;
        }

        // Solve Ax = x
        x = res.solve(x);

        double est_old = est;
        est = norm(x, 1);

        if (k > 0 && est <= est_old) {
            break;
        }

        // s elements are in {-1, 1}
        std::vector<double> s(N, 1.0);
        for (csint p = 0; p < N; p++) {
            if (x[p] < 0) {
                s[p] = -1.0;
            }
        }

        x = res.tsolve(s);
    }

    return est;
}


// Exercise 6.15
double cond1est(const CSCMatrix& A)
{
    auto [M, N] = A.shape();

    if (M != N) {
        throw std::runtime_error("Matrix must be square!");
    }

    if (A.nnz() == 0) {
        return 0.0;
    }

    // Compute the LU factorization
    SymbolicLU S = slu(A);
    LUResult res = lu(A, S);

    // Compute the 1-norm estimate
    for (csint i = 0; i < N; i++) {
        if (res.U(i, i) == 0) {
            return std::numeric_limits<double>::infinity();
        }
    }

    // κ = |A|_1 * |A^{-1}|_1
    return A.norm() * norm1est_inv(res);
}


// Exercise 8.1
std::vector<double> spsolve(
    const CSCMatrix& A,
    const std::vector<double>& B
)
{
    auto [M, N] = A.shape();
    csint MxK = static_cast<csint>(B.size());

    if (MxK % M != 0) {
        throw std::runtime_error("RHS vector size is not a multiple of matrix rows!");
    }

    if (M != N) {
        // Use QR factorization for rectangular matrices
        return qr_solve(A, B, AMDOrder::ATA).x;
    }

    // For square matrices, go through the decision tree
    csint is_tri = A.is_triangular();

    // Check if diagonal is structurally non-zero
    int diag_sign = 0;   // -1: all neg, 0: mixed, 1: all pos
    csint nnz_diag = 0;  // number of non-zeros on diagonal
    bool first_seen = false;

    for (auto v : A.diagonal()) {
        if (v == 0) {
            continue;
        }

        nnz_diag++;  // count non-zeros
                     //
        if (!first_seen) {
            diag_sign = (v > 0) ? 1 : -1;
            first_seen = true;
        } else if (diag_sign != 0) {
            if ((v > 0 && diag_sign == -1) || (v < 0 && diag_sign == 1)) {
                diag_sign = 0;  // mixed signs
            }
        }
    }

    // If triangular with non-zero diagonal, use triangular solve
    if (nnz_diag == N) {
        if (is_tri == -1) {
            return lsolve_opt(A, B);
        } else if (is_tri == 1) {
            return usolve_opt(A, B);
        }
    }

    // Matrix may be permuted triangular
    try {
        return tri_solve_perm(A, B);
    } catch (const PermutedTriangularMatrixError&) {
        // do nothing
    }

    // Cholesky factorization if symmetric positive definite
    if (A.is_symmetric() && diag_sign) {
        try {
            if (diag_sign == 1) {
                return chol_solve(A, B, AMDOrder::APlusAT);
            } else {  // diag_sign == -1
                return chol_solve(-A, -B, AMDOrder::APlusAT);
            }
        } catch (const CholeskyNotPositiveDefiniteError& e) {
            // do nothing
        }
    }

    // General LU Solver
    double sym = A.structural_symmetry();
    double diag_dens = static_cast<double>(nnz_diag) / N;

    // These thresholds are set in SuiteSparse/UMFPACK/Include/umfpack.h:311:
    //     // added for v6.0.0.  Default changed fro 0.5 to 0.3
    //     #define UMFPACK_DEFAULT_STRATEGY_THRESH_SYM 0.3         /* was 0.5 */
    //     #define UMFPACK_DEFAULT_STRATEGY_THRESH_NNZDIAG 0.9
    //
    double tsym = 0.3;
    double tnzd = 0.9;

    // These notes are from umfpack.h:602:
    //
    // UMFPACK_STRATEGY_UNSYMMETRIC:  Use the unsymmetric strategy.  COLAMD
    //     is used to order the columns of A, followed by a postorder of
    //     the column elimination tree.  No attempt is made to perform
    //     diagonal pivoting.  The column ordering is refined during
    //     factorization.
    //
    //     In the numerical factorization, the
    //     Control [UMFPACK_SYM_PIVOT_TOLERANCE] parameter is ignored.  A
    //     pivot is selected if its magnitude is >=
    //     Control [UMFPACK_PIVOT_TOLERANCE] (default 0.1) times the
    //     largest entry in its column.
    //
    // UMFPACK_STRATEGY_SYMMETRIC:  Use the symmetric strategy
    //     In this method, the approximate minimum degree
    //     ordering (AMD) is applied to A+A', followed by a postorder of
    //     the elimination tree of A+A'.  UMFPACK attempts to perform
    //     diagonal pivoting during numerical factorization.  No refinement
    //     of the column pre-ordering is performed during factorization.
    //
    //     In the numerical factorization, a nonzero entry on the diagonal
    //     is selected as the pivot if its magnitude is >= Control
    //     [UMFPACK_SYM_PIVOT_TOLERANCE] (default 0.001) times the largest
    //     entry in its column.  If this is not acceptable, then an
    //     off-diagonal pivot is selected with magnitude >= Control
    //     [UMFPACK_PIVOT_TOLERANCE] (default 0.1) times the largest entry
    //     in its column.
    //
    //  C++Sparse only allows a single pivot tolerance, so our symmetric
    //  strategy does not perform the secondary check on off-diagonal pivots.

    AMDOrder order;
    double tol;

    if ((sym >= tsym) && (diag_dens >= tnzd)) {
        // symmetric strategy
        order = AMDOrder::APlusAT;
        tol = 0.001;
    } else {
        // non-symmetric strategy
        order = AMDOrder::ATANoDenseRows;
        tol = 0.1;
    }

    csint ir_steps = 2;  // number of iterative refinement steps

    return lu_solve(A, B, order, tol, ir_steps);
}


// TODO update for multiple RHS
std::vector<double> spsolve(const CSCMatrix& A, const CSCMatrix& b)
{
    if (b.shape()[1] != 1) {
        throw std::runtime_error(
            std::format(
                "Sparse RHS matrix must have exactly one column!"
                " Got shape ({}, {})",
                b.shape()[0], b.shape()[1]
            )
        );
    }

    return spsolve(A, b.to_dense_vector());
}


}  // namespace cs

/*==============================================================================
 *============================================================================*/
