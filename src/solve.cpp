/*==============================================================================
 *     File: solve.cpp
 *  Created: 2025-01-30 13:52
 *   Author: Bernie Roesler
 *
 *  Description: Implementations of various matrix solvers.
 *
 *============================================================================*/

#include <cassert>
#include <ranges>  // for std::views::reverse

#include "solve.h"
#include "csc.h"
#include "utils.h"
#include "lu.h"

namespace cs {

/*------------------------------------------------------------------------------
 *      Triangular Matrix Solutions 
 *----------------------------------------------------------------------------*/
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


std::vector<double> usolve(const CSCMatrix& L, const std::vector<double>& b)
{
    assert(L.M_ == L.N_);
    assert(L.M_ == b.size());

    std::vector<double> x = b;

    for (csint j = L.N_ - 1; j >= 0; j--) {
        x[j] /= L.v_[L.p_[j+1] - 1];  // diagonal entry
        for (csint p = L.p_[j]; p < L.p_[j+1] - 1; p++) {
            x[L.i_[p]] -= L.v_[p] * x[j];
        }
    }

    return x;
}


std::vector<double> utsolve(const CSCMatrix& L, const std::vector<double>& b)
{
    assert(L.M_ == L.N_);
    assert(L.M_ == b.size());

    std::vector<double> x = b;

    for (csint j = 0; j < L.N_; j++) {
        for (csint p = L.p_[j]; p < L.p_[j+1] - 1; p++) {
            x[j] -= L.v_[p] * x[L.i_[p]];
        }
        x[j] /= L.v_[L.p_[j+1] - 1];  // diagonal entry
    }

    return x;
}


std::vector<double> lsolve_opt(const CSCMatrix& L, const std::vector<double>& b)
{
    assert(L.M_ == L.N_);
    assert(L.M_ == b.size());

    std::vector<double> x = b;

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

    return x;
}


std::vector<double> usolve_opt(const CSCMatrix& L, const std::vector<double>& b)
{
    assert(L.M_ == L.N_);
    assert(L.M_ == b.size());

    std::vector<double> x = b;

    for (csint j = L.N_ - 1; j >= 0; j--) {
        double& x_val = x[j];  // cache reference to value
        if (x_val != 0) {
            x_val /= L.v_[L.p_[j+1] - 1];  // diagonal entry
            for (csint p = L.p_[j]; p < L.p_[j+1] - 1; p++) {
                x[L.i_[p]] -= L.v_[p] * x_val;
            }
        }
    }

    return x;
}


std::vector<csint> find_lower_diagonals(const CSCMatrix& L)
{
    assert(L.M_ == L.N_);

    std::vector<bool> marked(L.N_, false);  // workspace
    std::vector<csint> p_diags(L.N_);  // diagonal indicies (inverse permutation)

    for (csint j = L.N_ - 1; j >= 0; j--) {
        csint N_unmarked = 0;

        for (csint p = L.p_[j]; p < L.p_[j+1]; p++) {
            csint i = L.i_[p];
            // Mark the rows viewed so far
            if (!marked[i]) {
                marked[i] = true;
                p_diags[j] = p;
                N_unmarked++;
            }
        }

        // If 0 or > 1 "diagonal" entries found, the matrix is not permuted.
        if (N_unmarked != 1) {
            throw std::runtime_error("Matrix is not a permuted lower triangular matrix!");
        }
    }

    return p_diags;
}


std::vector<double> lsolve_rows(const CSCMatrix& L, const std::vector<double>& b)
{
    assert(L.M_ == L.N_);
    assert(L.M_ == b.size());

    // First (backward) pass to find diagonal entries
    // p_diags is a vector of pointers to the diagonal entries
    std::vector<csint> p_diags = find_lower_diagonals(L);

    // Compute the row permutation vector
    std::vector<csint> permuted_rows(L.N_);
    for (csint i = 0; i < L.N_; i++) {
        permuted_rows[L.i_[p_diags[i]]] = i;
    }

    // Second (forward) pass to solve the system
    std::vector<double> x = b;

    // Perform the permuted forward solve
    for (csint j = 0; j < L.N_; j++) {
        csint d = p_diags[j];  // pointer to the diagonal entry
        double& x_val = x[j];  // cache diagonal value

        if (x_val != 0) {
            x_val /= L.v_[d];    // solve for x[d]
            for (csint p = L.p_[j]; p < L.p_[j+1]; p++) {
                csint i = permuted_rows[L.i_[p]];
                if (p != d) {
                    x[i] -= L.v_[p] * x_val;  // update the off-diagonals
                }
            }
        }
    }

    return x;
}


std::vector<double> lsolve_cols(const CSCMatrix& L, const std::vector<double>& b)
{
    assert(L.M_ == L.N_);
    assert(L.M_ == b.size());

    // First O(N) pass to find the diagonal entries
    // Assume that the first entry in each column has the smallest row index
    std::vector<csint> p_diags(L.N_, -1);
    for (csint j = 0; j < L.N_; j++) {
        if (p_diags[j] == -1) {
            p_diags[j] = L.p_[j];  // pointer to the diagonal entry
        } else {
            // We have seen this column index before
            throw std::runtime_error("Matrix is not a permuted lower triangular matrix!");
        }
    }

    // Compute the column permutation vector
    std::vector<csint> permuted_cols(L.N_);
    for (csint i = 0; i < L.N_; i++) {
        permuted_cols[L.i_[p_diags[i]]] = i;
    }

    // Second (forward) pass to solve the system
    std::vector<double> x = b;

    // Perform the permuted forward solve
    for (const auto& j : permuted_cols) {
        csint d = p_diags[j];      // pointer to the diagonal entry
        double& x_val = x[L.i_[d]];  // cache diagonal value

        if (x_val != 0) {
            x_val /= L.v_[d];  // solve for x[L.i_[d]]
            for (csint p = L.p_[j]+1; p < L.p_[j+1]; p++) {
                x[L.i_[p]] -= L.v_[p] * x_val;  // update the off-diagonals
            }
        }
    }

    return x;
}


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
            throw std::runtime_error("Matrix is not a permuted upper triangular matrix!");
        }
    }

    return p_diags;
}


std::vector<double> usolve_rows(const CSCMatrix& U, const std::vector<double>& b)
{
    assert(U.M_ == U.N_);
    assert(U.M_ == b.size());

    // First (backward) pass to find diagonal entries
    // p_diags is a vector of pointers to the diagonal entries
    std::vector<csint> p_diags = find_upper_diagonals(U);

    // Compute the row permutation vector
    std::vector<csint> permuted_rows(U.N_);
    for (csint i = 0; i < U.N_; i++) {
        permuted_rows[U.i_[p_diags[i]]] = i;
    }

    // Second (forward) pass to solve the system
    std::vector<double> x = b;

    // Perform the permuted forward solve
    for (csint j = U.N_ - 1; j >= 0; j--) {
        csint d = p_diags[j];  // pointer to the diagonal entry
        double& x_val = x[j];  // cache diagonal value

        if (x_val != 0) {
            x_val /= U.v_[d];    // solve for x[d]
            for (csint p = U.p_[j]; p < U.p_[j+1]; p++) {
                csint i = permuted_rows[U.i_[p]];
                if (p != d) {
                    x[i] -= U.v_[p] * x_val;  // update the off-diagonals
                }
            }
        }
    }

    return x;
}


std::vector<double> usolve_cols(const CSCMatrix& U, const std::vector<double>& b)
{
    assert(U.M_ == U.N_);
    assert(U.M_ == b.size());

    // First O(N) pass to find the diagonal entries
    // Assume that the last entry in each column has the largest row index
    std::vector<csint> p_diags(U.N_, -1);
    for (csint j = 0; j < U.N_; j++) {
        if (p_diags[j] == -1) {
            p_diags[j] = U.p_[j+1] - 1;  // pointer to the diagonal entry
        } else {
            // We have seen this column index before
            throw std::runtime_error("Matrix is not a permuted lower triangular matrix!");
        }
    }

    // Compute the column permutation vector
    std::vector<csint> permuted_cols(U.N_);
    for (csint i = 0; i < U.N_; i++) {
        permuted_cols[U.i_[p_diags[i]]] = i;
    }

    // Second (forward) pass to solve the system
    std::vector<double> x = b;

    // Perform the permuted forward solve
    for (const auto& j : std::views::reverse(permuted_cols)) {
        csint d = p_diags[j];      // pointer to the diagonal entry
        double& x_val = x[U.i_[d]];  // cache diagonal value

        if (x_val != 0) {
            x_val /= U.v_[d];  // solve for x[U.i_[d]]
            for (csint p = U.p_[j]; p < U.p_[j+1] - 1; p++) {
                x[U.i_[p]] -= U.v_[p] * x_val;  // update the off-diagonals
            }
        }
    }

    return x;
}


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
            throw std::runtime_error("Matrix is not a permuted triangular matrix!");
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


std::vector<double> tri_solve_perm(
    const CSCMatrix& A, 
    const std::vector<double>& b,
    bool is_upper
)
{
    assert(A.M_ == A.N_);
    assert(A.M_ == b.size());

    // Get the permutation vectors
    // NOTE If upper triangular, the permutation vectors are reversed
    auto [p_inv, q_inv, p_diags] = find_tri_permutation(A);

    // Get the non-inverse row-permutation vector O(N)
    std::vector<csint> p = inv_permute(p_inv);

    // Copy the RHS vector
    std::vector<double> x =
        (is_upper) ? std::vector<double>(b.rbegin(), b.rend()) : b;

    // Solve the system
    for (csint k = 0; k < A.N_; k++) {
        csint j = q_inv[k];    // permuted column
        csint d = p_diags[k];  // pointer to the diagonal entry

        // Update the solution
        double& x_val = x[k];  // diagonal of un-permuted row of x
        if (x_val != 0) {
            x_val /= A.v_[d];  // diagonal entry
            for (csint t = A.p_[j]; t < A.p_[j+1]; t++) {
                // off-diagonals from un-permuted row
                if (t != d) {
                    x[p[A.i_[t]]] -= A.v_[t] * x_val;
                }
            }
        }
    }

    if (is_upper) {
        std::reverse(x.begin(), x.end());
    }

    return x;
}


SparseSolution spsolve(
    const CSCMatrix& A, 
    const CSCMatrix& B,
    csint k,
    std::optional<const std::vector<csint>> p_inv,
    bool lo
)
{
    // Populate xi with the non-zero indices of x
    std::vector<csint> xi = reach(A, B, k, p_inv);
    std::vector<double> x(A.M_);  // dense output vector

    // scatter B(:, k) into x
    for (csint p = B.p_[k]; p < B.p_[k+1]; p++) {
        x[B.i_[p]] = B.v_[p];
    }

    // Solve Lx = b_k or Ux = b_k
    for (auto& j : xi) {  // x(j) is nonzero
        // j maps to col J of G
        csint J = p_inv.has_value() ? p_inv.value()[j] : j;
        if (J < 0) {
            continue;                                // x(j) is not in the pattern of G
        }
        x[j] /= A.v_[lo ? A.p_[J] : A.p_[J+1] - 1];  // x(j) /= G(j, j)
        csint p = lo ? A.p_[J] + 1 : A.p_[J];        // lo: L(j,j) 1st entry
        csint q = lo ? A.p_[J+1]   : A.p_[J+1] - 1;  // up: U(j,j) last entry
        for (; p < q; p++) {
            x[A.i_[p]] -= A.v_[p] * x[j];            // x[i] -= G(i, j) * x[j]
        }
    }

    return {xi, x};
}


std::vector<csint> reach(
    const CSCMatrix& A,
    const CSCMatrix& B,
    csint k,
    std::optional<const std::vector<csint>> p_inv
)
{
    std::vector<bool> marked(A.N_, false);
    std::vector<csint> xi;  // do not initialize for dfs call!
    xi.reserve(A.N_);

    for (csint p = B.p_[k]; p < B.p_[k+1]; p++) {
        csint j = B.i_[p];  // consider nonzero B(j, k)
        if (!marked[j]) {
            xi = dfs(A, j, marked, xi, p_inv);
        }
    }

    // xi is returned from dfs in reverse order, since it is a stack
    return std::vector<csint>(xi.rbegin(), xi.rend());
}


std::vector<csint>& dfs(
    const CSCMatrix& A, 
    csint j,
    std::vector<bool>& marked,
    std::vector<csint>& xi,
    std::optional<const std::vector<csint>> p_inv
)
{
    std::vector<csint> rstack, pstack;  // recursion and pause stacks
    rstack.reserve(A.N_);
    pstack.reserve(A.N_);

    rstack.push_back(j);       // initialize the recursion stack

    bool done = false;  // true if no unvisited neighbors

    while (!rstack.empty()) {
        j = rstack.back();  // get j from the top of the recursion stack
        // j maps to col jnew of G
        csint jnew = p_inv.has_value() ? p_inv.value()[j] : j;

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


// -----------------------------------------------------------------------------
//         Cholesky Factorization Solvers
// -----------------------------------------------------------------------------
// Exercise 4.3
SparseSolution chol_lsolve(
    const CSCMatrix& L,
    const CSCMatrix& b,
    std::vector<csint> parent
)
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

    // Get the order of the nodes from the elimination tree
    std::vector<csint> xi = topological_order(b, parent);

    // Solve Lx = b
    for (const auto& j : xi) {
        x[j] /= L.v_[L.p_[j]];
        for (csint p = L.p_[j] + 1; p < L.p_[j+1]; p++) {
            x[L.i_[p]] -= L.v_[p] * x[j];
        }
    }

    return {xi, x};
}



// Exercise 4.4
SparseSolution chol_ltsolve(
    const CSCMatrix& L,
    const CSCMatrix& b,
    std::vector<csint> parent
)
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

    // Get the order of the nodes from the elimination tree
    std::vector<csint> xi = topological_order(b, parent, false);

    // Solve Lx = b
    for (const auto& j : xi) {
        for (csint p = L.p_[j] + 1; p < L.p_[j+1]; p++) {
            x[j] -= L.v_[p] * x[L.i_[p]];
        }
        x[j] /= L.v_[L.p_[j]];
    }

    return {xi, x};
}


std::vector<csint> topological_order(
    const CSCMatrix& b,
    const std::vector<csint>& parent,
    bool forward
)
{
    assert(b.N_ == 1);
    csint N = b.M_;

    std::vector<bool> marked(N, false);
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

    assert(xi.size() == N);

    if (forward) {
        // Reverse the order of the stack to get the topological order
        return std::vector<csint>(xi.rbegin(), xi.rend());
    } else {
        return xi;
    }
}


// -----------------------------------------------------------------------------
//         LU Factorization Solvers
// -----------------------------------------------------------------------------
std::vector<double> lu_tsolve(const CSCMatrix& A, const std::vector<double>& b)
{
    if (A.shape()[0] != b.size()) {
        throw std::runtime_error("Matrix and RHS vector sizes do not match!");
    }

    // Compute the numeric factorization
    SymbolicLU S = slu(A);
    LUResult res = lu(A, S);

    // Solve A^T x = b == (P^T LU)^T x = b == U^T (L^T P x) = b
    const std::vector<double> y = utsolve(res.U, b);   // solve U^T y = b
    const std::vector<double> Px = ltsolve(res.L, y);  // solve L^T P x = y
    std::vector<double> x = pvec(res.p_inv, Px);       // permute back
    
    return x;
}


}  // namespace cs

/*==============================================================================
 *============================================================================*/
