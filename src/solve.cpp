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

namespace cs {

/*------------------------------------------------------------------------------
 *      Triangular Matrix Solutions 
 *----------------------------------------------------------------------------*/
std::vector<double> lsolve(const CSCMatrix& A, const std::vector<double>& b)
{
    assert(A.M_ == A.N_);
    assert(A.M_ == b.size());

    std::vector<double> x = b;

    for (csint j = 0; j < A.N_; j++) {
        x[j] /= A.v_[A.p_[j]];
        for (csint p = A.p_[j] + 1; p < A.p_[j+1]; p++) {
            x[A.i_[p]] -= A.v_[p] * x[j];
        }
    }

    return x;
}


std::vector<double> ltsolve(const CSCMatrix& A, const std::vector<double>& b)
{
    assert(A.M_ == A.N_);
    assert(A.M_ == b.size());

    std::vector<double> x = b;

    for (csint j = A.N_ - 1; j >= 0; j--) {
        for (csint p = A.p_[j] + 1; p < A.p_[j+1]; p++) {
            x[j] -= A.v_[p] * x[A.i_[p]];
        }
        x[j] /= A.v_[A.p_[j]];
    }

    return x;
}


std::vector<double> usolve(const CSCMatrix& A, const std::vector<double>& b)
{
    assert(A.M_ == A.N_);
    assert(A.M_ == b.size());

    std::vector<double> x = b;

    for (csint j = A.N_ - 1; j >= 0; j--) {
        x[j] /= A.v_[A.p_[j+1] - 1];  // diagonal entry
        for (csint p = A.p_[j]; p < A.p_[j+1] - 1; p++) {
            x[A.i_[p]] -= A.v_[p] * x[j];
        }
    }

    return x;
}


std::vector<double> utsolve(const CSCMatrix& A, const std::vector<double>& b)
{
    assert(A.M_ == A.N_);
    assert(A.M_ == b.size());

    std::vector<double> x = b;

    for (csint j = 0; j < A.N_; j++) {
        for (csint p = A.p_[j]; p < A.p_[j+1] - 1; p++) {
            x[j] -= A.v_[p] * x[A.i_[p]];
        }
        x[j] /= A.v_[A.p_[j+1] - 1];  // diagonal entry
    }

    return x;
}


std::vector<double> lsolve_opt(const CSCMatrix& A, const std::vector<double>& b)
{
    assert(A.M_ == A.N_);
    assert(A.M_ == b.size());

    std::vector<double> x = b;

    for (csint j = 0; j < A.N_; j++) {
        double& x_val = x[j];  // cache reference to value
        // Exercise 3.8: improve performance by checking for zeros
        if (x_val != 0) {
            x_val /= A.v_[A.p_[j]];
            for (csint p = A.p_[j] + 1; p < A.p_[j+1]; p++) {
                x[A.i_[p]] -= A.v_[p] * x_val;
            }
        }
    }

    return x;
}


std::vector<double> usolve_opt(const CSCMatrix& A, const std::vector<double>& b)
{
    assert(A.M_ == A.N_);
    assert(A.M_ == b.size());

    std::vector<double> x = b;

    for (csint j = A.N_ - 1; j >= 0; j--) {
        double& x_val = x[j];  // cache reference to value
        if (x_val != 0) {
            x_val /= A.v_[A.p_[j+1] - 1];  // diagonal entry
            for (csint p = A.p_[j]; p < A.p_[j+1] - 1; p++) {
                x[A.i_[p]] -= A.v_[p] * x_val;
            }
        }
    }

    return x;
}


std::vector<csint> find_lower_diagonals(const CSCMatrix& A)
{
    assert(A.M_ == A.N_);

    std::vector<bool> marked(A.N_, false);  // workspace
    std::vector<csint> p_diags(A.N_);  // diagonal indicies (inverse permutation)

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
            throw std::runtime_error("Matrix is not a permuted lower triangular matrix!");
        }
    }

    return p_diags;
}


std::vector<double> lsolve_rows(const CSCMatrix& A, const std::vector<double>& b)
{
    assert(A.M_ == A.N_);
    assert(A.M_ == b.size());

    // First (backward) pass to find diagonal entries
    // p_diags is a vector of pointers to the diagonal entries
    std::vector<csint> p_diags = find_lower_diagonals(A);

    // Compute the row permutation vector
    std::vector<csint> permuted_rows(A.N_);
    for (csint i = 0; i < A.N_; i++) {
        permuted_rows[A.i_[p_diags[i]]] = i;
    }

    // Second (forward) pass to solve the system
    std::vector<double> x = b;

    // Perform the permuted forward solve
    for (csint j = 0; j < A.N_; j++) {
        csint d = p_diags[j];  // pointer to the diagonal entry
        double& x_val = x[j];  // cache diagonal value

        if (x_val != 0) {
            x_val /= A.v_[d];    // solve for x[d]
            for (csint p = A.p_[j]; p < A.p_[j+1]; p++) {
                csint i = permuted_rows[A.i_[p]];
                if (p != d) {
                    x[i] -= A.v_[p] * x_val;  // update the off-diagonals
                }
            }
        }
    }

    return x;
}


std::vector<double> lsolve_cols(const CSCMatrix& A, const std::vector<double>& b)
{
    assert(A.M_ == A.N_);
    assert(A.M_ == b.size());

    // First O(N) pass to find the diagonal entries
    // Assume that the first entry in each column has the smallest row index
    std::vector<csint> p_diags(A.N_, -1);
    for (csint j = 0; j < A.N_; j++) {
        if (p_diags[j] == -1) {
            p_diags[j] = A.p_[j];  // pointer to the diagonal entry
        } else {
            // We have seen this column index before
            throw std::runtime_error("Matrix is not a permuted lower triangular matrix!");
        }
    }

    // Compute the column permutation vector
    std::vector<csint> permuted_cols(A.N_);
    for (csint i = 0; i < A.N_; i++) {
        permuted_cols[A.i_[p_diags[i]]] = i;
    }

    // Second (forward) pass to solve the system
    std::vector<double> x = b;

    // Perform the permuted forward solve
    for (const auto& j : permuted_cols) {
        csint d = p_diags[j];      // pointer to the diagonal entry
        double& x_val = x[A.i_[d]];  // cache diagonal value

        if (x_val != 0) {
            x_val /= A.v_[d];  // solve for x[A.i_[d]]
            for (csint p = A.p_[j]+1; p < A.p_[j+1]; p++) {
                x[A.i_[p]] -= A.v_[p] * x_val;  // update the off-diagonals
            }
        }
    }

    return x;
}


std::vector<csint> find_upper_diagonals(const CSCMatrix& A)
{
    assert(A.M_ == A.N_);

    std::vector<bool> marked(A.N_, false);  // workspace
    std::vector<csint> p_diags(A.N_);  // diagonal indicies (inverse permutation)

    for (csint j = 0; j < A.N_; j++) {
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
            throw std::runtime_error("Matrix is not a permuted upper triangular matrix!");
        }
    }

    return p_diags;
}


std::vector<double> usolve_rows(const CSCMatrix& A, const std::vector<double>& b)
{
    assert(A.M_ == A.N_);
    assert(A.M_ == b.size());

    // First (backward) pass to find diagonal entries
    // p_diags is a vector of pointers to the diagonal entries
    std::vector<csint> p_diags = find_upper_diagonals(A);

    // Compute the row permutation vector
    std::vector<csint> permuted_rows(A.N_);
    for (csint i = 0; i < A.N_; i++) {
        permuted_rows[A.i_[p_diags[i]]] = i;
    }

    // Second (forward) pass to solve the system
    std::vector<double> x = b;

    // Perform the permuted forward solve
    for (csint j = A.N_ - 1; j >= 0; j--) {
        csint d = p_diags[j];  // pointer to the diagonal entry
        double& x_val = x[j];  // cache diagonal value

        if (x_val != 0) {
            x_val /= A.v_[d];    // solve for x[d]
            for (csint p = A.p_[j]; p < A.p_[j+1]; p++) {
                csint i = permuted_rows[A.i_[p]];
                if (p != d) {
                    x[i] -= A.v_[p] * x_val;  // update the off-diagonals
                }
            }
        }
    }

    return x;
}


std::vector<double> usolve_cols(const CSCMatrix& A, const std::vector<double>& b)
{
    assert(A.M_ == A.N_);
    assert(A.M_ == b.size());

    // First O(N) pass to find the diagonal entries
    // Assume that the last entry in each column has the largest row index
    std::vector<csint> p_diags(A.N_, -1);
    for (csint j = 0; j < A.N_; j++) {
        if (p_diags[j] == -1) {
            p_diags[j] = A.p_[j+1] - 1;  // pointer to the diagonal entry
        } else {
            // We have seen this column index before
            throw std::runtime_error("Matrix is not a permuted lower triangular matrix!");
        }
    }

    // Compute the column permutation vector
    std::vector<csint> permuted_cols(A.N_);
    for (csint i = 0; i < A.N_; i++) {
        permuted_cols[A.i_[p_diags[i]]] = i;
    }

    // Second (forward) pass to solve the system
    std::vector<double> x = b;

    // Perform the permuted forward solve
    for (const auto& j : std::views::reverse(permuted_cols)) {
        csint d = p_diags[j];      // pointer to the diagonal entry
        double& x_val = x[A.i_[d]];  // cache diagonal value

        if (x_val != 0) {
            x_val /= A.v_[d];  // solve for x[A.i_[d]]
            for (csint p = A.p_[j]; p < A.p_[j+1] - 1; p++) {
                x[A.i_[p]] -= A.v_[p] * x_val;  // update the off-diagonals
            }
        }
    }

    return x;
}


std::tuple<std::vector<csint>, std::vector<csint>, std::vector<csint>>
find_tri_permutation(const CSCMatrix& A)
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

    return std::make_tuple(p_inv, q_inv, p_diags);
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


std::pair<std::vector<csint>, std::vector<double>> spsolve(
    const CSCMatrix& A, 
    const CSCMatrix& B,
    csint k,
    bool lo
)
{
    // Populate xi with the non-zero indices of x
    std::vector<csint> xi = reach(A, B, k);
    std::vector<double> x(A.N_);  // dense output vector

    // Clear non-zeros of x
    for (auto& i : xi) {
        x[i] = 0.0;
    }

    // scatter B(:, k) into x
    for (csint p = B.p_[k]; p < B.p_[k+1]; p++) {
        x[B.i_[p]] = B.v_[p];
    }

    // Solve Lx = b_k or Ux = b_k
    for (auto& j : xi) {  // x(j) is nonzero
        csint J = j;  // j maps to col J of G (NOTE ignore for now)
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

    return std::make_pair(xi, x);
}


std::vector<csint> reach(const CSCMatrix& A, const CSCMatrix& B, csint k)
{
    std::vector<bool> marked(A.N_, false);
    std::vector<csint> xi;  // do not initialize for dfs call!
    xi.reserve(A.N_);

    for (csint p = B.p_[k]; p < B.p_[k+1]; p++) {
        csint j = B.i_[p];  // consider nonzero B(j, k)
        if (!marked[j]) {
            xi = dfs(A, j, marked, xi);
        }
    }

    // xi is returned from dfs in reverse order, since it is a stack
    return std::vector<csint>(xi.rbegin(), xi.rend());
}


std::vector<csint>& dfs(
    const CSCMatrix& A, 
    csint j,
    std::vector<bool>& marked,
    std::vector<csint>& xi
)
{
    std::vector<csint> rstack, pstack;  // recursion and pause stacks
    rstack.reserve(A.N_);
    pstack.reserve(A.N_);

    rstack.push_back(j);       // initialize the recursion stack

    bool done = false;  // true if no unvisited neighbors

    while (!rstack.empty()) {
        j = rstack.back();  // get j from the top of the recursion stack
        csint jnew = j;  // j maps to col jnew of G (NOTE ignore p_inv for now)

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


}  // namespace cs

/*==============================================================================
 *============================================================================*/
