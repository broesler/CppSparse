/*==============================================================================
 *     File: decomposition.cpp
 *  Created: 2025-01-27 13:14
 *   Author: Bernie Roesler
 *
 *  Description: Implements the symbolic factorization for a sparse matrix.
 *
 *============================================================================*/

#include <cassert>
#include <cmath>    // std::sqrt
#include <numeric>  // std::iota

#include "decomposition.h"
#include "utils.h"

namespace cs {

/*------------------------------------------------------------------------------
 *         Cholesky Decomposition
 *----------------------------------------------------------------------------*/
std::vector<csint> etree(const CSCMatrix& A, bool ata)
{
    std::vector<csint> parent(A.N_, -1);  // parent of i is parent[i]
    std::vector<csint> ancestor(A.N_, -1);  // workspaces
    std::vector<csint> prev = ata ? std::vector<csint>(A.M_, -1) : std::vector<csint>();

    for (csint k = 0; k < A.N_; k++) {
        for (csint p = A.p_[k]; p < A.p_[k+1]; p++) {
            csint i = ata ? prev[A.i_[p]] : A.i_[p];  // A(i, k) is nonzero
            while (i != -1 && i < k) {      // only use upper triangular of A
                csint inext = ancestor[i];  // traverse up to the root
                ancestor[i] = k;            // path compression
                if (inext == -1) {
                    parent[i] = k;          // no ancestor
                }
                i = inext;
            }
            if (ata) {
                prev[A.i_[p]] = k;  // use prev for A^T
            }
        }
    }

    return parent;
}


std::vector<csint> ereach(
    const CSCMatrix& A,
    csint k,
    const std::vector<csint>& parent
)
{
    std::vector<bool> marked(A.N_, false);  // workspace
    std::vector<csint> s, xi;  // internal dfs stack, output stack
    s.reserve(A.N_);
    xi.reserve(A.N_);

    marked[k] = true;  // mark node k as visited

    for (csint p = A.p_[k]; p < A.p_[k+1]; p++) {
        csint i = A.i_[p];  // A(i, k) is nonzero
        if (i <= k) {     // only consider upper triangular part of A
            // Traverse up the etree
            while (!marked[i]) {
                s.push_back(i);  // L(k, i) is nonzero
                marked[i] = true;  // mark i as visited
                i = parent[i]; 
            }

            // Push path onto output stack
            while (!s.empty()) {
                xi.push_back(s.back());
                s.pop_back();
            }
        }
    }

    // Reverse the stack to get the topological order
    return std::vector<csint>(xi.rbegin(), xi.rend());
}


std::vector<csint> post(const std::vector<csint>& parent)
{
    const csint N = parent.size();

    std::vector<csint> postorder;  // postorder of elimination tree
    postorder.reserve(N);

    // Linked list representation of the children of each node in ascending
    // order of node number. 
    //   head[i] is the first child of node i. 
    //   next[j] is the next child of node j.
    std::vector<csint> head(N, -1);
    std::vector<csint> next(N);

    // Traverse nodes in reverse order
    for (csint j = N - 1; j >= 0; j--) {
        if (parent[j] != -1) {           // only operate on non-roots
            next[j] = head[parent[j]];   // add j to list of its parent
            head[parent[j]] = j;
        }
    }

    // Search from each root
    for (csint j = 0; j < N; j++) {
        if (parent[j] == -1) {  // only search from roots
            tdfs(j, head, next, postorder);
        }
    }

    assert(postorder.size() == N);
    postorder.shrink_to_fit();

    return postorder;
}


void tdfs(
    csint j,
    std::vector<csint>& head,
    const std::vector<csint>& next,
    std::vector<csint>& postorder
)
{
    std::vector<csint> stack;
    stack.push_back(j);              // place j on stack

    while (!stack.empty()) {
        csint p = stack.back();      // p = top of stack
        csint i = head[p];           // i = youngest child of p
        if (i == -1) {
            stack.pop_back();        // p has no unordered children left
            postorder.push_back(p);  // node p is the kth node in postorder
        } else {
            head[p] = next[i];       // remove i from children of p
            stack.push_back(i);      // start dfs on child node i
        }
    }
}


std::pair<std::vector<csint>, std::vector<csint>> firstdesc(
    const std::vector<csint>& parent,
    const std::vector<csint>& postorder
)
{
    assert(parent.size() == postorder.size());
    const csint N = parent.size();

    std::vector<csint> first(N, -1);
    std::vector<csint> level(N);

    for (csint k = 0; k < N; k++) {
        csint i = postorder[k];  // node i of etree is kth postordered node
        csint len = 0;      // traverse from i to root 
        csint r = i;
        while (r != -1 && first[r] == -1) {
            first[r] = k;
            r = parent[r];
            len++;
        }
        len += (r == -1) ? -1 : level[r];  // r is root of tree or end of path
        for (csint s = i; s != r; s = parent[s]) {
            level[s] = len--;
        }
    }

    return std::make_pair(first, level);
}


std::vector<csint> rowcnt(
    const CSCMatrix& A,
    const std::vector<csint>& parent,
    const std::vector<csint>& postorder
)
{
    assert(A.M_ == A.N_);
    assert(parent.size() == A.N_);
    assert(parent.size() == postorder.size());

    // Count of nonzeros in each row of L
    std::vector<csint> rowcount(A.N_, 1);   // count the diagonal to start

    std::vector<csint> ancestor(A.N_);  // every node is its own ancestor
    std::iota(ancestor.begin(), ancestor.end(), 0);

    std::vector<csint> maxfirst(A.N_, -1);  // max first[i] for nodes in subtree of i
    std::vector<csint> prevleaf(A.N_, -1);  // previous leaf of ith row subtree

    auto [first, level] = firstdesc(parent, postorder);

    for (const auto& j : postorder) {  // j is the kth node in postorder
        for (csint p = A.p_[j]; p < A.p_[j+1]; p++) {
            csint i = A.i_[p];  // A(i, j) is nonzero, consider ith row subtree
            auto [q, jleaf] = least_common_ancestor(i, j, first, maxfirst, prevleaf, ancestor);
            if (jleaf != LeafStatus::NotLeaf) {
                rowcount[i] += (level[j] - level[q]);
            }
        }

        // Merge j into the ancestor set containing j's parent
        if (parent[j] != -1) {
            ancestor[j] = parent[j];
        }
    }

    return rowcount;
}


std::pair<csint, LeafStatus> least_common_ancestor(
    csint i,
    csint j,
    const std::vector<csint>& first,
    std::vector<csint>& maxfirst,
    std::vector<csint>& prevleaf,
    std::vector<csint>& ancestor
)
{
    LeafStatus jleaf;

    // See "skeleton" function in Davis, p 48.
    if (i <= j || first[j] <= maxfirst[i]) {  // j is not a leaf
        return std::make_pair(-1, LeafStatus::NotLeaf);
    }

    maxfirst[i] = first[j];         // update max first[j] seen so far
    csint jprev = prevleaf[i];      // jprev is the previous leaf of i
    prevleaf[i] = j;                // j is now the previous leaf of i
    jleaf = (jprev == -1) ? LeafStatus::FirstLeaf : LeafStatus::SubsequentLeaf;

    if (jleaf == LeafStatus::FirstLeaf) {
        return std::make_pair(i, jleaf);  // if j is the first leaf, q = root of ith subtree
    }

    // Traverse up to the root of the subtree to find q
    csint q = jprev;
    while (q != ancestor[q]) {
        q = ancestor[q];
    }

    // Compress the path to the root
    csint s = jprev;
    while (s != q) {
        csint sparent = ancestor[s];  // path compression
        ancestor[s] = q;
        s = sparent;
    }

    return std::make_pair(q, jleaf);  // least common ancestor of j_prev and j
}


// TODO include option to count columns in ATA -> not needed until we implement
// symbolic QR decomposition.
std::vector<csint> counts(
    const CSCMatrix& A,
    const std::vector<csint>& parent,
    const std::vector<csint>& postorder
)
{
    std::vector<csint> delta(A.N_);  // allocate the result

    // Workspaces
    std::vector<csint> ancestor(A.N_),
                       maxfirst(A.N_, -1),  // max first[i] for nodes in subtree of i
                       prevleaf(A.N_, -1),  // previous leaf of ith row subtree
                       first(A.N_, -1);     // first descendant of each node in the tree

    // every node is its own ancestor
    std::iota(ancestor.begin(), ancestor.end(), 0);

    // Compute first descendent of each node in the tree
    for (csint k = 0; k < A.N_; k++) {
        csint j = postorder[k];  // node j of etree is kth postordered node
        delta[j] = (first[j] == -1) ? 1 : 0;  // delta[j] = 1 if j is a leaf
        while (j != -1 && first[j] == -1) {
            first[j] = k;   // first descendant of j
            j = parent[j];  // move up the etree
        }
    }

    // Operate on the transpose
    bool values = false;  // do not copy values in the transpose
    CSCMatrix AT = A.transpose(values);

    for (csint k = 0; k < A.N_; k++) {
        csint j = postorder[k];  // node j of etree is kth postordered node
        if (parent[j] != -1) {
            delta[parent[j]]--;  // j is not a root
        }

        for (csint p = AT.p_[j]; p < AT.p_[j+1]; p++) {
            csint i = AT.i_[p];  // AT(i, j) is nonzero
            auto [q, jleaf] = least_common_ancestor(i, j, first, maxfirst, prevleaf, ancestor);
            if (jleaf != LeafStatus::NotLeaf) {
                delta[j]++;  // A(i, j) is in skeleton
            }
            if (jleaf == LeafStatus::SubsequentLeaf) {
                delta[q]--;  // account for overlap in q
            }
        }

        // Merge j into the ancestor set containing j's parent
        if (parent[j] != -1) {
            ancestor[j] = parent[j];
        }
    }

    // sum up the counts for each child
    for (csint j = 0; j < A.N_; j++) {
        if (parent[j] != -1) {
            delta[parent[j]] += delta[j];
        }
    }

    return delta;
}


std::vector<csint> chol_rowcounts(const CSCMatrix& A)
{
    // Compute the elimination tree of A
    std::vector<csint> parent = etree(A);

    // Compute the post-order of the elimination tree
    std::vector<csint> postorder = post(parent);

    // Count the number of non-zeros in each row of L
    return rowcnt(A, parent, postorder);
}


std::vector<csint> chol_colcounts(const CSCMatrix& A)
{
    // Compute the elimination tree of A
    std::vector<csint> parent = etree(A);

    // Compute the post-order of the elimination tree
    std::vector<csint> postorder = post(parent);

    // Count the number of non-zeros in each column of L
    return counts(A, parent, postorder);
}


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
    S.parent = etree(C);                     // compute the elimination tree
    auto postorder = post(S.parent);          // postorder the elimination tree
    auto c = counts(C, S.parent, postorder);   // find column counts of L
    S.cp = cumsum(c);                         // find column pointers for L
    S.lnz = S.unz = S.cp.back();              // number of non-zeros in L and U

    return S;
}


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
        const std::vector<csint> s = ereach(C, k, S.parent);  // pattern of L(k, :)
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


// Exercise 4.1 O(|L|)-time elimination tree and row/column counts
// Use ereach to compute the elimination tree one node at a time (pp 43--44)
// TODO make type for the output?
std::tuple<std::vector<csint>, std::vector<csint>, std::vector<csint>> 
    chol_etree_counts(const CSCMatrix& A)
{
    assert(A.M_ == A.N_);
    csint N = A.N_;
    std::vector<csint> parent(N, -1);
    std::vector<csint> row_counts(N, 1);  // count diagonals
    std::vector<csint> col_counts(N, 1);

    // Note that we "mark" nodes by setting the flag to the column index k, so
    // we don't have to reset a bool array each time.
    std::vector<csint> flag(N, -1);  // workspace

    for (csint k = 0; k < N; k++) {
        flag[k] = k;  // mark node k as visited
        // Compute T_k from T_{k-1} by finding the children of node k
        for (csint p = A.p_[k]; p < A.p_[k+1]; p++) {
            // Follow path from node i to the root of the etree, or flagged node
            for (csint i = A.i_[p]; flag[i] != k && i < k; i = parent[i]) {
                if (parent[i] == -1) {
                    parent[i] = k;   // the parent of i must be k
                }
                row_counts[k]++;  // A[i, k] != 0 => L[k, i] != 0
                col_counts[i]++;
                flag[i] = k;  // mark node k as visited
            }
        }
    }

    return std::make_tuple(parent, row_counts, col_counts);
}

} // namespace cs

/*==============================================================================
 *============================================================================*/
