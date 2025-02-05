/*==============================================================================
 *     File: decomposition.cpp
 *  Created: 2025-01-27 13:14
 *   Author: Bernie Roesler
 *
 *  Description: Implements the symbolic factorization for a sparse matrix.
 *
 *============================================================================*/

#include <algorithm>  // std::copy
#include <cassert>
#include <cmath>      // std::sqrt
#include <iterator>   // std::back_inserter
#include <numeric>    // std::iota

#include "decomposition.h"
#include "csc.h"
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


// Exercise 4.6
csint etree_height(const std::vector<csint>& parent)
{
    assert(!parent.empty());

    csint height = 0;
    for (csint i = 0; i < parent.size(); i++) {
        csint h = 0;
        for (csint p = i; p != -1; p = parent[p]) {
            h++;
        }
        height = std::max(height, h);
    }

    return height;
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
            std::copy(s.rbegin(), s.rend(), std::back_inserter(xi));
            s.clear();
        }
    }

    // Reverse the stack to get the topological order
    return std::vector<csint>(xi.rbegin(), xi.rend());
}


std::vector<csint> ereach_post(
    const CSCMatrix& A,
    csint k,
    const std::vector<csint>& parent
)
{
    assert(A.has_sorted_indices_);

    std::vector<bool> marked(A.N_, false);  // workspace
    std::vector<csint> xi;  // internal dfs stack, output stack
    xi.reserve(A.N_);

    marked[k] = true;  // mark node k as visited

    for (csint p = A.p_[k]; p < A.p_[k+1]; p++) {
        csint i = A.i_[p];  // A(i, k) is nonzero
        csint i2 = A.i_[p+1];  // next row index
        if (i <= k) {     // only consider upper triangular part of A
            // Traverse up the etree i -> a = lca(i1, i2)
            while (i < i2 && i != -1 && !marked[i]) {
                xi.push_back(i);   // L(k, i) is nonzero
                marked[i] = true;  // mark i as visited
                i = parent[i]; 
            }
        }
    }

    return xi;
}


std::vector<csint> ereach_queue(
    const CSCMatrix& A,
    csint k,
    const std::vector<csint>& parent
)
{
    std::vector<bool> marked(A.N_, false);  // workspace
    std::vector<csint> s;  // internal dfs stack, output stack
    s.reserve(A.N_);

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
        }
    }

    return s;
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


FirstDesc firstdesc(
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

    return {first, level};
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


LCAStatus least_common_ancestor(
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
        return {-1, LeafStatus::NotLeaf};
    }

    maxfirst[i] = first[j];         // update max first[j] seen so far
    csint jprev = prevleaf[i];      // jprev is the previous leaf of i
    prevleaf[i] = j;                // j is now the previous leaf of i
    jleaf = (jprev == -1) ? LeafStatus::FirstLeaf : LeafStatus::SubsequentLeaf;

    if (jleaf == LeafStatus::FirstLeaf) {
        return {i, jleaf};  // if j is the first leaf, q = root of ith subtree
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

    return {q, jleaf};  // least common ancestor of j_prev and j
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


Symbolic schol(const CSCMatrix& A, AMDOrder order, bool use_postorder)
{
    Symbolic S;
    std::vector<csint> p(A.shape()[1]);  // the matrix permutation

    if (order == AMDOrder::Natural) {
        // TODO set to empty vector?
        // NOTE if we allow an empty p_inv here, we should support an empty
        // p or p_inv argument to all permute functions: pvec, ipvec,
        // inv_permute, permute, and symperm... and all of the upper/lower
        // triangular permuted solvers!!
        std::iota(p.begin(), p.end(), 0);  // identity permutation
        S.p_inv = p;  // identity is its own inverse
    } else {
        // TODO implement amd order (see Chapter 7)
        // p = amd(order, A);  // P = amd(A + A.T()) or natural
        // S.p_inv = inv_permute(p);
        throw std::runtime_error("Ordering method not implemented!");
    }

    // Find pattern of Cholesky factor
    CSCMatrix C = A.symperm(S.p_inv, false);  // C = spones(triu(A(p, p)))
    S.parent = etree(C);
    std::vector<csint> postorder = post(S.parent);

    // Exercise 4.9
    if (use_postorder) {
        p = pvec(postorder, p);         // combine the permutations
        S.p_inv = inv_permute(p);
        C = A.symperm(S.p_inv, false);  // apply combined permutation
        S.parent = etree(C);
        postorder = post(S.parent);     // should be identity for natural order
    }

    std::vector<csint> c = counts(C, S.parent, postorder);

    S.cp = cumsum(c);                         // find column pointers for L
    S.lnz = S.unz = S.cp.back();              // number of non-zeros in L and U

    return S;
}


CSCMatrix symbolic_cholesky(const CSCMatrix& A, const Symbolic& S)
{
    auto [M, N] = A.shape();
    CSCMatrix L(M, N, S.lnz);        // allocate result

    std::vector<csint> c(S.cp);      // column pointers for L

    const CSCMatrix C = A.symperm(S.p_inv);

    L.p_ = S.cp;  // column pointers for L

    // Compute L(:, k) for L*L' = C
    for (csint k = 0; k < N; k++) {
        // pattern of L(k, :) (order doesn't matter)
        for (const auto& j : ereach_queue(C, k, S.parent)) {
            L.i_[c[j]++] = k;  // store L(k, j) in column j
        }

        // Store the diagonal element
        L.i_[c[k]++] = k;
    }

    // Guaranteed by construction
    L.has_sorted_indices_ = true;
    L.has_canonical_format_ = true;

    return L;
}


CSCMatrix chol(const CSCMatrix& A, const Symbolic& S, double drop_tol)
{
    auto [M, N] = A.shape();
    CSCMatrix L(M, N, S.lnz);  // allocate result

    // Workspaces
    std::vector<csint> c(S.cp);  // column pointers for L
    std::vector<double> x(N);    // sparse accumulator

    const CSCMatrix C = A.symperm(S.p_inv);

    L.p_ = S.cp;  // column pointers for L

    // Compute L(k, :) for L*L' = C in up-looking order
    for (csint k = 0; k < N; k++) {
        //--- Nonzero pattern of L(k, :) ---------------------------------------
        x[k] = 0.0;  // x(0:k) is now zero

        // scatter C into x = full(triu(C(:,k)))
        // C does not have to be in sorted order (d = x[k] gets the diagonal)
        for (csint p = C.p_[k]; p < C.p_[k+1]; p++) {
            csint i = C.i_[p];
            if (i <= k) {
                x[i] = C.v_[p];
            }
        }

        double d = x[k];  // d = C(k, k)
        x[k] = 0.0;       // clear x for k + 1st iteration

        //--- Triangular Solve -------------------------------------------------
        // Solve L(0:k-1, 0:k-1) * x = C(0:k-1, k) == L[:k, :k] * x = C[:k, k]
        //   => L[k, :k] := x.T
        // ereach gives the pattern of L(k, :) in topological order
        for (const auto& i : ereach(C, k, S.parent)) {
            double lki = x[i] / L.v_[L.p_[i]];  // L(k, i) = x(i) / L(i, i)
            x[i] = 0.0;                         // clear x for k + 1st iteration

            for (csint p = L.p_[i] + 1; p < c[i]; p++) {
                x[L.i_[p]] -= L.v_[p] * lki;    // x -= L(i, :) * L(k, i)
            }

            // subtract the sparse dot product from the diagonal
            d -= lki * lki;                     // d -= L(k, i) * L(k, i)

            // We build L one *row* at a time, in topological order. All
            // i < k since they are reachable, so the diagonal is always the
            // first element in its column, and all other elements are in order.
            if (std::abs(lki) > drop_tol) {
            csint p = c[i]++;
            L.i_[p] = k;                        // store L(k, i) in column i
            L.v_[p] = lki;
        }
        }

        //--- Compute L(k, k) --------------------------------------------------
        if (d <= 0) {
            throw std::runtime_error("Matrix not positive definite!");
        }

        double sqrt_d = std::sqrt(d);
        if (std::abs(sqrt_d) > drop_tol) {
        csint p = c[k]++;
        L.i_[p] = k;  // store L(k, k) = sqrt(d) in column k
            L.v_[p] = sqrt_d;
        }
    }

    // Guaranteed by construction
    L.has_sorted_indices_ = true;
    L.has_canonical_format_ = true;  // L retains numerically 0 entries

    return L;
}


CSCMatrix& leftchol(const CSCMatrix& A, const Symbolic& S, CSCMatrix& L)
{
    // Ensure L has been allocated via symbolic_cholesky
    assert(!L.indptr().empty());
    assert(!L.indices().empty());
    assert(!L.data().empty());
    assert(L.has_sorted_indices_);

    csint N = A.shape()[1];

    // Workspaces
    std::vector<csint> c(S.cp);  // column pointers for L
    std::vector<double> x(N);    // sparse accumulator

    // Need the *lower* triangular part for a_{32}, so do a full permutation
    const CSCMatrix C = A.permute(S.p_inv, inv_permute(S.p_inv));

    L.p_ = S.cp;  // column pointers for L

    // Compute L(:, k) for L*L' = C in left-looking order
    for (csint k = 0; k < N; k++) {
        // scatter [ a22 | --- a32 --- ].T into x
        //            k    k+1  ...  N
        // x := full(tril(C(:, k))) == full(triu(C(k, :)))
        for (csint p = C.p_[k]; p < C.p_[k+1]; p++) {
            csint i = C.i_[p];
            if (i >= k) {  // only take lower triangular
                x[i] = C.v_[p];
            }
        }

        //--- Sparse Multiply --------------------------------------------------
        // Multiply L[k:, :k] @ L[k, :k].T, and subtract it from x
        //
        //        1    ...  k-1     k
        //   k   [--- l12.T ---] [  |  ]  == (N - k) x (k - 1) * (k - 1) x 1
        //   k+1 [             ] [  |  ]  => (N - k) x 1
        //   ... [     L31     ] [ l12 ]
        //   N   [             ] [  |  ]
        //
        // Result is: [a_22 - l_12.T @ l_12 | a_32 - L_31 @ l_12].T
        //                  x[k]            | ---- x[k+1:] ----

        // ereach_queue gives pattern of L(k, :) in no particular order
        for (const auto& j : ereach_queue(C, k, S.parent)) {
            // Compute x[k:] -= L[k:, j] * L[k, j]
            //
            // Row indices and values in L[k:, j] are stored in:
            //  L.i_ and L.v_[c[j] ... L.p_[j+1]-1]
            //
            double lkj = L.v_[c[j]];  // cache L(k, j)
            for (csint p = c[j]++; p < L.p_[j+1]; p++) {
                x[L.i_[p]] -= L.v_[p] * lkj;
            }
        }

        //--- Compute L[k:, k] -------------------------------------------------
        double Lkk = std::sqrt(x[k]);
        L.v_[c[k]++] = Lkk;
        x[k] = 0.0;  // clear x for k + 1st iteration

        // Compute the rest of the column L[k+1:, k] = x[k+1:] / L[k, k]
        for (csint p = c[k]; p < L.p_[k+1]; p++) {
            csint i = L.i_[p];
            L.v_[p] = x[i] / Lkk;
            x[i] = 0.0;  // clear x for k + 1st iteration
        }
    }

    // Guaranteed by construction
    L.has_sorted_indices_ = true;
    L.has_canonical_format_ = true;  // L retains numerically 0 entries

    return L;
}


CSCMatrix& rechol(const CSCMatrix& A, const Symbolic& S, CSCMatrix& L)
{
    // Ensure L has been allocated via symbolic_cholesky
    assert(!L.p_.empty());
    assert(!L.i_.empty());
    assert(!L.v_.empty());
    assert(L.has_sorted_indices_);

    csint N = A.shape()[1];

    // Workspaces
    std::vector<csint> c(S.cp);  // column pointers for L
    std::vector<double> x(N);    // sparse accumulator

    const CSCMatrix C = A.symperm(S.p_inv);

    L.p_ = S.cp;  // column pointers for L

    // Compute L(:, k) for L*L' = C
    for (csint k = 0; k < N; k++) {
        //--- Nonzero pattern of L(k, :) ---------------------------------------
        x[k] = 0.0;  // x(0:k) is now zero

        // scatter C into x = full(triu(C(:,k)))
        for (csint p = C.p_[k]; p < C.p_[k+1]; p++) {
            csint i = C.i_[p];
            if (i <= k) {
                x[i] = C.v_[p];
            }
        }

        double d = x[k];  // d = C(k, k)
        x[k] = 0.0;       // clear x for k + 1st iteration

        //--- Triangular Solve -------------------------------------------------
        // Solve L(0:k-1, 0:k-1) * x = C(0:k-1, k) =- L[:k, :k] * x = C[:k, k]
        //   => L[k, :k] := x.T == l_{12}.T
        // ereach gives the pattern of L(k, :) in topological order
        for (const auto& i : ereach(C, k, S.parent)) {
            double lki = x[i] / L.v_[L.p_[i]];  // L(k, i) = x(i) / L(i, i)
            x[i] = 0.0;                         // clear x for k + 1st iteration

            for (csint p = L.p_[i] + 1; p < c[i]; p++) {
                x[L.i_[p]] -= L.v_[p] * lki;    // x -= L(i, :) * L(k, i)
            }

            // subtract the sparse dot product from the diagonal
            d -= lki * lki;                     // d -= L(k, i) * L(k, i)

            // These pointers are incremented one at a time, guaranteeing that
            // the columns of L are sorted.
            L.v_[c[i]++] = lki;                 // store L(k, i) in column i
        }

        //--- Compute L(k, k) --------------------------------------------------
        if (d <= 0) {
            throw std::runtime_error("Matrix not positive definite!");
        }

        L.v_[c[k]++] = std::sqrt(d);  // store L(k, k) = sqrt(d) in column k
    }

    // Guaranteed by construction
    L.has_sorted_indices_ = true;
    L.has_canonical_format_ = true;  // L retains numerically 0 entries

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
CholCounts chol_etree_counts(const CSCMatrix& A)
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
            // For each nonzero in the strict upper triangular part of A,
            // follow path from node i to the root of the etree, or flagged node
            for (csint i = A.i_[p]; i < k && flag[i] != k; i = parent[i]) {
                if (parent[i] == -1) {
                    parent[i] = k;   // the parent of i must be k
                }
                row_counts[k]++;  // A[i, k] != 0 => L[k, i] != 0
                col_counts[i]++;
                flag[i] = k;  // mark node k as visited
            }
        }
    }

    return {parent, row_counts, col_counts};
}


CSCMatrix ichol(
    const CSCMatrix& A,
    ICholMethod method,
    double drop_tol
)
{
    switch (method) {
        case ICholMethod::NoFill:
            // L = A;  // just copy the matrix and update the values
            throw std::runtime_error("Not implemented!");
            break;

        case ICholMethod::ICT:
            return chol(A, schol(A), drop_tol);
            break;

        default:
            throw std::runtime_error("Invalid incomplete Cholesky method!");
    }
}



} // namespace cs

/*==============================================================================
 *============================================================================*/
