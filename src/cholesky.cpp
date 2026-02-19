/*==============================================================================
 *     File: cholesky.cpp
 *  Created: 2025-01-27 13:14
 *   Author: Bernie Roesler
 *
 *  Description: Implements the symbolic factorization for a sparse matrix.
 *
 *============================================================================*/

#include <algorithm>  // reverse_copy
#include <cassert>
#include <cmath>      // std::sqrt
#include <format>
#include <iterator>   // std::back_inserter
#include <numeric>    // std::iota, partial_sum

#include "cholesky.h"
#include "csc.h"
#include "utils.h"

namespace cs {

/*------------------------------------------------------------------------------
 *         Cholesky Decomposition
 *----------------------------------------------------------------------------*/
std::vector<csint> etree(const CSCMatrix& A, bool ata)
{
    const auto [M, N] = A.shape();
    std::vector<csint> parent(N, -1);    // parent of i is parent[i]
    std::vector<csint> ancestor(N, -1);  // workspaces
    std::vector<csint> prev;
    if (ata) {
        prev = std::vector<csint>(M, -1);
    }

    for (auto k : A.column_range()) {
        for (auto ip : A.row_indices(k)) {
            auto i = ata ? prev[ip] : ip;  // A(i, k) is nonzero
            while (i != -1 && i < k) {      // only use upper triangular of A
                auto inext = ancestor[i];  // traverse up to the root
                ancestor[i] = k;            // path compression
                if (inext == -1) {
                    parent[i] = k;          // no ancestor
                }
                i = inext;
            }
            if (ata) {
                prev[ip] = k;  // use prev for A^T
            }
        }
    }

    return parent;
}


// Exercise 4.6
// TODO this algorithm is O(N * avg height), but needs to be O(N)
csint etree_height(std::span<const csint> parent)
{
    assert(!parent.empty());

    csint height = 0;
    for (csint i = 0; i < std::ssize(parent); ++i) {
        csint h = 0;
        for (csint p = i; p != -1; p = parent[p]) {
            ++h;
        }
        height = std::max(height, h);
    }

    return height;
}


std::vector<csint> ereach(
    const CSCMatrix& A,
    csint k,
    std::span<const csint> parent
)
{
    const auto [M, N] = A.shape();
    std::vector<char> marked(M, false);  // workspace
    std::vector<csint> s, xi;  // internal dfs stack, output stack
    s.reserve(N);
    xi.reserve(N);

    marked[k] = true;  // mark node k as visited

    for (auto i : A.row_indices(k)) {
        if (i <= k) {     // only consider upper triangular part of A
            // Traverse up the etree
            while (!marked[i]) {
                s.push_back(i);  // L(k, i) is nonzero
                marked[i] = true;  // mark i as visited
                i = parent[i];
            }

            // Push path onto output stack
            std::ranges::reverse_copy(s, std::back_inserter(xi));
            s.clear();
        }
    }

    // Reverse the stack to get the topological order
    std::ranges::reverse(xi);
    return xi;
}


std::vector<csint> ereach_post(
    const CSCMatrix& A,
    csint k,
    std::span<const csint> parent
)
{
    if (!A.has_sorted_indices()) {
        throw std::invalid_argument(
            "Matrix A must have sorted row indices for ereach_post."
        );
    }

    const auto [M, N] = A.shape();
    std::vector<char> marked(M, false);  // workspace
    std::vector<csint> xi;  // internal dfs stack, output stack
    xi.reserve(N);

    marked[k] = true;  // mark node k as visited

    auto indices = A.row_indices(k);

    for (size_t idx = 0; idx < indices.size(); ++idx) {
        auto i = indices[idx];  // A(i, k) is nonzero
        csint i2 = (idx + 1) < indices.size() ? indices[idx + 1] : A.nnz();  // next row index
        if (i <= k) {  // only consider upper triangular part of A
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
    std::span<const csint> parent
)
{
    const auto [M, N] = A.shape();
    std::vector<char> marked(M, false);  // workspace
    std::vector<csint> s;  // internal dfs stack, output stack
    s.reserve(N);

    marked[k] = true;  // mark node k as visited

    for (auto i : A.row_indices(k)) {
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


std::vector<csint> post(std::span<const csint> parent)
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

    auto j_range = std::views::iota(csint{0}, N);

    // Traverse nodes in reverse order
    for (auto j : j_range | std::views::reverse) {
        if (parent[j] != -1) {           // only operate on non-roots
            next[j] = head[parent[j]];   // add j to list of its parent
            head[parent[j]] = j;
        }
    }

    // Search from each root
    for (auto j : j_range) {
        if (parent[j] == -1) {  // only search from roots
            tdfs(j, head, next, postorder);
        }
    }

    assert(std::ssize(postorder) == N);
    postorder.shrink_to_fit();

    return postorder;
}


void tdfs(
    csint j,
    std::span<csint> head,
    std::span<const csint> next,
    std::vector<csint>& postorder
)
{
    std::vector<csint> stack;
    stack.push_back(j);              // place j on stack

    while (!stack.empty()) {
        auto p = stack.back();      // p = top of stack
        auto i = head[p];           // i = youngest child of p
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
    std::span<const csint> parent,
    std::span<const csint> postorder
)
{
    assert(parent.size() == postorder.size());
    const csint N = parent.size();

    std::vector<csint> first(N, -1);
    std::vector<csint> level(N);

    for (csint k = 0; k < N; ++k) {
        auto i = postorder[k];  // node i of etree is kth postordered node
        csint len = 0;      // traverse from i to root
        auto r = i;
        while (r != -1 && first[r] == -1) {
            first[r] = k;
            r = parent[r];
            ++len;
        }
        len += (r == -1) ? -1 : level[r];  // r is root of tree or end of path
        for (csint s = i; s != r; s = parent[s]) {
            level[s] = len--;
        }
    }

    return {.first = first, .level = level};
}


std::vector<csint> rowcnt(
    const CSCMatrix& A,
    std::span<const csint> parent,
    std::span<const csint> postorder
)
{
    const auto [M, N] = A.shape();

    if (M != N) {
        throw std::invalid_argument(
            std::format("Matrix must be square. Got {} x {}.", M, N)
        );
    }

    if (std::ssize(parent) != N) {
        throw std::invalid_argument(
            "Parent vector size must match number of columns in A."
        );
    };

    if (parent.size() != postorder.size()) {
        throw std::invalid_argument(
            "Parent and postorder vectors must be the same size."
        );
    };

    // Count of nonzeros in each row of L
    std::vector<csint> rowcount(N, 1);   // count the diagonal to start

    std::vector<csint> ancestor(N);  // every node is its own ancestor
    std::iota(ancestor.begin(), ancestor.end(), 0);

    std::vector<csint> maxfirst(N, -1);  // max first[i] for nodes in subtree of i
    std::vector<csint> prevleaf(N, -1);  // previous leaf of ith row subtree

    auto [first, level] = firstdesc(parent, postorder);

    for (const auto& j : postorder) {  // j is the kth node in postorder
        for (auto i : A.row_indices(j)) {
            // A(i, j) is nonzero, consider ith row subtree
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
    std::span<const csint> first,
    std::span<csint> maxfirst,
    std::span<csint> prevleaf,
    std::span<csint> ancestor
)
{
    LeafStatus jleaf;

    // See "skeleton" function in Davis, p 48.
    if (i <= j || first[j] <= maxfirst[i]) {  // j is not a leaf
        return {.q = -1, .jleaf = LeafStatus::NotLeaf};
    }

    maxfirst[i] = first[j];         // update max first[j] seen so far
    auto jprev = prevleaf[i];      // jprev is the previous leaf of i
    prevleaf[i] = j;                // j is now the previous leaf of i
    jleaf = (jprev == -1) ? LeafStatus::FirstLeaf : LeafStatus::SubsequentLeaf;

    if (jleaf == LeafStatus::FirstLeaf) {
        return {.q = i, .jleaf = jleaf};  // if j is the first leaf, q = root of ith subtree
    }

    // Traverse up to the root of the subtree to find q
    auto q = jprev;
    while (q != ancestor[q]) {
        q = ancestor[q];
    }

    // Compress the path to the root
    auto s = jprev;
    while (s != q) {
        auto sparent = ancestor[s];  // path compression
        ancestor[s] = q;
        s = sparent;
    }

    return {.q = q, .jleaf = jleaf};  // least common ancestor of j_prev and j
}


void init_ata(
    const CSCMatrix& AT,
    std::span<const csint> post,
    std::vector<csint>& head,
    std::vector<csint>& next
)
{
    auto [N, M] = AT.shape();
    head.assign(N + 1, -1);
    next.assign(M, -1);
    std::vector<csint> w(N);

    // Invert postorder
    for (auto k : AT.row_range()) {
        w[post[k]] = k;
    }

    // Find the first non-zero row index in each column
    for (auto i : AT.column_range()) {
        auto k = N;
        for (auto ip : AT.row_indices(i)) {
            k = std::min(k, w[ip]);
        }
        next[i] = head[k];  // place row i in linked list k
        head[k] = i;
    }
}


std::vector<csint> counts(
    const CSCMatrix& A,
    std::span<const csint> parent,
    std::span<const csint> postorder,
    bool ata
)
{
    const auto [M, N] = A.shape();
    std::vector<csint> delta(N);  // allocate the result

    // Workspaces
    std::vector<csint> ancestor(N),
                       maxfirst(N, -1),  // max first[i] for nodes in subtree of i
                       prevleaf(N, -1),  // previous leaf of ith row subtree
                       first(N, -1),     // first descendant of each node in the tree
                       head,             // head of the linked list
                       next;             // next node of the linked list

    // every node is its own ancestor
    std::iota(ancestor.begin(), ancestor.end(), 0);

    // Compute first descendent of each node in the tree
    for (auto k : A.column_range()) {
        auto j = postorder[k];  // node j of etree is kth postordered node
        delta[j] = (first[j] == -1) ? 1 : 0;  // delta[j] = 1 if j is a leaf
        while (j != -1 && first[j] == -1) {
            first[j] = k;   // first descendant of j
            j = parent[j];  // move up the etree
        }
    }

    // Operate on the transpose
    auto AT = A.transpose(false);  // do not copy values in the transpose

    if (ata) {
        init_ata(AT, postorder, head, next);
    }

    for (auto k : A.column_range()) {
        auto j = postorder[k];  // node j of etree is kth postordered node
        if (parent[j] != -1) {
            delta[parent[j]]--;  // j is not a root
        }

        // J = j for LL' = A case, otherwise traverse the linked list
        for (csint J = ata ? head[k] : j; J != -1; J = ata ? next[J] : -1) {
            for (auto i : AT.row_indices(J)) {
                auto [q, jleaf] = least_common_ancestor(i, j, first, maxfirst, prevleaf, ancestor);
                if (jleaf != LeafStatus::NotLeaf) {
                    delta[j]++;  // A(i, j) is in skeleton
                }
                if (jleaf == LeafStatus::SubsequentLeaf) {
                    delta[q]--;  // account for overlap in q
                }
            }
        }

        // Merge j into the ancestor set containing j's parent
        if (parent[j] != -1) {
            ancestor[j] = parent[j];
        }
    }

    // sum up the counts for each child
    for (auto j : A.column_range()) {
        if (parent[j] != -1) {
            delta[parent[j]] += delta[j];
        }
    }

    return delta;
}


std::vector<csint> chol_rowcounts(const CSCMatrix& A)
{
    // Compute the elimination tree of A
    auto parent = etree(A);

    // Compute the post-order of the elimination tree
    auto postorder = post(parent);

    // Count the number of non-zeros in each row of L
    return rowcnt(A, parent, postorder);
}


std::vector<csint> chol_colcounts(const CSCMatrix& A, bool ata)
{
    // Compute the elimination tree of A
    auto parent = etree(A, ata);

    // Compute the post-order of the elimination tree
    auto postorder = post(parent);

    // Count the number of non-zeros in each column of L
    return counts(A, parent, postorder, ata);
}


SymbolicChol schol(const CSCMatrix& A, AMDOrder order, bool use_postorder)
{
    SymbolicChol S;
    std::vector<csint> p(A.shape()[1]);    // the matrix permutation

    if (order == AMDOrder::Natural) {
        std::iota(p.begin(), p.end(), 0);  // identity permutation
        S.p_inv = p;                       // identity is its own inverse
    } else {
        p = amd(A, order);                 // order = APlusAT for Cholesky
        S.p_inv = inv_permute(p);
    }

    // Find pattern of Cholesky factor
    auto C = A.symperm(S.p_inv, false);  // C = spones(triu(A(p, p)))
    S.parent = etree(C);
    auto postorder = post(S.parent);

    // Exercise 4.9
    if (use_postorder) {
        p = pvec(postorder, p);         // combine the permutations
        S.p_inv = inv_permute(p);
        C = A.symperm(S.p_inv, false);  // apply combined permutation
        S.parent = etree(C);
        postorder = post(S.parent);     // should be identity for natural order
    }

    auto c = counts(C, S.parent, postorder);

    // Compute column pointers for L
    S.cp.resize(c.size() + 1);
    std::partial_sum(c.cbegin(), c.cend(), S.cp.begin() + 1);
    S.lnz = S.cp.back();  // number of non-zeros in L

    return S;
}


CholResult symbolic_cholesky(const CSCMatrix& A, const SymbolicChol& S)
{
    const auto [M, N] = A.shape();
    CSCMatrix L{{M, N}, S.lnz};        // allocate result

    std::vector<csint> c(S.cp);      // column pointers for L

    const auto C = A.symperm(S.p_inv);

    L.p_ = S.cp;  // column pointers for L

    // Compute L(:, k) for L*L' = C
    for (auto k : L.column_range()) {
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

    return {.L = L, .p_inv = S.p_inv};
}


CholResult chol(const CSCMatrix& A, const SymbolicChol& S)
{
    const auto [M, N] = A.shape();
    CSCMatrix L{{M, N}, S.lnz};  // allocate result

    // Workspaces
    std::vector<csint> c(S.cp);  // column pointers for L
    std::vector<double> x(N);    // sparse accumulator

    const auto C = A.symperm(S.p_inv);

    L.p_ = S.cp;  // column pointers for L

    // Compute L(k, :) for L*L' = C in up-looking order
    for (auto k : L.column_range()) {
        //--- Nonzero pattern of L(k, :) ---------------------------------------
        x[k] = 0.0;  // x(0:k) is now zero

        // scatter C into x = full(triu(C(:,k)))
        // C does not have to be in sorted order (d = x[k] gets the diagonal)
        for (auto [i, v] : C.column(k)) {
            if (i <= k) {
                x[i] = v;
            }
        }

        auto d = x[k];  // d = C(k, k)
        x[k] = 0.0;       // clear x for k + 1st iteration

        //--- Triangular Solve -------------------------------------------------
        // Solve L(0:k-1, 0:k-1) * x = C(0:k-1, k) == L[:k, :k] * x = C[:k, k]
        //   => L[k, :k] := x.T
        // ereach gives the pattern of L(k, :) in topological order
        for (const auto& i : ereach(C, k, S.parent)) {
            auto lki = x[i] / L.v_[L.p_[i]];  // L(k, i) = x(i) / L(i, i)
            x[i] = 0.0;                         // clear x for k + 1st iteration

            for (csint p = L.p_[i] + 1; p < c[i]; ++p) {
                x[L.i_[p]] -= L.v_[p] * lki;    // x -= L(i, :) * L(k, i)
            }

            // subtract the sparse dot product from the diagonal
            d -= lki * lki;                     // d -= L(k, i) * L(k, i)

            // We build L one *row* at a time, in topological order. All
            // i < k since they are reachable, so the diagonal is always the
            // first element in its column, and all other elements are in order.
            auto p = c[i]++;
            L.i_[p] = k;                        // store L(k, i) in column i
            L.v_[p] = lki;
        }

        //--- Compute L(k, k) --------------------------------------------------
        if (d <= 0) {
            throw CholeskyNotPositiveDefiniteError(
                "Matrix not positive definite!"
            );
        }

        // store L(k, k) = sqrt(d) in column k
        auto p = c[k]++;
        L.i_[p] = k;
        L.v_[p] = std::sqrt(d);
    }

    // Guaranteed by construction
    L.has_sorted_indices_ = true;
    L.has_canonical_format_ = true;

    return {.L = L, .p_inv = S.p_inv};
}


CSCMatrix& leftchol(const CSCMatrix& A, const SymbolicChol& S, CSCMatrix& L)
{
    // Ensure L has been allocated via symbolic_cholesky
    assert(!L.indptr().empty());
    assert(!L.indices().empty());
    assert(!L.data().empty());
    assert(L.has_sorted_indices_);

    auto N = A.shape()[1];

    // Workspaces
    std::vector<csint> c(S.cp);  // column pointers for L
    std::vector<double> x(N);    // sparse accumulator

    // Need the *lower* triangular part for a_{32}, so do a full permutation
    const auto C = A.permute(S.p_inv, inv_permute(S.p_inv));

    // Compute L(:, k) for L*L' = C in left-looking order
    for (auto k : L.column_range()) {
        // scatter [ a22 | --- a32 --- ].T into x
        //            k    k+1  ...  N
        // x := full(tril(C(:, k))) == full(triu(C(k, :)))
        for (auto [i, v] : C.column(k)) {
            if (i >= k) {  // only take lower triangular
                x[i] = v;
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
            auto lkj = L.v_[c[j]++];  // cache L(k, j)
            for (auto [i, v] : L.column(j)) {
                x[i] -= v * lkj;
            }
        }

        //--- Compute L[k:, k] -------------------------------------------------
        auto Lkk = std::sqrt(x[k]);
        L.v_[c[k]++] = Lkk;
        x[k] = 0.0;  // clear x for k + 1st iteration

        // Compute the rest of the column L[k+1:, k] = x[k+1:] / L[k, k]
        for (csint p = c[k]; p < L.p_[k+1]; ++p) {
            auto i = L.i_[p];
            L.v_[p] = x[i] / Lkk;
            x[i] = 0.0;  // clear x for k + 1st iteration
        }
    }

    // Guaranteed by construction
    L.has_sorted_indices_ = true;
    L.has_canonical_format_ = true;  // L retains numerically 0 entries

    return L;
}


CSCMatrix& rechol(const CSCMatrix& A, const SymbolicChol& S, CSCMatrix& L)
{
    // Ensure L has been allocated via symbolic_cholesky
    assert(!L.p_.empty());
    assert(!L.i_.empty());
    assert(!L.v_.empty());
    assert(L.has_sorted_indices_);

    auto N = A.shape()[1];

    // Workspaces
    std::vector<csint> c(S.cp);  // column pointers for L
    std::vector<double> x(N);    // sparse accumulator

    const auto C = A.symperm(S.p_inv);

    L.p_ = S.cp;  // column pointers for L

    // Compute L(:, k) for L*L' = C
    for (auto k : L.column_range()) {
        //--- Nonzero pattern of L(k, :) ---------------------------------------
        x[k] = 0.0;  // x(0:k) is now zero

        // scatter C into x = full(triu(C(:,k)))
        for (auto [i, v] : C.column(k)) {
            if (i <= k) {
                x[i] = v;
            }
        }

        auto d = x[k];  // d = C(k, k)
        x[k] = 0.0;       // clear x for k + 1st iteration

        //--- Triangular Solve -------------------------------------------------
        // Solve L(0:k-1, 0:k-1) * x = C(0:k-1, k) =- L[:k, :k] * x = C[:k, k]
        //   => L[k, :k] := x.T == l_{12}.T
        // ereach gives the pattern of L(k, :) in topological order
        for (const auto& i : ereach(C, k, S.parent)) {
            auto lki = x[i] / L.v_[L.p_[i]];  // L(k, i) = x(i) / L(i, i)
            x[i] = 0.0;                         // clear x for k + 1st iteration

            for (csint p = L.p_[i] + 1; p < c[i]; ++p) {
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
    bool update,
    const CSCMatrix& C,
    std::span<const csint> parent
)
{
    if (L.shape()[0] != C.shape()[0]) {
        throw std::invalid_argument(
            std::format(
                "L and C must have the same number of rows."
                "Got {} and {}.",
                L.shape()[0], C.shape()[0]
            )
        );
    }

    if (C.shape()[1] != 1) {  // C must be a column vector
        throw std::invalid_argument(
            std::format("C must be a column vector. Got {} columns.", C.shape()[1])
        );
    }

    double α,
           β = 1.0,
           β2 = 1.0,
           δ,
           γ,
           σ = (update) ? 1.0 : -1.0;

    std::vector<double> w(L.shape()[0]);  // sparse accumulator workspace

    // Find the minimum row index in the update vector
    auto p = C.p_[0];
    auto f = C.i_[p];
    for (; p < C.p_[1]; ++p) {
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
        δ = update ? (β / β2) : (β2 / β);
        γ = σ * α / (β2 * β);
        L.v_[p] = δ * L.v_[p] + (update ? (γ * w[j]) : 0.0);
        β = β2;
        for (p++; p < L.p_[j+1]; ++p) {
            auto w1 = w[L.i_[p]];
            auto w2 = w1 - α * L.v_[p];
            w[L.i_[p]] = w2;
            L.v_[p] = δ * L.v_[p] + γ * (update ? w1 : w2);
        }
    }

    return L;
}


// Exercise 4.1 O(|L|)-time elimination tree and row/column counts
// Use ereach to compute the elimination tree one node at a time (pp 43--44)
CholCounts chol_etree_counts(const CSCMatrix& A)
{
    const auto [M, N] = A.shape();

    if (M != N) {
        throw std::runtime_error("Matrix must be square!");
    }

    std::vector<csint> parent(N, -1);
    std::vector<csint> row_counts(N, 1);  // count diagonals
    std::vector<csint> col_counts(N, 1);

    // Note that we "mark" nodes by setting the flag to the column index k, so
    // we don't have to reset a bool array each time.
    std::vector<csint> flag(N, -1);  // workspace

    for (auto k : A.column_range()) {
        flag[k] = k;  // mark node k as visited
        // Compute T_k from T_{k-1} by finding the children of node k
        for (auto i : A.row_indices(k)) {
            // For each nonzero in the strict upper triangular part of A,
            // follow path from node i to the root of the etree, or flagged node
            for (; i < k && flag[i] != k; i = parent[i]) {
                if (parent[i] == -1) {
                    parent[i] = k;   // the parent of i must be k
                }
                row_counts[k]++;  // A[i, k] != 0 => L[k, i] != 0
                col_counts[i]++;
                flag[i] = k;  // mark node k as visited
            }
        }
    }

    return {.parent = parent, .row_counts = row_counts, .col_counts = col_counts};
}


// Exercise 4.13
CholResult ichol_nofill(const CSCMatrix& A, const SymbolicChol& S)
{
    const auto [M, N] = A.shape();

    if (M != N) {
        throw std::runtime_error("Matrix must be square!");
    }

    const auto C = A.symperm(S.p_inv);

    // Get structure of "lower" tri of C (may be stored as upper only)
    const auto C_tril = C.band(0, N).T();

    CSCMatrix L{{N, N}, C_tril.nnz()};  // allocate result

    // Workspaces
    std::vector<csint> c(C_tril.p_);  // column pointers for L
    std::vector<csint> w(N, -1);      // row indices for column of C
    std::vector<double> x(N);         // values for column of C

    L.p_ = C_tril.p_;  // column pointers for L (same pattern as A)

    // Compute L(k, :) for L*L' = C in up-looking order
    for (auto k : L.column_range()) {
        //--- Nonzero pattern of L(k, :) ---------------------------------------
        x[k] = 0.0;  // x(0:k) is now zero

        // scatter C into x = full(triu(C(:,k)))
        // C does not have to be in sorted order (d = x[k] gets the diagonal)
        for (auto [i, v] : C.column(k)) {
            if (i <= k) {
                x[i] = v;
            }
            w[i] = k;
        }

        auto d = x[k];  // d = C(k, k)
        x[k] = 0.0;       // clear x for k + 1st iteration

        //--- Triangular Solve -------------------------------------------------
        // Solve L(0:k-1, 0:k-1) * x = C(0:k-1, k) == L[:k, :k] * x = C[:k, k]
        //   => L[k, :k] := x.T
        // ereach gives the pattern of L(k, :) in topological order
        for (const auto& i : ereach(C, k, S.parent)) {
            // Only keep entries that are in the column of C
            if (w[i] != k) {
                continue;
            }

            auto lki = x[i] / L.v_[L.p_[i]];  // L(k, i) = x(i) / L(i, i)
            x[i] = 0.0;                         // clear x for k + 1st iteration

            for (csint p = L.p_[i] + 1; p < c[i]; ++p) {
                x[L.i_[p]] -= L.v_[p] * lki;    // x -= L(i, :) * L(k, i)
            }

            // subtract the sparse dot product from the diagonal
            d -= lki * lki;                     // d -= L(k, i) * L(k, i)

            // We build L one *row* at a time, in topological order. All
            // i < k since they are reachable, so the diagonal is always the
            // first element in its column, and all other elements are in order.
            auto p = c[i]++;
            L.i_[p] = k;                        // store L(k, i) in column i
            L.v_[p] = lki;
        }

        //--- Compute L(k, k) --------------------------------------------------
        if (d <= 0) {
            throw std::runtime_error("Matrix not positive definite!");
        }

        auto p = c[k]++;
        L.i_[p] = k;  // store L(k, k) = sqrt(d) in column k
        L.v_[p] = std::sqrt(d);
    }

    // Guaranteed by construction
    L.has_sorted_indices_ = true;
    L.has_canonical_format_ = true;

    return {.L = L, .p_inv = S.p_inv};
}


// Exercise 4.13
CholResult icholt(const CSCMatrix& A, const SymbolicChol& S, double drop_tol)
{
    const auto [M, N] = A.shape();

    if (M != N) {
        throw std::runtime_error("Matrix must be square!");
    }

    if (drop_tol < 0) {
        throw std::runtime_error("drop_tol must be non-negative!");
    }

    CSCMatrix L{{M, N}, S.lnz};  // allocate result

    // Workspaces
    std::vector<csint> c(S.cp);  // column pointers for L
    std::vector<double> x(N);    // sparse accumulator

    const auto C = A.symperm(S.p_inv);

    L.p_ = S.cp;  // column pointers for L

    // Compute L(k, :) for L*L' = C in up-looking order
    for (auto k : L.column_range()) {
        //--- Nonzero pattern of L(k, :) ---------------------------------------
        x[k] = 0.0;  // x(0:k) is now zero

        // scatter C into x = full(triu(C(:,k)))
        // C does not have to be in sorted order (d = x[k] gets the diagonal)
        for (auto [i, v] : C.column(k)) {
            if (i <= k) {
                x[i] = v;
            }
        }

        // NOTE MATLAB compares elements to the 1-norm of drop_tol * A[k:, k]
        // (lower tri of column k)
        // however, we only access A[:k, k] (upper tri) on a given iteration, so
        // computing the 1-norm of the lower tri is more difficult.
        //
        // It is the same as computing the 1-norm of A[k, k:] since A is
        // symmetric, but that is not helpful as we haven't seen the column yet.
        //
        // We would have to re-write symperm, ereach, and this function to only
        // consider the lower tri of A.
        //
        // Instead, compare the elements of L to drop_tol * A[k, k], like
        // SuperLU does for ilu.
        //
        // IDEA: The leftchol algorithm scatters the *lower* triangular of
        // A for computation. We could use that algorithm instead to easily
        // compute the 1-norm of each column lower tri for comparison.
        //
        // Problem: MATLAB seems to use a dropping criteria that is different
        // than the one stated in their documentation. Filtering the full
        // L factor based on their criteria gives a different result than
        // computing L = ichol(A, options).

        auto d = x[k];  // d = C(k, k)
        x[k] = 0.0;       // clear x for k + 1st iteration

        auto abs_diag = std::fabs(d);  // store diagonal for drop_tol check

        //--- Triangular Solve -------------------------------------------------
        // Solve L(0:k-1, 0:k-1) * x = C(0:k-1, k) == L[:k, :k] * x = C[:k, k]
        //   => L[k, :k] := x.T
        // ereach gives the pattern of L(k, :) in topological order
        for (const auto& i : ereach(C, k, S.parent)) {
            auto lki = x[i] / L.v_[L.p_[i]];  // L(k, i) = x(i) / L(i, i)
            x[i] = 0.0;                         // clear x for k + 1st iteration

            for (csint p = L.p_[i] + 1; p < c[i]; ++p) {
                x[L.i_[p]] -= L.v_[p] * lki;    // x -= L(i, :) * L(k, i)
            }

            // We build L one *row* at a time, in topological order. All
            // i < k since they are reachable, so the diagonal is always the
            // first element in its column, and all other elements are in order.
            if (std::abs(lki) > drop_tol * abs_diag) {
                // subtract the sparse dot product from the diagonal
                d -= lki * lki;  // d -= L(k, i) * L(k, i)

                // store L(k, i) in column i
                auto p = c[i]++;
                L.i_[p] = k;
                L.v_[p] = lki;
            }
        }

        //--- Compute L(k, k) --------------------------------------------------
        if (d <= 0) {
            throw std::runtime_error("Matrix not positive definite!");
        }

        // store L(k, k) = sqrt(d) in column k
        auto p = c[k]++;
        L.i_[p] = k;
        L.v_[p] = std::sqrt(d);
    }

    if (drop_tol > 0) {
        L.dropzeros();   // remove numerically zero entries
    }

    // Guaranteed by construction
    L.has_sorted_indices_ = true;
    L.has_canonical_format_ = true;

    return {.L = L, .p_inv = S.p_inv};
}



} // namespace cs

/*==============================================================================
 *============================================================================*/
