//==============================================================================
//     File: cholesky.h
//  Created: 2025-01-27 13:01
//   Author: Bernie Roesler
//
//  Description: Implements the symbolic factorization for a sparse matrix.
//
//==============================================================================

#ifndef _CSPARSE_DECOMPOSITION_H_
#define _CSPARSE_DECOMPOSITION_H_

#include <vector>

#include "types.h"

namespace cs {

// ---------- Enums
enum class LeafStatus {
    NotLeaf,
    FirstLeaf,
    SubsequentLeaf
};


// ---------- Structs
struct FirstDesc {
    std::vector<csint> first, level;
};


struct LCAStatus {
    csint q;
    LeafStatus jleaf;
};


struct CholCounts {
    std::vector<csint> parent, row_counts, col_counts;
};


/** SymbolicChol Cholesky decomposition return struct (see cs_symbolic aka css) */
struct SymbolicChol
{
    std::vector<csint> p_inv,   ///< fill-reducing permutation
                       parent,  ///< elimination tree
                       cp;      ///< column pointers

    csint lnz;  ///< # entries in L
};


/*------------------------------------------------------------------------------
 *          Cholesky Decomposition
 *----------------------------------------------------------------------------*/
// ---------- Helpers
/** Post-order a tree non-recursively, in O(N) time.
 *
 * @param parent  the parent vector of the elimination tree
 *
 * @return post  the post-order of the elimination tree
 */
std::vector<csint> post(const std::vector<csint>& parent);


/** Depth-first search in a tree.
 *
 * @param j  the starting node
 * @param[in,out] head  the head of the linked list
 * @param next  the next vector of the linked list
 * @param[in,out] postorder  the post-order of the elimination tree
 */
void tdfs(
    csint j,
    std::vector<csint>& head,
    const std::vector<csint>& next,
    std::vector<csint>& postorder
);


// NOTE firstdesc and rowcnt are *not* officially part of CSparse, but are in
// the book for demonstrative purposes.
/** Find the first descendent of a node in a tree.
 *
 * @note The *first descendent* of a node `j` is the smallest postordering of
 * any descendant of `j`.
 *
 * @param parent  the parent vector of the elimination tree
 * @param post  the post-order of the elimination tree
 *
 * @return first  the first descendent of each node in the tree
 * @return level  the level of each node in the tree
 */
FirstDesc firstdesc(
    const std::vector<csint>& parent,
    const std::vector<csint>& postorder
);


/** Compute the least common ancestor of j_prev and j, if j is a leaf of the ith
 * row subtree.
 *
 * @param i  the row index
 * @param j  the column index
 * @param first  the first descendant of each node in the tree
 * @param maxfirst  the maximum first descendant of each node in the tree
 * @param prevleaf  the previous leaf of each node in the tree
 * @param ancestor  the ancestor of each node in the tree
 *
 * @return q lca(jprev, j)
 * @return jleaf  the leaf status of j:
 *                  0 (not a leaf), 1 (first leaf), 2 (subsequent leaf)
 *
 * @see cs_leaf Davis p 48.
 */
LCAStatus least_common_ancestor(
    csint i,
    csint j,
    const std::vector<csint>& first,
    std::vector<csint>& maxfirst,
    std::vector<csint>& prevleaf,
    std::vector<csint>& ancestor
);


// ---------- Matrix operations
/** Compute the elimination tree of A.
 *
 * @param A  the matrix to factorize
 * @param ata  if True, compute the elimination tree of A^T A
 *
 * @return parent  the parent vector of the elimination tree
 */
std::vector<csint> etree(const CSCMatrix& A, bool ata=false);


/** Compute the height of the elimination tree.
 *
 * The height is defined as the length of the longest path from the root to any
 * leaf of the tree.
 *
 * See: Davis, Exercise 4.6.
 *
 * @param parent  the parent vector of the elimination tree
 *
 * @return height  the height of the elimination tree
 */
csint etree_height(const std::vector<csint>& parent);


/** Compute the reachability set for the *k*th row of *L*, the Cholesky faxtcor
 * of `A`, in topological order.
 *
 * @param A  the matrix to factorize
 * @param k  the row index
 * @param parent  the parent vector of the elimination tree
 *
 * @return xi  the reachability set of the *k*th row of *L* in topological order
 */
std::vector<csint> ereach(
    const CSCMatrix& A,
    csint k,
    const std::vector<csint>& parent
);


/** Compute the reachability set for the *k*th row of *L*, the Cholesky faxtcor
 * of this matrix.
 *
 * `A` and `parent` are assumed to be postordered, and `A` is assumed to have
 * sorted columns.
 *
 * @param A  the matrix to factorize
 * @param k  the row index
 * @param parent  the parent vector of the elimination tree
 *
 * @return xi  the reachability set of the *k*th row of *L* in topological order
 */
std::vector<csint> ereach_post(
    const CSCMatrix& A,
    csint k,
    const std::vector<csint>& parent
);


/** Compute the reachability set for the *k*th row of *L*, the Cholesky faxtcor
 * of `A`, in no particular order.
 *
 * @param A  the matrix to factorize
 * @param k  the row index
 * @param parent  the parent vector of the elimination tree
 *
 * @return xi  the reachability set of the *k*th row of *L*
 */
std::vector<csint> ereach_queue(
    const CSCMatrix& A,
    csint k,
    const std::vector<csint>& parent
);


/** Count the number of non-zeros in each row of the Cholesky factor L of A.
 *
 * @param A  the matrix to factorize
 * @param parent  the parent vector of the elimination tree
 * @param postorder  the post-order of the elimination tree
 *
 * @return rowcount  the number of non-zeros in each row of L
 */
std::vector<csint> rowcnt(
    const CSCMatrix& A,
    const std::vector<csint>& parent,
    const std::vector<csint>& postorder
);


/** Initialize the linked list structure for the column counts of A^T A.
 *
 * @param AT  the transpose of the matrix to factorize
 * @param post  the post-order of the elimination tree
 * @param[out] head  the head of the linked list
 * @param[out] next  the next vector of the linked list
 */
void init_ata(
    const CSCMatrix& AT,
    const std::vector<csint>& post,
    std::vector<csint>& head,
    std::vector<csint>& next
);


/** Count the number of non-zeros in each column of the Cholesky factor L of A.
 *
 * @param A  the matrix to factorize
 * @param parent  the parent vector of the elimination tree
 * @param postorder  the post-order of the elimination tree
 * @param ata  if True, compute the counts for A^T A, otherwise A
 *
 * @return colcount  the number of non-zeros in each column of L
 */
std::vector<csint> counts(
    const CSCMatrix& A,
    const std::vector<csint>& parent,
    const std::vector<csint>& postorder,
    bool ata=false
);


/** Count the number of non-zeros in each row of the Cholesky factor L of A.
 *
 * @param A  the matrix to factorize
 *
 * @return rowcount  the number of non-zeros in each row of L
 */
std::vector<csint> chol_rowcounts(const CSCMatrix& A);


/** Count the number of non-zeros in each column of the Cholesky factor L of A.
 *
 * @param A  the matrix to factorize
 * @param ata  if True, compute the counts for A^T A, otherwise A
 *
 * @return colcount  the number of non-zeros in each column of L
 */
std::vector<csint> chol_colcounts(const CSCMatrix& A, bool ata=false);


/** Compute the symbolic Cholesky factorization of a sparse matrix.
 *
 * @note This function assumes that `A` is symmetric and positive definite.
 *
 * @param A the matrix to factorize
 * @param order the ordering method to use:
 *       - 0: natural ordering
 *       - 1: amd(A + A.T)
 *       - 2: amd(A.T * A) with no dense rows
 *       - 3: amd(A.T * A)
 * @param postorder  if True, postorder the matrix in addition to the AMD
 *        (or natural) ordering. See: Davis, Exercise 4.9.
 *
 * @return the SymbolicChol factorization
 *
 * @see cs_schol
 */
SymbolicChol schol(
    const CSCMatrix& A,
    AMDOrder order=AMDOrder::Natural,
    bool use_postorder=false
);


/** Compute the complete symbolic Cholesky factorization of a sparse matrix.
 *
 * This functions computes the entire sparsity pattern of `L` in *O(|L|)* time.
 * It returns the matrix with sorted columns.
 *
 * See: Davis, Exercise 4.10.
 *
 * @note This function assumes that `A` is symmetric and positive definite.
 *
 * @param A the matrix to factorize
 * @param S the SymbolicChol factorization of `A`, from `cs::schol()`
 *
 * @return L  a CSCMatrix with the sparsity pattern of the Cholesky
 *         factor of A, and the values vector zeroed out.
 *
 * @see cs_schol
 * @see cs_chol
 * @see cs::schol()
 * @see cs::chol()
 */
CSCMatrix symbolic_cholesky(const CSCMatrix& A, const SymbolicChol& S);


/** Compute the up-looking Cholesky factorization of a sparse matrix.
 *
 * @note This function assumes that `A` is symmetric and positive definite.
 *
 * @param A the matrix to factorize
 * @param S the SymbolicChol factorization of `A`
 * @param drop_tol  the drop tolerance for the factorization
 *
 * @return the numeric Cholesky factorization of `A`
 */
CSCMatrix chol(const CSCMatrix& A, const SymbolicChol& S);


/** Compute the left-looking Cholesky factorization of a sparse matrix, given
 * the non-zero pattern.
 *
 * See: Davis, Exercise 4.11.
 *
 * @note This function assumes that `A` is symmetric and positive definite.
 *
 * @param A the matrix to factorize
 * @param S the SymbolicChol factorization of `A` from `cs::schol()`
 * @param[in, out] L  the symbolic Cholesky factor of `A` from
 *        `cs::symbolic_cholesky()`. This matrix is modified in place.
 *
 * @return the numeric Cholesky factorization of `A`
 *
 * @see 'python/cholesky.py::chol_left_amp()'
 */
CSCMatrix& leftchol(const CSCMatrix& A, const SymbolicChol& S, CSCMatrix& L);


/** Compute the up-looking Cholesky factorization of a sparse matrix, given the
 * non-zero pattern.
 *
 * See: Davis, Exercise 4.12.
 *
 * @note This function assumes that `A` is symmetric and positive definite.
 *
 * @param A the matrix to factorize
 * @param S the SymbolicChol factorization of `A` from `cs::schol()`
 * @param[in, out] L  the symbolic Cholesky factor of `A` from
 *        `cs::symbolic_cholesky()`. This matrix is modified in place.
 *
 * @return the numeric Cholesky factorization of `A`
 *
 * @see 'python/cholesky.py::chol_left_amp()'
 */
CSCMatrix& rechol(const CSCMatrix& A, const SymbolicChol& S, CSCMatrix& L);


/** Update the Cholesky factor for \f$ A = A + σ w w^T \f$.
 *
 * @param L  the Cholesky factor of A
 * @param update  true for update, false for downdate
 * @param C  the update vector, as the first column in a CSCMatrix
 * @param parent  the elimination tree of A
 *
 * @return L  the updated Cholesky factor of A
 */
CSCMatrix& chol_update(
    CSCMatrix& L,
    bool update,
    const CSCMatrix& C,
    const std::vector<csint>& parent
);


/** Compute the elimination tree of L and row and column counts using ereach.
 *
 * This function takes O(|L|) time and O(N) space.
 *
 * See: Davis, Exercise 4.1, and pp 43--44.
 *
 * See: Davis (2005)
 *  *Algorithm 849: A Concise Sparse Cholesky Factorization Package*
 *  Figure 1, p 590.
 *
 * See: SuiteSparse-7.7.0/LDL/Source/ldl.c
 *
 * @param A  the matrix to factorize
 *
 * @return parent  the parent vector of the elimination tree
 * @return rowcount  the number of non-zeros in each row of L
 * @return colcount  the number of non-zeros in each column of L
 */
CholCounts chol_etree_counts(const CSCMatrix& A);


/** Compute the incomplete Cholesky factorization with no fill-in.
 *
 * See: Davis, Exercise 4.13.
 *
 * This function uses the up-looking algorithm, like `chol`.
 *
 * @param A  the matrix to factorize. Only the upper triangle is used.
 * @param S  the SymbolicChol factorization of `A` from `cs::schol()`
 *
 * @return L  the incomplete Cholesky factor of `A`
 *
 * @throws std::runtime_error if `A` is not square or positive definite.
 *
 * @see cs::chol()
 * @see cs::leftchol()
 * @see cs::ilu()
 */
CSCMatrix ichol_nofill(const CSCMatrix& A, const SymbolicChol& S);


/** Compute the incomplete Cholesky factorization with drop tolerance.
 *
 * See: Davis, Exercise 4.13.
 *
 * This function uses the up-looking algorithm, like `chol`.
 *
 * @param A  the matrix to factorize. Only the upper triangle is used.
 * @param S  the SymbolicChol factorization of `A` from `cs::schol()`
 * @param drop_tol  the drop tolerance for the factorization. Any element that
 *        is smaller than `drop_tol` will not be included in `L`.
 *
 * @return L  the incomplete Cholesky factor of `A`
 *
 * @throws std::runtime_error if `A` is not square or positive definite.
 *
 * @see cs::chol()
 * @see cs::leftchol()
 * @see cs::ilu()
 */
CSCMatrix icholt(const CSCMatrix& A, const SymbolicChol& S, double drop_tol=0);


}  // namespace cs

#endif // _DECOMPOSITION_H_

//==============================================================================
//==============================================================================
