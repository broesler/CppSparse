//==============================================================================
//     File: decomposition.h
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
enum class AMDOrder
{
    Natural,
    APlusAT,
    ATimesA
};


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


// See cs_symbolic aka css
struct Symbolic
{
    std::vector<csint> p_inv,     // inverse row permutation for QR, fill-reducing permutation for Cholesky
                       q,         // fill-reducting column permutation for LU and QR
                       parent,    // elimination tree
                       cp,        // column pointers for Cholesky
                       leftmost,  // leftmost[i] = min(find(A(i,:))) for QR
                       m2;        // # of rows for QR, after adding fictitious rows

    double lnz,   // # entries in L for LU or Cholesky, in V for QR
           unz;   // # entries in U for LU, in R for QR
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
 * of this matrix.
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


/** Count the number of non-zeros in each column of the Cholesky factor L of A.
 *
 * @param A  the matrix to factorize
 * @param parent  the parent vector of the elimination tree
 * @param postorder  the post-order of the elimination tree
 *
 * @return colcount  the number of non-zeros in each column of L
 */
std::vector<csint> counts(
    const CSCMatrix& A,
    const std::vector<csint>& parent,
    const std::vector<csint>& postorder
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
 *
  * @return colcount  the number of non-zeros in each column of L
  */
std::vector<csint> chol_colcounts(const CSCMatrix& A);


/** Compute the symbolic Cholesky factorization of a sparse matrix.
 *
 * @note This function assumes that `A` is symmetric and positive definite.
 *
 * @param A the matrix to factorize
 * @param order the ordering method to use:
 *       - 0: natural ordering
 *       - 1: amd(A + A.T())
 *       - 2: amd(??)  FIXME
 *       - 3: amd(A^T A)
 * @param postorder  if True, postorder the matrix in addition to the AMD
 *        (or natural) ordering. See: Davis, Exercise 4.9.
 *
 * @return the Symbolic factorization
 *
 * @see cs_schol
 */
Symbolic schol(
    const CSCMatrix& A,
    AMDOrder order=AMDOrder::Natural,
    bool use_postorder=false
);


/** Compute the up-looking Cholesky factorization of a sparse matrix.
 *
 * @note This function assumes that `A` is symmetric and positive definite.
 *
 * @param A the matrix to factorize
 * @param S the Symbolic factorization of `A`
 *
 * @return the Cholesky factorization of `A`
 */
CSCMatrix chol(const CSCMatrix& A, const Symbolic& S);


/** Update the Cholesky factor for \f$ A = A + σ w w^T \f$.
 *
 * @param L  the Cholesky factor of A
 * @param σ  +1 for an update, or -1 for a downdate
 * @param C  the update vector, as the first column in a CSCMatrix
 * @param parent  the elimination tree of A
 *
 * @return L  the updated Cholesky factor of A
 */
CSCMatrix& chol_update(
    CSCMatrix& L,
    int sigma,
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



}  // namespace cs

#endif // _DECOMPOSITION_H_

//==============================================================================
//==============================================================================
