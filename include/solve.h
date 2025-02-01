//==============================================================================
//     File: solve.h
//  Created: 2025-01-30 13:46
//   Author: Bernie Roesler
//
//  Description: Implement various matrix solvers.
//
//==============================================================================

#ifndef _CSPARSE_SOLVE_H_
#define _CSPARSE_SOLVE_H_

#include <vector>

#include "types.h"


namespace cs {

struct SparseSolution {
    std::vector<csint> xi;
    std::vector<double> x;
};


struct TriPerm {
    std::vector<csint> p_inv, q_inv, p_diags;
};


//------------------------------------------------------------------------------
//        Triangular Matrix Solutions
//------------------------------------------------------------------------------
/** Forward solve a lower-triangular system \f$ Lx = b \f$.
 *
 * @note This function assumes that the diagonal entry of `L` is always
 * present and is the first entry in each column. Otherwise, the row
 * indices in each column of `L` may appear in any order.
 *
 * @param L  a lower-triangular matrix
 * @param b  a dense vector
 *
 * @return x  the solution vector
 */
std::vector<double> lsolve(const CSCMatrix& L, const std::vector<double>& b);


/** Backsolve a lower-triangular system \f$ L^Tx = b \f$.
 *
 * @note This function assumes that the diagonal entry of `L` is always
 * present and is the first entry in each column. Otherwise, the row
 * indices in each column of `L` may appear in any order.
 *
 * @param L  a lower-triangular matrix
 * @param b  a dense vector
 *
 * @return x  the solution vector
 */
std::vector<double> ltsolve(const CSCMatrix& L, const std::vector<double>& b);


/** Backsolve an upper-triangular system \f$ Ux = b \f$.
 *
 * @note This function assumes that the diagonal entry of `U` is always
 * present and is the last entry in each column. Otherwise, the row
 * indices in each column of `U` may appear in any order.
 *
 * @param U  an upper-triangular matrix
 * @param b  a dense vector
 *
 * @return x  the solution vector
 */
std::vector<double> usolve(const CSCMatrix& U, const std::vector<double>& b);


/** Forward solve an upper-triangular system \f$ U^T x = b \f$.
 *
 * @note This function assumes that the diagonal entry of `U` is always present
 * and is the last entry in each column. Otherwise, the row indices in each
 * column of `U` may appear in any order.
 *
 * @param U  an upper-triangular matrix
 * @param b  a dense vector
 *
 * @return x  the solution vector
 */
std::vector<double> utsolve(const CSCMatrix& U, const std::vector<double>& b);


/** Forward solve a lower-triangular system \f$ Lx = b \f$, but
 * optimized for cache efficiency.
 *
 * See: Davis, Exercise 3.8
 *
 * @note This function assumes that the diagonal entry of `L` is always
 * present and is the first entry in each column. Otherwise, the row
 * indices in each column of `L` may appear in any order.
 *
 * @param L  a lower-triangular matrix
 * @param b  a dense vector
 *
 * @return x  the solution vector
 */
std::vector<double> lsolve_opt(const CSCMatrix& L, const std::vector<double>& b);


/** Backsolve an upper-triangular system \f$ Ux = b \f$, but optimized for cache
 * efficiency.
 *
 * @note This function assumes that the diagonal entry of `U` is always present
 * and is the last entry in each column. Otherwise, the row indices in each
 * column of `U` may appear in any order.
 *
 * @param U  an upper-triangular matrix
 * @param b  a dense vector
 *
 * @return x  the solution vector
 */
std::vector<double> usolve_opt(const CSCMatrix& U, const std::vector<double>& b);


/** Solve Lx = b with a row-permuted L. The permutation is unknown.
 *
 * See: Davis, Exercise 3.3
 *
 * @param L  a lower-triangular matrix
 * @param b  a dense RHS vector, *not* permuted.
 *
 * @return x  the dense solution vector, also *not* permuted.
 */
std::vector<double> lsolve_rows(const CSCMatrix& L, const std::vector<double>& b);


/** Solve Ux = b with a row-permuted U. The permutation is unknown.
 *
 * See: Davis, Exercise 3.4
 *
 * @param U  an upper-triangular matrix
 * @param b  a dense RHS vector, *not* permuted.
 *
 * @return x  the dense solution vector, also *not* permuted.
 */
std::vector<double> usolve_rows(const CSCMatrix& U, const std::vector<double>& b);


/** Solve Lx = b with a column-permuted L. The permutation is unknown.
 *
 * See: Davis, Exercise 3.5
 *
 * @param L  a lower-triangular matrix
 * @param b  a dense RHS vector, *not* permuted.
 *
 * @return x  the dense solution vector, also *not* permuted.
 */
std::vector<double> lsolve_cols(const CSCMatrix& L, const std::vector<double>& b);


/** Solve Ux = b with a column-permuted U. The permutation is unknown.
 *
 * See: Davis, Exercise 3.6
 *
 * @param U  an upper-triangular matrix
 * @param b  a dense RHS vector, *not* permuted.
 *
 * @return x  the dense solution vector, also *not* permuted.
 */
std::vector<double> usolve_cols(const CSCMatrix& U, const std::vector<double>& b);


/** Find the diagonal indices of a row-permuted lower triangular matrix.
 *
 * See: Davis, Exercise 3.3
 *
 * @param L  a permuted lower-triangular matrix
 *
 * @return p_diags  a vector of pointers to the indices of the diagonal entries.
 */
std::vector<csint> find_lower_diagonals(const CSCMatrix& L);


/** Find the diagonal indices of a row-permuted upper triangular matrix.
 *
 * See: Davis, Exercise 3.4
 *
 * @param U  a permuted upper-triangular matrix
 *
 * @return p_diags  a vector of pointers to the indices of the diagonal entries.
 */
std::vector<csint> find_upper_diagonals(const CSCMatrix& U);


/** Solve a row- and column-permuted triangular system P A Q x = b, for unknown
 * P and Q.
 *
 * See: Davis, Exercise 3.7
 *
 * @param A  a permuted triangular matrix
 * @param b  a dense RHS vector, *not* permuted.
 * @param is_upper  true if the matrix is upper triangular, false otherwise.
 *
 * @return x  the dense solution vector, also *not* permuted.
 */
std::vector<double> tri_solve_perm(
    const CSCMatrix& A,
    const std::vector<double>& b,
    bool is_upper=false
);


/** Find the permutation vectors of a permuted triangular matrix.
 *
 * See: Davis, Exercise 3.7
 *
 * @param A  a permuted triangular matrix
 *
 * @return p_inv, q_inv  the inverse row and column permutation vectors.
 * @return p_diags  the pointers to the diagonal entries.
 */
TriPerm find_tri_permutation(const CSCMatrix& A);


// TODO return `x` as a 1-column CSCMatrix?
/** Solve a triangular system \f$ Lx = b_k \f$ for column `k` of `B`,
 * where `L` and `B` are sparse.
 *
 * @note If `lo` is non-zero, this function assumes that the diagonal entry of
 * `L` is always present and is the first entry in each column. Otherwise, the
 * row indices in each column of `L` may appear in any order.
 * If `lo` is zero, the function assumes that the diagonal entry of `U` is
 * always present and is the last entry in each column.
 *
 * @param A  the sparse, triangular system matrix
 * @param B  the sparse RHS matrix
 * @param k  the column index of `B` to solve
 * @param lo  the lower bound of the diagonal entries of `G`. If `lo` is
 *        true, the function solves \f$ Lx = b_k`, otherwise it solves
 *        \f$ Ux = b_k \f$.
 *
 * @return xi the row indices of the non-zero entries in `x`.
 * @return x  the numerical values of the solution vector, as a dense vector.
 */
SparseSolution spsolve(
    const CSCMatrix& A,
    const CSCMatrix& B,
    csint k,
    bool lo=true
);


/** Compute the reachability indices of a column `k` in a sparse matrix `B`,
 * given a sparse matrix `A` that defines the graph.
 *
 * @param A  a sparse system matrix
 * @param B  a sparse matrix containing the RHS in column `k`
 * @param k  the column index of `B` containing the RHS
 *
 * @return xi  the row indices of the non-zero entries in `x`, in topological
 *         order of the graph.
 */
std::vector<csint> reach(const CSCMatrix& A, const CSCMatrix& B, csint k);


/** Perform depth-first search on the matrix graph.
 *
 * @param A  a sparse matrix
 * @param j  the starting node
 * @param marked  a boolean vector of length `N_` that marks visited nodes
 * @param[in,out] xi  the row indices of the non-zero entries in `x`. This
 *       vector is used as a stack to store the output. It should not be
 *       initialized, other than by a previous call to `dfs`.
 *
 * @return xi  a reference to the row indices of the non-zero entries in `x`.
 */
std::vector<csint>& dfs(
    const CSCMatrix& A,
    csint j,
    std::vector<bool>& marked,
    std::vector<csint>& xi
);


/** Solve \f$ Lx = b \f$ with sparse RHS `b`, where `L` is a lower-triangular
 * Cholesky factor.
 *
 * See: Davis, Exercise 4.3.
 *
 * @param L  a lower-triangular matrix from a Cholesky factorization. `L` must
 *        be in canonical format.
 * @param b  a sparse RHS vector, stored as an Nx1 CSCMatrix.
 * @param parent  the parent vector of the elimination tree of `L`. If not
 *        given, the function will compute it from `L`.
 *
 * @return xi  the row indices of the non-zero entries in `x`.
 * @return x  the solution vector, stored as a dense vector.
 */
SparseSolution chol_lsolve(
    const CSCMatrix& L,
    const CSCMatrix& b,
    std::vector<csint> parent = {}
);


/** Solve \f$ L^T x = b \f$ with sparse RHS `b`, where `L` is a lower-triangular
 * Cholesky factor.
 *
 * See: Davis, Exercise 4.4.
 *
 * @param L  a lower-triangular matrix from a Cholesky factorization. `L` must
 *        be in canonical format.
 * @param b  a sparse RHS vector, stored as an Nx1 CSCMatrix.
 * @param parent  the parent vector of the elimination tree of `L`. If not
 *        given, the function will compute it from `L`.
 *
 * @return xi  the row indices of the non-zero entries in `x`.
 * @return x  the solution vector, stored as a dense vector.
 */
SparseSolution chol_ltsolve(
    const CSCMatrix& L,
    const CSCMatrix& b,
    std::vector<csint> parent = {}
);


/** Find the topological order of the nodes in the elimination tree.
 *
 * @param b  a sparse matrix
 * @param parent  the parent vector of the elimination tree
 * @param forward  if true, return the topological order of the forward tree,
 *        (from lower nodes to higher nodes). If false, return the reverse
 *        topological order (from higher nodes to lower nodes). `forward=false`
 *        is useful for computing the solution to \f$ L^T x = b \f$.
 *
 * @return xi  the row indices of the non-zero entries in `x`, in topological
 *      order of the graph of `b`.
 */
std::vector<csint> topological_order(
    const CSCMatrix& b,
    const std::vector<csint>& parent,
    bool forward=true
);


}  // namespace cs

#endif  // _SOLVE_H_

//==============================================================================
//==============================================================================
