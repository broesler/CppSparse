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

#include <optional>
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
 * See: Davis, Exercise 3.8
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


/** Solve a triangular system \f$ Lx = b_k \f$ for column `k` of `B`,
 * where `L` and `B` are sparse.
 *
 * @note If `lo` is non-zero, this function assumes that the diagonal entry of
 * `L` is always present and is the first entry in each column. Otherwise, the
 * row indices in each column of `L` may appear in any order.
 * If `lo` is zero, the function assumes that the diagonal entry of `U` is
 * always present and is the last entry in each column.
 *
 * @note In the CSparse library, this function is only called within `cs_lu`
 *       using a pre-allocated dense `x` vector, since it is called in a loop
 *       over the columns of `A`. The dense `x` vector can then be indexed
 *       directly by the row indices stored in `xi`.
 *
 * @param A  the sparse, triangular system matrix
 * @param B  the sparse RHS matrix
 * @param k  the column index of `B` to solve
 * @param p_inv  the inverse permutation vector of the matrix `A`. If not given,
 *        A is taken in natural order.
 * @param lo  the lower bound of the diagonal entries of `G`. If `lo` is
 *        true, the function solves \f$ Lx = b_k`, otherwise it solves
 *        \f$ Ux = b_k \f$.
 *
 * @return res  a struct containing:
 *         * xi the row indices of the non-zero entries in `x`.
 *         * x  the numerical values of the solution vector, as a dense vector.
 */
SparseSolution spsolve(
    const CSCMatrix& A,
    const CSCMatrix& B,
    csint k,
    OptVectorRef<csint> p_inv_ref=std::nullopt,
    bool lo=true
);


/** Compute the reachability indices of a column `k` in a sparse matrix `B`,
 * given a sparse matrix `A` that defines the graph.
 *
 * @param A  a sparse system matrix
 * @param B  a sparse matrix containing the RHS in column `k`
 * @param k  the column index of `B` containing the RHS
 * @param p_inv  the inverse permutation vector of the matrix `A`. If not given,
 *        A is taken in natural order.
 *
 * @return xi  the row indices of the non-zero entries in `x`, in topological
 *         order of the graph.
 */
std::vector<csint> reach(
    const CSCMatrix& A,
    const CSCMatrix& B,
    csint k,
    OptVectorRef<csint> p_inv_ref=std::nullopt
);


/** Perform depth-first search on the matrix graph.
 *
 * @param A  a sparse matrix
 * @param j  the starting node
 * @param marked  a boolean vector of length `N_` that marks visited nodes
 * @param[in,out] xi  the row indices of the non-zero entries in `x`. This
 *       vector is used as a stack to store the output. It should not be
 *       initialized, other than by a previous call to `dfs`.
 * #param p_inv  the inverse permutation vector of the matrix `A`. If not given,
 *        A is taken in natural order.
 *
 * @return xi  a reference to the row indices of the non-zero entries in `x`.
 */
std::vector<csint>& dfs(
    const CSCMatrix& A,
    csint j,
    std::vector<char>& marked,
    std::vector<csint>& xi,
    OptVectorRef<csint> p_inv_ref=std::nullopt
);


// -----------------------------------------------------------------------------
//        Cholesky Factorization Solutions
// -----------------------------------------------------------------------------
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


/** Solve the system Ax = b using the Cholesky factorization.
 *
 * @param A  a symmetric, positive-definite matrix
 * @param b  a dense vector
 * @param order  the fill-reducing ordering of the matrix to compute
 *
 * @return x  the dense solution vector
 *
 * @see cs_cholsol
 */
std::vector<double> chol_solve(
    const CSCMatrix& A,
    const std::vector<double>& b,
    AMDOrder order=AMDOrder::Natural
);


// -----------------------------------------------------------------------------
//         QR Factorization Solvers
// -----------------------------------------------------------------------------

/** Solve the system Ax = b using the QR factorization.
 *
 * This method is useful for solving least-squares problems where the matrix `A`
 * is `M`-by-`N` and `M` > `N`. It can also be used to solve under-determined
 * systems where `M` < `N`. In the under-determined case, the solution is
 * the minimum-norm solution.
 *
 * @param A  a matrix
 * @param b  a dense vector
 * @param order  the fill-reducing ordering of the matrix to compute
 *
 * @return x  the solution vector
 *
 * @see cs_qrsol
 */
std::vector<double> qr_solve(
    const CSCMatrix& A,
    const std::vector<double>& b,
    AMDOrder order=AMDOrder::Natural
);

// -----------------------------------------------------------------------------
//         LU Factorization Solutions
// -----------------------------------------------------------------------------

/** Solve a system \f$ A x = b \f$ using the LU factorization of `A`.
 *
 * See also: Davis, Exercise 6.1.
 *
 * @param A  a square matrix
 * @param b  a dense vector
 * @param order  the fill-reducing ordering of the matrix to compute
 * #param tol  the tolerance for pivoting. If `tol` is 1.0, partial pivoting is
 *        used. If `tol` is less than 1.0, diagonal pivoting is used. 
 *
 * @return x  the solution vector
 */
std::vector<double> lu_solve(
    const CSCMatrix& A,
    const std::vector<double>& b,
    AMDOrder order=AMDOrder::Natural,
    double tol=1.0
);


/** Solve a system \f$ A^T x = b \f$ using the LU factorization of `A`.
 *
 * See: Davis, Exercise 6.1.
 *
 * @param A  a square matrix
 * @param b  a dense vector
 * @param order  the fill-reducing ordering of the matrix to compute
 * #param tol  the tolerance for pivoting. If `tol` is 1.0, partial pivoting is
 *        used. If `tol` is less than 1.0, diagonal pivoting is used. 
 *
 * @return x  the solution vector
 */
std::vector<double> lu_tsolve(
    const CSCMatrix& A,
    const std::vector<double>& b,
    AMDOrder order=AMDOrder::Natural,
    double tol=1.0
);


/** Estimate the 1-norm of the *inverse* of a sparse matrix.
 *
 * See: Davis, Exercise 6.15.
 *
 * @param res  the LU factorization result of a matrix `A`
 *
 * @return norm  the estimated 1-norm of the *inverse* of `A`
 */
double norm1est_inv(const LUResult& res);


/** Estimate the 1-norm condition number of a sparse matrix.
 *
 * The condition number for a non-symmetric matrix is defined as:
 * \f$ \kappa_p(A) = ||A||_p ||A^{-1}||_p \f$.
 * This function chooses \f$ p = 1 \f$.
 *
 * See: Davis, Exercise 6.15.
 *
 * @param A  a real, square matrix
 *
 * @return cond  the estimated 1-norm condition number of `A`
 */
double cond1est(const CSCMatrix& A);


}  // namespace cs

#endif  // _SOLVE_H_

//==============================================================================
//==============================================================================
