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
#include <span>
#include <vector>

#include "csc.h"
#include "types.h"


namespace cs {

struct SparseSolution {
    std::vector<csint> xi;  // non-zero indices of x
    std::vector<double> x;  // (N,) dense solution vector
};


struct TriPerm {
    std::vector<csint> p_inv, q_inv, p_diags;
};


/** Exception raised when a matrix is *not* permuted triangular.
 */
class PermutedTriangularMatrixError : public std::runtime_error {
public:
    PermutedTriangularMatrixError(const std::string& msg)
        : std::runtime_error(msg) {}
};


struct QRSolveResult {
    std::vector<double> x;  // solution vector
    std::vector<double> r;  // residual b - A * x
    double rnorm;           // residual 2-norm
};


//------------------------------------------------------------------------------
//        Triangular Matrix Solutions
//------------------------------------------------------------------------------

namespace detail {

/** Solve a triangular linear system \f$ Tx = B \f$ for multiple RHS columns.
 *
 * @tparam InplaceTriSolve  a function that performs an in-place triangular
 *         solve on a single RHS vector.
 *
 * @param L  a triangular matrix
 * @param B  a dense matrix with multiple RHS columns, stored column-wise
 * @param inplace_solver  a function that performs an in-place triangular
 *         solve on a single RHS vector.
 *
 * @return X  the solution matrix with multiple columns, stored column-wise
 */
template <typename InplaceTriSolve>
std::vector<double> trisolve_dense(
    const CSCMatrix& L,
    const std::vector<double>& B,
    InplaceTriSolve inplace_solver
)
{
    auto [M, N] = L.shape();
    csint MxK = static_cast<csint>(B.size());

    if (MxK % M != 0) {
        throw std::runtime_error("RHS vector size is not a multiple of matrix rows!");
    }

    csint K = MxK / M;            // number of RHS columns
    std::vector<double> X = B;    // NOTE only works if M >= N
    std::span<double> X_span(X);  // view onto X

    for (csint k = 0; k < K; k++) {
        auto X_k = X_span.subspan(k * N, N);
        inplace_solver(L, X_k);
    }

    return X;
};


/** Solve a triangular linear system \f$ Tx = B \f$ for multiple RHS columns.
 *
 * @tparam Lower  True if input is lower triangular, otherwise upper.
 *
 * @param L  a triangular matrix
 * @param B  a sparse matrix with multiple RHS columns
 *
 * @return X  the solution matrix with multiple columns, stored column-wise
 */
template <bool Lower>
std::vector<double> trisolve_sparse(const CSCMatrix& L, const CSCMatrix& B)
{
    auto [M, N] = L.shape();
    csint K = B.shape()[1];

    csint Nx = std::max(M, N);  // enough space for non-square solutions
    std::vector<double> X(Nx * K);
    std::span<double> X_span(X);

    for (csint k = 0; k < K; k++) {
        auto X_k = X_span.subspan(k * Nx, Nx);
        // TODO rewrite spsolve to take xi and x vectors as inputs
        SparseSolution sol = spsolve(L, B, k, std::nullopt, Lower);
        for (auto& i : sol.xi) {
            X_k[i] = sol.x[i];
        }
    }

    return X;
}

}  // namespace detail

/** Forward solve a lower-triangular system \f$ Lx = b \f$, in-place.
 *
 * @note This function assumes that the diagonal entry of `L` is always
 * present and is the first entry in each column. Otherwise, the row
 * indices in each column of `L` may appear in any order.
 *
 * @param L  a lower-triangular matrix
 * @param x[in,out]  RHS vector on input, solution on output.
 */
void lsolve_inplace(const CSCMatrix& L, std::span<double> x);


/** Forward solve a lower-triangular system \f$ Lx = b \f$.
 *
 * @note This function assumes that the diagonal entry of `L` is always
 * present and is the first entry in each column. Otherwise, the row
 * indices in each column of `L` may appear in any order.
 *
 * @param L  a lower-triangular matrix
 * @param B  the RHS matrix (in column-major order if dense)
 *
 * @return x  the solution matrix, in column-major order.
 */
std::vector<double> lsolve(const CSCMatrix& L, const std::vector<double>& B);
std::vector<double> lsolve(const CSCMatrix& L, const CSCMatrix& B);


/** Backsolve a lower-triangular system \f$ L^Tx = b \f$.
 *
 * @note This function assumes that the diagonal entry of `L` is always
 * present and is the first entry in each column. Otherwise, the row
 * indices in each column of `L` may appear in any order.
 *
 * @param L  a lower-triangular matrix
 * @param x[in,out]  RHS vector on input, solution on output.
 */
void ltsolve_inplace(const CSCMatrix& L, std::span<double> x);


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
 * @param x[in,out]  RHS vector on input, solution on output.
 */
void usolve_inplace(const CSCMatrix& U, std::span<double> x);


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
std::vector<double> usolve(const CSCMatrix& U, const CSCMatrix& B);


/** Forward solve an upper-triangular system \f$ U^T x = b \f$.
 *
 * @note This function assumes that the diagonal entry of `U` is always present
 * and is the last entry in each column. Otherwise, the row indices in each
 * column of `U` may appear in any order.
 *
 * @param U  an upper-triangular matrix
 * @param x[in,out]  RHS vector on input, solution on output.
 */
void utsolve_inplace(const CSCMatrix& U, std::span<double> x);


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
 * @param x[in,out]  the RHS vector on input, solution on output.
 */
void lsolve_inplace_opt(const CSCMatrix& A, std::span<double> x);


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
 * @param x[in,out]  the RHS vector on input, solution on output.
 */
void usolve_inplace_opt(const CSCMatrix& A, std::span<double> x);


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
 *
 * @throws PermutedTriangularMatrixError if L is not a permuted lower triangular
 * matrix.
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
 *
 * @throws PermutedTriangularMatrixError if U is not a permuted upper triangular
 * matrix.
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
 *
 * @throws PermutedTriangularMatrixError if L is not a permuted lower triangular
 * matrix.
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
 *
 * #throws PermutedTriangularMatrixError if U is not a permuted upper triangular
 * matrix.
 */
std::vector<double> usolve_cols(const CSCMatrix& U, const std::vector<double>& b);


/** Find the diagonal indices of a row-permuted lower triangular matrix.
 *
 * See: Davis, Exercise 3.3
 *
 * @param L  a permuted lower-triangular matrix
 *
 * @return p_diags  a vector of pointers to the indices of the diagonal entries.
 *
 * @throws PermutedTriangularMatrixError if L is not a permuted lower triangular
 * matrix.
 */
std::vector<csint> find_lower_diagonals(const CSCMatrix& L);


/** Find the diagonal indices of a row-permuted upper triangular matrix.
 *
 * See: Davis, Exercise 3.4
 *
 * @param U  a permuted upper-triangular matrix
 *
 * @return p_diags  a vector of pointers to the indices of the diagonal entries.
 *
 * @throws PermutedTriangularMatrixError if U is not a permuted upper triangular
 * matrix.
 */
std::vector<csint> find_upper_diagonals(const CSCMatrix& U);


/** Solve a row- and column-permuted triangular system P A Q x = b, for unknown
 * P and Q.
 *
 * See: Davis, Exercise 3.7
 *
 * @param A  a permuted triangular matrix
 * @param tri_perm  the permutation vectors of A, as returned by
 *        find_tri_permutation
 * @param b  a dense RHS vector, *not* permuted.
 * @param x  the dense solution vector, also *not* permuted.
 *
 * @see find_tri_permutation
 */
void tri_solve_perm_inplace(
    const CSCMatrix& A,
    const TriPerm& tri_perm,
    std::span<double> b,
    std::span<double> x
);


/** Solve a row- and column-permuted triangular system P A Q X = B, for unknown
 * P and Q.
 *
 * See: Davis, Exercise 3.7
 *
 * @param A  a permuted triangular matrix
 * @param b  a dense RHS matrix, *not* permuted, in column-major order.
 *
 * @return x  the dense solution matrix, also *not* permuted, in column-major
 *         order.
 *
 * @throws PermutedTriangularMatrixError if A is not a permuted triangular
 * matrix.
 */
std::vector<double> tri_solve_perm(
    const CSCMatrix& A,
    const std::vector<double>& B
);


/** Find the permutation vectors of a permuted triangular matrix.
 *
 * See: Davis, Exercise 3.7
 *
 * @param A  a permuted triangular matrix
 *
 * @return p_inv, q_inv  the inverse row and column permutation vectors.
 * @return p_diags  the pointers to the diagonal entries.
 *
 * @throws PermutedTriangularMatrixError if A is not a permuted triangular
 * matrix.
 */
TriPerm find_tri_permutation(const CSCMatrix& A);


/** Solve a triangular system \f$ Lx = b_k \f$ for column `k` of `B`,
 * where `L` and `B` are sparse.
 *
 * @note If `lower` is true, this function assumes that the diagonal entry of
 * `L` is always present and is the first entry in each column. Otherwise, the
 * row indices in each column of `L` may appear in any order.
 * If `lower` is false, the function assumes that the diagonal entry of `U` is
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
 * @param lower  If `lower` is true, the function solves \f$ Lx = b_k`, otherwise it
 *        solves \f$ Ux = b_k \f$.
 *
 * @return res  a struct containing:
 *         * xi the row indices of the non-zero entries in `x`.
 *         * x  the numerical values of the solution vector, as a dense vector.
 */
SparseSolution spsolve(
    const CSCMatrix& A,
    const CSCMatrix& B,
    csint k,
    OptionalVectorRef<csint> p_inv_ref=std::nullopt,
    bool lower=true
);


namespace detail {

/** Compute the reachability indices of a column `k` in a sparse matrix `B`,
 * given a sparse matrix `A` that defines the graph.
 *
 * This function is a recursive version of `cs::reach`. It is not intended to be
 * used directly, but rather as a demonstration.
 *
 * @param A  a sparse system matrix
 * @param B  a sparse matrix containing the RHS in column 0.
 *
 * @return xi  the row indices of the non-zero entries in `x`, in topological
 *         order of the graph.
 */
std::vector<csint> reach_r(const CSCMatrix& A, const CSCMatrix& B);


/** Perform depth-first search on the matrix graph.
 *
 * This function is a recursive version of `cs::dfs`. It is not intended to be
 * used directly, but rather as a demonstration.
 *
 * @param A  a sparse matrix
 * @param j  the starting node
 * @param marked  a boolean vector of length `N` that marks visited nodes
 * @param[in,out] xi  the row indices of the non-zero entries in `x`. This
 *       vector is used as a stack to store the output. It should not be
 *       initialized, other than by a previous call to `dfs`.
 * @param pstack  memory for the pause stack, reserved to length `N`.
 *
 * @return xi  a reference to the row indices of the non-zero entries in `x`.
 */
std::vector<csint>& dfs_r(
    const CSCMatrix& A,
    csint j,
    std::vector<char>& marked,
    std::vector<csint>& xi
);

}  // namespace detail


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
    OptionalVectorRef<csint> p_inv_ref=std::nullopt
);


/** Perform depth-first search on the matrix graph.
 *
 * @param A  a sparse matrix
 * @param j  the starting node
 * @param marked  a boolean vector of length `N` that marks visited nodes
 * @param[in,out] xi  the row indices of the non-zero entries in `x`. This
 *       vector is used as a stack to store the output. It should not be
 *       initialized, other than by a previous call to `dfs`.
 * @param pstack  memory for the pause stack, reserved to length `N`.
 * @param rstack  memory for the recursion stack, reserved to length `N`.
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
    std::vector<csint>& pstack,
    std::vector<csint>& rstack,
    OptionalVectorRef<csint> p_inv_ref=std::nullopt
);


// -----------------------------------------------------------------------------
//        Cholesky Factorization Solutions
// -----------------------------------------------------------------------------
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
 * @param A  (N, N) a symmetric, positive-definite matrix
 * @param B  (N, K) a dense matrix, in column-major format
 * @param order  the fill-reducing ordering of the matrix to compute
 *
 * @return x  (N, K) the dense solution matrix
 *
 * @see cs_cholsol
 */
std::vector<double> chol_solve(
    const CSCMatrix& A,
    const std::vector<double>& B,
    AMDOrder order=AMDOrder::Natural
);


/** Solve the system AX = B using the Cholesky factorization.
 *
 * @param A  (N, N) a symmetric, positive-definite matrix
 * @param B  (N, K) a sparse matrix
 * @param order  the fill-reducing ordering of the matrix to compute
 *
 * @return X  (N, K) the dense solution matrix
 *
 * @see cs_cholsol
 */
std::vector<double> chol_solve(
    const CSCMatrix& A,
    const CSCMatrix& B,
    AMDOrder order=AMDOrder::Natural
);


// -----------------------------------------------------------------------------
//         QR Factorization Solvers
// -----------------------------------------------------------------------------

/** Solve the system AX = B using the QR factorization.
 *
 * This method is useful for solving least-squares problems where the matrix `A`
 * is `M`-by-`N` and `M` > `N`. It can also be used to solve under-determined
 * systems where `M` < `N`. In the under-determined case, the solution is
 * the minimum-norm solution.
 *
 * @param A  (M, N) a sparse matrix
 * @param B  (M, K) a dense vector
 * @param order  the fill-reducing ordering of the matrix to compute
 *
 * @return res  a struct containing:
 *        * x  (N, K) the solution matrix
 *        * r  (M, K) the residual matrix (b - A * x)
 *        * rnorm  the residual 2-norm (in the flattened vector sense)
 *
 * @see cs_qrsol
 */
QRSolveResult qr_solve(
    const CSCMatrix& A,
    const std::vector<double>& B,
    AMDOrder order=AMDOrder::Natural
);


/** Solve the system AX = B using the QR factorization.
 *
 * This method is useful for solving least-squares problems where the matrix `A`
 * is `M`-by-`N` and `M` > `N`. It can also be used to solve under-determined
 * systems where `M` < `N`. In the under-determined case, the solution is
 * the minimum-norm solution.
 *
 * @param A  (M, N) the sparse system matrix
 * @param B  (M, K) the sparse RHS matrix
 * @param order  the fill-reducing ordering of the matrix to compute
 *
 * @return res  a struct containing:
 *        * x  (N, K) the solution matrix
 *        * r  (M, K) the residual matrix (b - A * x)
 *        * rnorm  the residual 2-norm (in the flattened vector sense)
 *
 * @see cs_qrsol
 */
QRSolveResult qr_solve(
    const CSCMatrix& A,
    const CSCMatrix& B,
    AMDOrder order=AMDOrder::Natural
);


// -----------------------------------------------------------------------------
//         LU Factorization Solutions
// -----------------------------------------------------------------------------

/** Solve a system \f$ A X = B \f$ using the LU factorization of `A`.
 *
 * See also: Davis, Exercise 6.1.
 *
 * @param A  (N, N) the sparse system matrix
 * @param B  (N, K) the dense RHS matrix in column-major format.
 * @param order  the fill-reducing ordering of the matrix to compute
 * @param tol  the tolerance for pivoting. If `tol` is 1.0, partial pivoting is
 *        used. If `tol` is less than 1.0, diagonal pivoting is used. 
 * @param ir_steps  the maximum number of iterative refinement steps to perform.
 *
 * @return X  (N, K) the dense solution matrix, in column-major format.
 */
std::vector<double> lu_solve(
    const CSCMatrix& A,
    const std::vector<double>& B,
    AMDOrder order=AMDOrder::Natural,
    double tol=1.0,
    csint ir_steps=0
);


/** Solve a system \f$ A X = B \f$ using the LU factorization of `A`.
 *
 * See also: Davis, Exercise 6.1.
 *
 * @param A  (N, N) the sparse system matrix
 * @param B  (N, K) the sparse RHS matrix in column-major format.
 * @param order  the fill-reducing ordering of the matrix to compute
 * @param tol  the tolerance for pivoting. If `tol` is 1.0, partial pivoting is
 *        used. If `tol` is less than 1.0, diagonal pivoting is used. 
 * @param ir_steps  the maximum number of iterative refinement steps to perform.
 *
 * @return X  (N, K) the dense solution matrix, in column-major format.
 */
std::vector<double> lu_solve(
    const CSCMatrix& A,
    const CSCMatrix& B,
    AMDOrder order=AMDOrder::Natural,
    double tol=1.0,
    csint ir_steps=0
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


/** Solve a sparse linear system AX = B.
 *
 * See: Davis, Exercise 8.1 and 8.10.
 *
 * This function mimics the behavior of MATLAB's `\` operator for sparse
 * matrices.
 *
 * @param A  (M, N) the sparse system matrix
 * @param b  (M, K) the dense RHS matrix
 *
 * @return x (N, K) the dense solution matrix
 */
std::vector<double> spsolve(const CSCMatrix& A, const std::vector<double>& B);


/** Solve a sparse linear system Ax = b.
 *
 * See: Davis, Exercise 8.1.
 *
 * This function mimics the behavior of MATLAB's `\` operator for sparse
 * matrices.
 *
 * @param A  (M, N) the sparse system matrix
 * @param b  (M, 1) the sparse RHS matrix
 *
 * @return x (N,) the dense solution vector
 */
std::vector<double> spsolve(const CSCMatrix& A, const CSCMatrix& b);



}  // namespace cs

#endif  // _SOLVE_H_

//==============================================================================
//==============================================================================
