//==============================================================================
//     File: lu.h
//  Created: 2025-03-19 12:32
//   Author: Bernie Roesler
//
//  Description: Declarations for LU decomposition.
//
//==============================================================================

#ifndef _CSPARSE_LU_H_
#define _CSPARSE_LU_H_

#include <span>
#include <vector>

#include "types.h"
#include "utils.h"  // pvec, ipvec
#include "csc.h"


namespace cs {


/** Symbolic LU decomposition return struct (see: cs_symbolic aka css) */
struct SymbolicLU
{
    std::vector<csint> q;  ///< fill-reducting column permutation

    csint lnz,  ///< # entries in L
          unz;  ///< # entries in U
};


/** Numeric LU decomposition return struct (see: cs_numeric aka csn) */
struct LUResult
{
    CSCMatrix L,               ///< lower triangular matrix
              U;               ///< upper triangular matrix
    std::vector<csint> p_inv,  ///< row permutation of A
                       q;      ///< column permutation of A

    /** Solve the linear system Ax = b in-place.
     *
     * See also: Davis, Exercise 6.1.
     *
     * @param b[in,out]  right-hand side vector on input, solution on output.
     */
    void solve_inplace(std::span<double> b) const; 

    /** Solve the linear system A^T x = b.
     *
     * See: Davis, Exercise 6.1.
     *
     * @param b  the right-hand side vector
     *
     * @return x  the solution vector
     */
    std::vector<double> tsolve(const std::vector<double>& b) const; 
};


/** Compute the symolic LU decomposition of A.
 *
 * This function computes the column permutation `q` and the non-zero counts of
 * the matrices `L` and `U` in the LU decomposition of `A`. If `order` is
 * `AMDOrder::APlusAT`, the function computes the symbolic Cholesky
 * factorization of `A + A^T`. Use this estimate when the matrix is nearly
 * structrually symmetric.
 *
 * @param A  the matrix to factorize
 * @param order  the ordering method to use
 * @param qr_bound  if true, use the QR factorization bound for the non-zero
 *       counts of L and U. This option is only applicable when `order` is
 *       not `AMDOrder::APlusAT`.
 * @param alpha  the scaling factor for the optimistic estimate of the number
 *       of the non-zero counts of L and U. This option is only applicable when
 *       `order` is not `AMDOrder::APlusAT` and `qr_bound` is false.
 *
 * @return the symbolic factorization
 */
SymbolicLU slu(
    const CSCMatrix& A,
    AMDOrder order=AMDOrder::Natural,
    bool qr_bound=false,
    double alpha=1.0
);


/** Compute the numeric LU decomposition of A, such that \f$PA = LU\f$.
 *
 * The implementation of `cs_lu` from CSparse (Davis, pp 86-87). This function
 * does not support non-square matrices or singular matrices, and does not do
 * column pivoting or symmetric pruning. 
 *
 * See `cs::lu` and related functions `cs::lu_col` and `cs::lu_symprune` for
 * more general LU decompositions per the Davis exercises.
 *
 * @param A  the matrix to factorize
 * @param S  the symbolic factorization
 * #param tol  the tolerance for pivoting. If `tol` is 1.0, partial pivoting is
 *        used. If `tol` is less than 1.0, diagonal pivoting is used. 
 *        See: 
 *          - MATLAB: `thresh` in `lu`, or `options.thresh` in `ilu`.
 *          - Python: `diag_pivot_thresh` argument to `scipy.sparse.linalg.splu`.
 *
 * @return the LU decomposition
 */
LUResult lu_original(const CSCMatrix& A, const SymbolicLU& S, double tol=1.0);


/** Compute the numeric LU decomposition of A, such that \f$PAQ = LU\f$.
 *
 * This function computes the LU decomposition of `A` using the symbolic
 * factorization `S`.
 *
 * @param A  the matrix to factorize
 * @param S  the symbolic factorization
 * @param tol  the tolerance for pivoting. If `tol` is 1.0, partial pivoting is
 *        used. If `tol` is less than 1.0, diagonal pivoting is used.
 *        See: 
 *          - MATLAB: `thresh` in `lu`, or `options.thresh` in `ilu`.
 *          - Python: `diag_pivot_thresh` argument to `scipy.sparse.linalg.splu`.
 * @param col_tol  the column tolerance for pivoting. If `col_tol` is 0.0, then
 *        no column pivoting will occur. If `col_tol` is greater than 0.0, then
 *        a column will be pivoted to the end of the matrix if the absolute
 *        value of the pivot candidate is less than `col_tol`.
 *
 * @return the LU decomposition
 */
LUResult lu(
    const CSCMatrix& A,
	const SymbolicLU& S,
	double tol=1.0,
	double col_tol=0.0
);


/** Reallocate memory for L or U in the LU decomposition.
 *
 * This function reallocates memory for the L or U matrix in the LU
 * decomposition. It is used to ensure that the matrix has enough space for
 * the non-zero entries. It begins with an overly conservative estimate of the
 * number of non-zero entries, and then halves the size until it finds a 
 * size that can be allocated, but is at least the minimum size.
 *
 * @param R  the matrix to reallocate
 * @param k  the current column of the matrix
 * @param lower  true if reallocating L, false if reallocating U
 *
 * @throws std::bad_alloc if memory cannot be allocated.
 */
void lu_realloc(CSCMatrix& R, csint k, bool lower);


/** Compute the numeric LU decomposition of A with known sparsity pattern.
 *
 * See: Davis, Exercise 6.4.
 *
 * This function assumes that the sparsity pattern of `A` is the same as that
 * used to compute the numeric factorization `R`, and the symbolic factorization
 * `S`. It uses the same pivot permutation.
 *
 * @param A  the matrix to factorize
 * @param N  the numeric factorization of the sparsity pattern of A, assumed to
 *        be computed in a prior call to cs::lu.
 * @param S  the symbolic factorization of A
 *
 * @return the LU decomposition
 */
LUResult relu(const CSCMatrix& A, const LUResult& R, const SymbolicLU& S);


/** Compute the numeric LU decomposition of A, such that \f$PA = LU\f$.
 *
 * See: Davis, Exercise 6.7.
 *
 * This function computes the LU decomposition of `A` using the symbolic
 * factorization `S`. The numeric factorization is computed using Crout's
 * method, where each step computes one column of L and one row of U.
 *
 * @param A  the matrix to factorize
 * @param S  the symbolic factorization
 *
 * @return the LU decomposition
 */
LUResult lu_crout(const CSCMatrix& A, const SymbolicLU& S);


/** Incomplete LU decomposition using a drop threshold and pivoting.
 *
 * See: Davis, Exercise 6.13, and MATLAB `ilu` with option `type = 'ilutp'`.
 *
 * This function computes the incomplete LU decomposition of `A` with drop
 * tolerance `drop_tol` and pivoting tolerance `tol`. The function uses the
 * symbolic factorization `S` to compute the LU decomposition.
 *
 * Entries of U and L are dropped if their absolute value is less than
 * `drop_tol`,
 * as opposed to MATLAB, which compares entries to 
 *   `droptol * norm(A(:, j))` (aka the 2-norm) for each column `j`,
 * or SuperLU (used in `scipy.sparse.linalg.splu`, which compares entries to 
 *   `drop_tol * norm(A[:, j], np.inf)` (aka the pivot) for each column, such
 *   that `0 <= drop_tol <= 1`.
 *
 * @param A  the matrix to factorize
 * @param S  the symbolic factorization of A
 * @param drop_tol  the drop tolerance for the incomplete factorization.
 * @param tol  the tolerance for pivoting. If `tol` is 1.0, partial pivoting is
 *        used. If `tol` is less than 1.0, diagonal pivoting is used. 
 *        See: 
 *          - MATLAB: `thresh` in `lu`, or `options.thresh` in `ilu`.
 *          - Python: `diag_pivot_thresh` argument to `scipy.sparse.linalg.splu`.
 *
 * @return the LU decomposition
 */
LUResult ilutp(
    const CSCMatrix& A,
    const SymbolicLU& S,
    double drop_tol=0.0,
    double tol=1.0
);


/** Incomplete LU decomposition with zero fill-in.
 *
 * See: Davis, Exercise 6.13, and MATLAB `ilu` with option `type = 'nofill'`.
 *
 * This function computes the incomplete LU decomposition of `A` with zero
 * fill-in, *e.g.* using the same non-zero pattern as `A`. The function uses the
 * symbolic factorization `S` to compute the LU decomposition.
 *
 * No pivoting is performed.
 *
 * @param A  the matrix to factorize
 * @param S  the symbolic factorization of A
 *
 * @return the LU decomposition
 *
 * @throws std::runtime_error if the matrix is not square, or if the matrix is
 *         singular (has a zero on the diagonal).
 */
LUResult ilu_nofill(
    const CSCMatrix& A,
    const SymbolicLU& S
);


}  // namespace cs

#endif  // _CSPARSE_LU_H_

//==============================================================================
//==============================================================================
