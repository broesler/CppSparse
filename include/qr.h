//==============================================================================
//     File: qr.h
//  Created: 2025-02-11 12:41
//   Author: Bernie Roesler
//
//  Description: Declarations for QR decomposition using Householder
//      reflections and Givens rotations.
//
//==============================================================================

#ifndef _CSPARSE_QR_H_
#define _CSPARSE_QR_H_

#include <span>
#include <vector>

#include "csc.h"
#include "types.h"


namespace cs {


/** Householder reflection return struct. */
struct Householder {
    std::vector<double> v;  ///< the Householder vector
    double beta;            ///< the scaling factor
    double s;               ///< the first element of v
};


/** Symbolic QR decomposition return struct (see: cs_symbolic aka css) */
struct SymbolicQR
{
    std::vector<csint> p_inv,     ///< inverse row permutation
                       q,         ///< fill-reducting column permutation
                       parent,    ///< elimination tree
                       leftmost;  ///< leftmost[i] = min(find(A(i,:)))

    csint m2,   ///< # of rows for QR, after adding fictitious rows
          vnz,  ///< # entries in V
          rnz;  ///< # entries in R
};


/** Numeric QR decomposition return struct. */
struct QRResult {
    CSCMatrix V;                  ///< the Householder vectors
    std::vector<double> beta;     ///< the scaling factors
    CSCMatrix R;                  ///< the upper triangular matrix
    std::vector<csint> p_inv, q;  ///< row and column permutations
};


/** Compute the Householder reflection matrix for a given vector.
 *
 * The Householder reflection matrix is defined as
 * \f[
 *      H = I - \beta v v^T
 * \f]
 * where \f$ \beta = 2 / (v^T v) \f$.
 *
 * The result is defined such that applying the Householder reflection gives
 * \f$ Hx = s e_1 \f$, where \f$ s = \pm \|x\|_2 \f$.
 *
 * The choice of sign is determined to be consistent with the LAPACK DLARFG
 * subroutine. When \f$ x = \alpha e_1 \f$, \f$ \beta = 0 \f$ so that 
 * \f$ Hx = x \f$. Otherwise, \f$ 1 \le \beta \le 2 \f$ and 
 * \f$ s = -\text{sign}(x_1) \|x\| \f$. The Householder vector is chosen as
 * \f$ v = x + \text{sign}(x_1) \|x\| e_1 \f$
 * and then normalized such that \f$ v_1 = 1 \f$.
 *
 * See: LAPACK DLARFG, and Algorithm 5.1.1, Golub & Van Loan, 3rd ed.
 *
 * @param x  the input vector, may be a subspan of a larger vector
 *
 * @return beta  the scaling factor
 * @return v  the Householder vector
 * @return s  the first element of v, which is guaranteed to be
 *         \f$ \pm \|x\|_2 \f$
 */
Householder house(std::span<const double> x);


/** Apply a Householder reflection to a dense vector `x` with a sparse `v`.
 *
 * The Householder reflection is applied as
 * \f[
 *     Hx = x - v \beta v^T x = s e_1
 * \f]
 * where \f$ s = \pm \|x\|_2 \f$, `v` is the Householder vector and `beta` is
 * the scaling factor.
 *
 * @param V  a CSCMatrix with a single column containing the Householder vector
 * @param j  the column index of the Householder vector in `V`
 * @param beta  the scaling factor
 * @param[in,out] x  the dense vector to which to apply the reflection. This
 *        vector will be overwritten with the result.
 *
 * @return Hx  the result of applying the Householder reflection to `x`
 */
std::vector<double>& happly(
    const CSCMatrix& V,
	csint j,
	double beta,
	std::vector<double>& x
);


/** Compute the leftmost non-zero row index of each row in `A`.
 *
 * @param A  a CSCMatrix
 *
 * @return leftmost  the leftmost non-zero row index of each row in `A`.
 */
std::vector<csint> find_leftmost(const CSCMatrix& A);


/** Compute the column counts of the matrix V containing Householder vectors.
 *
 * This function also computes a row permutation vector `S.p_inv`, so that the
 * diagonal entries of PA are all structurally non-zero.
 *
 * If `A` is structurally rank-deficient, then this function adds fictitious
 * rows to `A` to make it structurally full rank. The total row count including
 * these fictitious rows is stored in `S.m2`.
 *
 * @note `S.parent` is expected to have been computed by `cs::etree`, and
 * `S.leftmost` is expected to have been computed by `cs::find_leftmost`. Only
 * `p_inv`, `lnz`, and `m2` values are updated by this function.
 *
 * @param A  the CSCMatrix that will be decomposed
 * @param[in,out] S  the symbolic QR decomposition of A.
 */
void vcount(const CSCMatrix& A, SymbolicQR& S);


/** Perform symbolic analysis for the QR decomposition of a matrix.
 *
 * This function calls `vcount` to compute the column counts of the matrix `V`;
 * the total non-zeros in `V`, `S.lnz`; the row permutation vector `S.p_inv`;
 * the leftmost row index of each row in `A`, `S.leftmost`; and the total row
 * count (including fictitious rows), `S.m2`.
 *
 * It also computes the elimination tree of the matrix `A.T @ A`, `S.parent`;
 * a column permutation vector `S.q` per the desired `order`.
 *
 * Note that for a matrix `A` with `M` rows and `N` columns, if `M < N`, then 
 * `sqr` will only compute the symbolic analysis of the first `M` columns of
 * `A`. The numeric factorization in `cs::qr` will compute the remaining `N - M`
 * columns of `R`. This method is more efficient than using `S.m2` ficticious
 * rows when `M << N`.
 *
 * @param A  the matrix to factorize
 * @param order  the ordering method to use:
 *       - 0: natural ordering
 *       - 1: amd(A + A.T)
 *       - 2: amd(A.T * A) with no dense rows
 *       - 3: amd(A.T * A)
 * @param use_postorder  if true, postorder the matrix in addition to the AMD
 *        (or natural) ordering. See: Davis, Exercise 5.5.
 *
 * @return the symbolic factorization
 */
SymbolicQR sqr(
    const CSCMatrix& A,
    AMDOrder order=AMDOrder::Natural,
    bool use_postorder=false
);


/** Perform the symbolic QR decomposition of a matrix.
 *
 * See: Davis, Exercise 5.1.
 *
 * Note that for a matrix `A` with `M` rows and `N` columns, if `M < N`, then 
 * `sqr` will only compute the symbolic analysis of the first `M` columns of
 * `A`, and the `QRResult` from `symbolic_qr` will only contain the first `M`
 * columns of `R`. The numeric factorization in `cs::reqr` will compute the
 * remaining `N - M` columns of `R`. This method is more efficient than using
 * `S.m2` ficticious rows when `M << N`.
 *
 * @param A  the matrix to factorize
 * @param S  the symbolic analysis of A
 *
 * @return the symbolic factorization
 */
QRResult symbolic_qr(const CSCMatrix& A, const SymbolicQR& S);


/** Perform the numeric QR decomposition of a matrix.
 *
 * This function computes the Householder vectors `V`, the scaling factors
 * `beta`, and the upper triangular matrix `R` of the full QR decomposition of `A`.
 *
 * If the size of A is `M`-by-`N`, with `M >= N`, then `V` is `M`-by-`N`, `beta`
 * is length `N`, and `R` is `M`-by-`N`. 
 *
 * To form the full `Q` matrix, apply the Householder vectors to the identity
 * matrix `I_M` to form `Q = I_M - V @ diag(beta) @ V.T`,
 * using the python function `csparse.apply_qright(V, beta)`.
 *
 * This process forms the "full" QR decomposition with `Q` size `M`-by-`M`. If
 * `M > N`, `R` will have `M - N` rows of zeros at the bottom.
 *
 * @param A  the matrix to factorize
 * @param S  the symbolic analysis of A
 *
 * @return the numeric factorization
 */
QRResult qr(const CSCMatrix& A, const SymbolicQR& S);


/** Perform the numeric QR decomposition of a matrix, given the non-zero pattern
 * of V and R.
 *
 * See: Davis, Exercise 5.3.
 *
 * @param A  the matrix to factorize
 * @param S  the symbolic analysis of A
 * @param[in,out] res the symbolic QR decomposition with the non-zero pattern of
 *        V and R
 *
 * @return the numeric factorization
 */
void reqr(const CSCMatrix& A, const SymbolicQR& S, QRResult& res);


/** Apply \f$ Q \f$ to a dense vector `y`.
 *
 * This function computes \f$ x = H_1 \dots H_N y = Q y \f$ where 
 * `Q` is represented by the Householder vectors `V`, the scaling factors
 * `beta`.
 *
 * @param V  the Householder vectors
 * @param beta  the scaling factors
 * @param x[in,out]  On input, the dense vector to which to apply the
 *        transformation. On output, the result of applying the transformation.
 *
 * @see cs::happly, cs::qrsol
 */
void apply_qleft(
    const CSCMatrix& V,
    const std::vector<double>& beta,
    std::vector<double>& x
);


/** Apply \f$ Q^T \f$ to a dense vector `y`.
 *
 * This function computes \f$ x = H_N \dots H_1 y = Q^T y \f$ where 
 * `Q` is represented by the Householder vectors `V`, the scaling factors
 * `beta`.
 *
 * @param V  the Householder vectors
 * @param beta  the scaling factors
 * @param x[in,out]  On input, the dense vector to which to apply the
 *        transformation. On output, the result of applying the transformation.
 *
 * @see cs::happly, cs::qrsol
 */
void apply_qtleft(
    const CSCMatrix& V,
    const std::vector<double>& beta,
    std::vector<double>& x
);


/** Apply \f$ Q^T \f$ to a sparse matrix `Y`.
 *
 * This function computes \f$ X = H_N \dots H_1 P Y = Q^T Y \f$ where 
 * `Q` is represented by the Householder vectors `V`, the scaling factors
 * `beta`, and the row permutation vector `p_inv`. To obtain \f$ Q^T \f$ itself,
 * pass `Y` as the identity matrix.
 *
 * @param V  the Householder vectors
 * @param beta  the scaling factors
 * @param p_inv  the row permutation vector
 * @param Y  the CSCMatrix to which to apply the transformation
 *
 * @return X  the result of applying the transformation to `Y`
 *
 * @see cs::happly, cs::qrsol
 */
CSCMatrix apply_qtleft(
    const CSCMatrix& V,
    const std::vector<double>& beta,
    const std::vector<csint>& p_inv,
    const CSCMatrix& Y
);


}  // namespace cs

#endif  // _CSPARSE_QR_H_

//==============================================================================
//==============================================================================
