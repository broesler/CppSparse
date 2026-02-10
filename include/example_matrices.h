//==============================================================================
//     File: example_matrices.h
//  Created: 2025-03-20 15:14
//   Author: Bernie Roesler
//
//  Description: Declarations of example matrices.
//
//==============================================================================

#pragma once

#include "types.h"

namespace cs {


/** Define the 4x4 matrix from Davis Equation (2.1) [p 7--8].
 *
 *  A = [[4.5,   0, 3.2,   0],
 *       [3.1, 2.9,   0, 0.9],
 *       [  0, 1.7,   3,   0],
 *       [3.5, 0.4,   0,   1]]
 *
 * See: Davis, Timothy A. "Direct Methods for Sparse Linear Systems",
 *      Eqn (2.1), p. 7--8.
 *
 * @return A  A 4x4 matrix in COO format.
 */
COOMatrix davis_example_small();


/** Define the 11x11 matrix in Davis, Figure 4.2, p 39.
 *
 * This matrix is sparse and symmetric positive definite. We arbitrarily assign
 * the diagonal to the 0-based index values + 10, and the off-diagonals to 1.
 *
 *  A = [[10,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0],
 *       [ 0, 11,  1,  0,  0,  0,  0,  1,  0,  0,  0],
 *       [ 0,  1, 12,  0,  0,  0,  0,  0,  0,  1,  1],
 *       [ 0,  0,  0, 13,  0,  1,  0,  0,  0,  1,  0],
 *       [ 0,  0,  0,  0, 14,  0,  0,  1,  0,  0,  1],
 *       [ 1,  0,  0,  1,  0, 15,  0,  0,  1,  1,  0],
 *       [ 1,  0,  0,  0,  0,  0, 16,  0,  0,  0,  1],
 *       [ 0,  1,  0,  0,  1,  0,  0, 17,  0,  1,  1],
 *       [ 0,  0,  0,  0,  0,  1,  0,  0, 18,  0,  0],
 *       [ 0,  0,  1,  1,  0,  1,  0,  1,  0, 19,  1],
 *       [ 0,  0,  1,  0,  1,  0,  1,  1,  0,  1, 20]]
 *
 * See: Davis, Timothy A. "Direct Methods for Sparse Linear Systems",
 *      Figure 4.2, p 39.
 *
 * @return A  An 11x11 matrix in CSC format.
 */
CSCMatrix davis_example_chol();


/** Define the 8x8 matrix in Davis, Figure 5.1, p 74.
 *
 * This matrix is sparse, unsymmetric positive definite. We arbitrarily assign
 * the diagonal to the 1-based index values (except 8), and off-diagonals to 1.
 *
 *  A = [[1., 0., 0., 1., 0., 0., 1., 0.],
 *       [0., 2., 1., 0., 0., 0., 1., 0.],
 *       [0., 0., 3., 1., 0., 0., 0., 0.],
 *       [1., 0., 0., 4., 0., 0., 1., 0.],
 *       [0., 0., 0., 0., 5., 1., 0., 0.],
 *       [0., 0., 0., 0., 1., 6., 0., 1.],
 *       [0., 1., 1., 0., 0., 0., 7., 1.],
 *       [0., 0., 0., 0., 1., 1., 1., 0.]]
 *
 * See: Davis, Timothy A. "Direct Methods for Sparse Linear Systems",
 *      Figure 5.1, p 74.
 *
 * @param add_diag  If non-zero, add this value to the diagonal of the matrix.
 *        Can be used to make the matrix positive definite.
 * @param random_vals  If true, randomize the non-zero values in the matrix,
 *        before adding to the diagonal.
 *
 * @return A  An 8x8 matrix in CSC format.
 */
CSCMatrix davis_example_qr(double add_diag=0.0, bool random_vals=false);


/** Build the 10 x 10 symmetric, positive definite AMD example matrix.
 *
 * A = [[10.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.],
 *      [ 0., 11.,  0.,  0.,  1.,  1.,  0.,  0.,  1.,  0.],
 *      [ 0.,  0., 12.,  0.,  1.,  1.,  1.,  0.,  0.,  0.],
 *      [ 1.,  0.,  0., 13.,  0.,  0.,  1.,  1.,  0.,  0.],
 *      [ 0.,  1.,  1.,  0., 14.,  0.,  1.,  0.,  1.,  0.],
 *      [ 1.,  1.,  1.,  0.,  0., 15.,  0.,  0.,  0.,  0.],
 *      [ 0.,  0.,  1.,  1.,  1.,  0., 16.,  1.,  1.,  1.],
 *      [ 0.,  0.,  0.,  1.,  0.,  0.,  1., 17.,  1.,  1.],
 *      [ 0.,  1.,  0.,  0.,  1.,  0.,  1.,  1., 18.,  1.],
 *      [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1., 19.]]
 *
 * See: Davis, Figure 7.1, p 101.
 *
 * @return the 10x10 matrix in CSC format.
 */
CSCMatrix davis_example_amd();


/** Define the 3x3 matrix E from Strang, p 25.
 *
 * E = [[ 1, 0, 0],
 *      [-2, 1, 0],
 *      [ 0, 0, 1]]
 *
 * @return E  A 3x3 matrix in CSC format.
 */
CSCMatrix E_mat();


/** Define the 3x3 matrix A from Strang, p 25.
 *
 * A = [[ 2, 1, 1],
 *      [ 4,-6, 0],
 *      [-2, 7, 2]]
 *
 * @return A  A 3x3 matrix in CSC format.
 */
CSCMatrix A_mat();


}  // namespace cs


//==============================================================================
//==============================================================================
