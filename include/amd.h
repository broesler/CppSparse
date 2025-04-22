//==============================================================================
//     File: amd.h
//  Created: 2025-04-18 08:53
//   Author: Bernie Roesler
//
//  Description: Declarations for AMD ordering.
//
//==============================================================================

#ifndef _CSPARSE_AMD_H_
#define _CSPARSE_AMD_H_

#include <vector>

#include "types.h"

namespace cs {


/** Maximum matching permutation struct. */
struct MaxMatch {
    std::vector<csint> jmatch;  ///< (M,) column matches for each row
    std::vector<csint> imatch;  ///< (N,) row matches for each column

    MaxMatch() = default;
    MaxMatch(csint M, csint N, csint v=0) : jmatch(M, v), imatch(N, v) {}
};


/** Strongly connected components result struct. */
struct SCCResult {
    std::vector<csint> p,  ///< (M,) row permutation
                       r;  ///< (Nb+1,) block k is rows r[k] to r[k+1]-1 in A[p, q]
    csint Nb;              ///< # of blocks in fine dmperm decomposition

    SCCResult() = default;
    SCCResult(csint M, csint N) : Nb(0) {
        p.reserve(M);  // used as stacks in dfs, queues in bfs, so start empty
        r.reserve(M + 6);
    }
};


/** Compute the approximate minimum degree ordering of a matrix.
 *
 * This function computes the approximate minimum degree ordering of a matrix
 * using the AMD algorithm. The ordering is used to reduce the fill-in in the LU
 * decomposition of the matrix.
 *
 * @param A  the matrix to reorder
 * @param order  the ordering method to use:
 *       - `AMDOrder::Natural`: natural ordering (no permutation)
 *       - `AMDOrder::APlusAT`: AMD ordering of A + A^T. This option is appropriate for
 *         Cholesky factorization, or LU factorization with substantial entries
 *         on the diagonal and a roughly symmetric nonzero pattern. If `cs::lu`
 *         is used, `tol < 1.0` should be used to prefer the diagonal entries
 *         for partial pivoting.
 *       - `AMDOrder::ATANoDenseRows`: AMD ordering of A^T * A, with "dense"
 *         rows removed from `A`. This option is appropriate for LU
 *         factorization of unsymmetric matrices and produces a similar ordering
 *         to that of `COLAMD`.
 *       - `AMDOrder::ATA`: AMD ordering of A^T * A. This option is appropriate
 *         for QR factorization, or for LU factorization if `A` has no "dense"
 *         rows. A "dense" row is defined as a row with more than 
 *         \f$ 10 \sqrt{N} \f$ nonzeros, where \f$N\f$ is the number of columns
 *         in the matrix.
 *
 * @return the permutation vector
 */
std::vector<csint> amd(const CSCMatrix& A, const AMDOrder order=AMDOrder::Natural);


/** Find an augmenting path starting at column k and extend the match if found.
 *
 * This function uses a depth-first search (DFS) to find an augmenting path
 * in the bipartite graph represented by the CSCMatrix A. If an augmenting
 * path is found, it extends the match by flipping the matched edges along
 * the path.
 *
 * @param k  the starting column index for the DFS
 * @param A  the CSCMatrix representing the bipartite graph
 * @param jmatch  the current matching vector for the rows
 * @param cheap  the cheap assignment vector
 * @param w  the workspace vector
 * @param js  the row indices stack of the current matching
 * @param is  the column indices stack of the current matching
 * @param ps  the pause stack for the DFS
 */
void augment(
    csint k,
    const CSCMatrix& A,
    std::vector<csint>& jmatch,
    std::vector<csint>& cheap,
    std::vector<csint>& w,
    std::vector<csint>& js,
    std::vector<csint>& is,
    std::vector<csint>& ps
);


/** Find the maximum matching permutation of a matrix.
 *
 * This function finds the maximum matching permutation of a matrix using
 * the augmenting path algorithm. The matching is also known as a "maximum
 * transversal".
 *
 * @param A  the matrix to reorder
 * @param seed  the seed for the random number generator. If `seed` is 0, no
 *        permutation is applied. If `seed` is -1, the permutation is the
 *        reverse of the identity. Otherwise, a random permutation is generated.
 *
 * @return the matching permutation vector
 */
MaxMatch maxtrans(const CSCMatrix& A, csint seed=0);


/** Find the strongly connected components of a matrix.
 *
 * @param A  the matrix to reorder
 *
 * @return the strongly connected components of the matrix
 */
SCCResult scc(const CSCMatrix& A);


}  // namespace cs

#endif  // _CSPARSE_AMD_H_

//==============================================================================
//==============================================================================
