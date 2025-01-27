//==============================================================================
//     File: symbolic.h
//  Created: 2025-01-27 13:01
//   Author: Bernie Roesler
//
//  Description: Implements the symbolic factorization for a sparse matrix.
//
//==============================================================================

#ifndef _SYMBOLIC_H_
#define _SYMBOLIC_H_

namespace cs {

enum class AMDOrder
{
    Natural,
    APlusAT,
    ATimesA
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


// See cs_schol
Symbolic symbolic_cholesky(const CSCMatrix& A, AMDOrder order);


}  // namespace cs

#endif // _SYMBOLIC_H_

//==============================================================================
//==============================================================================
