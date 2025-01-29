//==============================================================================
//     File: decomposition.h
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


/*------------------------------------------------------------------------------
 *          Cholesky Decomposition
 *----------------------------------------------------------------------------*/
// ---------- Helpers
enum class LeafStatus {
    NotLeaf,
    FirstLeaf,
    SubsequentLeaf
};

std::vector<csint> post(const std::vector<csint>& parent);

void tdfs(
    csint j,
    std::vector<csint>& head,
    const std::vector<csint>& next,
    std::vector<csint>& postorder
);

// NOTE firstdesc and rowcnt are *not* officially part of CSparse, but are in
// the book for demonstrative purposes.
std::pair<std::vector<csint>, std::vector<csint>> firstdesc(
    const std::vector<csint>& parent,
    const std::vector<csint>& post
);

// See: cs_leaf
std::pair<csint, LeafStatus> least_common_ancestor(
    csint i,
    csint j,
    const std::vector<csint>& first,
    std::vector<csint>& maxfirst,
    std::vector<csint>& prevleaf,
    std::vector<csint>& ancestor
);

// ---------- Matrix operations
std::vector<csint> etree(const CSCMatrix& A, bool ata=false);
std::vector<csint> ereach(
    const CSCMatrix& A,
    csint k,
    const std::vector<csint>& parent
);

std::vector<csint> rowcnt(
    const CSCMatrix& A,
    const std::vector<csint>& parent,
    const std::vector<csint>& postorder
);

std::vector<csint> counts(
    const CSCMatrix& A,
    const std::vector<csint>& parent,
    const std::vector<csint>& postorder
);

std::vector<csint> chol_rowcounts(const CSCMatrix& A);
std::vector<csint> chol_colcounts(const CSCMatrix& A);

// See cs_schol
Symbolic symbolic_cholesky(const CSCMatrix& A, AMDOrder order);

CSCMatrix chol(const CSCMatrix& A, const Symbolic& S);  // up-looking Cholesky

CSCMatrix& chol_update(
    CSCMatrix& L,
    int sigma,
    const CSCMatrix& C,
    const std::vector<csint>& parent
);

}  // namespace cs

#endif // _SYMBOLIC_H_

//==============================================================================
//==============================================================================
