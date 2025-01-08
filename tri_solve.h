//==============================================================================
//     File: tri_solve.h
//  Created: 2025-01-07 20:49
//   Author: Bernie Roesler
//
//  Description: Triangular matrix solve functions.
//
//==============================================================================

#ifndef _TRI_SOLVE_H_
#define _TRI_SOLVE_H_

std::vector<double> lsolve(const CSCMatrix& L, const std::vector<double>& b);
std::vector<double> ltsolve(const CSCMatrix& L, const std::vector<double>& b);

#endif // _TRI_SOLVE_H_

//==============================================================================
//==============================================================================
