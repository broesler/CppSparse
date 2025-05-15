//==============================================================================
//     File: demo.h
//  Created: 2025-05-15 10:16
//   Author: Bernie Roesler
//
//  Description: Header file for C++Sparse demo programs.
//
//==============================================================================

#ifndef _CSPARSE_DEMO_H
#define _CSPARSE_DEMO_H

#include <algorithm>  // max
#include <chrono>
#include <iomanip>    // format
#include <iostream>
#include <limits>     // numeric_limits
#include <vector>

#include "csparse.h"


namespace cs {


/** Data structure to an Ax = b problem. */
struct Problem
{
    CSCMatrix A,                // /< original matrix
              C;                // /< symmetric version of original matrix
    csint is_sym;               // /< -1 if lower, 1 if upper, 0 otherwise
    std::vector<double> x,      // /< solution
                        b,      // /< rhs
                        resid;  // /< residuals

    /** Construct a Problem from an input matrix.
     *
     * @param T  The input matrix in COO format.
     * @param tol  The tolerance for dropping small entries.
     *
     * @return  A Problem object containing the matrix and other data.
     */
    static Problem from_matrix(const COOMatrix& T, double tol=0);
};


// Time-keeping functions
using Clock = std::chrono::steady_clock;  // never goes backwards
using TimePoint = Clock::time_point;

/** Start and stop a timer */
TimePoint tic();
double toc(TimePoint start_time);


/** Make a matrix symmetric.
 *
 * This function takes a matrix stored as either a lower or upper triangular,
 * and creates a symmetric matrix by adding the transpose of the matrix to
 * itself.
 *
 * @param A  The input matrix to be made symmetric.
 *
 * @return  A symmetric matrix.
 */
CSCMatrix make_sym(const CSCMatrix& A);


/** Compute and print the norm of the residuals of the solution to `Ax = b`.
 *
 * This function computes and prints the following:
 *      `norm(A*x - b, inf) / (norm(A, 1) * norm(x, inf) + norm(b, inf))`.
 *
 * @param A  The coefficient matrix
 * @param x  The solution vector
 * @param b  The right-hand side vector
 * @param[out] resid  A reference to the (empty) residual vector.
 */
void print_resid(
    const CSCMatrix& A,
    const std::vector<double>& x,
    const std::vector<double>& b,
    std::vector<double>& resid
);


/** Print AMDOrder */
std::ostream& operator<<(std::ostream& os, const AMDOrder& order);


}  // namespace cs

#endif  // _CSPARSE_DEMO_H

//==============================================================================
//==============================================================================
