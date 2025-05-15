/*==============================================================================
 *     File: demo.cpp
 *  Created: 2025-05-15 10:18
 *   Author: Bernie Roesler
 *
 *  Description: 
 *
 *============================================================================*/

#include <chrono>
#include <iomanip>    // format
#include <iostream>
#include <limits>     // numeric_limits
#include <vector>

#include "csparse.h"
#include "demo.h"


namespace cs {


std::ostream& operator<<(std::ostream& os, const AMDOrder& order)
{
    switch (order) {
        case AMDOrder::Natural:
            os << "Natural        ";
            break;
        case AMDOrder::APlusAT:
            os << "APlusAT        ";
            break;
        case AMDOrder::ATANoDenseRows:
            os << "ATANoDenseRows ";
            break;
        case AMDOrder::ATA:
            os << "ATA            ";
            break;
        default:
            os << "UnknownAMDOrder";
            break;
    }

    return os;
}


TimePoint tic() { return Clock::now(); }


double toc(TimePoint start_time)
{
    TimePoint end_time = Clock::now();
    auto duration = end_time - start_time;
    // Convert to a double
    std::chrono::duration<double> seconds = duration;
    return seconds.count();
}


CSCMatrix make_sym(const CSCMatrix& A)
{
    CSCMatrix AT = A.T();
    // Drop diagonal entries from AT
    AT.fkeep([](csint i, csint j, double aij) { return i != j; });
    return A + AT;
}


// Get a problem from the input stream
Problem Problem::from_matrix(const COOMatrix& T, double tol)
{
    CSCMatrix A = T.tocsc();                   // convert to CSC format
    A.sum_duplicates();                        // sum up duplicates
    csint is_sym = A.is_triangular();          // determine if A is symmetric
    auto [M, N] = A.shape();
    csint nz1 = A.nnz();
    A.dropzeros();                             // drop zero entries
    csint nz2 = A.nnz();

    if (tol > 0) {
        A.droptol(tol);  // drop tiny entries (just to test)
    }

    CSCMatrix C = is_sym ? make_sym(A) : A;  // C = A + triu(A,1)'

    // Print title
    std::cout << std::format(
        "--- Matrix: {}-by-{}, nnz: {} (sym: {}: nnz: {}), norm: {:8.2e}\n",
        M,
        N,
        A.nnz(),
        is_sym,
        is_sym ? C.nnz() : 0,
        C.norm()
    );

    if (nz1 != nz2) {
        std::cout << "zero entries dropped: " << nz1 - nz2 << std::endl;
    }

    if (nz2 != A.nnz()) {
        std::cout << "tiny entries dropped: " << nz2 - A.nnz() << std::endl;
    }

    // Compute the RHS
    std::vector<double> b(M);
    for (csint i = 0; i < M; i++) {
        b[i] = 1.0 + (double) i / M;
    }

    return {A, C, is_sym, {}, b, {}};
}


void print_resid(
    const CSCMatrix& A,
    const std::vector<double>& x,
    const std::vector<double>& b,
    std::vector<double>& resid
)
{
    resid = A * x - b;
    constexpr double inf = std::numeric_limits<double>::infinity();
    double norm_resid = norm(resid, inf);
    double norm_denom = A.norm() * norm(x, inf) + norm(b, inf);
    std::cout << "residual: " << std::format("{: 8.2e}", norm_resid / norm_denom);
    std::cout << std::endl;
}


}  // namespace cs

/*==============================================================================
 *============================================================================*/
