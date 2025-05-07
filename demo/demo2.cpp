/*==============================================================================
 *     File: demo2.cpp
 *  Created: 2025-05-06 12:45
 *   Author: Bernie Roesler
 *
 *  Description: Solve a linear system using Cholesky, LU, and QR, with various
 *  orderings.
 *
 *============================================================================*/

#include <array>
#include <algorithm>  // max
#include <chrono>
#include <iomanip>    // format
#include <iostream>
#include <limits>     // numeric_limits
#include <vector>

#include "csparse.h"
// #include "demo.h"

using namespace cs;


// Operator overload for std::ostream
static std::ostream& operator<<(std::ostream& os, const AMDOrder& order)
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


// Time-keeping functions
using Clock = std::chrono::steady_clock;  // never goes backwards
using TimePoint = Clock::time_point;

static TimePoint tic(void) { return Clock::now(); }

static double toc(TimePoint start_time)
{
    TimePoint end_time = Clock::now();
    auto duration = end_time - start_time;
    // Convert to a double
    std::chrono::duration<double> seconds = duration;
    return seconds.count();
}


// Make a matrix symmetric
static CSCMatrix make_sym(const CSCMatrix& A)
{
    CSCMatrix AT = A.T();
    // Drop diagonal entries from AT
    AT.fkeep([](csint i, csint j, double aij) { return i != j; });
    return A + AT;
}


// Get a problem from the input stream
struct Problem
{
    CSCMatrix A,                // /< original matrix
              C;                // /< symmetric version of original matrix
    csint is_sym;               // /< -1 if lower, 1 if upper, 0 otherwise
    std::vector<double> x,      // /< solution
                        b,      // /< rhs
                        resid;  // /< residuals
};


static Problem get_problem(std::istream& in, double tol)
{
    COOMatrix T(in);              // read the matrix
    CSCMatrix A = T.tocsc();      // convert to CSC format
    A.sum_duplicates();           // sum up duplicates
    csint is_sym = A.is_triangular();  // determine if A is symmetric
    auto [M, N] = A.shape();
    csint nz1 = A.nnz();
    A.dropzeros();                // drop zero entries
    csint nz2 = A.nnz();

    if (tol > 0) {
        A.droptol(tol);           // drop tiny entries (just to test)
    }

    CSCMatrix C = is_sym ? make_sym(A) : A;  // C = A + triu(A,1)'

    // Print title
    std::cout << "--- Matrix: " << M << "-by-" << N
              << ", nnz: " << A.nnz()
              << " (is_sym: " << (double) is_sym
              << ": nnz " << (is_sym ? C.nnz() : 0)
              << "), norm: " << std::format("{:8.2e}", C.norm())
              << std::endl;

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


// Compute residual:
//      norm(A*x - b, inf) / (norm(A, 1) * norm(x, inf) + norm(b, inf))
static void print_resid(
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
    std::cout << "residual: " << std::format("{:8.2e}", norm_resid / norm_denom);
}


// -----------------------------------------------------------------------------
//         Run the test
// -----------------------------------------------------------------------------
int main(void)
{
    Problem prob = get_problem(std::cin, 1e-14);

    auto [M, N] = prob.A.shape();
    double tol = prob.is_sym ? 0.001 : 1.0;  // partial pivoting tolerance

    DMPermResult D = dmperm(prob.C, 1);      // randomized dmperm analysis

    // Check matrix structure
    csint sprank = D.rr[3];
    csint ns = 0;  // number of singletons
    for (csint k = 0; k < D.Nb; k++)
    {
        ns += ((D.r[k+1] == D.r[k] + 1) && (D.s[k+1] == D.s[k] + 1));
    }
    std::cout << "blocks: " << D.Nb
              << " singletons: " << ns
              << " structural rank: " << sprank
              << std::endl;

    // Solve linear system using QR
    for (auto order : {AMDOrder::Natural, AMDOrder::ATA}) {
        if (order == AMDOrder::Natural && M > 1000) {
            continue;
        }
        std::cout << "QR   " << order;
        auto t = tic();
        prob.x = qr_solve(prob.C, prob.b, order);
        std::cout << std::format("time: {:.2e} ", toc(t));
        print_resid(prob.C, prob.x, prob.b, prob.resid);
        std::cout << std::endl;
    }

    if (M != N || sprank < N) {
        return EXIT_SUCCESS;  // return if rect. or singular
    }

    // Solve linear system using LU
    const std::array<AMDOrder, 4> all_orders = {
        AMDOrder::Natural,
        AMDOrder::APlusAT,
        AMDOrder::ATANoDenseRows,
        AMDOrder::ATA
    };

    for (auto order : all_orders) {
        if (order == AMDOrder::Natural && M > 1000) {
            continue;
        }
        // FIXME residual large for AMDOrder::ATANoDenseRows!
        // Also residuals for APlusAT and ATA are exactly 0?
        std::cout << "LU   " << order;
        auto t = tic();
        prob.x = lu_solve(prob.C, prob.b, order, tol);
        std::cout << std::format("time: {:.2e} ", toc(t));
        print_resid(prob.C, prob.x, prob.b, prob.resid);
        std::cout << std::endl;
    }

    if (!prob.is_sym) {
        return EXIT_SUCCESS;
    }

    // Solve linear system using Cholesky
    for (auto order : {AMDOrder::Natural, AMDOrder::APlusAT}) {
        if (order == AMDOrder::Natural && M > 1000) {
            continue;
        }
        std::cout << "Cholesky   " << order;
        auto t = tic();
        prob.x = chol_solve(prob.C, prob.b, order);
        std::cout << std::format("time: {:.2e} ", toc(t));
        print_resid(prob.C, prob.x, prob.b, prob.resid);
        std::cout << std::endl;
    }

    std::cout << std::endl;  // extra newline for readability

    return EXIT_SUCCESS;
}


/*==============================================================================
 *============================================================================*/
