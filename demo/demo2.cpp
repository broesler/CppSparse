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
#include <iomanip>    // format
#include <iostream>

#include "csparse.h"
#include "demo.h"


using namespace cs;


int main(int argc, char* argv[])
{
    COOMatrix T;

    if (argc > 1) {
        // Read the filename as a string from the argument
        T = COOMatrix::from_file(argv[1]);
    } else if (argc == 1) {
        // Read from stdin
        T = COOMatrix::from_stream(std::cin);
    } else {
        std::cerr << "Usage: demo2 [filename] or demo2 < [filename]" << std::endl;
        return EXIT_FAILURE;
    }

    auto prob = Problem::from_matrix(T, 1e-14);

    auto [M, N] = prob.A.shape();
    auto tol = prob.is_sym ? 0.001 : 1.0;  // partial pivoting tolerance

    auto D = dmperm(prob.C, 1);      // randomized dmperm analysis

    // Check matrix structure
    auto sprank = D.rr[3];
    csint ns = 0;  // number of singletons
    for (csint k = 0; k < D.Nb; ++k)
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
        std::cout << "QR    " << order;
        auto t = tic();
        prob.x = qr_solve(prob.C, prob.b, order).x;
        std::cout << std::format("time: {:.2e} ", toc(t));
        auto resid = residual_norm(prob.C, prob.x, prob.b, prob.resid);
        std::cout << std::format("residual: {:8.2e}", resid) << std::endl;
    }

    if (M != N || sprank < N) {
        std::cout << std::endl;
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
        std::cout << "LU    " << order;
        auto t = tic();
        prob.x = lu_solve(prob.C, prob.b, order, tol);
        std::cout << std::format("time: {:.2e} ", toc(t));
        auto resid = residual_norm(prob.C, prob.x, prob.b, prob.resid);
        std::cout << std::format("residual: {:8.2e}", resid) << std::endl;
    }

    if (!prob.is_sym) {
        std::cout << std::endl;
        return EXIT_SUCCESS;
    }

    // Solve linear system using Cholesky
    for (auto order : {AMDOrder::Natural, AMDOrder::APlusAT}) {
        if (order == AMDOrder::Natural && M > 1000) {
            continue;
        }
        std::cout << "Chol  " << order;
        auto t = tic();
        prob.x = chol_solve(prob.C, prob.b, order);
        std::cout << std::format("time: {:.2e} ", toc(t));
        auto resid = residual_norm(prob.C, prob.x, prob.b, prob.resid);
        std::cout << std::format("residual: {:8.2e}", resid) << std::endl;
    }

    std::cout << std::endl;  // extra newline for readability

    return EXIT_SUCCESS;
}


/*==============================================================================
 *============================================================================*/
