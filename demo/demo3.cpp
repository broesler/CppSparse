/*==============================================================================
 *     File: demo3.cpp
 *  Created: 2025-05-15 11:00
 *   Author: Bernie Roesler
 *
 *  Description: Cholesky update/downdate demo for C++Sparse.
 *
 *============================================================================*/

#include <algorithm>  // generate
#include <format>
#include <iostream>  // cerr
#include <print>
#include <random>
#include <span>

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
        std::println(std::cerr, "Usage: demo2 [filename] or demo2 < [filename]");
        return EXIT_FAILURE;
    }

    auto prob = Problem::from_matrix(T, 1e-14);

    auto [M, N] = prob.A.shape();
    auto order = AMDOrder::APlusAT;  // AMD ordering for Cholesky

    if (!prob.is_sym) {
        std::print(std::cerr, "Matrix is not symmetric.");
        return EXIT_FAILURE;
    }

    std::println("Cholesky then update/downdate, order = {:15}", order);

    // Symbolic Cholesky
    auto t = tic();
    auto S = schol(prob.C, order);
    std::println("symbolic chol time: {:8.2e}", toc(t));

    // Numeric Cholesky
    t = tic();
    auto L = chol(prob.C, S).L;
    std::println("numeric  chol time: {:8.2e}", toc(t));

    // Solve Ax = b part by part
    t = tic();

    auto Pb = ipvec(S.p_inv, prob.b);  // P*b
    auto y = lsolve(L, Pb);            // y = L \ Pb
    auto PTx = ltsolve(L, y);          // P^T x = L^T \ y
    prob.x = pvec(S.p_inv, PTx);                      // x = P P^T x

    std::println("solve    chol time: {:8.2e}", toc(t));
    auto resid = residual_norm(prob.C, prob.x, prob.b, prob.resid);
    std::println("original: residual: {:8.2e}", resid);

    // Construct W: W has the pattern of L[k:, k],
    // but with random values on [0, 1), scaled by the diagonal entry L[k, k]
    csint k = N / 2;
    double s = L(k, k);  // scale by the diagonal entry

    // Get the column indices of L
    auto p0 = L.indptr()[k];    // index reference for W
    auto p1 = L.indptr()[k+1];  // index reference for W

    std::vector<csint> W_rows(L.indices().begin() + p0, L.indices().begin() + p1);

    // Create a random vector of values the same size as W_rows
    std::default_random_engine rng(1);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    std::vector<double> W_vals(W_rows.size());
    std::generate(W_vals.begin(), W_vals.end(), [&]() { return unif(rng) * s; });

    std::vector<csint> W_cols(W_rows.size());  // all zeros for column vector

    // Construct the sparse vector
    auto W = COOMatrix{W_vals, W_rows, W_cols, {N, 1}}.tocsc();

    // Perform the update
    t = tic();
    L = chol_update(L, true, W, S.parent);
    auto t1 = toc(t);
    std::println("update:   time: {:8.2e}", t1);

    // Perform the solve again
    t = tic();

    Pb = ipvec(S.p_inv, prob.b);  // P*b
    y = lsolve(L, Pb);            // y = L \ Pb
    PTx = ltsolve(L, y);          // P^T x = L^T \ y
    prob.x = pvec(S.p_inv, PTx);  // x = P P^T x

    std::print("update:   time: {:8.2e} (incl solve) ", toc(t) + t1);

    // Check: E = C + (P'W)*(P'W)'
    auto PTW = W.permute_rows(inv_permute(S.p_inv));
    auto E = prob.C + PTW * PTW.T();

    resid = residual_norm(E, prob.x, prob.b, prob.resid);
    std::println("residual: {:8.2e}", resid);

    // Compute with rechol
    t = tic();

    // Factor and solve
    L = chol(E, S).L;
    Pb = ipvec(S.p_inv, prob.b);  // P*b
    y = lsolve(L, Pb);            // y = L \ Pb
    PTx = ltsolve(L, y);          // P^T x = L^T \ y
    prob.x = pvec(S.p_inv, PTx);  // x = P P^T x

    std::print("rechol:   time: {:8.2e} (incl solve) ", toc(t));
    resid = residual_norm(E, prob.x, prob.b, prob.resid);
    std::println("residual: {:8.2e}", resid);

    // Downdate: L @ L.T - W @ W.T
    t = tic();

    L = chol_update(L, false, W, S.parent);

    t1 = toc(t);

    std::println("downdate: time: {:8.2e}", t1);

    // Solve again
    Pb = ipvec(S.p_inv, prob.b);  // P*b
    y = lsolve(L, Pb);            // y = L \ Pb
    PTx = ltsolve(L, y);          // P^T x = L^T \ y
    prob.x = pvec(S.p_inv, PTx);  // x = P P^T x

    std::print("downdate: time: {:8.2e} (incl solve ", toc(t) + t1);
    resid = residual_norm(prob.C, prob.x, prob.b, prob.resid);
    std::println("residual: {:8.2e}", resid);

    return EXIT_SUCCESS;
}

/*==============================================================================
 *============================================================================*/
