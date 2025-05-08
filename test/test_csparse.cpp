/*==============================================================================
 *     File: test_csparse.cpp
 *  Created: 2024-10-01 21:07
 *   Author: Bernie Roesler
 *
 *  Description: Basic test of my CSparse implementation.
 *
 *============================================================================*/

#define CATCH_CONFIG_MAIN  // tell the compiler to define `main()`

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>  // reverse
#include <cmath>
#include <iomanip>  // setprecision, scientific, etc.
#include <iostream>
#include <fstream>
#include <map>
#include <new>         // bad_alloc
#include <numeric>    // iota
#include <optional>   // nullopt
#include <random>
#include <ranges>     // span
#include <string>
#include <sstream>
#include <vector>
#include <utility>  // as_const

#include "csparse.h"
#include "test_helpers.h"

using Catch::Approx;
using Catch::Matchers::AllTrue;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::UnorderedEquals;
using Catch::Matchers::RangeEquals;

namespace cs {


/*------------------------------------------------------------------------------
 *         Chapter 4: Cholesky Factorization
 *----------------------------------------------------------------------------*/
TEST_CASE("Cholesky Factorization")
{
    // Define the test matrix A (See Davis, Figure 4.2, p 39)
    CSCMatrix A = davis_example_chol();
    csint N = A.shape()[1];

    CHECK(A.is_symmetric());
    // CHECK(A.has_canonical_format());

    SECTION("Elimination Tree") {
        std::vector<csint> expect_A = {5, 2, 7, 5, 7, 6, 8, 9, 9, 10, -1};
        std::vector<csint> expect_ATA = {3, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1};
        REQUIRE(etree(A) == expect_A);
        REQUIRE(etree(A, true) == expect_ATA);
        REQUIRE(etree(A.T() * A) == etree(A, true));
    }

    SECTION("Reachability of Elimination Tree") {
        // Map defines the row subtrees.
        // See Davis Figure 4.4, p 40.
        std::map<csint, std::vector<csint>> expect_map = {
            {0, {}},
            {1, {}},
            {2, {1}},
            {3, {}},
            {4, {}},
            {5, {3, 0}},
            {6, {0, 5}},
            {7, {4, 1, 2}},
            {8, {5, 6}},
            {9, {3, 5, 6, 8, 2, 7}},
            {10, {6, 8, 4, 2, 7, 9}},
        };

        std::vector<csint> parent = etree(A);

        for (const auto& [key, expect] : expect_map) {
            std::vector<csint> xi = ereach(A, key, parent);
            // REQUIRE_THAT(xi, UnorderedEquals(expect));
            REQUIRE(xi == expect);  // in exact order
        }
    }

    SECTION("Post-order of Elimination Tree") {
        std::vector<csint> parent = etree(A);
        std::vector<csint> expect = {1, 2, 4, 7, 0, 3, 5, 6, 8, 9, 10};
        std::vector<csint> postorder = post(parent);
        REQUIRE(postorder == expect);
    }

    SECTION("Reachability of Post-ordered Elimination Tree") {
        // Map defines the post-ordered row subtrees.
        // See Davis Figure 4.8, p 49.
        std::map<csint, std::vector<csint>> expect_map = {
            {0, {}},
            {1, {0}},
            {2, {}},
            {3, {0, 1, 2}},
            {4, {}},
            {5, {}},
            {6, {4, 5}},
            {7, {4, 6}},
            {8, {6, 7}},
            {9, {1, 3, 5, 6, 7, 8}},
            {10, {1, 2, 3, 7, 8, 9}},
        };

        // Post-order A and recompute the elimination tree
        std::vector<csint> parent = etree(A);
        std::vector<csint> p = post(parent);

        // NOTE that we cannot just permute parent, as the post-ordering
        // is not a permutation of the elimination tree.
        A = A.permute(inv_permute(p), p).to_canonical();
        parent = etree(A);

        for (const auto& [key, expect] : expect_map) {
            std::vector<csint> xi = ereach_post(A, key, parent);
            CHECK(xi == expect);
        }
    }

    SECTION("First descendants and levels") {
        std::vector<csint> expect_firsts = {4, 0, 0, 5, 2, 4, 4, 0, 4, 0, 0};
        std::vector<csint> expect_levels = {5, 4, 3, 5, 3, 4, 3, 2, 2, 1, 0};
        std::vector<csint> parent = etree(A);
        auto [firsts, levels] = firstdesc(parent, post(parent));
        REQUIRE(firsts == expect_firsts);
        REQUIRE(levels == expect_levels);
    }

    SECTION("Rowcounts of L") {
        std::vector<csint> expect = {1, 1, 2, 1, 1, 3, 3, 4, 3, 7, 7};
        REQUIRE(chol_rowcounts(A) == expect);
    }

    SECTION("Column counts of L") {
        std::vector<csint> expect = {3, 3, 4, 3, 3, 4, 4, 3, 3, 2, 1};
        REQUIRE(chol_colcounts(A) == expect);
    }

    SECTION("Column counts of L from A^T A") {
        std::vector<csint> expect = {7, 6, 8, 8, 7, 6, 5, 4, 3, 2, 1};
        REQUIRE(chol_colcounts(A.T() * A) == expect);
        REQUIRE(chol_colcounts(A, true) == expect);
    }

    SECTION("Symbolic analysis") {
        SymbolicChol S = schol(A, AMDOrder::Natural);

        std::vector<csint> expect_p_inv(A.shape()[1]);
        std::iota(expect_p_inv.begin(), expect_p_inv.end(), 0);

        auto c = chol_colcounts(A);
        csint expect_nnz = std::accumulate(c.begin(), c.end(), 0);

        REQUIRE(S.p_inv == expect_p_inv);
        REQUIRE(S.parent == etree(A));
        REQUIRE(S.cp == cumsum(chol_colcounts(A)));
        REQUIRE(S.cp.back() == expect_nnz);
        REQUIRE(S.lnz == expect_nnz);
    }

    SECTION("Numeric factorization of non-positive definite matrix") {
        // Decrease the diagonal to make A non-positive definite
        for (csint i = 0; i < N; i++) {
            A.assign(i, i, 1.0);
        }
        SymbolicChol S = schol(A, AMDOrder::Natural);
        CHECK_THROWS(chol(A, S));  // A is not positive definite
    }

    SECTION("Numeric factorization") {
        // CSparse only uses AMDOrder::Natural and AMDOrder::APlusAT for
        // Cholesky factorization in cs_chol.m (see Test/test3.m)
        AMDOrder order = AMDOrder::Natural;

        // should be no permutation with AMDOrder::Natural
        std::vector<csint> expect_p(N);
        std::iota(expect_p.begin(), expect_p.end(), 0);

        SECTION("Natural") {}

        SECTION("APlusAT ordering") {
            order = AMDOrder::APlusAT;
            // MATLAB [L, p] = cs_chol(A) -> order = 1
            expect_p = {1, 4, 6, 8, 0, 3, 5, 2, 9, 10, 7};
        }

        SymbolicChol S = schol(A, order);

        std::vector<csint> expect_p_inv = inv_permute(expect_p);
        CHECK(S.p_inv == expect_p_inv);

        // Now compute the numeric factorization
        CSCMatrix L = chol(A, S);

        CHECK(L.has_sorted_indices());
        CHECK(L._test_sorted());

        // Check that the factorization is correct
        CSCMatrix LLT = (L * L.T()).droptol().to_canonical();

        // Permute the input matrix to match LL^T = P^T A P
        CSCMatrix PAPT = A.permute(S.p_inv, inv_permute(S.p_inv)).to_canonical();

        compare_matrices(LLT, PAPT);
    }

    SECTION("Update Cholesky") {
        SymbolicChol S = schol(A, AMDOrder::Natural);
        CSCMatrix L = chol(A, S);

        // Create a random vector with the sparsity of a column of L
        csint k = 3;  // arbitrary column index
        std::default_random_engine rng(56);
        std::uniform_real_distribution<double> unif(0.0, 1.0);

        COOMatrix w {{L.shape()[0], 1}};

        for (csint p = L.indptr()[k]; p < L.indptr()[k + 1]; p++) {
            w.assign(L.indices()[p], 0, unif(rng));
        }

        CSCMatrix W = w.tocsc();  // for arithmetic operations

        // Update the input matrix for testing
        CSCMatrix A_up = (A + W * W.T()).to_canonical();

        // Update the factorization in-place
        CSCMatrix L_up = chol_update(L.to_canonical(), true, W, S.parent);

        CSCMatrix LLT_up = (L_up * L_up.T()).droptol().to_canonical();
        CHECK(LLT_up.nnz() == A_up.nnz());

        compare_matrices(LLT_up, A_up);
    }

    SECTION("Exercise 4.1: etree and counts from ereach") {
        std::vector<csint> expect_parent = etree(A);
        std::vector<csint> expect_rowcounts = chol_rowcounts(A);
        std::vector<csint> expect_colcounts = chol_colcounts(A);

        auto [parent, rowcounts, colcounts] = chol_etree_counts(A);

        CHECK(parent == expect_parent);
        CHECK(rowcounts == expect_rowcounts);
        REQUIRE(colcounts == expect_colcounts);
    }

    SECTION("Exercise 4.3: Solve Lx = b") {
        AMDOrder order;

        SECTION("Natural") {
            order = AMDOrder::Natural;
        }

        SECTION("APlusAT") {
            order = AMDOrder::APlusAT;
        }

        // Compute the numeric factorization
        SymbolicChol S = schol(A, order);
        CSCMatrix L = chol(A, S);

        // Create RHS for Lx = b
        std::vector<double> expect(N);
        std::iota(expect.begin(), expect.end(), 1);
        // zero-out a few rows of expect to make it "sparse"
        for (const auto& i : {2, 4, 7, 9}) {
            expect[i] = 0;
        }

        const std::vector<double> b_vals = L * expect;

        // Create the sparse RHS matrix
        CSCMatrix b {b_vals, {N, 1}};

        // Solve Lx = b
        auto [xi, x] = chol_lsolve(L, b, S.parent);

        CHECK_THAT(is_close(x, expect, tol), AllTrue());

        // Solve Lx = b, inferring parent from L
        auto [xi_s, x_s] = chol_lsolve(L, b);

        CHECK(xi == xi_s);
        REQUIRE_THAT(is_close(x_s, expect, tol), AllTrue());
    }

    SECTION("Exercise 4.4: Solve L^T x = b") {
        AMDOrder order;

        SECTION("Natural") {
            order = AMDOrder::Natural;
        }

        SECTION("APlusAT") {
            order = AMDOrder::APlusAT;
        }

        // Compute the numeric factorization
        SymbolicChol S = schol(A, order);
        CSCMatrix L = chol(A, S);

        // Create RHS for Lx = b
        std::vector<double> expect(N);
        std::iota(expect.begin(), expect.end(), 1);
        // zero-out a few rows of expect to make it "sparse"
        for (const auto& i : {2, 4, 7, 9}) {
            expect[i] = 0;
        }

        const std::vector<double> b_vals = L.T() * expect;

        // Create the sparse RHS matrix
        CSCMatrix b {b_vals, {N, 1}};

        // Solve Lx = b
        auto [xi, x] = chol_ltsolve(L, b, S.parent);

        CHECK_THAT(is_close(x, expect, tol), AllTrue());

        // Solve Lx = b, inferring parent from L
        auto [xi_s, x_s] = chol_ltsolve(L, b);

        CHECK(xi == xi_s);
        REQUIRE_THAT(is_close(x_s, expect, tol), AllTrue());
    }

    SECTION("Exercise 4.6: etree height") {
        std::vector<csint> parent = etree(A);
        REQUIRE(etree_height(parent) == 6);
    }

    SECTION("Exercise 4.9: Use post-ordering with natural ordering") {
        // Compute the symbolic factorization with postordering
        bool use_postorder = true;

        AMDOrder order = AMDOrder::Natural;

        SECTION("Natural") {}

        SECTION("APlusAT") {
            order = AMDOrder::APlusAT;
        }

        SymbolicChol S = schol(A, order, use_postorder);
        CSCMatrix L = chol(A, S);
        CSCMatrix LLT = (L * L.T()).droptol().to_canonical();

        // The factorization will be postordered!
        CSCMatrix PAPT = A.permute(S.p_inv, inv_permute(S.p_inv)).to_canonical();

        compare_matrices(LLT, PAPT);
    }

    SECTION("Exercise 4.10: Symbolic Cholesky") {
        AMDOrder order;

        SECTION("Natural") {
            order = AMDOrder::Natural;
        }

        SECTION("APlusAT") {
            order = AMDOrder::APlusAT;
        }

        SymbolicChol S = schol(A, order);
        CSCMatrix L = chol(A, S);  // numeric factorization
        CSCMatrix Ls = symbolic_cholesky(A, S);

        CHECK(Ls.nnz() == L.nnz());
        CHECK(Ls.shape() == L.shape());
        CHECK(Ls.indptr() == L.indptr());
        CHECK(Ls.indices() == L.indices());
        CHECK(Ls.data().size() == L.data().size());  // allocation only
    }

    SECTION("Exercise 4.11: Left-looking Cholesky") {
        AMDOrder order;

        SECTION("Natural") {
            order = AMDOrder::Natural;
        }

        SECTION("APlusAT") {
            order = AMDOrder::APlusAT;
        }

        SymbolicChol S = schol(A, order);
        CSCMatrix L = symbolic_cholesky(A, S);

        // Compute the numeric factorization using the non-zero pattern
        L = leftchol(A, S, L);

        CSCMatrix LLT = (L * L.T()).droptol().to_canonical();
        CSCMatrix PAPT = A.permute(S.p_inv, inv_permute(S.p_inv)).to_canonical();

        compare_matrices(LLT, PAPT);
    }

    SECTION("Exercise 4.12: Up-looking Cholesky with Pattern") {
        AMDOrder order;

        SECTION("Natural") {
            order = AMDOrder::Natural;
        }

        SECTION("APlusAT") {
            order = AMDOrder::APlusAT;
        }

        SymbolicChol S = schol(A, order);
        CSCMatrix L = symbolic_cholesky(A, S);

        // Compute the numeric factorization using the non-zero pattern
        L = rechol(A, S, L);

        CSCMatrix LLT = (L * L.T()).droptol().to_canonical();
        CSCMatrix PAPT = A.permute(S.p_inv, inv_permute(S.p_inv)).to_canonical();

        compare_matrices(LLT, PAPT);
    }

    SECTION("Exercise 4.13: Incomplete Cholesky") {
        const SymbolicChol S = schol(A, AMDOrder::Natural);

        SECTION("IC0: No Fill") {
            // Compute the incomplete Cholesky factorization with no fill-in.
            const CSCMatrix L = ichol_nofill(A, S);

            // Compute the complete Cholesky factorization for comparison
            const CSCMatrix Lf = chol(A, schol(A));

            // std::cout << "Lf:" << std::endl;
            // Lf.print_dense();
            // std::cout << "L:" << std::endl;
            // L.print_dense();

            // std::cout << "A:" << std::endl;
            // A.print_dense();

            // L is lower triangular with the same sparsity pattern as A
            const CSCMatrix A_tril = std::as_const(A).band(-N, 0);

            csint fill_in = 6;  // shown in book example
            CHECK(L._test_sorted());
            CHECK(L.nnz() == Lf.nnz() - fill_in);  // fill-in is 6
            CHECK(L.nnz() == A_tril.nnz());
            CHECK(L.indptr() == A_tril.indptr());
            CHECK(L.indices() == A_tril.indices());

            // Test norm just on non-zero pattern of A
            // MATLAB >> norm(A - (Lf * Lf') * spones(A), "fro") / norm(A, "fro")

            const CSCMatrix LLT = (L * L.T()).droptol().to_canonical();
            const CSCMatrix LLT_Anz = LLT.fkeep(
                [A](csint i, csint j, double x) {
                    return std::as_const(A)(i, j) != 0.0;
                }
            );
            const CSCMatrix AmLLT = (A - LLT).droptol(tol).to_canonical();

            // std::cout << "A - LLT:" << std::endl;
            // AmLLT.print_dense();

            CHECK(LLT_Anz.nnz() == A.nnz());
            CHECK(AmLLT.is_symmetric());
            CHECK(AmLLT.nnz() == fill_in);

            double nz_norm = (A - LLT_Anz).fronorm() / A.fronorm();
            double norm = AmLLT.fronorm() / A.fronorm();  // total norm

            // std::cout << "nz_norm: " << std::format("{:6.4g}", nz_norm) << std::endl;
            // std::cout << "   norm: " << std::format("{:6.4g}", norm) << std::endl;

            CHECK_THAT(nz_norm, WithinAbs(0.0, 1e-15));
            CHECK(norm > nz_norm * 1e10);  // hack number
        }

        SECTION("ICT: Threshold") {
            SECTION("Full Cholesky (drop_tol = 0)") {
                // Should match full decomposition
                double drop_tol = 0.0;
                const CSCMatrix L = icholt(A, S, drop_tol);
                const CSCMatrix Lf = chol(A, S);
                compare_matrices(L, Lf);
            }

            SECTION("Drop all non-diagonal entries (drop_tol = inf)") {
                double drop_tol = 1.0;
                const CSCMatrix L = icholt(A, S, drop_tol);
                const CSCMatrix LLT = (L * L.T()).droptol().to_canonical();
                CHECK(L.nnz() == N);
                for (csint k = 0; k < N; k++) {
                    CHECK(LLT(k, k) == Approx(A(k, k)));
                }
            }

            SECTION("Arbitrary Tolerance") {
                // Compute the incomplete Cholesky factorization with a threshold
                double drop_tol = 0.005;
                const CSCMatrix L = icholt(A, S, drop_tol);

                // Compute the complete Cholesky factorization for comparison
                const CSCMatrix Lf = chol(A, S);
                const CSCMatrix LLT = (L * L.T()).droptol().to_canonical();

                // std::cout << "Lf:" << std::endl;
                // Lf.print_dense();
                // std::cout << "L:" << std::endl;
                // L.print_dense();

                // std::cout << "LLT:" << std::endl;
                // LLT.print_dense();
                // std::cout << "A:" << std::endl;
                // A.print_dense();

                // std::cout << "A - LLT:" << std::endl;
                // (A - LLT).print_dense();

                csint expect_drops = 6;  // rel_drop_tol = 0.005;

                CHECK(L._test_sorted());
                CHECK(L.nnz() == Lf.nnz() - expect_drops);

                // Only true for absolute drop tolerance
                // for (const auto& x : L.data()) {
                //     CHECK(std::fabs(x) > drop_tol);
                // }

                // Test the norm just on the pattern of A
                const CSCMatrix LLT_Anz = LLT.fkeep(
                    [A](csint i, csint j, double x) {
                        return std::as_const(A)(i, j) != 0.0;
                    }
                );

                const CSCMatrix AmLLTnz = (A - LLT_Anz).droptol(tol).to_canonical();
                const CSCMatrix AmLLT = (A - LLT).droptol(tol).to_canonical();

                CHECK(LLT_Anz.nnz() == A.nnz());
                CHECK(AmLLT.is_symmetric());
                CHECK(AmLLT.nnz() == expect_drops);

                double nz_norm = AmLLTnz.fronorm() / A.fronorm();
                double norm = (A - LLT).fronorm() / A.fronorm();

                // Only really true for realistically small drop_tol
                CHECK_THAT(nz_norm, WithinAbs(0.0, 1e-15));
                CHECK(norm < drop_tol);  // not always true!
            }
        }
    }
}  // cholesky


// -----------------------------------------------------------------------------
//         Chapter 5: QR Factorization
// -----------------------------------------------------------------------------
TEST_CASE("Householder Reflection")
{
    SECTION("Unit x") {
        std::vector<double> x = {1, 0, 0};

        std::vector<double> expect_v = {1, 0, 0};
        double expect_beta = 0.0;
        double expect_s = 1.0;

        Householder H = house(x);

        CHECK_THAT(is_close(H.v, expect_v, tol), AllTrue());
        CHECK_THAT(H.beta, WithinAbs(expect_beta, tol));
        CHECK_THAT(H.s, WithinAbs(expect_s, tol));

        // Apply the reflection
        CSCMatrix V = COOMatrix(H.v, {0, 1, 2}, {0, 0, 0}).tocsc();
        std::vector<double> Hx = happly(V, 0, H.beta, x);

        REQUIRE_THAT(is_close(Hx, x, tol), AllTrue());
    }

    SECTION("Negative unit x") {
        std::vector<double> x = {-1, 0, 0};

        std::vector<double> expect_v = {1, 0, 0};
        double expect_beta = 0.0;
        double expect_s = -1.0;

        Householder H = house(x);

        CHECK_THAT(is_close(H.v, expect_v, tol), AllTrue());
        CHECK_THAT(H.beta, WithinAbs(expect_beta, tol));
        CHECK_THAT(H.s, WithinAbs(expect_s, tol));

        // Apply the reflection
        CSCMatrix V = COOMatrix(H.v, {0, 1, 2}, {0, 0, 0}).tocsc();
        std::vector<double> Hx = happly(V, 0, H.beta, x);

        REQUIRE_THAT(is_close(Hx, x, tol), AllTrue());
    }

    SECTION("Arbitrary x, x[0] > 0") {
        std::vector<double> x = {3, 4};  // norm(x) == 5

        // These are the *unscaled* values from Octave
        // std::vector<double> expect_v = {8, 4};
        // double expect_beta = 0.025;
        //
        // To get the scaled values, we need to multiply beta by v(1)**2, and
        // then divide v by v(1).
        //
        // In Octave/MATLAB:
        // >> x = [3, 4]';
        // >> [v, beta] = gallery('house', x);
        // >> v / v(1)
        // >> beta * v(1)^2
        //
        // To get the values from scipy.linalg.qr, use:
        // >>> x = np.c_[[3, 4]]  # qr expects 2D array
        // >>> (Qraw, beta), Rraw = scipy.linalg.qr(x, mode='raw')
        // >>> v = np.vstack([1, Qraw[1:]])
        //
        // The relevant LAPACK routines are DGEQRF, DLARFG

        // These are the values from python's scipy.linalg.qr (via LAPACK):
        std::vector<double> expect_v {1, 0.5};
        double expect_beta = 1.6;

        // These are the values from Davis/Golub & Van Loan:
        // std::vector<double> expect_v {1, -2};
        // double expect_beta = 0.4;

        // s is the 2-norm of x == x.T @ x
        double expect_s = -5;

        Householder H = house(x);

        CHECK_THAT(is_close(H.v, expect_v, tol), AllTrue());
        CHECK_THAT(H.beta, WithinAbs(expect_beta, tol));
        CHECK_THAT(H.s, WithinAbs(expect_s, tol));

        // Apply the vector
        // Hx = [±norm(x), 0, 0]
        std::vector<double> expect = {-5, 0}; // LAPACK
        // std::vector<double> expect = {5, 0};  // Davis

        // Use column 0 of V to apply the Householder reflection
        CSCMatrix V = COOMatrix(H.v, {0, 1}, {0, 0}).tocsc();
        std::vector<double> Hx = happly(V, 0, H.beta, x);

        REQUIRE_THAT(is_close(Hx, expect, tol), AllTrue());
    }

    SECTION("Arbitrary x, x[0] < 0") {
        std::vector<double> x = {-3, 4};  // norm(x) == 5

        // These are the values from python's scipy.linalg.qr (via LAPACK):
        std::vector<double> expect_v {1, -0.5};
        double expect_beta = 1.6;
        double expect_s = 5;

        Householder H = house(x);

        CHECK_THAT(is_close(H.v, expect_v, tol), AllTrue());
        CHECK_THAT(H.beta, WithinAbs(expect_beta, tol));
        CHECK_THAT(H.s, WithinAbs(expect_s, tol));

        // Apply the vector
        std::vector<double> expect = {5, 0}; // LAPACK or Davis

        CSCMatrix V = COOMatrix(H.v, {0, 1}, {0, 0}).tocsc();
        std::vector<double> Hx = happly(V, 0, H.beta, x);

        REQUIRE_THAT(is_close(Hx, expect, tol), AllTrue());
    }
}


TEST_CASE("QR factorization of the Identity Matrix")
{
    csint N = 8;
    std::vector<csint> rows(N);
    std::iota(rows.begin(), rows.end(), 0);
    std::vector<double> vals(N, 1.0);

    CSCMatrix I = COOMatrix(vals, rows, rows).tocsc();

    AMDOrder order = GENERATE(
        AMDOrder::Natural,
        AMDOrder::APlusAT,
        AMDOrder::ATANoDenseRows,
        AMDOrder::ATA
    );

    SymbolicQR S = sqr(I, order);

    SECTION("Symbolic analysis") {
        std::vector<csint> expect_identity = {0, 1, 2, 3, 4, 5, 6, 7};
        CHECK(S.p_inv == expect_identity);
        CHECK(S.q == expect_identity);
        CHECK(S.parent == std::vector<csint>(N, -1));
        CHECK(S.leftmost == expect_identity);
        CHECK(S.m2 == N);
        CHECK(S.vnz == N);
        CHECK(S.rnz == N);
    }

    SECTION("Numeric factorization") {
        std::vector<double> expect_beta(N, 0.0);

        QRResult res = qr(I, S);

        compare_matrices(res.V, I);
        CHECK_THAT(is_close(res.beta, expect_beta, tol), AllTrue());
        compare_matrices(res.R, I);
    }
}


TEST_CASE("Symbolic QR Decomposition of Square, Non-symmetric A", "[qr][M == N]")
{
    CSCMatrix A = davis_example_qr();
    csint N = A.shape()[1];  // == 8

    // See etree in Figure 5.1, p 74
    std::vector<csint> parent = {3, 2, 3, 6, 5, 6, 7, -1};

    std::vector<csint> expect_leftmost = {0, 1, 2, 0, 4, 4, 1, 4};
    std::vector<csint> expect_p_inv = {0, 1, 3, 7, 4, 5, 2, 6};  // cs_qr MATLAB

    SECTION("find_leftmost") {
        REQUIRE(find_leftmost(A) == expect_leftmost);
    }

    SECTION("vcount") {
        SymbolicQR S;
        S.parent.assign(parent.begin(), parent.end());
        S.leftmost = find_leftmost(A);
        vcount(A, S);

        CHECK(S.p_inv == expect_p_inv);
        CHECK(S.vnz == 16);
        REQUIRE(S.m2 == N);
    }

    SECTION("Symbolic analysis") {
        std::vector<csint> expect_q = {0, 1, 2, 3, 4, 5, 6, 7};  // natural
        std::vector<csint> expect_parent = parent;

        SymbolicQR S = sqr(A, AMDOrder::Natural);

        CHECK(S.p_inv == expect_p_inv);
        CHECK(S.q == expect_q);
        CHECK(S.parent == expect_parent);
        CHECK(S.leftmost == expect_leftmost);
        CHECK(S.m2 == N);
        CHECK(S.vnz == 16);  // manual counts Figure 5.1, p 74
        REQUIRE(S.rnz == 24);
    }
}


TEST_CASE("Numeric QR Decomposition of Square, Non-symmetric A", "[qr][M == N]")
{
    CSCMatrix A = davis_example_qr();
    csint N = A.shape()[1];  // == 8

    // CSparse only uses 2 possible orders for QR factorization:
    AMDOrder order = GENERATE(
        AMDOrder::Natural,
        AMDOrder::ATA
    );
    CAPTURE(order);

    // ---------- Factor the matrix
    SymbolicQR S = sqr(A, order);
    QRResult res = qr(A, S);

    // Create the identity matrix for testing
    std::vector<csint> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::vector<double> vals(N, 1.0);
    CSCMatrix I = COOMatrix(vals, idx, idx).tocsc();

    SECTION("Numeric Factorization") {
        CSCMatrix Q = apply_qtleft(res.V, res.beta, res.p_inv, I).T();
        CSCMatrix QR = (Q * res.R).droptol().to_canonical();
        CSCMatrix Aq = A.permute_cols(res.q).to_canonical();

        compare_matrices(QR, Aq);
    }

    SECTION("Exercise 5.1: Symbolic factorization") {
        QRResult sym_res = symbolic_qr(A, S);

        CHECK(sym_res.V.indptr() == res.V.indptr());
        CHECK(sym_res.V.indices() == res.V.indices());
        CHECK(sym_res.V.data().empty());
        CHECK(sym_res.beta.empty());
        CHECK(sym_res.R.indptr() == res.R.indptr());
        CHECK(sym_res.R.indices() == res.R.indices());
        REQUIRE(sym_res.R.data().empty());
    }

    SECTION("Exercise 5.3: Re-QR factorization") {
        res = symbolic_qr(A, S);

        // Compute the numeric factorization using the symbolic result
        reqr(A, S, res);

        CSCMatrix Q = apply_qtleft(res.V, res.beta, res.p_inv, I).T();
        CSCMatrix QR = (Q * res.R).droptol().to_canonical();
        CSCMatrix Aq = A.permute_cols(res.q).to_canonical();

        compare_matrices(QR, Aq);
    }

    SECTION("Exercise 5.5: Use post-ordering") {
        // Compute the symbolic factorization with postordering
        bool use_postorder = true;
        SymbolicQR S = sqr(A, order, use_postorder);
        QRResult res = qr(A, S);

        // The postordering of this matrix *is* the natural ordering.
        // TODO Find an example with a different postorder for testing
        CSCMatrix Q = apply_qtleft(res.V, res.beta, res.p_inv, I).T();
        CSCMatrix QR = (Q * res.R).droptol().to_canonical();
        CSCMatrix Aq = A.permute_cols(res.q).to_canonical();

        compare_matrices(QR, Aq);
    }
}


TEST_CASE("Square, rank-deficient A", "[qr][rank-deficient]") 
{
    CSCMatrix A = davis_example_qr();
    auto [M, N] = A.shape();

    // CSparse only uses 2 possible orders for QR factorization:
    AMDOrder order = AMDOrder::Natural;

    SECTION("Single Zero Row") {
        // Zero out an arbitrary row to make A rank-deficient
        csint k = 3;
        for (csint j = 0; j < N; j++) {
            A.assign(k, j, 0.0);
        }
        A = A.to_canonical();
    }

    SECTION("Single Zero Column") {
        // Zero out an arbitrary row to make A rank-deficient
        csint k = 3;
        for (csint i = 0; i < M; i++) {
            A.assign(i, k, 0.0);
        }
        A = A.to_canonical();
    }

    SECTION("Multiple Zero Rows") {
        // Zero out an arbitrary row to make A rank-deficient
        for (const auto& k : {2, 3, 5}) {
            for (csint j = 0; j < N; j++) {
                A.assign(k, j, 0.0);
            }
        }
        A = A.to_canonical();
    }

    SECTION("Multiple Zero Columns") {
        // Zero out an arbitrary row to make A rank-deficient
        for (const auto& k : {2, 3, 5}) {
            for (csint i = 0; i < M; i++) {
                A.assign(i, k, 0.0);
            }
        }
        A = A.to_canonical();
    }

    SymbolicQR S = sqr(A, order);
    QRResult res = qr(A, S);

    // M2 - M is the number of dependent rows in the matrix
    // V and R will be size (M2, N), so Q will be (M2, M2), and QR (M2, N).
    // The last rows will just be zeros, so slice QR to (M, N) to match A.

    csint M2 = res.V.shape()[0];

    // Identity matrix for building Q
    std::vector<csint> idx(M2);
    std::iota(idx.begin(), idx.end(), 0);
    std::vector<double> vals(M2, 1.0);
    CSCMatrix I = COOMatrix(vals, idx, idx).tocsc();

    REQUIRE(res.V.shape() == Shape {M2, N});
    REQUIRE(res.R.shape() == Shape {M2, N});

    CSCMatrix Q = apply_qtleft(res.V, res.beta, res.p_inv, I).T();

    REQUIRE(Q.shape() == Shape {M2, M2});

    CSCMatrix QR = (Q * res.R).slice(0, M, 0, N).droptol(tol).to_canonical();
    CSCMatrix Aq = A.permute_cols(res.q).to_canonical();

    compare_matrices(QR, Aq);
}


TEST_CASE("Symbolic QR factorization of overdetermined matrix M > N", "[qr][M > N]")
{
    // Define the test matrix A (See Davis, Figure 5.1, p 74)
    // except remove the last 2 columns
    csint M = 8;
    csint N = 5;
    CSCMatrix A = davis_example_qr().slice(0, M, 0, N);

    CHECK(A.shape() == Shape {M, N});

    // See etree in Figure 5.1, p 74
    std::vector<csint> parent = {3, 2, 3, -1, -1};

    std::vector<csint> expect_leftmost = {0, 1, 2, 0, 4, 4, 1, 4};
    std::vector<csint> expect_p_inv = {0, 1, 3, 5, 4, 6, 2, 7};

    SECTION("find_leftmost") {
        REQUIRE(find_leftmost(A) == expect_leftmost);
    }

    SECTION("vcount") {
        SymbolicQR S;
        S.parent.assign(parent.begin(), parent.end());
        S.leftmost = find_leftmost(A);
        vcount(A, S);

        CHECK(S.p_inv == expect_p_inv);
        CHECK(S.vnz == 11);
        REQUIRE(S.m2 == M);
    }

    SECTION("Symbolic analysis") {
        std::vector<csint> expect_q = {0, 1, 2, 3, 4};  // natural
        std::vector<csint> expect_parent = parent;

        SymbolicQR S = sqr(A);

        CHECK(S.p_inv == expect_p_inv);
        CHECK(S.q == expect_q);
        CHECK(S.parent == expect_parent);
        CHECK(S.leftmost == expect_leftmost);
        CHECK(S.m2 == M);
        CHECK(S.vnz == 11);  // manual counts Figure 5.1, p 74
        REQUIRE(S.rnz == 8);
    }
}


TEST_CASE("Numeric QR factorization of overdetermined matrix M > N", "[qr][M > N]")
{
    // Define the test matrix A (See Davis, Figure 5.1, p 74)
    // except remove the last 2 columns
    csint M = 8;
    csint N = 5;
    CSCMatrix A = davis_example_qr().slice(0, M, 0, N);

    CHECK(A.shape() == Shape {M, N});

    // CSparse only uses 2 possible orders for QR factorization:
    AMDOrder order = GENERATE(
        AMDOrder::Natural,
        AMDOrder::ATA
    );
    CAPTURE(order);

    // ---------- Factor the matrix 
    SymbolicQR S = sqr(A, order);
    QRResult res = qr(A, S);

    // Create the identity matrix for testing
    std::vector<csint> idx(M);
    std::iota(idx.begin(), idx.end(), 0);
    std::vector<double> vals(M, 1.0);
    CSCMatrix I = COOMatrix(vals, idx, idx).tocsc();

    SECTION("Numeric factorization") {
        CSCMatrix Q = apply_qtleft(res.V, res.beta, res.p_inv, I).T();
        CSCMatrix QR = (Q * res.R).droptol().to_canonical();
        CSCMatrix Aq = A.permute_cols(res.q).to_canonical();

        compare_matrices(QR, Aq);
    }

    SECTION("Exercise 5.1: Symbolic factorization") {
        QRResult sym_res = symbolic_qr(A, S);

        CHECK(sym_res.V.indptr() == res.V.indptr());
        CHECK(sym_res.V.indices() == res.V.indices());
        CHECK(sym_res.V.data().empty());
        CHECK(sym_res.beta.empty());
        CHECK(sym_res.R.indptr() == res.R.indptr());
        CHECK(sym_res.R.indices() == res.R.indices());
        REQUIRE(sym_res.R.data().empty());
    }

    SECTION("Exercise 5.3: Re-QR factorization") {
        QRResult res = symbolic_qr(A, S);

        // Compute the numeric factorization using the symbolic result
        reqr(A, S, res);

        CSCMatrix Q = apply_qtleft(res.V, res.beta, res.p_inv, I).T();
        CSCMatrix QR = (Q * res.R).droptol().to_canonical();
        CSCMatrix Aq = A.permute_cols(res.q).to_canonical();

        compare_matrices(QR, Aq);
    }
}


TEST_CASE("Symbolic QR Factorization of Underdetermined Matrix M < N", "[qr][M < N][symbolic]")
{
    // NOTE in order to perform QR factorization efficiently for M < N, instead
    // of adding extra rows of zeros, we take the first M columns of A so that
    // we have a square factor, and find Q1 R1 = A1 = A[:, :M]. Then,
    // A = [ A1 | A2 ] = Q1 [ R1 | Q1.T A2 ] = Q R, so in cs::qr, we can just
    // multiply Q1.T by A2 to get the last N - M columns of R.

    // Define the test matrix A (See Davis, Figure 5.1, p 74)
    // except remove the last 3 rows
    csint M = 5;
    csint N = 8;
    CSCMatrix A = davis_example_qr().slice(0, M, 0, N);
    CHECK(A.shape() == Shape {M, N});

    // See etree in Figure 5.1, p 74
    std::vector<csint> parent = {3, 2, 3, -1, -1};
    std::vector<csint> expect_leftmost = {0, 1, 2, 0, 4};
    std::vector<csint> expect_p_inv = {0, 1, 2, 3, 4};  // natural

    SECTION("find_leftmost") {
        REQUIRE(find_leftmost(A) == expect_leftmost);
    }

    SECTION("vcount") {
        SymbolicQR S;
        S.parent.assign(parent.begin(), parent.end());
        S.leftmost = find_leftmost(A);
        // Only operate on the first M columns
        vcount(A.slice(0, M, 0, M), S);

        CHECK(S.p_inv == expect_p_inv);
        CHECK(S.vnz == 6);
        REQUIRE(S.m2 == M);
    }

    SECTION("Symbolic analysis") {
        std::vector<csint> expect_q = {0, 1, 2, 3, 4};  // natural
        std::vector<csint> expect_parent = parent;

        SymbolicQR S = sqr(A);

        CHECK(S.p_inv == expect_p_inv);
        CHECK(S.q == expect_q);
        CHECK(S.parent == expect_parent);
        CHECK(S.leftmost == expect_leftmost);
        CHECK(S.m2 == M);
        CHECK(S.vnz == 6);
        REQUIRE(S.rnz == 8);
    }
}


TEST_CASE("Numeric QR Factorization of Underdetermined Matrix M < N", "[qr][M < N][numeric]")
{
    // Define the test matrix A (See Davis, Figure 5.1, p 74)
    // except remove the last 3 rows
    csint M = 5;
    csint N = 8;
    CSCMatrix A = davis_example_qr().slice(0, M, 0, N);
    CHECK(A.shape() == Shape {M, N});

    // CSparse only uses 2 possible orders for QR factorization:
    AMDOrder order = GENERATE(
        AMDOrder::Natural,
        AMDOrder::ATA
    );
    CAPTURE(order);

    // ---------- Factor the matrix
    SymbolicQR S = sqr(A, order);
    QRResult res = qr(A, S);

    // M2 - M is the number of dependent rows in the matrix
    // V and R will be size (M2, N), so Q will be (M2, M2), and QR (M2, N).
    // The last rows will just be zeros, so slice QR to (M, N) to match A.

    csint M2 = res.V.shape()[0];

    // Create the identity matrix for testing
    std::vector<csint> idx(M2);
    std::iota(idx.begin(), idx.end(), 0);
    std::vector<double> vals(M2, 1.0);
    CSCMatrix I = COOMatrix(vals, idx, idx).tocsc();

    SECTION("Numeric factorization") {
        CSCMatrix Q = apply_qtleft(res.V, res.beta, res.p_inv, I).T();
        CSCMatrix QR = (Q * res.R).slice(0, M, 0, N).droptol(tol).to_canonical();
        CSCMatrix Aq = A.permute_cols(res.q).to_canonical();

        compare_matrices(QR, Aq);
    }

    SECTION("Exercise 5.1: Symbolic factorization") {
        // NOTE symbolic_qr will only compute the factorization of A(:, :M) for
        // M < N, whereas `res` is the full factorization of A.
        QRResult sym_res = symbolic_qr(A, S);

        CHECK(sym_res.V.indptr() == res.V.indptr());
        CHECK(sym_res.V.indices() == res.V.indices());
        CHECK(sym_res.V.data().empty());
        CHECK(sym_res.beta.empty());

        // sym_res does not include the last N - M columns of R
        auto res_indptr = std::span(res.R.indptr().data(), M + 1);
        CHECK_THAT(sym_res.R.indptr(), RangeEquals(res_indptr));

        // NOTE hstack sorts the indices, whereas qr/symbolic_qr does not, so we
        // can either:
        //   * remove sorting from hstack
        //   * check the unordered sets of indices (not as robust)
        //   * sort the indices of sym_res.R (for testing only)
        auto res_indices = std::span(res.R.indices().data(), sym_res.R.nnz());
        sym_res.R.sort();  // sort columns in-place
        CHECK_THAT(sym_res.R.indices(), RangeEquals(res_indices));

        REQUIRE(sym_res.R.data().empty());
    }

    SECTION("Exercise 5.3: Re-QR factorization") {
        res = symbolic_qr(A, S);

        // Compute the numeric factorization using the symbolic result
        reqr(A, S, res);

        CSCMatrix Q = apply_qtleft(res.V, res.beta, res.p_inv, I).T();
        CSCMatrix QR = (Q * res.R).slice(0, M, 0, N).droptol(tol).to_canonical();
        CSCMatrix Aq = A.permute_cols(res.q).to_canonical();

        compare_matrices(QR, Aq);
    }
}


// -----------------------------------------------------------------------------
//         Chapter 6: LU Factorization
// -----------------------------------------------------------------------------
/** Define a helper function to test LU decomposition */
LUResult lu_test(const CSCMatrix& A, AMDOrder order=AMDOrder::Natural)
{
    SymbolicLU S = slu(A, order);
    LUResult res = lu(A, S);
    CSCMatrix LU = (res.L * res.U).droptol().to_canonical();
    CSCMatrix PAQ = A.permute(res.p_inv, res.q).to_canonical();
    compare_matrices(LU, PAQ);
    return res;
}


TEST_CASE("LU Factorization of Square Matrix", "[lu]")
{
    const CSCMatrix A = davis_example_qr(10);
    auto [M, N] = A.shape();

    std::vector<csint> expect_q = {0, 1, 2, 3, 4, 5, 6, 7};

    SECTION("Symbolic Factorization") {
        AMDOrder order = AMDOrder::Natural;
        csint expect_lnz;

        SECTION("Natural") {
            order = AMDOrder::Natural;
            expect_lnz = 4 * A.nnz() + N;
        }

        SECTION("APlusAT") {
            order = AMDOrder::APlusAT;
            // MATLAB [L, U, p, q] = cs_lu(A, 1.0); -> order = 1
            expect_q = {4, 5, 7, 1, 2, 0, 6, 3};
            expect_lnz = schol(A, order).lnz;
        }

        SECTION("ATANoDenseRows") {
            order = AMDOrder::ATANoDenseRows;
            // MATLAB [L, U, p, q] = cs_lu(A); -> order = 2
            expect_q = {0, 3, 1, 2, 4, 5, 7, 6};
            expect_lnz = 4 * A.nnz() + N;
        }

        SymbolicLU S = slu(A, order);

        CHECK(S.q == expect_q);
        CHECK(S.lnz == expect_lnz);
        REQUIRE(S.unz == S.lnz);
    }
}


TEST_CASE("Numeric LU Factorization of Square Matrix", "[lu_numeric]")
{
    CSCMatrix A = davis_example_qr(10);
    CSCMatrix Ap = A;
    std::vector<csint> expect_p,
                       expect_q;

    // Cycle through each order and row permutation
    AMDOrder order = GENERATE(
        AMDOrder::Natural,
        AMDOrder::APlusAT,
        AMDOrder::ATANoDenseRows
    );
    bool row_perm = GENERATE(true, false);
    CAPTURE(order, row_perm);  // track which order and row_perm are being used

    std::vector<csint> p = {5, 1, 7, 0, 2, 6, 4, 3};  // arbitrary
    std::vector<csint> p_inv = inv_permute(p);

    if (row_perm) {
        Ap = A.permute_rows(p_inv);
    }

    if (order == AMDOrder::Natural) {
        expect_q = {0, 1, 2, 3, 4, 5, 6, 7};  // no column permutation
        if (row_perm) {
            // LU *should* select pivots to recover the original A matrix
            expect_p = p_inv;
        }
    } else if (order == AMDOrder::APlusAT) {
        // MATLAB [L, U, p, q] = cs_lu(A, 1.0); -> order = 1
        if (row_perm) {
            expect_p = {1, 4, 3, 7, 0, 5, 2, 6};
            expect_q = {1, 2, 0, 3, 5, 6, 7, 4};
        } else {
            expect_q = {4, 5, 7, 1, 2, 0, 6, 3};
        }
    } else if (order == AMDOrder::ATANoDenseRows) {
        // MATLAB [L, U, p, q] = cs_lu(A); -> order = 2
        if (row_perm) {
            expect_p = {3, 7, 1, 4, 6, 0, 2, 5};
            expect_q = {0, 3, 1, 2, 4, 5, 7, 6};
        } else {
            expect_q = {0, 3, 1, 2, 4, 5, 7, 6};
        }
    }

    // Expect the permutation to be the same as the column permutation due to
    // the pivoting algorithm for a square, positive definite matrix.
    if (!row_perm) {
        expect_p = expect_q;
    }

    // Test the factorization
    LUResult res = lu_test(Ap, order);

    // Check the permutations
    CHECK(res.p_inv == inv_permute(expect_p));
    CHECK(res.q == expect_q);
}


TEST_CASE("Solve Ax = b with LU")
{
    const CSCMatrix A = davis_example_qr(10).to_canonical();

    // Create RHS for A x = b
    const std::vector<double> expect = {1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<double> b = A * expect;

    AMDOrder order = AMDOrder::Natural;
    CSCMatrix Ap;
    std::vector<double> bp;

    std::vector<double> x; 
    std::vector<double> x_ov; 

    SECTION("Natural Order") {
        Ap = A;
        bp = b;
    }

    SECTION("Row-Permuted A") {
        // Permuting the rows of A requires permuting the columns of b, but the
        // solution vector will *not* be permuted.
        std::vector<csint> p = {5, 1, 7, 0, 2, 6, 4, 3};  // arbitrary
        std::vector<csint> p_inv = inv_permute(p);

        Ap = A.permute_rows(p_inv);
        bp = pvec(p, b);
    }

    SECTION("Column-Permuted A") {
        // If A has permuted columns, then the RHS vector b is not affected,
        // but the *solution* vector will be permuted.
        order = AMDOrder::APlusAT;
        Ap = A;
        bp = b;
    }

    // Solve the system
    x = lu_solve(Ap, bp, order);

    // Test overload
    SymbolicLU S = slu(Ap, order);
    LUResult res = lu(Ap, S);
    x_ov = res.solve(bp);

    CHECK_THAT(is_close(x, x_ov, tol), AllTrue());
    REQUIRE_THAT(is_close(x, expect, tol), AllTrue());
}


TEST_CASE("Exercise 6.1: Solve A^T x = b with LU")
{
    const CSCMatrix A = davis_example_qr(10).to_canonical();

    // Create RHS for A^T x = b
    const std::vector<double> expect = {1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<double> b = A.T() * expect;

    bool row_perm = false;
    AMDOrder order = AMDOrder::Natural;
    CSCMatrix Ap;

    std::vector<double> x; 
    std::vector<double> x_ov; 
    std::vector<csint> p_inv;

    SECTION("Natural Order") {
        Ap = A;
    }

    SECTION("Row-Permuted A") {
        row_perm = true;
        std::vector<csint> p = {5, 1, 7, 0, 2, 6, 4, 3};  // arbitrary
        p_inv = inv_permute(p);
        Ap = A.permute_rows(p_inv);
    }

    SECTION("Column-Permuted A") {
        order = AMDOrder::APlusAT;
        Ap = A;
    }

    // Solve the system
    x = lu_tsolve(Ap, b, order);

    // Test overload
    SymbolicLU S = slu(Ap, order);
    LUResult res = lu(Ap, S);
    x_ov = res.tsolve(b);

    // Permuting the rows of A is the same as permuting the columns of A^T, so
    // the RHS vector is not affected, but the solution vector will be permuted,
    // so permute it back for comparison.
    if (row_perm) {
        x = pvec(p_inv, x);         // permute back to match expect
        x_ov = pvec(p_inv, x_ov);
    }

    REQUIRE_THAT(is_close(x, x_ov, tol), AllTrue());
    REQUIRE_THAT(is_close(x, expect, tol), AllTrue());
}


TEST_CASE("Exercise 6.3: Column Pivoting in LU", "[ex6.3]")
{
    CSCMatrix A = davis_example_qr(10);
    auto [M, N] = A.shape();

    double col_tol;
    std::vector<csint> expect_p,
                       expect_q;

    // Cases to test:
    //   1. no pivot found in a column (zero column)
    //   2. pivot in a column is below given tolerance

    auto lu_col_test = [](
        const CSCMatrix& A,
        double col_tol,
        const std::vector<csint>& expect_p_inv,
        const std::vector<csint>& expect_q
    ) {
        SymbolicLU S = slu(A);
        double tol = 1.0;  // the row pivot tolerance
        LUResult res = lu(A, S, tol, col_tol);

        CSCMatrix LU = (res.L * res.U).droptol().to_canonical();
        CSCMatrix PAQ = A.permute(res.p_inv, res.q).to_canonical();

        CHECK(res.p_inv == expect_p_inv);
        CHECK(res.q == expect_q);
        compare_matrices(LU, PAQ);
    };

    SECTION("No Column Pivoting") {
        col_tol = 0.0;
        expect_q = {0, 1, 2, 3, 4, 5, 6, 7};  // no pivoting
    }

    SECTION("Zero Column") {
        col_tol = 1e-10;  // only pivot empty columns

        SECTION("Single zero column") {
            // Remove a column to test the column pivoting
            csint k = 3;
            for (csint i = 0; i < M; i++) {
                A(i, k) = 0.0;
            }
            A = A.dropzeros();

            expect_q = {0, 1, 2, 4, 5, 6, 7, 3};
        }

        SECTION("Multiple zero columns") {
            // Remove a column to test the column pivoting
            for (const auto& k : {2, 3, 5}) {
                for (csint i = 0; i < M; i++) {
                    A(i, k) = 0.0;
                }
            }
            A = A.dropzeros();

            expect_q = {0, 1, 4, 6, 7, 2, 3, 5};
        }
    }

    SECTION("Threshold") {
        // Absolute threshold below which to pivot a column to the end
        col_tol = 0.1;

        SECTION("No small columns") {
            // No columns are small enough to pivot
            expect_q = {0, 1, 2, 3, 4, 5, 6, 7};
        }

        SECTION("Single small column") {
            // Scale down a column to test the column pivoting
            csint k = 3;
            double A_kk = A(k, k);
            for (csint i = 0; i < M; i++) {
                A(i, k) *= 0.95 * col_tol / A_kk;
            }

            expect_q = {0, 1, 2, 4, 5, 6, 7, 3};
        }

        SECTION("Multiple small columns") {
            // Scale down multiple columns
            for (const auto& k : {2, 3, 5}) {
                double A_kk = A(k, k);
                for (csint i = 0; i < M; i++) {
                    A(i, k) *= 0.95 * col_tol / A_kk;
                }
            }

            expect_q = {0, 1, 4, 6, 7, 2, 3, 5};
        }
    }

    // Run the Tests
    expect_p = expect_q;  // diagonals are pivots
    std::vector<csint> expect_p_inv = inv_permute(expect_p);

    lu_col_test(A, col_tol, expect_p_inv, expect_q);
}


TEST_CASE("Exercise 6.4: relu", "[ex6.4]")
{
    CSCMatrix A = davis_example_qr(10);
    auto [M, N] = A.shape();

    std::vector<csint> expect_q(N);
    std::iota(expect_q.begin(), expect_q.end(), 0);

    // Create new matrix with same sparsity pattern as A
    std::vector<double> B_data(A.data());
    for (auto& x : B_data) {
        x += 1;
    }
    CSCMatrix B {B_data, A.indices(), A.indptr(), A.shape()};

    CSCMatrix Ap, Bp;
    std::vector<csint> expect_p;

    SECTION("no pivoting") {
        // Compute the LU factorization of B using the pattern of LU = A
        Ap = A;
        Bp = B;
        expect_p = expect_q;
    }

    SECTION("permuted") {
        // Permute the rows of A to test pivoting
        std::vector<csint> p = {5, 1, 7, 0, 2, 6, 4, 3};  // arbitrary
        std::vector<csint> p_inv = inv_permute(p);

        // Permute the rows of A and B to test pivoting
        Ap = A.permute_rows(p_inv);
        Bp = B.permute_rows(p_inv);
        expect_p = p;
    }

    // Compute LU = PA
    SymbolicLU S = slu(Ap);
    LUResult R = lu(Ap, S);

    // Compute the LU factorization of Bp using the pattern of LU = PA
    LUResult res = relu(Bp, R, S);

    CSCMatrix LU = (res.L * res.U).droptol().to_canonical();

    // Permute the rows of the input Bp to compare with LU
    CSCMatrix PBp = Bp.permute_rows(res.p_inv).to_canonical();

    CHECK(res.q == expect_q);
    CHECK(res.p_inv == expect_p);
    compare_matrices(B, PBp);  // LU should match the un-permuted B
    compare_matrices(LU, B.to_canonical());
}


TEST_CASE("Exercise 6.5: LU for square, singular matrices", "[ex6.5]")
{
    CSCMatrix A = davis_example_qr(10).to_canonical();
    auto [M, N] = A.shape();

    CSCMatrix B = A;  // create a copy to edit

    SECTION("Single pair of linearly dependent columns") {
        // Create a singular matrix by setting column 3 = 2 * column 5
        for (csint i = 0; i < M; i++) {
            B(i, 3) = 2 * B(i, 5);
        }
    }

    SECTION("Two pairs of linearly dependent columns") {
        for (csint i = 0; i < M; i++) {
            B(i, 3) = 2 * B(i, 5);
            B(i, 2) = 3 * B(i, 4);
        }
    }

    SECTION("Single pair of linearly dependent rows") {
        // Create a singular matrix by setting row 3 = 2 * row 5
        for (csint j = 0; j < N; j++) {
            B(3, j) = 2 * B(5, j);
        }
    }

    SECTION("Two pairs of linearly dependent rows") {
        for (csint j = 0; j < N; j++) {
            B(3, j) = 2 * B(5, j);
            B(2, j) = 3 * B(4, j);
        }
    }

    SECTION("Single zero column") {
        for (csint i = 0; i < M; i++) {
            B(i, 3) = 0.0;
        }

        SECTION("Structural") {
            B = B.dropzeros();
        }
    }

    SECTION("Multiple zero columns") {
        for (csint i = 0; i < M; i++) {
            for (const auto& j : {2, 3, 4}) {
                B(i, j) = 0.0;
            }
        }

        SECTION("Structural") {
            B = B.dropzeros();
        }
    }

    SECTION("Single zero row") {
        for (csint j = 0; j < N; j++) {
            B(3, j) = 0.0;
        }

        SECTION("Structural") {
            B = B.dropzeros();
        }
    }

    SECTION("Multiple zero rows") {
        for (const auto& i : {2, 3, 4}) {
            for (csint j = 0; j < N; j++) {
                B(i, j) = 0.0;
            }
        }

        SECTION("Structural") {
            B = B.dropzeros();
        }
    }

    lu_test(B);
}


TEST_CASE("Exercise 6.6: LU Factorization of Rectangular Matrices", "[ex6.6]")
{
    CSCMatrix A = davis_example_qr(10).to_canonical();
    auto [M, N] = A.shape();

    csint r = 3;  // number of rows or columns to remove

    SECTION("M < N") {
        A = A.slice(0, M - r, 0, N);
    }

    SECTION("M > N") {
        A = A.slice(0, M, 0, N - r);
    }

    lu_test(A);
}


TEST_CASE("Exercise 6.7: Crout's method LU Factorization", "[ex6.7]")
{
    const CSCMatrix A = davis_example_qr(10);

    SECTION("No pivoting") {
        // Compute the LU factorization of A using Crout's method
        SymbolicLU S = slu(A);
        LUResult res = lu_crout(A, S);

        // std::cout << "A:" << std::endl;
        // A.print_dense();
        // std::cout << "L:" << std::endl;
        // res.L.print_dense();
        // std::cout << "U:" << std::endl;
        // res.U.print_dense();

        CSCMatrix LU = (res.L * res.U).droptol().to_canonical();
        CSCMatrix PA = A.permute_rows(res.p_inv).to_canonical();

        compare_matrices(LU, PA);
    }
}


TEST_CASE("Exercise 6.11: lu_realloc", "[ex6.11]")
{
    // Subclass CSCMatrix to override the realloc function
    class LowMemoryMatrix : public CSCMatrix {
        csint fail_thresh_ = 1000;             // threshold for realloc failure
        std::vector<csint> realloc_attempts_;  // log realloc attempts

    public:
        LowMemoryMatrix(Shape shape, csint fail_thresh) 
            : CSCMatrix(shape), fail_thresh_(fail_thresh) 
        {
            for (csint i = 0; i < shape[0]; i++) {
                assign(i, i, 1.0);
            }
        }

        /** Override realloc to log the number of attempts.
         * @param request  new capacity of the matrix
         */
        void realloc(csint request) override {
            realloc_attempts_.push_back(request);

            if (request > fail_thresh_) {
                throw std::bad_alloc();
            } else {
                CSCMatrix::realloc(request);
            }
        }

        /** Get the number of realloc attempts.
         * @return vector of realloc attempts
         */
        std::vector<csint> get_realloc_attempts() const {
            return realloc_attempts_;
        }
    };

    // Create a low memory matrix with a small nzmax threshold
    csint N = 100;
    csint k = 3;    // arbitrary column index
    csint nnz = N;  // arbitrary, defined the class to assign diagonal

    bool lower = GENERATE(true, false);
    CAPTURE(lower);

    csint max_request = 2 * nnz + N;
    csint min_request = lower ? nnz + N - k : nnz + k + 1;

    double diff = max_request - min_request;
    REQUIRE(diff > 0);
    csint max_total_requests = 1 + static_cast<csint>(std::log2(diff));

    // allocation failure threshold, requesets above threshold will fail
    csint thresh;

    // Expect results (see python/scripts/lu_realloc_calcs.py
    // N: 100
    // max request:  300
    // --- lower:
    // min request:  197
    // requests:  [300, 248, 222, 209, 203, 200, 198]
    // --- upper:
    // min request:  104
    // requests:  [300, 202, 153, 128, 116, 110, 107, 105]

    SECTION("Test without failure: Single Request") {
        thresh = 1000;  // min_request < max_request < threshold 
        LowMemoryMatrix L {Shape {N, N}, thresh};

        csint original_nzmax = L.nzmax();

        REQUIRE_NOTHROW(lu_realloc(L, k, lower));
        REQUIRE(L.nzmax() > original_nzmax);
        REQUIRE(L.nzmax() == max_request);
    }

    SECTION("Test without failure: Multiple Requests") {
        thresh = 200;  // min_request < threshold < max_request
        LowMemoryMatrix L {Shape {N, N}, thresh};

        csint original_nzmax = L.nzmax();

        REQUIRE_NOTHROW(lu_realloc(L, k, lower));

        REQUIRE(L.nzmax() > original_nzmax);

        std::vector<csint> requests = L.get_realloc_attempts();

        // std::cout << "--- Test without failure:" << std::endl;
        // if (lower) {
        //     std::cout << "(lower) ";
        // } else {
        //     std::cout << "(upper) ";
        // }
        // std::cout << "requests: " << requests << std::endl;

        CHECK(requests.front() == max_request);
        CHECK(requests.back() >= min_request);
        CHECK(requests.size() <= max_total_requests);
    }

    SECTION("Test with failure") {
        thresh = 75;  // threshold < min_request < max_request
        LowMemoryMatrix L {Shape {N, N}, thresh};

        // --- Redirect std::cerr to capture the error message ---
        // Save the original cerr buffer
        std::streambuf* original_cerr = std::cerr.rdbuf();

        // Redirect cerr to a stringstream
        std::stringstream captured_cerr;
        std::cerr.rdbuf(captured_cerr.rdbuf());

        REQUIRE_THROWS_AS(lu_realloc(L, k, lower), std::bad_alloc);

        // Restore the original cerr buffer
        std::cerr.rdbuf(original_cerr);

        std::vector<csint> requests = L.get_realloc_attempts();

        // std::cout << "--- Test with failure:" << std::endl;
        // if (lower) {
        //     std::cout << "(lower) ";
        // } else {
        //     std::cout << "(upper) ";
        // }
        // std::cout << "requests: " << requests << std::endl;

        REQUIRE(requests.front() == max_request);
        REQUIRE(requests.back() >= min_request);
        CHECK(requests.size() <= static_cast<csint>(std::log2(max_request - min_request)) + 1);
    }
}


TEST_CASE("Exercise 6.13: Incomplete LU Decomposition", "[ex6.13]")
{
    const CSCMatrix A = davis_example_qr(10).to_canonical();
    auto [M, N] = A.shape();

    SECTION("ILUTP: Threshold with Pivoting") {
        // Default is no pivoting
        CSCMatrix Ap = A;

        // Permute the rows of A to test pivoting
        std::vector<csint> p = {5, 1, 7, 0, 2, 6, 4, 3};  // arbitrary
        std::vector<csint> p_inv = inv_permute(p);

        SECTION("Full LU (tolerance = 0)") {
            double drop_tol = 0.0;

            SECTION("With pivoting") {
                Ap = A.permute_rows(p_inv);
            }

            SymbolicLU S = slu(Ap);
            LUResult res = lu(Ap, S);
            LUResult ires = ilutp(Ap, S, drop_tol);
            CSCMatrix iLU = (ires.L * ires.U).droptol().to_canonical();

            compare_matrices(res.L, ires.L);
            compare_matrices(res.U, ires.U);
            compare_matrices(iLU, A);
        }

        SECTION("Drop all non-digonal entries (tolerance = inf)") {
            double drop_tol = std::numeric_limits<double>::infinity();

            SECTION("With pivoting") {
                Ap = A.permute_rows(p_inv);
            }

            SymbolicLU S = slu(Ap);
            LUResult ires = ilutp(Ap, S, drop_tol);

            REQUIRE(ires.L.nnz() == N);
            REQUIRE(ires.U.nnz() == N);
            for (csint i = 0; i < N; i++) {
                CHECK(ires.L(i, i) == 1.0);
                CHECK_THAT(ires.U(i, i), WithinAbs(A(i, i), tol));
            }
        }

        SECTION("Arbitrary drop tolerance") {
            double drop_tol = 0.08;  // quite large to drop many entries

            SECTION("With pivoting") {
                Ap = A.permute_rows(p_inv);
            }

            SymbolicLU S = slu(Ap);
            LUResult res = lu(Ap, S);
            LUResult ires = ilutp(Ap, S, drop_tol);
            CSCMatrix iLU = (ires.L * ires.U).droptol().to_canonical();

            csint expect_L_drops = 6;  // abs_drop_tol = 0.08
            csint expect_U_drops = 0;

            CHECK(ires.L.nnz() == res.L.nnz() - expect_L_drops);
            CHECK(ires.U.nnz() == res.U.nnz() - expect_U_drops);
            // NOTE only true to absolute tolerance
            CHECK_THAT(ires.L.data() >= drop_tol, AllTrue());
            CHECK_THAT(ires.U.data() >= drop_tol, AllTrue());
            REQUIRE((iLU - A).fronorm() / A.fronorm() < drop_tol);
        }
    }

    SECTION("ILU0: Zero-fill with no pivoting") {
        // Compute the LU factorization of A using the pattern of LU = A
        const SymbolicLU S = slu(A);
        const LUResult ires = ilu_nofill(A, S);

        const CSCMatrix LU = (ires.L * ires.U).droptol().to_canonical();

        // L + U and A are *structurally* identical (no fill-in!)
        const CSCMatrix LpU = (ires.L + ires.U).droptol().to_canonical();

        CHECK(LpU.nnz() == A.nnz());
        CHECK(LpU.indices() == A.indices());
        CHECK(LpU.indptr() == A.indptr());

        // Test norm just on non-zero pattern of A
        // MATLAB >> norm(A - (L * U) * spones(A), "fro") / norm(A, "fro")

        const CSCMatrix LU_Anz = LU.fkeep(
            [A](csint i, csint j, double Aij) { return A(i, j) != 0.0; }
        );
        const CSCMatrix AmLU = (A - LU).droptol(tol).to_canonical();

        CHECK(AmLU.nnz() == 1);  // LU(6, 3) == 0.0705 , A(6, 3) == 0.0

        double nz_norm = (A - LU_Anz).fronorm() / A.fronorm();
        double norm = AmLU.fronorm() / A.fronorm();  // total norm

        // std::cout << "   norm: " << std::format("{:6.4g}", norm) << std::endl;

        CHECK_THAT(nz_norm, WithinAbs(0.0, 1e-16));
        CHECK(norm > nz_norm * 1e10);  // hack number
    }

}


TEST_CASE("Exercise 6.15: 1-norm condition number estimate", "[ex6.15]")
{
    const CSCMatrix A = davis_example_qr(10).to_canonical();

    SECTION("Estimate 1-norm of A inverse") {
        // Compute the LU decomposition
        SymbolicLU S = slu(A);
        LUResult res = lu(A, S);

        double expect = 0.11537500551678347;  // MATLAB and python calcs
        double exact_norm = A.norm();         // 1-norm == maximum column sum

        double est_norm = norm1est_inv(res);

        CHECK(exact_norm >= est_norm);  // estimate is a lower bound
        REQUIRE(est_norm == Approx(expect));
    }

    SECTION("Estimate condition number of A") {
        double kappa = cond1est(A);
        double expect = 2.422875115852452;  // MATLAB and python calcs

        REQUIRE(kappa == Approx(expect));
    }
}


/*------------------------------------------------------------------------------
 *          Chapter 7: Fill-Reducing Orderings
 *----------------------------------------------------------------------------*/
TEST_CASE("Build Graph", "[amd][build_graph]")
{
    CSCMatrix A = davis_example_amd();
    auto [M, N] = A.shape();

    // Number of entries required for a dense row
    csint dense = GENERATE(
        100,  // keep all rows
        0,    // drop all rows
        4     // drop some rows
    );
    CAPTURE(dense);

    CSCMatrix expect_C;
    AMDOrder order = AMDOrder::Natural;
    bool values = false;

    SECTION("Natural") {
        order = AMDOrder::Natural;
        expect_C = CSCMatrix {{}, A.indices(), A.indptr(), A.shape()};
    }

    SECTION("A + A^T") {
        order = AMDOrder::APlusAT;
        expect_C = A + A.transpose(values);
    }

    SECTION("A^T A (no dense)") {
        order = AMDOrder::ATANoDenseRows;

        if (dense == 0) {
            // Drop all rows
            A = CSCMatrix {A.shape()};
        } else if (dense == 4) {
            // Remove some rows (manual count of row_nnz > 4)
            for (csint i : {4, 6, 7, 8}) {
                for (csint j = 0; j < N; j++) {
                    A(i, j) = 0.0;
                }
            }
            A = A.to_canonical();
        }

        expect_C = A.transpose(values) * A;
    }


    SECTION("A^T A") {
        order = AMDOrder::ATA;
        expect_C = A.transpose(values) * A;
    }

    // Remove diagonal elements
    expect_C.fkeep([] (csint i, csint j, double v) { return i != j; });

    const CSCMatrix C = build_graph(A, order, dense);

    CHECK(C.data().empty());
    compare_matrices(C, expect_C, values);
}


TEST_CASE("Approximate Minimum Degree (AMD)", "[amd]")
{
    const CSCMatrix A = davis_example_amd();
    auto [M, N] = A.shape();
    AMDOrder order;
    std::vector<csint> expect_p;

    SECTION("Natural") {
        order = AMDOrder::Natural;
        // ordering should be the same as the original
        expect_p.resize(N);
        std::iota(expect_p.begin(), expect_p.end(), 0);
    }

    SECTION("A + A^T") {
        order = AMDOrder::APlusAT;
        expect_p = {0, 5, 9, 7, 3, 2, 4, 6, 8, 1};  // MATLAB cs_amd
    }

    SECTION("A^T A (no dense)") {
        order = AMDOrder::ATANoDenseRows;
        expect_p = {0, 3, 4, 5, 7, 8, 9, 1, 2, 6};
    }

    SECTION("A^T A") {
        order = AMDOrder::ATA;
        expect_p = {0, 3, 4, 5, 7, 8, 9, 1, 2, 6};
    }

    std::vector<csint> p = amd(A, order);
    CHECK(p == expect_p);
}


TEST_CASE("Maximum Matching", "[maxmatch]")
{
    CSCMatrix A = davis_example_amd();
    auto [M, N] = A.shape();

    csint seed = GENERATE(-1, 0, 1);
    CAPTURE(seed);

    csint expect_rank = M;
    std::vector<csint> expect_jmatch(M);
    std::vector<csint> expect_imatch(N);

    SECTION("Square, Symmetric Full rank") {
        // Matrix is full rank, so should be identity permutation
        expect_rank = 10;
        std::iota(expect_jmatch.begin(), expect_jmatch.end(), 0);
        std::iota(expect_imatch.begin(), expect_imatch.end(), 0);
    }

    SECTION("Square, Rank-deficient (zero rows)") {
        // Zero-out some rows
        for (csint i = 2; i < 5; i++) {
            for (csint j = 0; j < N; j++) {
                A(i, j) = 0.0;
            }
        }
        A = A.to_canonical();

        expect_rank = 7;
        expect_jmatch = {0, 1, -1, -1, -1,  2, 3, 6,  4,  7};
        expect_imatch = {0, 1,  5,  6,  8, -1, 7, 9, -1, -1};
    }

    SECTION("Square, Rank-deficient (zero columns)") {
        // Zero-out some columns
        for (csint j = 2; j < 5; j++) {
            for (csint i = 0; i < M; i++) {
                A(i, j) = 0.0;
            }
        }
        A = A.to_canonical();

        expect_rank = 7;
        expect_jmatch = {0, 1,  5,  6,  8, -1, 7, 9, -1, -1};
        expect_imatch = {0, 1, -1, -1, -1,  2, 3, 6,  4,  7};
    }

    SECTION("M < N, Full Row Rank") {
        // Slice some rows
        A = A.slice(0, M - 3, 0, N);

        expect_rank = 7;
        expect_jmatch = {0, 1, 2, 3, 4, 5, 6};
        expect_imatch = {0, 1, 2, 3, 4, 5, 6, -1, -1, -1};
    }

    SECTION("M > N, Full Column Rank") {
        // Slice some columns
        A = A.slice(0, M, 0, N - 3);

        expect_rank = 7;
        expect_jmatch = {0, 1, 2, 3, 4, 5, 6, -1, -1, -1};
        expect_imatch = {0, 1, 2, 3, 4, 5, 6};
    }

    SECTION("M < N, Row Rank-Deficient") {
        // Slice some rows
        A = A.slice(0, M - 3, 0, N);

        // Zero out some rows
        for (csint i = 2; i < 5; i++) {
            for (csint j = 0; j < N; j++) {
                A(i, j) = 0.0;
            }
        }
        A = A.to_canonical();

        expect_rank = 4;
        expect_jmatch = {0, 1, -1, -1, -1,  2,  3};
        expect_imatch = {0, 1,  5,  6, -1, -1, -1, -1, -1, -1};
    }

    SECTION("M > N, Column Rank-Deficient") {
        // Slice some columns
        A = A.slice(0, M, 0, N - 3);

        // Zero out some columns
        for (csint j = 2; j < 5; j++) {
            for (csint i = 0; i < M; i++) {
                A(i, j) = 0.0;
            }
        }
        A = A.to_canonical();

        expect_rank = 4;
        expect_jmatch = {0, 1,  5,  6, -1, -1, -1, -1, -1, -1};
        expect_imatch = {0, 1, -1, -1, -1,  2,  3};
    }

    MaxMatch res = maxtrans(A, seed);

    // Count number of non-negative entries in jmatch
    csint row_rank = std::accumulate(
        res.jmatch.begin(), res.jmatch.end(), 0,
        [](csint sum, csint i) { return sum + (i >= 0); }
    );

    csint col_rank = std::accumulate(
        res.imatch.begin(), res.imatch.end(), 0,
        [](csint sum, csint j) { return sum + (j >= 0); }
    );

    csint sprank = std::min(row_rank, col_rank);

    CHECK(sprank == expect_rank);

    if (seed == 0) {
        CHECK(res.jmatch == expect_jmatch);
        CHECK(res.imatch == expect_imatch);
    }

    // Check that the matchings are valid
    for (csint i = 0; i < A.shape()[0]; i++) {
        if (res.jmatch[i] >= 0) {
            CHECK(res.imatch[res.jmatch[i]] == i);
        }
    }

    for (csint j = 0; j < A.shape()[1]; j++) {
        if (res.imatch[j] >= 0) {
            CHECK(res.jmatch[res.imatch[j]] == j);
        }
    }
}


TEST_CASE("Strongly Connected Components", "[scc]")
{
    CSCMatrix A = davis_example_amd();
    auto [M, N] = A.shape();

    csint expect_Nb = 0;
    std::vector<csint> expect_p(N);
    std::vector<csint> expect_r(N);

    SECTION("Full Rank") {
        expect_Nb = 1;  // full rank
        expect_p = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        expect_r = {0, 10};
    }

    SECTION("Zero Rows") {
        // Zero out some rows
        for (csint i = 2; i < 5; i++) {
            for (csint j = 0; j < N; j++) {
                A(i, j) = 0.0;
            }
        }
        A = A.to_canonical();

        expect_Nb = 4;  // test with cs_scc
        expect_p = {0, 1, 5, 6, 7, 8, 9, 2, 3, 4};
        expect_r = {0, 7, 8, 9, 10};
    }

    SCCResult D = scc(A);

    CHECK(D.p == expect_p);
    CHECK(D.r == expect_r);
    CHECK(D.Nb == expect_Nb);
}


TEST_CASE("Dulmage-Mendelsohn Permutation", "[dmperm]")
{
    CSCMatrix A = davis_example_amd();
    auto [M, N] = A.shape();

    csint seed = GENERATE(-1, 0, 1);
    CAPTURE(seed);

    csint expect_Nb = 0;
    std::vector<csint> expect_p(N),
                       expect_q(N),
                       expect_r(N),
                       expect_s(N);
    std::array<csint, 5> expect_cc;
    std::array<csint, 5> expect_rr;

    SECTION("Full Rank") {
        // MATLAB results
        expect_Nb = 1;  // full rank
        expect_p = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        expect_q = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        expect_r = {0, 10};
        expect_s = {0, 10};
        expect_cc = {0, 0,  0, 10, 10};
        expect_rr = {0, 0, 10, 10, 10};
    }

    SECTION("Zero Rows") {
        // Zero out some rows
        for (csint i = 2; i < 5; i++) {
            for (csint j = 0; j < N; j++) {
                A(i, j) = 0.0;
            }
        }
        A = A.to_canonical();

        expect_Nb = 2;  // test with MATLAB dmperm
        expect_p = {0, 1, 5, 6, 8, 7, 9, 2, 3, 4};
        expect_q = {5, 8, 9, 0, 1, 2, 3, 4, 6, 7};
        expect_r = {0,  7, 10};
        expect_s = {0, 10, 10};
        expect_cc = {0, 3, 10, 10, 10};
        expect_rr = {0, 7,  7,  7, 10};
    }

    DMPermResult D = dmperm(A, seed);

    if (seed == 0) {
        CHECK(D.p == expect_p);
        CHECK(D.q == expect_q);
    } else {
        CHECK_THAT(D.p, UnorderedEquals(expect_p));
        CHECK_THAT(D.q, UnorderedEquals(expect_q));
    }
    CHECK(D.r == expect_r);
    CHECK(D.s == expect_s);
    CHECK(D.Nb == expect_Nb);
    CHECK(D.rr == expect_rr);
    CHECK(D.cc == expect_cc);
}


// -----------------------------------------------------------------------------
//         Chapter 8: Solving Sparse Linear Systems
// -----------------------------------------------------------------------------
// TODO write a test function that takes A, and a solve function, creates expect
// and b, and runs the test

TEST_CASE("Cholesky Solution", "[cholsol]")
{
    CSCMatrix A = davis_example_chol();
    auto [M, N] = A.shape();

    AMDOrder order = GENERATE(
        AMDOrder::Natural,
        AMDOrder::APlusAT
    );

    // Create RHS for Ax = b
    std::vector<double> expect(N);
    std::iota(expect.begin(), expect.end(), 1);

    const std::vector<double> b = A * expect;

    // Solve Ax = b
    std::vector<double> x = chol_solve(A, b, order);

    // Check that Ax = b
    REQUIRE_THAT(is_close(x, expect, tol), AllTrue());
}


TEST_CASE("QR Solution", "[qrsol]")
{
    CSCMatrix A = davis_example_qr();
    auto [M, N] = A.shape();

    AMDOrder order = GENERATE(
        AMDOrder::Natural,
        AMDOrder::ATA
    );

    std::vector<double> expect(N);
    std::iota(expect.begin(), expect.end(), 1);

    std::vector<double> b, x;

    SECTION("Square") {
        // Create RHS for Ax = b
        b = A * expect;

        // Solve Ax = b
        x = qr_solve(A, b, order);

        // Check that Ax = b
        REQUIRE_THAT(is_close(x, expect, 1e-13), AllTrue());
    }

    SECTION("Over-determined") {
        // Create a new matrix with more rows than columns
        csint k = 3;
        A = A.slice(0, M, 0, N - k);

        // Take only the first N - k elements of expect
        expect = std::vector<double>(expect.begin(), expect.end() - k);

        b = A * expect;

        // Solve Ax = b
        x = qr_solve(A, b);

        // Check that Ax = b
        REQUIRE_THAT(is_close(x, expect, 1e-13), AllTrue());
    }

    SECTION("Under-determined") {
        // Create a new matrix with more rows than columns
        csint k = 3;
        A = A.slice(0, M - k, 0, N);

        b = A * expect;

        // Solve Ax = b
        x = qr_solve(A, b);  // (M - k, N)

        // Actual expect (python and MATLAB)
        const std::vector<double> min_norm_x = {3.2222222222222143,
            3.1111111111111125,
            3.                ,
            4.000000000000004 ,
            5.961538461538462 ,
            1.192307692307692 ,
            4.7777777777777715,
            0.                
        };

        // Check that Ax = b
        REQUIRE_THAT(is_close(x, min_norm_x, tol), AllTrue());
    }
}


TEST_CASE("LU Solution", "[lusol]") {
    CSCMatrix A = davis_example_qr();
    auto [M, N] = A.shape();

    AMDOrder order = GENERATE(
        AMDOrder::Natural,
        AMDOrder::ATANoDenseRows,
        AMDOrder::ATA
    );

    // Create RHS for Ax = b
    std::vector<double> expect(N);
    std::iota(expect.begin(), expect.end(), 1);

    const std::vector<double> b = A * expect;

    // Solve Ax = b
    std::vector<double> x = lu_solve(A, b, order);

    // Check that Ax = b
    REQUIRE_THAT(is_close(x, expect, tol), AllTrue());
}


}  // namespace cs


/*==============================================================================
 *============================================================================*/
