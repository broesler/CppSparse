/*==============================================================================
 *     File: test_cholesky.cpp
 *  Created: 2025-05-08 13:19
 *   Author: Bernie Roesler
 *
 *  Description: Test Chapter 4: Cholesky Factorization.
 *
 *============================================================================*/

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include <algorithm>  // reverse
#include <map>
#include <numeric>    // iota
#include <optional>   // nullopt
#include <random>
#include <utility>    // as_const
#include <vector>

#include "csparse.h"
#include "test_helpers.h"

using Catch::Matchers::AllTrue;
using Catch::Matchers::WithinAbs;

namespace cs {


TEST_CASE("Cholesky Factorization", "[cholesky]")
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
        CholResult res = chol(A, S);
        CSCMatrix L = res.L;

        CHECK(L.has_sorted_indices());
        CHECK(L._test_sorted());

        // Check that the factorization is correct
        CSCMatrix LLT = (L * L.T()).droptol().to_canonical();

        // Permute the input matrix to match LL^T = P^T A P
        CSCMatrix PAPT = A.permute(res.p_inv, inv_permute(res.p_inv)).to_canonical();

        check_sparse_allclose(LLT, PAPT);
    }

    SECTION("Update Cholesky") {
        SymbolicChol S = schol(A, AMDOrder::Natural);
        CholResult res = chol(A, S);
        CSCMatrix L = res.L;

        // Create a random vector with the sparsity of a column of L
        csint k = 3;  // arbitrary column index
        std::default_random_engine rng(56);
        std::uniform_real_distribution<double> unif(0.0, 1.0);

        COOMatrix w {{L.shape()[0], 1}};

        for (csint p = L.indptr()[k]; p < L.indptr()[k + 1]; p++) {
            w.insert(L.indices()[p], 0, unif(rng));
        }

        CSCMatrix W = w.tocsc();  // for arithmetic operations

        // Update the input matrix for testing
        CSCMatrix A_up = (A + W * W.T()).to_canonical();

        // Update the factorization in-place
        CSCMatrix L_up = chol_update(L.to_canonical(), true, W, res.parent);

        CSCMatrix LLT_up = (L_up * L_up.T()).droptol().to_canonical();
        CHECK(LLT_up.nnz() == A_up.nnz());

        check_sparse_allclose(LLT_up, A_up);
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
        CholResult res = chol(A, S);
        CSCMatrix& L = res.L;

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
        auto [xi, x] = chol_lsolve(L, b, res.parent);

        check_vectors_allclose(x, expect, tol);

        // Solve Lx = b, inferring parent from L
        auto [xi_s, x_s] = chol_lsolve(L, b);

        CHECK(xi == xi_s);
        check_vectors_allclose(x_s, expect, tol);
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
        CholResult res = chol(A, S);
        CSCMatrix& L = res.L;

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
        auto [xi, x] = chol_ltsolve(L, b, res.parent);

        check_vectors_allclose(x, expect, tol);

        // Solve Lx = b, inferring parent from L
        auto [xi_s, x_s] = chol_ltsolve(L, b);

        CHECK(xi == xi_s);
        check_vectors_allclose(x_s, expect, tol);
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
        CholResult res = chol(A, S);
        CSCMatrix& L = res.L;
        CSCMatrix LLT = (L * L.T()).droptol().to_canonical();

        // The factorization will be postordered!
        CSCMatrix PAPT = A.permute(res.p_inv, inv_permute(res.p_inv)).to_canonical();

        check_sparse_allclose(LLT, PAPT);
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
        CSCMatrix L = chol(A, S).L;  // numeric factorization
        CSCMatrix Ls = symbolic_cholesky(A, S).L;

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
        CholResult res = symbolic_cholesky(A, S);
        CSCMatrix& L = res.L;

        // Compute the numeric factorization using the non-zero pattern
        L = leftchol(A, S, L);

        CSCMatrix LLT = (L * L.T()).droptol().to_canonical();
        CSCMatrix PAPT = A.permute(res.p_inv, inv_permute(res.p_inv)).to_canonical();

        check_sparse_allclose(LLT, PAPT);
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
        CholResult res = symbolic_cholesky(A, S);
        CSCMatrix& L = res.L;

        // Compute the numeric factorization using the non-zero pattern
        L = rechol(A, S, L);

        CSCMatrix LLT = (L * L.T()).droptol().to_canonical();
        CSCMatrix PAPT = A.permute(res.p_inv, inv_permute(res.p_inv)).to_canonical();

        check_sparse_allclose(LLT, PAPT);
    }

    SECTION("Exercise 4.13: Incomplete Cholesky") {
        const SymbolicChol S = schol(A, AMDOrder::Natural);

        SECTION("IC0: No Fill") {
            // Compute the incomplete Cholesky factorization with no fill-in.
            const CSCMatrix L = ichol_nofill(A, S).L;

            // Compute the complete Cholesky factorization for comparison
            const CSCMatrix Lf = chol(A, schol(A)).L;

            // std::cout << "Lf:" << std::endl;
            // Lf.print_dense();
            // std::cout << "L:" << std::endl;
            // L.print_dense();

            // std::cout << "A:" << std::endl;
            // A.print_dense();

            // L is lower triangular with the same sparsity pattern as A
            const CSCMatrix A_tril = std::as_const(A).band(-N, 0).to_canonical();

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
                [A](csint i, csint j, [[maybe_unused]] double x) {
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
                const CSCMatrix L = icholt(A, S, drop_tol).L;
                const CSCMatrix Lf = chol(A, S).L;
                check_sparse_allclose(L, Lf);
            }

            SECTION("Drop all non-diagonal entries (drop_tol = inf)") {
                double drop_tol = 1.0;
                const CSCMatrix L = icholt(A, S, drop_tol).L;
                const CSCMatrix LLT = (L * L.T()).droptol().to_canonical();
                CHECK(L.nnz() == N);
                for (csint k = 0; k < N; k++) {
                    CHECK_THAT(LLT(k, k), WithinAbs(A(k, k), tol));
                }
            }

            SECTION("Arbitrary Tolerance") {
                // Compute the incomplete Cholesky factorization with a threshold
                double drop_tol = 0.005;
                const CSCMatrix L = icholt(A, S, drop_tol).L;

                // Compute the complete Cholesky factorization for comparison
                const CSCMatrix Lf = chol(A, S).L;
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
                    [A](csint i, csint j, [[maybe_unused]] double x) {
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


}  // namespace cs

/*==============================================================================
 *============================================================================*/
