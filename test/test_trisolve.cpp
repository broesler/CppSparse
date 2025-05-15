/*==============================================================================
 *     File: test_trisolve.cpp
 *  Created: 2025-05-08 13:14
 *   Author: Bernie Roesler
 *
 *  Description: Test Chapter 3: Triangular Matrix Solutions.
 *
 *============================================================================*/

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include <algorithm>  // reverse
#include <numeric>    // iota
#include <optional>   // nullopt
#include <vector>

#include "csparse.h"
#include "test_helpers.h"

using Catch::Matchers::AllTrue;

namespace cs {


TEST_CASE("Triangular solve with dense RHS", "[trisolve_dense]")
{
    const CSCMatrix L = COOMatrix(
        std::vector<double> {1, 2, 3, 4, 5, 6},
        std::vector<csint>  {0, 1, 1, 2, 2, 2},
        std::vector<csint>  {0, 0, 1, 0, 1, 2}
    ).tocsc();

    const CSCMatrix U = L.T();

    const std::vector<double> expect = {1, 1, 1};

    SECTION("Forward solve L x = b") {
        const std::vector<double> b = {1, 5, 15};  // row sums of L
        const std::vector<double> x = lsolve(L, b);

        REQUIRE(x.size() == expect.size());
        REQUIRE_THAT(is_close(x, expect, tol), AllTrue());
    }

    SECTION("Backsolve L.T x = b") {
        const std::vector<double> b = {7, 8, 6};  // row sums of L.T == col sums of L
        const std::vector<double> x = ltsolve(L, b);

        REQUIRE(x.size() == expect.size());
        REQUIRE_THAT(is_close(x, expect, tol), AllTrue());
    }

    SECTION("Backsolve U x = b") {
        const std::vector<double> b = {7, 8, 6};  // row sums of L.T == col sums of L
        const std::vector<double> x = usolve(U, b);

        REQUIRE(x.size() == expect.size());
        REQUIRE_THAT(is_close(x, expect, tol), AllTrue());
    }

    SECTION("Forward solve U.T x = b") {
        const std::vector<double> b = {1, 5, 15};  // row sums of L
        const std::vector<double> x = utsolve(U, b);

        REQUIRE(x.size() == expect.size());
        REQUIRE_THAT(is_close(x, expect, tol), AllTrue());
    }
}


TEST_CASE("Reachability and DFS", "[dfs][reach]")
{
    csint N = 14;  // size of L

    // Define a lower-triangular matrix L with arbitrary non-zeros
    std::vector<csint> rows = {2, 3, 4, 6, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13};
    std::vector<csint> cols = {0, 1, 2, 1, 2, 4, 1, 3, 5, 5, 6, 7,  6,  9,  8, 10,  8,  9, 10, 11,  9, 12};

    // Add the diagonals
    std::vector<csint> diags(N);
    std::iota(diags.begin(), diags.end(), 0);
    rows.insert(rows.end(), diags.begin(), diags.end());
    cols.insert(cols.end(), diags.begin(), diags.end());

    // All values are 1
    std::vector<double> vals(rows.size(), 1);

    CSCMatrix L = COOMatrix(vals, rows, cols).tocsc();
    CSCMatrix U = L.T();

    // Define the rhs matrix B
    CSCMatrix B {Shape {N, 1}};

    SECTION("dfs from a single node") {
        // Assign non-zeros to rows 3 and 5 in column 0
        csint j = 3;
        B.assign(j, 0, 1.0);
        std::vector<csint> expect = {13, 12, 11, 8, 3};  // reversed in stack

        std::vector<char> marked(N, false);
        std::vector<csint> xi,      // do not initialize!
                           pstack,  // pause and recursion stacks
                           rstack;
        xi.reserve(N);
        pstack.reserve(N);
        rstack.reserve(N);

        xi = dfs(L, j, marked, xi, pstack, rstack);

        REQUIRE(xi == expect);
    }

    SECTION("Reachability from a single node") {
        // Assign non-zeros to rows 3 and 5 in column 0
        B.assign(3, 0, 1.0);
        std::vector<csint> expect = {3, 8, 11, 12, 13};

        std::vector<csint> xi = reach(L, B, 0);

        REQUIRE(xi == expect);
    }

    SECTION("Reachability from multiple nodes") {
        // Assign non-zeros to rows 3 and 5 in column 0
        B.assign(3, 0, 1.0).assign(5, 0, 1.0).to_canonical();
        std::vector<csint> expect = {5, 9, 10, 3, 8, 11, 12, 13};

        std::vector<csint> xi = reach(L, B, 0);

        REQUIRE(xi == expect);
    }

    SECTION("spsolve Lx = b with dense RHS") {
        // Create RHS from sums of rows of L, so that x == ones(N)
        std::vector<double> b = {1., 1., 2., 2., 2., 1., 2., 3., 4., 4., 3., 3., 5., 3.};
        for (int i = 0; i < N; i++) {
            B.assign(i, 0, b[i]);
        }
        std::vector<double> expect(N, 1.0);

        // Use structured bindings to unpack the result
        auto [xi, x] = spsolve(L, B, 0);

        REQUIRE(x == expect);
    }

    SECTION("spsolve Lx = b with sparse RHS") {
        // RHS is just B with non-zeros in the first column
        B.assign(3, 0, 1.0);

        std::vector<double> expect = { 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  1.,  0.,  0.};

        // Use structured bindings to unpack the result
        auto [xi, x] = spsolve(L, B, 0);

        REQUIRE(x == expect);
    }

    SECTION("spsolve Ux = b with sparse RHS") {
        // RHS is just B with non-zeros in the first column
        B.assign(3, 0, 1.0);

        std::vector<double> expect = {0., -1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.};

        auto [xi, x] = spsolve(U, B, 0, std::nullopt, false);

        REQUIRE(x == expect);
    }
}


TEST_CASE("Permuted triangular solvers", "[trisolve_perm]")
{
    // >>> L.toarray()
    // === array([[1, 0, 0, 0, 0, 0],
    //            [2, 2, 0, 0, 0, 0],
    //            [3, 3, 3, 0, 0, 0],
    //            [4, 4, 4, 4, 0, 0],
    //            [5, 5, 5, 5, 5, 0],
    //            [6, 6, 6, 6, 6, 6]])
    // >>> (P @ L).toarray().astype(int)
    // === array([[ 6,  6,  6,  6,  6, *6],
    //            [ 4,  4,  4, *4,  0,  0],
    //            [*1,  0,  0,  0,  0,  0],
    //            [ 2, *2,  0,  0,  0,  0],
    //            [ 5,  5,  5,  5, *5,  0],
    //            [ 3,  3, *3,  0,  0,  0]])
    //
    // Starred elements are the diagonals of the un-permuted matrix

    // Create full matrix with row numbers as values
    const std::vector<double> row_vals = {1, 2, 3, 4, 5, 6};
    const csint N = row_vals.size();

    std::vector<double> A_vals;
    A_vals.reserve(N * N);

    for (csint i = 0; i < N; i++) {
        A_vals.insert(A_vals.end(), row_vals.begin(), row_vals.end());
    }

    const CSCMatrix A = CSCMatrix(A_vals, {N, N});

    // Un-permuted matrices
    const CSCMatrix L = A.band(-N, 0);
    const CSCMatrix U = A.band(0, N);

    // TODO I am curious what happens when there is more than one singleton row
    // at a time in find_tri_permutation(). Try removing a few entries from each
    // matrix to make them more sparse and see if the permutation order is still
    // correct.

    const std::vector<csint> p = {5, 3, 0, 1, 4, 2};
    const std::vector<csint> q = {1, 4, 0, 2, 5, 3};

    // Permute the rows (non-canonical form works too)
    const CSCMatrix PL = L.permute_rows(inv_permute(p)).to_canonical();
    const CSCMatrix PU = U.permute_rows(inv_permute(p)).to_canonical();

    // Permute the columns (non-canonical form works too)
    const CSCMatrix LQ = L.permute_cols(p).to_canonical();
    const CSCMatrix UQ = U.permute_cols(p).to_canonical();

    // Permute both rows and columns
    const CSCMatrix PLQ = L.permute(inv_permute(p), q).to_canonical();
    const CSCMatrix PUQ = U.permute(inv_permute(p), q).to_canonical();

    SECTION("Find diagonals of permuted L") {
        std::vector<csint> expect = {2, 8, 14, 16, 19, 20};
        std::vector<csint> p_diags = find_lower_diagonals(PL);
        CHECK(p_diags == expect);

        // Check that we can get the inverse permutation
        std::vector<csint> p_inv = inv_permute(p);  // {2, 3, 5, 1, 4, 0};
        std::vector<csint> diags;
        for (const auto& p : p_diags) {
            diags.push_back(PL.indices()[p]);
        }
        REQUIRE(diags == p_inv);
    }

    SECTION("Find diagonals of permuted U") {
        std::vector<csint> expect = {0, 2, 5, 6, 13, 15};
        std::vector<csint> p_diags = find_upper_diagonals(PU);
        CHECK(p_diags == expect);

        // Check that we can get the inverse permutation
        std::vector<csint> p_inv = inv_permute(p);  // {2, 3, 5, 1, 4, 0};
        std::vector<csint> diags;
        for (const auto& p : p_diags) {
            diags.push_back(PU.indices()[p]);
        }
        REQUIRE(diags == p_inv);
    }

    SECTION("Find diagonals of non-triangular matrix") {
        const CSCMatrix A = davis_example_small().tocsc();
        REQUIRE_THROWS(find_lower_diagonals(A));
        REQUIRE_THROWS(find_upper_diagonals(A));
        REQUIRE_THROWS(find_tri_permutation(A));
    }

    SECTION("Find permutation vectors of permuted L") {
        std::vector<csint> expect_p = inv_permute(p);
        std::vector<csint> expect_q = inv_permute(q);

        auto [p_inv, q_inv, p_diags] = find_tri_permutation(PLQ);

        CHECK(p_inv == expect_p);
        CHECK(q_inv == expect_q);
        compare_matrices(L, PLQ.permute(inv_permute(p_inv), q_inv));
        compare_matrices(PLQ, L.permute(p_inv, inv_permute(q_inv)));
    }

    SECTION("Find permutation vectors of permuted U") {
        std::vector<csint> expect_p = inv_permute(p);
        std::vector<csint> expect_q = inv_permute(q);

        // NOTE returns *reversed* vectors for an upper triangular matrix!!
        auto [p_inv, q_inv, p_diags] = find_tri_permutation(PUQ);
        std::reverse(p_inv.begin(), p_inv.end());
        std::reverse(q_inv.begin(), q_inv.end());

        CHECK(p_inv == expect_p);
        CHECK(q_inv == expect_q);
        compare_matrices(U, PUQ.permute(inv_permute(p_inv), q_inv));
        compare_matrices(PUQ, U.permute(p_inv, inv_permute(q_inv)));
    }

    SECTION("Permuted P L x = b, with unknown P") {
        // Create RHS for Lx = b
        // Set b s.t. x == {1, 2, 3, 4, 5, 6} to see output permutation
        const std::vector<double> b = { 1,  6, 18, 40, 75, 126};
        const std::vector<double> expect = {1, 2, 3, 4, 5, 6};

        // Solve Lx = b
        const std::vector<double> x = lsolve(L, b);
        CHECK_THAT(is_close(x, expect, tol), AllTrue());

        // Solve PLx = b
        const std::vector<double> xp = lsolve_rows(PL, b);

        REQUIRE_THAT(is_close(xp, expect, tol), AllTrue());
    }

    SECTION("Permuted L Q x = b, with unknown Q") {
        // Create RHS for Lx = b
        // Set b s.t. x == {1, 2, 3, 4, 5, 6} to see output permutation
        const std::vector<double> b = { 1,  6, 18, 40, 75, 126};
        const std::vector<double> expect = {1, 2, 3, 4, 5, 6};

        // Solve L Q.T x = b
        const std::vector<double> xp = lsolve_cols(LQ, b);

        REQUIRE_THAT(is_close(xp, expect, tol), AllTrue());
    }

    SECTION("Permuted P U x = b, with unknown P") {
        // Create RHS for Ux = b
        // Set b s.t. x == {1, 2, 3, 4, 5, 6} to see output permutation
        const std::vector<double> b = {21, 40, 54, 60, 55, 36};
        const std::vector<double> expect = {1, 2, 3, 4, 5, 6};

        // Solve Ux = b (un-permuted)
        const std::vector<double> x = usolve(U, b);
        CHECK_THAT(is_close(x, expect, tol), AllTrue());

        // Solve PUx = b
        const std::vector<double> xp = usolve_rows(PU, b);
        REQUIRE_THAT(is_close(xp, expect, tol), AllTrue());
    }

    SECTION("Permuted U Q x = b, with unknown Q") {
        // Create RHS for Ux = b
        // Set b s.t. x == {1, 2, 3, 4, 5, 6} to see output permutation
        const std::vector<double> b = {21, 40, 54, 60, 55, 36};
        const std::vector<double> expect = {1, 2, 3, 4, 5, 6};

        // Solve U Q.T x = b
        const std::vector<double> xp = usolve_cols(UQ, b);

        REQUIRE_THAT(is_close(xp, expect, tol), AllTrue());
    }

    SECTION("Permuted P L Q x = b, with unknown P and Q") {
        // Create RHS for Lx = b
        // Set b s.t. x == {1, 2, 3, 4, 5, 6} to see output permutation
        const std::vector<double> b = { 1,  6, 18, 40, 75, 126};
        const std::vector<double> expect = {1, 2, 3, 4, 5, 6};

        // Solve P L Q x = b
        const std::vector<double> xt = tri_solve_perm(PLQ, b);
        REQUIRE_THAT(is_close(xt, expect, tol), AllTrue());
    }

    SECTION("Permuted P U Q x = b, with unknown P and Q") {
        // Create RHS for Ux = b
        // Set b s.t. x == {1, 2, 3, 4, 5, 6} to see output permutation
        const std::vector<double> b = {21, 40, 54, 60, 55, 36};
        const std::vector<double> expect = {1, 2, 3, 4, 5, 6};

        // Solve P U Q x = b
        std::vector<double> xp = tri_solve_perm(PUQ, b, true);
        REQUIRE_THAT(is_close(xp, expect, tol), AllTrue());
    }
}


}  // namespace cs

/*==============================================================================
 *============================================================================*/
