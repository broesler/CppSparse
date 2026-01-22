/*==============================================================================
 *     File: test_fillreducing.cpp
 *  Created: 2025-05-08 13:34
 *   Author: Bernie Roesler
 *
 *  Description: Test Chapter 7: Fill-Reducing Orderings.
 *
 *============================================================================*/

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <numeric>   // accumulate, iota
#include <vector>

#include "csparse.h"
#include "test_helpers.h"

using Catch::Matchers::UnorderedEquals;

namespace cs {


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
        expect_C = CSCMatrix{{}, A.indices(), A.indptr(), A.shape()};
    }

    SECTION("A + A^T") {
        order = AMDOrder::APlusAT;
        expect_C = A + A.transpose(values);
    }

    SECTION("A^T A (no dense)") {
        order = AMDOrder::ATANoDenseRows;

        if (dense == 0) {
            // Drop all rows
            A = CSCMatrix{A.shape()};
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
    expect_C.fkeep([] (csint i, csint j, [[maybe_unused]] double v) { return i != j; });

    const CSCMatrix C = build_graph(A, order, dense);

    CHECK(C.data().empty());
    check_sparse_allclose(C, expect_C, values);
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

TEST_CASE("AMD with M < N", "[amd][M < N]")
{
    csint M = 7;
    csint N = 10;
    const CSCMatrix A = davis_example_amd().slice(0, M, 0, N);

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
    }

    SECTION("A^T A (no dense)") {
        order = AMDOrder::ATANoDenseRows;
        expect_p = {9, 4, 8, 0, 3, 5, 7, 1, 2, 6};
    }

    SECTION("A^T A") {
        order = AMDOrder::ATA;
        expect_p = {9, 4, 8, 0, 3, 5, 7, 1, 2, 6};
    }

    if (order == AMDOrder::APlusAT) {
        REQUIRE_THROWS_WITH(
            amd(A, order),
            "Matrix must be square for APlusAT!"
        );
    } else {
        std::vector<csint> p = amd(A, order);
        CHECK(p == expect_p);
    }
}


TEST_CASE("AMD with M > N", "[amd][M > N]")
{
    csint M = 10;
    csint N = 7;
    const CSCMatrix A = davis_example_amd().slice(0, M, 0, N);

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
    }

    SECTION("A^T A (no dense)") {
        order = AMDOrder::ATANoDenseRows;
        expect_p = {4, 0, 1, 3, 2, 5, 6};
    }

    SECTION("A^T A") {
        order = AMDOrder::ATA;
        expect_p = {4, 0, 1, 3, 2, 5, 6};
    }

    if (order == AMDOrder::APlusAT) {
        REQUIRE_THROWS_WITH(
            amd(A, order),
            "Matrix must be square for APlusAT!"
        );
    } else {
        std::vector<csint> p = amd(A, order);
        CHECK(p == expect_p);
    }
}


TEST_CASE("Maximum Matching", "[maxmatch]")
{
    CSCMatrix A = davis_example_amd();
    auto [M, N] = A.shape();

    bool recursive = GENERATE(true, false);
    CAPTURE(recursive);

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

    MaxMatch res;
    if (recursive) {
        res = detail::maxtrans_r(A, seed);
    } else {
        res = maxtrans(A, seed);
    }

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

    if (seed == 0 && !recursive) {
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


}  // namespace cs

/*==============================================================================
 *============================================================================*/
