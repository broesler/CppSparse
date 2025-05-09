/*==============================================================================
 *     File: test_solve.cpp
 *  Created: 2025-05-08 13:40
 *   Author: Bernie Roesler
 *
 *  Description: Test Chapter 8: Solving Sparse Linear Systems.
 *
 *============================================================================*/

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <numeric>  // iota
#include <vector>

#include "csparse.h"
#include "test_helpers.h"

using Catch::Matchers::AllTrue;

namespace cs {


TEST_CASE("Cholesky Solution", "[cholsol]")
{
    CSCMatrix A = davis_example_chol();
    auto [M, N] = A.shape();

    AMDOrder order = GENERATE(
        AMDOrder::Natural,
        AMDOrder::APlusAT
    );
    CAPTURE(order);

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
    CAPTURE(order);

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
        const std::vector<double> min_norm_x = {
            3.2222222222222143,
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
    CAPTURE(order);

    // Create RHS for Ax = b
    std::vector<double> expect(N);
    std::iota(expect.begin(), expect.end(), 1);

    const std::vector<double> b = A * expect;

    // TODO test with different pivot tolerance
    // Solve Ax = b
    std::vector<double> x = lu_solve(A, b, order);

    // Check that Ax = b
    REQUIRE_THAT(is_close(x, expect, tol), AllTrue());
}


}  // namespace cs

/*==============================================================================
 *============================================================================*/
