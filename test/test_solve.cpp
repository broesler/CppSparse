/*==============================================================================
 *     File: test_solve.cpp
 *  Created: 2025-05-08 13:40
 *   Author: Bernie Roesler
 *
 *  Description: Test Chapter 8: Solving Sparse Linear Systems.
 *
 *============================================================================*/

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
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
    check_vectors_allclose(x, expect, tol);
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
        check_vectors_allclose(x, expect, 1e-13);
    }

    SECTION("Over-determined") {
        // Create a new matrix with more rows than columns
        csint k = 3;
        A = A.slice(0, M, 0, N - k);

        // Take only the first N - k elements of expect
        expect = std::vector<double>(expect.begin(), expect.end() - k);

        b = A * expect;

        // Solve Ax = b
        x = qr_solve(A, b, order);

        // Check that Ax = b
        check_vectors_allclose(x, expect, 1e-13);
    }

    SECTION("Under-determined") {
        // Create a new matrix with more rows than columns
        csint k = 3;
        A = A.slice(0, M - k, 0, N);

        b = A * expect;

        // Solve Ax = b
        x = qr_solve(A, b, order);  // (M - k, N)

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
        check_vectors_allclose(x, min_norm_x, tol);
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
    check_vectors_allclose(x, expect, tol);
}


/*------------------------------------------------------------------------------
 *         Exercise 8.1: General Sparse Solver
 *----------------------------------------------------------------------------*/
struct DenseRHS {};
struct SparseRHS {};

CSCMatrix sparse_from_dense(const std::vector<double>& b) {
    csint N = static_cast<csint>(b.size());
    return CSCMatrix(b, {N, 1});
}


TEMPLATE_TEST_CASE("Backslash: Triangular", "[spsol-tri]", DenseRHS, SparseRHS)
{
    CSCMatrix A = davis_example_small().tocsc();
    auto [M, N] = A.shape();

    // Create RHS for Lx = b
    std::vector<double> expect_x(N);
    std::iota(expect_x.begin(), expect_x.end(), 1);

    const std::vector<csint> p = {3, 0, 1, 2};
    const std::vector<csint> q = {1, 2, 0, 3};

    auto solve_and_check = [&](const CSCMatrix& A) {
        std::vector<double> b = A * expect_x;
        std::vector<double> x;
        if constexpr (std::is_same_v<TestType, DenseRHS>) {
            x = spsolve(A, b);
        } else {
            x = spsolve(A, sparse_from_dense(b));
        }
        check_vectors_allclose(x, expect_x, tol);
    };

    // Triangular with non-zero diagonal
    SECTION("Lx = b") {
        CSCMatrix L = A.band(-N, 0);
        solve_and_check(L);
    }

    SECTION("Ux = b") {
        CSCMatrix U = A.band(0, N);
        solve_and_check(U);
    }

    SECTION("Permuted lower triangular") {
        CSCMatrix L = A.band(-N, 0);
        CSCMatrix PLQ = L.permute(inv_permute(p), q).to_canonical();
        solve_and_check(PLQ);
    }

    SECTION("Permuted upper triangular") {
        CSCMatrix U = A.band(0, N);
        CSCMatrix PUQ = U.permute(inv_permute(p), q).to_canonical();
        solve_and_check(PUQ);
    }
}


TEMPLATE_TEST_CASE("Backslash: Cholesky", "[spsol-chol]", DenseRHS, SparseRHS)
{
    CSCMatrix A = davis_example_chol();
    auto [M, N] = A.shape();

    // Create RHS for Ax = b
    std::vector<double> expect_x(N);
    std::iota(expect_x.begin(), expect_x.end(), 1);

    const std::vector<double> b = A * expect_x;

    std::vector<double> x;
    if constexpr (std::is_same_v<TestType, DenseRHS>) {
        x = spsolve(A, b);
    } else {
        x = spsolve(A, sparse_from_dense(b));
    }

    check_vectors_allclose(x, expect_x, tol);
}


TEMPLATE_TEST_CASE("Backslash: LU Symmetric", "[spsol-lu-sym]", DenseRHS, SparseRHS)
{
    CSCMatrix A = davis_example_chol();
    auto [M, N] = A.shape();

    // Change one element to make unsymmetric
    A(1, 2) = 0.0;
    A.dropzeros();

    CHECK(A.structural_symmetry() < 1.0);

    // Create RHS for Ax = b
    std::vector<double> expect_x(N);
    std::iota(expect_x.begin(), expect_x.end(), 1);

    const std::vector<double> b = A * expect_x;

    std::vector<double> x;
    if constexpr (std::is_same_v<TestType, DenseRHS>) {
        x = spsolve(A, b);
    } else {
        x = spsolve(A, sparse_from_dense(b));
    }

    check_vectors_allclose(x, expect_x, tol);
}


TEMPLATE_TEST_CASE("Backslash: LU Unsymmetric", "[spsol-lu-unsym]", DenseRHS, SparseRHS)
{
    CSCMatrix A = davis_example_chol();
    auto [M, N] = A.shape();

    // Drop upper bands to make less symmetric (below 0.3 threshold)
    A.band(-M, 1);

    CHECK(A.structural_symmetry() < 0.3);

    // Create RHS for Ax = b
    std::vector<double> expect_x(N);
    std::iota(expect_x.begin(), expect_x.end(), 1);

    const std::vector<double> b = A * expect_x;

    std::vector<double> x;
    if constexpr (std::is_same_v<TestType, DenseRHS>) {
        x = spsolve(A, b);
    } else {
        x = spsolve(A, sparse_from_dense(b));
    }

    check_vectors_allclose(x, expect_x, tol);
}


TEMPLATE_TEST_CASE("Backslash: QR", "[spsol-qr]", DenseRHS, SparseRHS)
{
    CSCMatrix A = davis_example_qr();
    auto [M, N] = A.shape();

    // Create RHS for Ax = b
    std::vector<double> expect_x(N);
    std::iota(expect_x.begin(), expect_x.end(), 1);

    std::vector<double> b, x;

    SECTION("Square") {
        b = A * expect_x;

        std::vector<double> x;
        if constexpr (std::is_same_v<TestType, DenseRHS>) {
            x = spsolve(A, b);
        } else {
            x = spsolve(A, sparse_from_dense(b));
        }

        check_vectors_allclose(x, expect_x, 1e-13);
    }

    SECTION("Over-determined") {
        // Create a new matrix with more rows than columns
        csint k = 3;
        A = A.slice(0, M, 0, N - k);

        // Take only the first N - k elements of expect_x
        expect_x = std::vector<double>(expect_x.begin(), expect_x.end() - k);

        b = A * expect_x;

        std::vector<double> x;
        if constexpr (std::is_same_v<TestType, DenseRHS>) {
            x = spsolve(A, b);
        } else {
            x = spsolve(A, sparse_from_dense(b));
        }

        check_vectors_allclose(x, expect_x, 1e-13);
    }

    SECTION("Under-determined") {
        // Create a new matrix with more rows than columns
        csint k = 3;
        A = A.slice(0, M - k, 0, N);

        b = A * expect_x;

        std::vector<double> x;  // (M - k, N)
        if constexpr (std::is_same_v<TestType, DenseRHS>) {
            x = spsolve(A, b);
        } else {
            x = spsolve(A, sparse_from_dense(b));
        }

        // Actual expect_x (python and MATLAB)
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

        check_vectors_allclose(x, min_norm_x, tol);
    }
}

}  // namespace cs

/*==============================================================================
 *============================================================================*/
