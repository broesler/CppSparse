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

constexpr double solve_tol = 1e-12;

struct DenseRHS {};
struct SparseRHS {};


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
    check_vectors_allclose(x, expect, solve_tol);
}


TEMPLATE_TEST_CASE(
    "Cholesky Solution with Matrix RHS",
    "[cholsol-matrix]",
    DenseRHS,
    SparseRHS
)
{
    CSCMatrix A = davis_example_chol();
    auto [M, N] = A.shape();

    AMDOrder order = GENERATE(
        AMDOrder::Natural,
        AMDOrder::APlusAT
    );
    CAPTURE(order);

    // Create RHS for Ax = b
    csint K = 3;  // arbitrary number of RHS columns
    std::vector<double> expect_x(N * K);
    std::iota(expect_x.begin(), expect_x.end(), 1);

    const std::vector<double> b = A * expect_x;
    std::vector<double> x;

    if constexpr (std::is_same_v<TestType, DenseRHS>) {
        x = chol_solve(A, b, order);  // solve Ax = b
    } else {
        x = chol_solve(A, CSCMatrix(b, {M, K}), order);
    }

    // Check that Ax = b
    check_vectors_allclose(x, expect_x, solve_tol);
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

    std::vector<double> b;
    QRSolveResult res;

    SECTION("Square") {
        // Create RHS for Ax = b
        b = A * expect;

        // Solve Ax = b
        res = qr_solve(A, b, order);

        // Check that Ax = b
        check_vectors_allclose(res.x, expect, 1e-13);
        REQUIRE(res.rnorm < 1e-13);
    }

    SECTION("Over-determined") {
        // Create a new matrix with more rows than columns
        csint k = 3;
        A = A.slice(0, M, 0, N - k);

        // Take only the first N - k elements of expect
        expect = std::vector<double>(expect.begin(), expect.end() - k);

        b = A * expect;

        // Solve Ax = b
        res = qr_solve(A, b, order);

        // Check that Ax = b
        check_vectors_allclose(res.x, expect, 1e-13);
        REQUIRE(res.rnorm < 1e-13);
    }

    SECTION("Under-determined") {
        // Create a new matrix with more rows than columns
        csint k = 3;
        A = A.slice(0, M - k, 0, N);

        b = A * expect;

        // Solve Ax = b
        res = qr_solve(A, b, order);  // (M - k, N)

        // Actual expect (python and MATLAB)
        const std::vector<double> min_norm_x{
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
        check_vectors_allclose(res.x, min_norm_x, solve_tol);
        REQUIRE(res.rnorm < 1e-13);
    }
}


TEMPLATE_TEST_CASE(
    "QR Solution with Matrix RHS",
    "[qrsol-matrix]",
    DenseRHS,
    SparseRHS
)
{
    CSCMatrix A = davis_example_qr();
    auto [M, N] = A.shape();

    AMDOrder order = GENERATE(
        AMDOrder::Natural,
        AMDOrder::ATA
    );
    CAPTURE(order);

    // Create RHS for Ax = b
    csint K = 3;  // arbitrary number of RHS columns
    std::vector<double> expect(N * K);
    std::iota(expect.begin(), expect.end(), 1);


    std::vector<double> b;
    QRSolveResult res;

    SECTION("Square") {
        // Create RHS for Ax = b
        b = A * expect;

        // Solve Ax = b
        if constexpr (std::is_same_v<TestType, DenseRHS>) {
            res = qr_solve(A, b, order);
        } else {
            res = qr_solve(A, CSCMatrix(b, {M, K}), order);
        }

        // Check that Ax = b
        check_vectors_allclose(res.x, expect, 1e-12);
        REQUIRE(res.rnorm < 1e-12);
    }

    SECTION("Over-determined") {
        std::vector<double> expect_x = expect;  // copy the original expect

        // Create a new matrix with more rows than columns
        csint k = 3;
        A = A.slice(0, M, 0, N - k);

        // Take only the first N - k rows of expect_x
        csint Nmk = N - k;
        for (csint j = 0; j < K; j++) {
            auto read_start = expect_x.begin() + j * N;
            auto write_start = expect_x.begin() + j * Nmk;
            std::move(read_start, read_start + Nmk, write_start);
        } 
        expect_x.resize(Nmk * K);

        b = A * expect_x;

        // Solve Ax = b
        // Solve Ax = b
        if constexpr (std::is_same_v<TestType, DenseRHS>) {
            res = qr_solve(A, b, order);
        } else {
            res = qr_solve(A, CSCMatrix(b, {M, K}), order);
        }

        // Check that Ax = b
        check_vectors_allclose(res.x, expect_x, 1e-12);
        REQUIRE(res.rnorm < 1e-12);
    }

    SECTION("Under-determined") {
        // Create a new matrix with more rows than columns
        csint k = 3;
        A = A.slice(0, M - k, 0, N);

        b = A * expect;

        // Solve Ax = b
        if constexpr (std::is_same_v<TestType, DenseRHS>) {
            res = qr_solve(A, b, order);
        } else {
            res = qr_solve(A, CSCMatrix(b, {M - k, K}), order);
        }

        // Actual expect_x la.lstsq(A.toarray(), b)
        const std::vector<double> min_norm_x{
             3.2222222222222143,  3.1111111111111125,  3.                ,  4.000000000000004 ,  5.961538461538462 ,  1.192307692307692 ,  4.7777777777777715,  0.,
             9.444444444444414 , 10.222222222222229 , 10.999999999999996 , 12.000000000000016 , 15.192307692307692 ,  3.0384615384615374, 14.55555555555553  ,  0.,
            15.666666666666636 , 17.333333333333343 , 19.                , 20.000000000000018 , 24.423076923076923 ,  4.884615384615383 , 24.33333333333331  ,  0.
        };

        // Check that Ax = b
        check_vectors_allclose(res.x, min_norm_x, 1e-12);
        REQUIRE(res.rnorm < 1e-12);
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

    const std::vector<double> piv_tols{0.0, 1e-3, 1.0};

    for (const auto& piv_tol : piv_tols) {
        CAPTURE(piv_tol);

        // Solve Ax = b
        std::vector<double> x = lu_solve(A, b, order, piv_tol);

        // Check that Ax = b
        check_vectors_allclose(x, expect, solve_tol);
    }
}


TEST_CASE("LU with Iterative Refinement", "[lusol-ir]") {
    double add_diag = 0.0;
    bool randomized = true;
    CSCMatrix A = davis_example_qr(add_diag, randomized);
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

    // Solve Ax = b
    double piv_tol = 1e-3;
    std::vector<double> x = lu_solve(A, b, order, piv_tol, 0);
    std::vector<double> x_ir = lu_solve(A, b, order, piv_tol, 2);

    // Check that Ax = b
    check_vectors_allclose(x, expect, 1e-12);
    check_vectors_allclose(x_ir, expect, 1e-12);

#ifdef DEBUG
    double r_norm = norm(b - A * x);
    double r_ir_norm = norm(b - A * x_ir);

    std::cout << "\nIterative Refinement: " << std::endl;
    std::cout << "x    = " << x << std::endl;
    std::cout << "x_ir = " << x_ir << std::endl;
    std::cout << "norm(r)    = " << r_norm << std::endl;
    std::cout << "norm(r_ir) = " << r_ir_norm << std::endl;
#endif
}


TEMPLATE_TEST_CASE(
    "LU Solution with Matrix RHS",
    "[lusol-matrix]",
    DenseRHS,
    SparseRHS
)
{
    CSCMatrix A = davis_example_qr();
    auto [M, N] = A.shape();

    AMDOrder order = GENERATE(
        AMDOrder::Natural,
        AMDOrder::ATANoDenseRows,
        AMDOrder::ATA
    );
    CAPTURE(order);

    // Create RHS for Ax = b
    csint K = 3;  // arbitrary number of RHS columns
    std::vector<double> expect_x(N * K);
    std::iota(expect_x.begin(), expect_x.end(), 1);

    const std::vector<double> b = A * expect_x;
    std::vector<double> x;

    // Solve Ax = b
    if constexpr (std::is_same_v<TestType, DenseRHS>) {
        x = lu_solve(A, b, order);
    } else {
        x = lu_solve(A, CSCMatrix(b, {M, K}), order);
    }

    // Check that Ax = b
    check_vectors_allclose(x, expect_x, 1e-10);
}


/*------------------------------------------------------------------------------
 *         Exercise 8.1: General Sparse Solver
 *----------------------------------------------------------------------------*/
struct SingleRHS { static constexpr csint K = 1; };
struct MultipleRHS { static constexpr csint K = 3; };

struct PositiveA {};
struct NegativeA {};

using RhsCombinations = std::tuple<
    std::tuple<DenseRHS, SingleRHS>,
    std::tuple<SparseRHS, SingleRHS>,
    std::tuple<DenseRHS, MultipleRHS>,
    std::tuple<SparseRHS, MultipleRHS>
>;


TEMPLATE_LIST_TEST_CASE("Backslash: Triangular", "[spsolve-tri]", RhsCombinations)
{
    using RhsType = std::tuple_element_t<0, TestType>;
    using RhsCountType = std::tuple_element_t<1, TestType>;
    constexpr csint K = RhsCountType::K;

    CSCMatrix A = davis_example_small().tocsc();
    auto [M, N] = A.shape();

    // Create RHS for Lx = b
    std::vector<double> expect_x(N * K);
    std::iota(expect_x.begin(), expect_x.end(), 1);

    const std::vector<csint> p{3, 0, 1, 2};
    const std::vector<csint> q{1, 2, 0, 3};

    auto solve_and_check = [&](const CSCMatrix& A) {
        std::vector<double> b = A * expect_x;
        std::vector<double> x;
        if constexpr (std::is_same_v<RhsType, DenseRHS>) {
            x = spsolve(A, b);
        } else {
            x = spsolve(A, CSCMatrix(b, {M, K}));
        }
        check_vectors_allclose(x, expect_x, solve_tol);
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


using CholeskyCombinations = std::tuple<
    std::tuple<DenseRHS, SingleRHS, PositiveA>,
    std::tuple<SparseRHS, SingleRHS, PositiveA>,
    std::tuple<DenseRHS, SingleRHS, NegativeA>,
    std::tuple<SparseRHS, SingleRHS, NegativeA>,
    std::tuple<DenseRHS, MultipleRHS, PositiveA>,
    std::tuple<SparseRHS, MultipleRHS, PositiveA>,
    std::tuple<DenseRHS, MultipleRHS, NegativeA>,
    std::tuple<SparseRHS, MultipleRHS, NegativeA>
>;


TEMPLATE_LIST_TEST_CASE("Backslash: Cholesky", "[spsolve-chol]", CholeskyCombinations)
{
    using RhsType = std::tuple_element_t<0, TestType>;
    using RhsCountType = std::tuple_element_t<1, TestType>;
    using ASignType = std::tuple_element_t<2, TestType>;
    constexpr csint K = RhsCountType::K;

    CSCMatrix A = davis_example_chol();
    auto [M, N] = A.shape();

    // Create RHS for Ax = b
    std::vector<double> expect_x(N * K);
    std::iota(expect_x.begin(), expect_x.end(), 1);

    std::vector<double> b = A * expect_x;

    if constexpr (std::is_same_v<ASignType, NegativeA>) {
        A = -A;
        b = -b;
    }

    std::vector<double> x;
    if constexpr (std::is_same_v<RhsType, DenseRHS>) {
        x = spsolve(A, b);
    } else {
        x = spsolve(A, CSCMatrix(b, {M, K}));
    }

    check_vectors_allclose(x, expect_x, solve_tol);
}


TEMPLATE_LIST_TEST_CASE("Backslash: LU Symmetric", "[spsolve-lu-sym]", RhsCombinations)
{
    using RhsType = std::tuple_element_t<0, TestType>;
    using RhsCountType = std::tuple_element_t<1, TestType>;
    constexpr csint K = RhsCountType::K;

    CSCMatrix A = davis_example_chol();
    auto [M, N] = A.shape();

    // Change one element to make unsymmetric
    A(1, 2) = 0.0;
    A.dropzeros();

    CHECK(A.structural_symmetry() < 1.0);

    // Create RHS for Ax = b
    std::vector<double> expect_x(N * K);
    std::iota(expect_x.begin(), expect_x.end(), 1);

    const std::vector<double> b = A * expect_x;

    std::vector<double> x;
    if constexpr (std::is_same_v<RhsType, DenseRHS>) {
        x = spsolve(A, b);
    } else {
        x = spsolve(A, CSCMatrix(b, {M, K}));
    }

    check_vectors_allclose(x, expect_x, solve_tol);
}


TEMPLATE_LIST_TEST_CASE("Backslash: LU Unsymmetric", "[spsolve-lu-unsym]", RhsCombinations)
{
    using RhsType = std::tuple_element_t<0, TestType>;
    using RhsCountType = std::tuple_element_t<1, TestType>;
    constexpr csint K = RhsCountType::K;

    CSCMatrix A = davis_example_chol();
    auto [M, N] = A.shape();

    // Drop upper bands to make less symmetric (below 0.3 threshold)
    A.band(-M, 1);

    CHECK(A.structural_symmetry() < 0.3);

    // Create RHS for Ax = b
    std::vector<double> expect_x(N * K);
    std::iota(expect_x.begin(), expect_x.end(), 1);

    const std::vector<double> b = A * expect_x;

    std::vector<double> x;
    if constexpr (std::is_same_v<RhsType, DenseRHS>) {
        x = spsolve(A, b);
    } else {
        x = spsolve(A, CSCMatrix(b, {M, K}));
    }

    check_vectors_allclose(x, expect_x, solve_tol);
}


TEMPLATE_LIST_TEST_CASE("Backslash: QR", "[spsolve-qr]", RhsCombinations)
{
    using RhsType = std::tuple_element_t<0, TestType>;
    using RhsCountType = std::tuple_element_t<1, TestType>;
    constexpr csint K = RhsCountType::K;

    CSCMatrix A = davis_example_qr();
    auto [M, N] = A.shape();

    // Create RHS for Ax = b
    std::vector<double> expect_x(N * K);
    std::iota(expect_x.begin(), expect_x.end(), 1);

    std::vector<double> b, x;

    SECTION("Square") {
        b = A * expect_x;

        if constexpr (std::is_same_v<RhsType, DenseRHS>) {
            x = spsolve(A, b);
        } else {
            x = spsolve(A, CSCMatrix(b, {M, K}));
        }

        check_vectors_allclose(x, expect_x, solve_tol);
    }

    SECTION("Over-determined") {
        // Create a new matrix with more rows than columns
        csint k = 3;
        A = A.slice(0, M, 0, N - k);

        // Take only the first N - k rows of expect_x
        csint Nmk = N - k;
        for (csint j = 0; j < K; j++) {
            auto read_start = expect_x.begin() + j * N;
            auto write_start = expect_x.begin() + j * Nmk;
            std::move(read_start, read_start + Nmk, write_start);
        } 
        expect_x.resize(Nmk * K);

        b = A * expect_x;

        if constexpr (std::is_same_v<RhsType, DenseRHS>) {
            x = spsolve(A, b);
        } else {
            x = spsolve(A, CSCMatrix(b, {M, K}));
        }

        check_vectors_allclose(x, expect_x, solve_tol);
    }

    SECTION("Under-determined") {
        // Create a new matrix with more rows than columns
        csint k = 3;
        A = A.slice(0, M - k, 0, N);

        b = A * expect_x;

        if constexpr (std::is_same_v<RhsType, DenseRHS>) {
            x = spsolve(A, b);
        } else {
            x = spsolve(A, CSCMatrix(b, {M - k, K}));
        }

        // Actual expect_x (python and MATLAB)
        std::vector<double> min_norm_x;

        if constexpr (std::is_same_v<RhsCountType, SingleRHS>) {
            min_norm_x = {
                3.2222222222222143,
                3.1111111111111125,
                3.                ,
                4.000000000000004 ,
                5.961538461538462 ,
                1.192307692307692 ,
                4.7777777777777715,
                0.                
            };
        } else {
            min_norm_x = {
                 3.2222222222222143,  3.1111111111111125,  3.                ,  4.000000000000004 ,  5.961538461538462 ,  1.192307692307692 ,  4.7777777777777715,  0.,
                 9.444444444444414 , 10.222222222222229 , 10.999999999999996 , 12.000000000000016 , 15.192307692307692 ,  3.0384615384615374, 14.55555555555553  ,  0.,
                15.666666666666636 , 17.333333333333343 , 19.                , 20.000000000000018 , 24.423076923076923 ,  4.884615384615383 , 24.33333333333331  ,  0.
            };
        }

        check_vectors_allclose(x, min_norm_x, solve_tol);
    }
}

}  // namespace cs

/*==============================================================================
 *============================================================================*/
