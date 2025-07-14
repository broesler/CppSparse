/*==============================================================================
 *     File: test_lu.cpp
 *  Created: 2025-05-08 13:28
 *   Author: Bernie Roesler
 *
 *  Description: Test Chapter 6: LU Factorization.
 *
 *============================================================================*/

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>     // log2
#include <iostream>  // cerr
#include <numeric>   // iota
#include <sstream>
#include <vector>

#include "csparse.h"
#include "test_helpers.h"

using Catch::Matchers::AllTrue;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::RangeEquals;
using Catch::Matchers::UnorderedEquals;


namespace cs {


/** Define a helper function to test LU decomposition */
LUResult lu_test(const CSCMatrix& A, AMDOrder order=AMDOrder::Natural)
{
    SymbolicLU S = slu(A, order);
    LUResult res = lu(A, S);
    CSCMatrix LU = (res.L * res.U).droptol().to_canonical();

    // Test that permutations are valid
    auto [M, N] = A.shape();
    std::vector<csint> row_perm(M);
    std::vector<csint> col_perm(N);
    std::iota(row_perm.begin(), row_perm.end(), 0);
    std::iota(col_perm.begin(), col_perm.end(), 0);

    CHECK_THAT(res.p_inv, UnorderedEquals(row_perm));
    CHECK_THAT(res.q,     UnorderedEquals(col_perm));

    CSCMatrix PAQ = A.permute(res.p_inv, res.q).to_canonical();
    check_sparse_allclose(LU, PAQ);
    return res;
}


TEST_CASE("Symbolic LU Factorization of Square Matrix", "[lu][M == N][symbolic]")
{
    const CSCMatrix A = davis_example_qr(10);
    auto [M, N] = A.shape();

    std::vector<csint> expect_q = {0, 1, 2, 3, 4, 5, 6, 7};

    SECTION("Symbolic Factorization") {
        AMDOrder order = AMDOrder::Natural;
        csint expect_lnz = 0;

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


TEST_CASE("Numeric LU Factorization of Square Matrix", "[lu][M == N][numeric]")
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


TEST_CASE("Solve Ax = b with LU", "[lu_solve]")
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

    check_vectors_allclose(x, x_ov, tol);
    check_vectors_allclose(x, expect, tol);
}


TEST_CASE("Exercise 6.1: Solve A^T x = b with LU", "[ex6.1][lu_tsolve]")
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

    check_vectors_allclose(x, x_ov, tol);
    check_vectors_allclose(x, expect, tol);
}


TEST_CASE("Exercise 6.3: Column Pivoting in LU", "[ex6.3][lu_colpiv]")
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
        check_sparse_allclose(LU, PAQ);
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
            A.dropzeros();

            expect_q = {0, 1, 2, 4, 5, 6, 7, 3};
        }

        SECTION("Multiple zero columns") {
            // Remove a column to test the column pivoting
            for (const auto& k : {2, 3, 5}) {
                for (csint i = 0; i < M; i++) {
                    A(i, k) = 0.0;
                }
            }
            A.dropzeros();

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


TEST_CASE("Exercise 6.4: relu", "[ex6.4][relu]")
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
    check_sparse_allclose(B, PBp);  // LU should match the un-permuted B
    check_sparse_allclose(LU, B.to_canonical());
}


// Exercise 6.5: LU for square, singular matrices
void run_lu_singular_test(
    std::function<void(CSCMatrix&, csint, csint)> matrix_modifier
) {
    CSCMatrix A = davis_example_qr(10).to_canonical();
    auto [M, N] = A.shape();

    AMDOrder order = GENERATE(
        AMDOrder::Natural,
        AMDOrder::APlusAT,
        AMDOrder::ATANoDenseRows
    );
    bool permute_rows = GENERATE(true, false);
    bool structural = GENERATE(true, false);
    CAPTURE(order, permute_rows, structural);

    if (permute_rows) {
        std::vector<csint> p = {5, 1, 7, 0, 2, 6, 4, 3};  // arbitrary
        A = A.permute_rows(inv_permute(p));
    }

    // Do the actual modification to make the matrix singular
    matrix_modifier(A, M, N);

    if (structural) {
        A.dropzeros();
    }

    lu_test(A, order);
}


TEST_CASE("Exercise 6.5: LU with Single pair of linearly dependent columns", "[ex6.5][lu_singular]")
{
    run_lu_singular_test(
        [](CSCMatrix& A, csint M, csint N) {
            for (csint i = 0; i < M; i++) {
                A(i, 3) = 2 * A(i, 5);
            }
        }
    );
}


TEST_CASE("Exercise 6.5: LU with Two pairs of linearly dependent columns", "[ex6.5][lu_singular]")
{
    run_lu_singular_test(
        [](CSCMatrix& A, csint M, csint N) {
            for (csint i = 0; i < M; i++) {
                // These two sets create a zero row:
                // A(i, 3) = 2 * A(i, 5);
                // A(i, 2) = 3 * A(i, 4);
                // These two sets *do not* create a zero row (separate test)
                A(i, 2) = 2 * A(i, 6);
                A(i, 4) = 3 * A(i, 5);
            }
        }
    );
}


TEST_CASE("Exercise 6.5: LU with Single pair of linearly dependent rows", "[ex6.5][lu_singular]")
{
    run_lu_singular_test(
        [](CSCMatrix& A, csint M, csint N) {
            for (csint j = 0; j < N; j++) {
                A(3, j) = 2 * A(5, j);
            }
        }
    );
}


TEST_CASE("Exercise 6.5: LU with Two pairs of linearly dependent rows", "[ex6.5][lu_singular]")
{
    run_lu_singular_test(
        [](CSCMatrix& A, csint M, csint N) {
            for (csint j = 0; j < N; j++) {
                A(3, j) = 2 * A(5, j);
                A(2, j) = 3 * A(4, j);
            }
        }
    );
}


TEST_CASE("Exercise 6.5: LU with Single zero column", "[ex6.5][lu_singular]")
{
    run_lu_singular_test(
        [](CSCMatrix& A, csint M, csint N) {
            for (csint i = 0; i < M; i++) {
                A(i, 3) = 0.0;
            }
        }
    );
}


TEST_CASE("Exercise 6.5: LU with Multiple zero columns", "[ex6.5][lu_singular]")
{
    run_lu_singular_test(
        [](CSCMatrix& A, csint M, csint N) {
            for (csint i = 0; i < M; i++) {
                for (const auto& j : {2, 3, 5}) {
                    A(i, j) = 0.0;
                }
            }
        }
    );
}


TEST_CASE("Exercise 6.5: LU with Single zero row", "[ex6.5][lu_singular]")
{
    run_lu_singular_test(
        [](CSCMatrix& A, csint M, csint N) {
            for (csint j = 0; j < N; j++) {
                A(3, j) = 0.0;
            }
        }
    );
}


TEST_CASE("Exercise 6.5: LU with Multiple zero rows", "[ex6.5][lu_singular]")
{
    run_lu_singular_test(
        [](CSCMatrix& A, csint M, csint N) {
            for (const auto& i : {2, 3, 5}) {
                for (csint j = 0; j < N; j++) {
                    A(i, j) = 0.0;
                }
            }
        }
    );
}


TEST_CASE("Exercise 6.6: LU Factorization of Rectangular Matrices", "[ex6.6][lu][M < N][M > N]")
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


TEST_CASE("Exercise 6.7: Crout's method LU Factorization", "[ex6.7][crout]")
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

        check_sparse_allclose(LU, PA);
    }
}


TEST_CASE("Exercise 6.11: lu_realloc", "[ex6.11][lu_realloc]")
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


TEST_CASE("Exercise 6.13: Incomplete LU Decomposition", "[ex6.13][ilu]")
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

            check_sparse_allclose(res.L, ires.L);
            check_sparse_allclose(res.U, ires.U);
            check_sparse_allclose(iLU, A);
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
            check_all_greater_equal(ires.L.data(), drop_tol);
            check_all_greater_equal(ires.U.data(), drop_tol);
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


TEST_CASE("Exercise 6.15: 1-norm condition number estimate", "[ex6.15][norm1est][cond1est]")
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
        REQUIRE_THAT(est_norm, WithinAbs(expect, tol));
    }

    SECTION("Estimate condition number of A") {
        double kappa = cond1est(A);
        double expect = 2.422875115852452;  // MATLAB and python calcs

        REQUIRE_THAT(kappa, WithinAbs(expect, tol));
    }
}


}  // namespace cs

/*==============================================================================
 *============================================================================*/
