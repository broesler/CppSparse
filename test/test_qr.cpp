/*==============================================================================
 *     File: test_qr.cpp
 *  Created: 2025-05-08 13:23
 *   Author: Bernie Roesler
 *
 *  Description: Test Chapter 5: QR Factorization.
 *
 *============================================================================*/

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <numeric>    // iota
#include <span>
#include <vector>

#include "csparse.h"
#include "test_helpers.h"

using Catch::Matchers::AllTrue;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::RangeEquals;

namespace cs {


TEST_CASE("Householder Reflection", "[house]")
{
    SECTION("Unit x") {
        std::vector<double> x{1, 0, 0};

        std::vector<double> expect_v{1, 0, 0};
        double expect_beta = 0.0;
        double expect_s = 1.0;

        Householder H = house(x);

        check_vectors_allclose(H.v, expect_v, tol);
        CHECK_THAT(H.beta, WithinAbs(expect_beta, tol));
        CHECK_THAT(H.s, WithinAbs(expect_s, tol));

        // Apply the reflection
        CSCMatrix V = COOMatrix(H.v, {0, 1, 2}, {0, 0, 0}).tocsc();
        std::vector<double> Hx = happly(V, 0, H.beta, x);

        check_vectors_allclose(Hx, x, tol);
    }

    SECTION("Negative unit x") {
        std::vector<double> x{-1, 0, 0};

        std::vector<double> expect_v{1, 0, 0};
        double expect_beta = 0.0;
        double expect_s = -1.0;

        Householder H = house(x);

        check_vectors_allclose(H.v, expect_v, tol);
        CHECK_THAT(H.beta, WithinAbs(expect_beta, tol));
        CHECK_THAT(H.s, WithinAbs(expect_s, tol));

        // Apply the reflection
        CSCMatrix V = COOMatrix(H.v, {0, 1, 2}, {0, 0, 0}).tocsc();
        std::vector<double> Hx = happly(V, 0, H.beta, x);

        check_vectors_allclose(Hx, x, tol);
    }

    SECTION("Arbitrary x, x[0] > 0") {
        std::vector<double> x{3, 4};  // norm(x) == 5

        // These are the *unscaled* values from Octave
        // std::vector<double> expect_v{8, 4};
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

        check_vectors_allclose(H.v, expect_v, tol);
        CHECK_THAT(H.beta, WithinAbs(expect_beta, tol));
        CHECK_THAT(H.s, WithinAbs(expect_s, tol));

        // Apply the vector
        // Hx = [±norm(x), 0, 0]
        std::vector<double> expect{-5, 0}; // LAPACK
        // std::vector<double> expect{5, 0};  // Davis

        // Use column 0 of V to apply the Householder reflection
        CSCMatrix V = COOMatrix(H.v, {0, 1}, {0, 0}).tocsc();
        std::vector<double> Hx = happly(V, 0, H.beta, x);

        check_vectors_allclose(Hx, expect, tol);
    }

    SECTION("Arbitrary x, x[0] < 0") {
        std::vector<double> x{-3, 4};  // norm(x) == 5

        // These are the values from python's scipy.linalg.qr (via LAPACK):
        std::vector<double> expect_v {1, -0.5};
        double expect_beta = 1.6;
        double expect_s = 5;

        Householder H = house(x);

        check_vectors_allclose(H.v, expect_v, tol);
        CHECK_THAT(H.beta, WithinAbs(expect_beta, tol));
        CHECK_THAT(H.s, WithinAbs(expect_s, tol));

        // Apply the vector
        std::vector<double> expect{5, 0}; // LAPACK or Davis

        CSCMatrix V = COOMatrix(H.v, {0, 1}, {0, 0}).tocsc();
        std::vector<double> Hx = happly(V, 0, H.beta, x);

        check_vectors_allclose(Hx, expect, tol);
    }
}


TEST_CASE("QR factorization of the Identity Matrix", "[qr][identity]")
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
        std::vector<csint> expect_identity{0, 1, 2, 3, 4, 5, 6, 7};
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

        check_sparse_allclose(res.V, I);
        check_vectors_allclose(res.beta, expect_beta, tol);
        check_sparse_allclose(res.R, I);
    }
}


TEST_CASE("Symbolic QR Decomposition of Square, Non-symmetric A", "[qr][M == N][symbolic]")
{
    CSCMatrix A = davis_example_qr();
    csint N = A.shape()[1];  // == 8

    // See etree in Figure 5.1, p 74
    std::vector<csint> parent{3, 2, 3, 6, 5, 6, 7, -1};

    std::vector<csint> expect_leftmost{0, 1, 2, 0, 4, 4, 1, 4};
    std::vector<csint> expect_p_inv{0, 1, 3, 7, 4, 5, 2, 6};  // cs_qr MATLAB

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
        std::vector<csint> expect_q{0, 1, 2, 3, 4, 5, 6, 7};  // natural
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


TEST_CASE("Numeric QR Decomposition of Square, Non-symmetric A", "[qr][M == N][numeric]")
{
    CSCMatrix A = davis_example_qr();
    csint N = A.shape()[1];  // == 8
    double tol = 1e-12;  // QR factorization is sensitive to numerical errors

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
        CSCMatrix QR = (Q * res.R).droptol(tol).to_canonical();
        CSCMatrix Aq = A.permute_cols(res.q).to_canonical();

        check_sparse_allclose(QR, Aq);
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
        CSCMatrix QR = (Q * res.R).droptol(tol).to_canonical();
        CSCMatrix Aq = A.permute_cols(res.q).to_canonical();

        check_sparse_allclose(QR, Aq);
    }

    SECTION("Exercise 5.5: Use post-ordering") {
        // Compute the symbolic factorization with postordering
        bool use_postorder = true;
        SymbolicQR S = sqr(A, order, use_postorder);
        QRResult res = qr(A, S);

        // The postordering of this matrix *is* the natural ordering.
        // TODO Find an example with a different postorder for testing
        CSCMatrix Q = apply_qtleft(res.V, res.beta, res.p_inv, I).T();
        CSCMatrix QR = (Q * res.R).droptol(tol).to_canonical();
        CSCMatrix Aq = A.permute_cols(res.q).to_canonical();

        check_sparse_allclose(QR, Aq);
    }
}


TEST_CASE("Square, rank-deficient A", "[qr][rank-deficient][numeric]") 
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

    check_sparse_allclose(QR, Aq);
}


TEST_CASE("Symbolic QR factorization of overdetermined matrix M > N", "[qr][M > N][symbolic]")
{
    // Define the test matrix A (See Davis, Figure 5.1, p 74)
    // except remove the last 2 columns
    csint M = 8;
    csint N = 5;
    CSCMatrix A = davis_example_qr().slice(0, M, 0, N);

    CHECK(A.shape() == Shape {M, N});

    // See etree in Figure 5.1, p 74
    std::vector<csint> parent{3, 2, 3, -1, -1};

    std::vector<csint> expect_leftmost{0, 1, 2, 0, 4, 4, 1, 4};
    std::vector<csint> expect_p_inv{0, 1, 3, 5, 4, 6, 2, 7};

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
        std::vector<csint> expect_q{0, 1, 2, 3, 4};  // natural
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


TEST_CASE("Numeric QR factorization of overdetermined matrix M > N", "[qr][M > N][numeric]")
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

        check_sparse_allclose(QR, Aq);
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

        check_sparse_allclose(QR, Aq);
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
    std::vector<csint> parent{3, 2, 3, -1, -1};
    std::vector<csint> expect_leftmost{0, 1, 2, 0, 4};
    std::vector<csint> expect_p_inv{0, 1, 2, 3, 4};  // natural

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
        std::vector<csint> expect_q{0, 1, 2, 3, 4};  // natural
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

        check_sparse_allclose(QR, Aq);
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
        auto res_indptr = std::span(res.R.indptr()).subspan(0, M + 1);
        CHECK_THAT(sym_res.R.indptr(), RangeEquals(res_indptr));

        // NOTE hstack sorts the indices, whereas qr/symbolic_qr does not, so we
        // can either:
        //   * remove sorting from hstack
        //   * check the unordered sets of indices (not as robust)
        //   * sort the indices of sym_res.R (for testing only)
        auto res_indices = std::span(res.R.indices()).subspan(0, sym_res.R.nnz());
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

        check_sparse_allclose(QR, Aq);
    }
}


}  // namespace cs

/*==============================================================================
 *============================================================================*/
