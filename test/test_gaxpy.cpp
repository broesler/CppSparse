/*==============================================================================
 *     File: test_gaxpy.cpp
 *  Created: 2025-05-08 12:36
 *   Author: Bernie Roesler
 *
 *  Description: Test matrix/vector multiplication and addition.
 *
 *============================================================================*/

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include <vector>

#include "csparse.h"
#include "test_helpers.h"

using Catch::Matchers::AllTrue;
using Catch::Matchers::WithinAbs;


namespace cs {


TEST_CASE("gaxpy for dense vector x, y", "[math][gaxpy]")
{
    auto multiply_test = [](
        const CSCMatrix& A,
        const std::vector<double>& x,
        const std::vector<double>& y,
        const std::vector<double>& expect_Ax,
        const std::vector<double>& expect_Axpy
        )
    {
        std::vector<double> zero(y.size());
        check_vectors_allclose(gaxpy(A, x, zero),   expect_Ax,   tol);
        check_vectors_allclose(gaxpy(A, x, y),      expect_Axpy, tol);
        check_vectors_allclose(gatxpy(A.T(), x, y), expect_Axpy, tol);
        check_vectors_allclose(A.dot(x),            expect_Ax,   tol);
        check_vectors_allclose((A * x),             expect_Ax,   tol);
        check_vectors_allclose((A * x + y),         expect_Axpy, tol);
    };

    SECTION("A non-square matrix.") {
        CSCMatrix A = COOMatrix(
            std::vector<double> {1, 1, 2},
            std::vector<csint>  {0, 1, 2},
            std::vector<csint>  {0, 1, 1}
        ).tocsc();

        std::vector<double> x = {1, 2};
        std::vector<double> y = {1, 2, 3};

        // A @ x + y
        std::vector<double> expect_Ax   = {1, 2, 4};
        std::vector<double> expect_Axpy = {2, 4, 7};

        multiply_test(A, x, y, expect_Ax, expect_Axpy);
    }

    SECTION("A symmetric (diagonal) matrix.") {
        CSCMatrix A = COOMatrix(
            std::vector<double> {1, 2, 3},
            std::vector<csint>  {0, 1, 2},
            std::vector<csint>  {0, 1, 2}
        ).compress();

        std::vector<double> x = {1, 2, 3};
        std::vector<double> y = {9, 6, 1};

        // A @ x + y
        std::vector<double> expect_Ax   = {1, 4, 9};
        std::vector<double> expect_Axpy = {10, 10, 10};

        multiply_test(A, x, y, expect_Ax, expect_Axpy);
        check_vectors_allclose(sym_gaxpy(A, x, y),  expect_Axpy, tol);
    }

    SECTION("An arbitrary non-symmetric matrix.") {
        COOMatrix Ac = davis_example_small();
        CSCMatrix A = Ac.compress();

        std::vector<double> x = {1, 2, 3, 4};
        std::vector<double> y = {1, 1, 1, 1};

        // A @ x + y
        std::vector<double> expect_Ax   = {14.1, 12.5, 12.4,  8.3};
        std::vector<double> expect_Axpy = {15.1, 13.5, 13.4,  9.3};

        multiply_test(A, x, y, expect_Ax, expect_Axpy);

        // COOMatrix
        check_vectors_allclose(Ac.dot(x), expect_Ax, tol);
        check_vectors_allclose((Ac * x),  expect_Ax, tol);
    }

    SECTION("An arbitrary symmetric matrix.") {
        // See Davis pp 7-8, Eqn (2.1)
        std::vector<csint>  i = {  0,   1,   3,   0,   1,   2,   1,   2,   0,   3};
        std::vector<csint>  j = {  0,   0,   0,   1,   1,   1,   2,   2,   3,   3};
        std::vector<double> v = {4.5, 3.1, 3.5, 3.1, 2.9, 1.7, 1.7, 3.0, 3.5, 1.0};
        CSCMatrix A = COOMatrix(v, i, j).compress();

        std::vector<double> x = {1, 2, 3, 4};
        std::vector<double> y = {1, 1, 1, 1};

        // A @ x + y
        std::vector<double> expect_Axpy = {25.7, 15.0, 13.4,  8.5};

        check_vectors_allclose(sym_gaxpy(A, x, y), expect_Axpy, tol);
    }
}


TEST_CASE("Exercise 2.27: gaxpy for dense matrix X, Y", "[ex2.27][math][gaxpy]")
{
    CSCMatrix A = davis_example_small().compress();

    SECTION("Identity op") {
        std::vector<double> I = {
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        };

        std::vector<double> Z(16, 0);

        CSCMatrix expect = A;

        check_sparse_allclose(CSCMatrix(gaxpy_col(A, I, Z), {4, 4}), expect);
        check_sparse_allclose(CSCMatrix(gatxpy_col(A.T(), I, Z), {4, 4}), expect);
    }

    SECTION("Arbitrary square matrix in column-major format") {
        std::vector<double> A_dense = A.to_dense_vector();

        // A.T @ A + A in column-major format
        std::vector<double> expect = {
            46.61, 13.49, 14.4 ,  9.79,
            10.39, 14.36,  6.8 ,  3.41,
            17.6 ,  5.1 , 22.24,  0.0 ,
             6.29,  3.91,  0.0 ,  2.81
        };

        std::vector<double> C_col = gaxpy_col(A.T(), A_dense, A_dense);
        std::vector<double> C_block = gaxpy_block(A.T(), A_dense, A_dense);
        std::vector<double> CT_col = gatxpy_col(A, A_dense, A_dense);
        std::vector<double> CT_block = gatxpy_block(A, A_dense, A_dense);

        check_vectors_allclose(C_col, expect, tol);
        check_vectors_allclose(C_block, expect, tol);
        check_vectors_allclose(CT_col, expect, tol);
        check_vectors_allclose(CT_block, expect, tol);
    }

    SECTION("Arbitrary square matrix in row-major format") {
        std::vector<double> A_dense = A.to_dense_vector('C');

        // A.T @ A + A in row-major format
        std::vector<double> expect = {
            46.61, 10.39, 17.6 ,  6.29,
            13.49, 14.36,  5.1 ,  3.91,
            14.4 ,  6.8 , 22.24,  0.0 ,
             9.79,  3.41,  0.0 ,  2.81
        };

        std::vector<double> C = gaxpy_row(A.T(), A_dense, A_dense);
        std::vector<double> CT = gatxpy_row(A, A_dense, A_dense);

        check_vectors_allclose(C, expect, tol);
        check_vectors_allclose(CT, expect, tol);
    }

    SECTION("Non-square matrix in column-major format.") {
        CSCMatrix Ab = A.slice(0, 4, 0, 3);  // {4, 3}
        std::vector<double> Ac_dense = A.slice(0, 3, 0, 4).to_dense_vector();
        std::vector<double> A_dense = A.to_dense_vector();

        // Ab @ Ac + A in column-major format
        std::vector<double> expect = {
            24.75, 26.04,  5.27, 20.49,
             5.44, 11.31, 11.73,  1.56,
            27.2 ,  9.92, 12.0 , 11.2 ,
             0.0 ,  3.51,  1.53,  1.36
        };

        check_vectors_allclose(gaxpy_col(Ab, Ac_dense, A_dense), expect, tol);
        check_vectors_allclose(gaxpy_block(Ab, Ac_dense, A_dense), expect, tol);
        check_vectors_allclose(gatxpy_col(Ab.T(), Ac_dense, A_dense), expect, tol);
        check_vectors_allclose(gatxpy_block(Ab.T(), Ac_dense, A_dense), expect, tol);
    }

    SECTION("Non-square matrix in row-major format.") {
        CSCMatrix Ab = A.slice(0, 4, 0, 3);  // {4, 3}
        std::vector<double> Ac_dense = A.slice(0, 3, 0, 4).to_dense_vector('C');
        std::vector<double> A_dense = A.to_dense_vector('C');

        // Ab @ Ac + A in row-major format
        std::vector<double> expect = {
            24.75,  5.44, 27.2 ,  0.0 ,
            26.04, 11.31,  9.92,  3.51,
             5.27, 11.73, 12.0 ,  1.53,
            20.49,  1.56, 11.2 ,  1.36
        };

        check_vectors_allclose(gaxpy_row(Ab, Ac_dense, A_dense), expect, tol);
        check_vectors_allclose(gatxpy_row(Ab.T(), Ac_dense, A_dense), expect, tol);
    }
}


TEST_CASE("Sparse matrix-matrix multiply.", "[math][dot]")
{
    SECTION("Square matrices") {
        // Build matrices with sorted columns
        CSCMatrix E = E_mat();
        CSCMatrix A = A_mat();

        // See: Strang, p 25
        // EA = [[ 2, 1, 1],
        //       [ 0,-8,-2],
        //       [-2, 7, 2]]

        CSCMatrix expect = COOMatrix(
            std::vector<double> {2, -2, 1, -8, 7, 1, -2, 2},  // vals
            std::vector<csint>  {0,  2, 0,  1, 2, 0,  1, 2},  // rows
            std::vector<csint>  {0,  0, 1,  1, 1, 2,  2, 2}   // cols
        ).tocsc();

        auto multiply_test = [](
            const CSCMatrix& C,
            const CSCMatrix& E,
            const CSCMatrix& A,
            const CSCMatrix& expect
        ) {
            auto [M, N] = C.shape();

            REQUIRE(M == E.shape()[0]);
            REQUIRE(N == A.shape()[1]);

            for (csint i = 0; i < M; i++) {
                for (csint j = 0; j < N; j++) {
                    REQUIRE_THAT(C(i, j), WithinAbs(expect(i, j), tol));
                }
            }
        };

        SECTION("CSCMatrix::dot (aka cs_multiply)") {
            CSCMatrix C = E * A;
            multiply_test(C, E, A, expect);
        }

        SECTION("Dot_2x two-pass multiply") {
            CSCMatrix C = E.dot_2x(A);
            multiply_test(C, E, A, expect);
        }
    }

    SECTION("Arbitrary size matrices") {
        // >>> A
        // ===
        // array([[1, 2, 3, 4],
        //        [5, 6, 7, 8]])
        // >>> B
        // ===
        // array([[ 1,  2,  3],
        //        [ 4,  5,  6],
        //        [ 7,  8,  9],
        //        [10, 11, 12]])
        // >>> A @ B
        // ===
        // array([[ 70,  80,  90],
        //        [158, 184, 210]])

        CSCMatrix A = COOMatrix(
            std::vector<double> {1, 2, 3, 4, 5, 6, 7, 8},  // vals
            std::vector<csint>  {0, 0, 0, 0, 1, 1, 1, 1},  // rows
            std::vector<csint>  {0, 1, 2, 3, 0, 1, 2, 3}   // cols
        ).compress();

        CSCMatrix B = COOMatrix(
            std::vector<double> {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},  // vals
            std::vector<csint>  {0, 0, 0, 1, 1, 1, 2, 2, 2,  3,  3,  3},  // rows
            std::vector<csint>  {0, 1, 2, 0, 1, 2, 0, 1, 2,  0,  1,  2}   // cols
        ).compress();

        CSCMatrix expect = COOMatrix(
            std::vector<double> {70, 80, 90, 158, 184, 210},  // vals
            std::vector<csint>  { 0,  0,  0,   1,   1,   1},  // rows
            std::vector<csint>  { 0,  1,  2,   0,   1,   2}   // cols
        ).compress();

        SECTION("M < N") {
            CSCMatrix C = A * B;
            auto [M, N] = C.shape();

            REQUIRE(M == A.shape()[0]);
            REQUIRE(N == B.shape()[1]);

            check_sparse_allclose(C, expect);
        }

        SECTION("M > N") {
            CSCMatrix CT = B.T() * A.T();
            auto [N, M] = CT.shape();

            REQUIRE(M == A.shape()[0]);
            REQUIRE(N == B.shape()[1]);

            check_sparse_allclose(CT, expect.T());
        }
    }

    SECTION("Symbolic Multiply") {
        CSCMatrix A = COOMatrix(
            std::vector<double> {},  // vals
            std::vector<csint>  {0, 0, 0, 0, 1, 1, 1, 1},  // rows
            std::vector<csint>  {0, 1, 2, 3, 0, 1, 2, 3}   // cols
        ).tocsc();

        CSCMatrix B = COOMatrix(
            std::vector<double> {},  // vals
            std::vector<csint>  {0, 0, 0, 1, 1, 1, 2, 2, 2,  3,  3,  3},  // rows
            std::vector<csint>  {0, 1, 2, 0, 1, 2, 0, 1, 2,  0,  1,  2}   // cols
        ).tocsc();

        CSCMatrix expect = COOMatrix(
            std::vector<double> {},  // vals
            std::vector<csint>  { 0,  0,  0,   1,   1,   1},  // rows
            std::vector<csint>  { 0,  1,  2,   0,   1,   2}   // cols
        ).tocsc();

        // M < N
        CSCMatrix C = A * B;
        auto [M, N] = C.shape();

        REQUIRE(M == A.shape()[0]);
        REQUIRE(N == B.shape()[1]);
        check_sparse_allclose(C, expect, false);
    }
}


TEST_CASE("Exercise 2.18: Sparse Vector-Vector Multiply", "[ex2.18][math]")
{
    // Sparse column vectors *without* sorting columns.
    CSCMatrix x = COOMatrix(
        std::vector<double> {4.5, 3.1, 3.5, 2.9, 1.7, 0.4},
        std::vector<csint>  {0, 1, 3, 5, 6, 7},
        std::vector<csint>  (6, 0)
    ).compress();

    CSCMatrix y = COOMatrix(
        std::vector<double> {3.2, 3.0, 0.9, 1.0},
        std::vector<csint>  {0, 2, 5, 7},
        std::vector<csint>  (4, 0)
    ).compress();

    double expect = 17.41;

    SECTION("Unsorted indices") {
        CHECK_THAT(x.T() * y, WithinAbs(expect, tol));
        REQUIRE_THAT(x.vecdot(y), WithinAbs(expect, tol));
    }

    SECTION("Sorted indices") {
        x.sort();
        y.sort();

        CHECK_THAT(x.T() * y, WithinAbs(expect, tol));
        REQUIRE_THAT(x.vecdot(y), WithinAbs(expect, tol));
    }
}


TEST_CASE("Scale a sparse matrix by a constant", "[math][scale]")
{
    std::vector<csint> i{{0, 0, 0, 1, 1, 1}};  // rows
    std::vector<csint> j{{0, 1, 2, 0, 1, 2}};  // cols

    CSCMatrix A = COOMatrix(
        std::vector<double> {1, 2, 3, 4, 5, 6},
        i, j
    ).compress();

    CSCMatrix expect = COOMatrix(
        std::vector<double> {0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
        i, j
    ).compress();

    auto [M, N] = A.shape();

    // Operator overloading
    CSCMatrix C = 0.1 * A;

    for (csint i = 0; i < M; i++) {
        for (csint j = 0; j < N; j++) {
            REQUIRE_THAT(C(i, j), WithinAbs(expect(i, j), tol));
        }
    }
}


TEST_CASE("Exercise 2.4: Scale rows and columns", "[ex2.4][math][scale]")
{
    CSCMatrix A = davis_example_small().compress();

    // Diagonals of R and C to compute RAC
    std::vector<double> r = {1, 2, 3, 4},
                        c = {1.0, 0.5, 0.25, 0.125};

    // expect_RAC = array([[ 4.5  ,  0.   ,  0.8  ,  0.   ],
    //                     [ 6.2  ,  2.9  ,  0.   ,  0.225],
    //                     [ 0.   ,  2.55 ,  2.25 ,  0.   ],
    //                     [14.   ,  0.8  ,  0.   ,  0.5  ]])

    std::vector<csint> expect_i = {0, 1, 3, 1, 2, 3, 0, 2, 1, 3};
    std::vector<csint> expect_j = {0, 0, 0, 1, 1, 1, 2, 2, 3, 3};
    std::vector<double> expect_v = {4.5, 6.2, 14.0, 2.9, 2.55, 0.8, 0.8, 2.25, 0.225, 0.5};
    CSCMatrix expect_RAC = COOMatrix(expect_v, expect_i, expect_j).compress();

    CSCMatrix RAC = A.scale(r, c);

    auto [M, N] = A.shape();

    for (csint i = 0; i < M; i++) {
        for (csint j = 0; j < N; j++) {
            REQUIRE_THAT(RAC(i, j), WithinAbs(expect_RAC(i, j), tol));
        }
    }
}


// See: cs_add
TEST_CASE("Matrix-matrix addition.", "[math][add_scaled]")
{
    // Add arbitrary-sized matrices
    // >>> A
    // ===
    // array([[1, 2, 3],
    //        [4, 5, 6]])
    // >>> B
    // ===
    // array([[1, 1, 1]
    //        [1, 1, 1]])
    // >>> 0.1 * B + 9.0 * C
    // ===
    // array([[9.1, 9.2, 9.3],
    //        [9.4, 9.5, 9.6]])

    std::vector<csint> i{{0, 0, 0, 1, 1, 1}};  // rows
    std::vector<csint> j{{0, 1, 2, 0, 1, 2}};  // cols

    CSCMatrix A = COOMatrix(
        std::vector<double> {1, 2, 3, 4, 5, 6},
        i, j
    ).compress();

    CSCMatrix B = COOMatrix(
        std::vector<double> {1, 1, 1, 1, 1, 1},
        i, j
    ).compress();

    SECTION("Addition") {
        CSCMatrix expect = COOMatrix(
            std::vector<double> {9.1, 9.2, 9.3, 9.4, 9.5, 9.6},
            i, j
        ).compress();

        // Function definition
        CSCMatrix Cf = add_scaled(A, B, 0.1, 9.0);

        // Operator overloading
        CSCMatrix C = 0.1 * A + 9.0 * B;
        
        // Add method
        CSCMatrix Cm = (0.1 * A).add(9.0 * B);

        check_sparse_allclose(C, expect);
        check_sparse_allclose(Cf, expect);
        check_sparse_allclose(Cm, expect);
    }

    SECTION("Subtraction") {
        CSCMatrix expect = COOMatrix(
            std::vector<double> {0, 1, 2, 3, 4, 5},
            i, j
        ).compress();

        // Function definition
        CSCMatrix Cf = add_scaled(A, B, 1.0, -1.0);

        // Operator overloading
        CSCMatrix C = A - B;

        check_sparse_allclose(C, expect);
        check_sparse_allclose(Cf, expect);
    }

    SECTION("Symbolic Addition") {
        CSCMatrix As = COOMatrix(std::vector<double> {}, i, j).tocsc();
        CSCMatrix expect = COOMatrix(std::vector<double> {}, i, j).tocsc();

        CSCMatrix Cs = add_scaled(As, B, 1.0, 1.0);

        check_sparse_allclose(Cs, expect, false);  // don't compare values
    }
}


TEST_CASE("Add sparse column vectors", "[math][add_scaled]")
{
    CSCMatrix a = COOMatrix(
        std::vector<double> {4.5, 3.1, 3.5, 2.9, 0.4},
        std::vector<csint>  {0, 1, 3, 5, 7},
        std::vector<csint>  (5, 0)
    ).tocsc();

    CSCMatrix b = COOMatrix(
        std::vector<double> {3.2, 3.0, 0.9, 1.0},
        std::vector<csint>  {0, 2, 5, 7},
        std::vector<csint>  (4, 0)
    ).tocsc();

    CSCMatrix expect = COOMatrix(
        std::vector<double> {7.7, 3.1, 3.0, 3.5, 3.8, 1.4},
        std::vector<csint>  {0, 1, 2, 3, 5, 7},
        std::vector<csint>  (6, 0)
    ).tocsc();

    auto [M, N] = a.shape();

    SECTION("Operator") {
        CSCMatrix C = a + b;

        REQUIRE(C.shape() == a.shape());

        for (csint i = 0; i < M; i++) {
            for (csint j = 0; j < N; j++) {
                REQUIRE_THAT(C(i, j), WithinAbs(expect(i, j), tol));
            }
        }
    }

    SECTION("Exercise 2.21: Saxpy") {
        std::vector<csint> expect_w(M);
        for (auto i : expect.indices()) {
            expect_w[i] = 1;
        }

        // Initialize workspaces
        std::vector<csint> w(M);
        std::vector<double> x(M);

        w = saxpy(a, b, w, x);

        REQUIRE(w == expect_w);
    }
}


TEST_CASE("Multiply Sparse by Dense Matrix", "[math][dot]")
{
    CSCMatrix A = davis_example_small().tocsc();

    SECTION("M < N, N > K") {
        A = A.slice(0, 3, 0, 4);  // 3 x 4

        std::vector<double> X = {
            1, 2, 3, 4,
            2, 4, 6, 8
        };  // 4 x 2 in column-major format

        std::vector<double> expect = {
            14.1, 12.5, 12.4,
            28.2, 25. , 24.8
        };  // 3 x 2 in column-major format

        std::vector<double> C = A * X;

        check_vectors_allclose(C, expect, tol);
    }

    SECTION("M < N, N < K") {
        A = A.slice(0, 3, 0, 4);  // 3 x 4

        std::vector<double> X = {
            1, 2,   3,  4,
            2, 4,   6,  8,
            3, 6,   9, 12,
            4, 8,  12, 16,
            5, 10, 15, 20
        };  // 4 x 5 in column-major format

        std::vector<double> expect = {
            14.1, 12.5, 12.4,
            28.2, 25.0, 24.8,
            42.3, 37.5, 37.2,
            56.4, 50.0, 49.6,
            70.5, 62.5, 62.0
        };  // 3 x 5 in column-major format

        std::vector<double> C = A * X;

        check_vectors_allclose(C, expect, tol);
    }

    SECTION("M > N, N < K") {
        A = A.slice(0, 4, 0, 3);  // 4 x 3

        std::vector<double> X = {
            1, 2,   3,
            2, 4,   6,
            3, 6,   9,
            4, 8,  12,
            5, 10, 15,
        };  // 3 x 5 in column-major format

        std::vector<double> expect = {
            14.1,  8.9, 12.4,  4.3,
            28.2, 17.8, 24.8,  8.6,
            42.3, 26.7, 37.2, 12.9,
            56.4, 35.6, 49.6, 17.2,
            70.5, 44.5, 62.0, 21.5
        };  // 4 x 5 in column-major format

        std::vector<double> C = A * X;

        check_vectors_allclose(C, expect, tol);
    }

    SECTION("M > N, N > K") {
        A = A.slice(0, 4, 0, 3);  // 4 x 3

        std::vector<double> X = {
            1, 2,   3,
            2, 4,   6,
        };  // 3 x 2 in column-major format

        std::vector<double> expect = {
            14.1,  8.9, 12.4,  4.3,
            28.2, 17.8, 24.8,  8.6,
        };  // 4 x 2 in column-major format

        std::vector<double> C = A * X;

        check_vectors_allclose(C, expect, tol);
    }
}


}  // namespace cs

/*==============================================================================
 *============================================================================*/
