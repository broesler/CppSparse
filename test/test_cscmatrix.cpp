/*==============================================================================
 *     File: test_cscmatrix.cpp
 *  Created: 2025-05-08 12:03
 *   Author: Bernie Roesler
 *
 *  Description: Test CSCMatrix constructors and other basic functions.
 *
 *============================================================================*/

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <sstream>
#include <string>
#include <stdexcept>  // out_of_range
#include <vector>

#include "csparse.h"
#include "test_helpers.h"

using Catch::Matchers::AllTrue;
using Catch::Matchers::WithinAbs;


namespace cs {


TEST_CASE("CSCMatrix Constructor", "[CSCMatrix]")
{
    COOMatrix A = davis_example_small();
    CSCMatrix C = A.compress();  // unsorted columns

    SECTION("Attributes") {
        std::vector<csint> indptr_expect  = {  0,             3,             6,        8,  10};
        std::vector<csint> indices_expect = {  1,   3,   0,   1,   3,   2,   2,   0,   3,   1};
        std::vector<double> data_expect   = {3.1, 3.5, 4.5, 2.9, 0.4, 1.7, 3.0, 3.2, 1.0, 0.9};

        REQUIRE(C.nnz() == 10);
        REQUIRE(C.nzmax() >= 10);
        REQUIRE(C.shape() == Shape{4, 4});
        REQUIRE(C.indptr() == indptr_expect);
        REQUIRE(C.indices() == indices_expect);
        REQUIRE(C.data() == data_expect);
    }

    SECTION ("Printing") {
        std::stringstream s;

        SECTION("Print short") {
            std::string expect =
                "<C++Sparse Compressed Sparse Column matrix\n"
                "        with 10 stored elements and shape (4, 4)>\n";

            C.print(s);  // default verbose=false

            REQUIRE(s.str() == expect);
        }

        SECTION("Print verbose") {
            std::string expect =
                "<C++Sparse Compressed Sparse Column matrix\n"
                "        with 10 stored elements and shape (4, 4)>\n"
                "(1, 0):  3.1\n"
                "(3, 0):  3.5\n"
                "(0, 0):  4.5\n"
                "(1, 1):  2.9\n"
                "(3, 1):  0.4\n"
                "(2, 1):  1.7\n"
                "(2, 2):  3\n"
                "(0, 2):  3.2\n"
                "(3, 3):  1\n"
                "(1, 3):  0.9\n";

            SECTION("Print from function") {
                C.print(s, true);
                REQUIRE(s.str() == expect);
            }

            SECTION("Print from operator<< overload") {
                s << C;
                REQUIRE(s.str() == expect);
            }
        }

        // Clear the stringstream to prevent memory leaks
        s.str("");
        s.clear();
    }

    // The transpose -> use indexing to test A(i, j) == A(j, i)
    SECTION("Transpose") {
        // lambda to test on M == N, M < N, M > N
        auto transpose_test = [](CSCMatrix C) {
            CSCMatrix C_T = C.transpose();

            auto [M, N] = C.shape();

            REQUIRE(C.nnz() == C_T.nnz());
            REQUIRE(M == C_T.shape()[1]);
            REQUIRE(N == C_T.shape()[0]);

            for (csint i = 0; i < M; i++) {
                for (csint j = 0; j < N; j++) {
                    double C_val = C(i, j);
                    double C_T_val = C_T(j, i);
                    CAPTURE(i, j, C_val, C_T_val);
                    // Results should be exact since we aren't doing actual
                    // floating point operations, just copying data.
                    CHECK(C_val == C_T_val);
                }
            }
        };

        SECTION("Square matrix M == N") {
            transpose_test(C);  // shape = {4, 4}
        }

        SECTION("Non-square matrix M < N") {
            transpose_test(A.insert(0, 4, 1.6).compress()); // shape = {4, 5}
        }

        SECTION("Non-square matrix M > N") {
            transpose_test(A.insert(4, 0, 1.6).compress()); // shape = {5, 4}
        }

        SECTION("Symbolic Transpose") {
            CSCMatrix C_T = C.transpose(false);

            auto [M, N] = C.shape();

            REQUIRE(C_T.data().empty());
            REQUIRE(C.nnz() == C_T.nnz());
            REQUIRE(M == C_T.shape()[1]);
            REQUIRE(N == C_T.shape()[0]);
        }
    }

    SECTION("Sort rows/columns") {
        // On non-square matrix M != N
        C = A.insert(0, 4, 1.6).compress();  // {4, 5}

        auto sort_test = [](const CSCMatrix& Cs) {
            Shape shape_expect = {4, 5};
            std::vector<csint> indptr_expect  = {  0,             3,             6,        8,       10, 11};
            std::vector<csint> indices_expect = {  0,   1,   3,   1,   2,   3,   0,   2,   1,   3,   0};
            std::vector<double> data_expect   = {4.5, 3.1, 3.5, 2.9, 1.7, 0.4, 3.2, 3.0, 0.9, 1.0, 1.6};

            CHECK(Cs.shape() == shape_expect);
            CHECK(Cs.has_sorted_indices());
            CHECK(Cs.indptr() == indptr_expect);
            CHECK(Cs.indices() == indices_expect);
            REQUIRE(Cs.data() == data_expect);
        };

        SECTION("Two transposes") {
            sort_test(C.tsort());
        }

        SECTION("qsort") {
            sort_test(C.qsort());
        }

        SECTION("Efficient two transposes") {
            sort_test(C.sort());
        }

        SECTION("Brute force check") {
            CHECK_FALSE(C.has_sorted_indices());
            CHECK_FALSE(C._test_sorted());
            REQUIRE(C.sort()._test_sorted());
        }
    }

    SECTION("Sum duplicates") {
        C = A.insert(0, 2, 100.0)
             .insert(3, 0, 100.0)
             .insert(2, 1, 100.0)
             .compress()
             .sum_duplicates();

        REQUIRE_THAT(C(0, 2), WithinAbs(103.2, tol));
        REQUIRE_THAT(C(3, 0), WithinAbs(103.5, tol));
        REQUIRE_THAT(C(2, 1), WithinAbs(101.7, tol));
    }

    SECTION("Droptol") {
        C = davis_example_small().compress().droptol(2.0);

        REQUIRE(C.nnz() == 6);
        REQUIRE(C.shape() == Shape{4, 4});
        REQUIRE_THAT(C.data() >= 2.0, AllTrue());
    }

    SECTION("Dropzeros") {
        // Insert explicit zeros
        C = davis_example_small()
            .insert(0, 1, 0.0)
            .insert(2, 1, 0.0)
            .insert(3, 1, 0.0)
            .compress();

        REQUIRE(C.nnz() == 13);

        C.dropzeros();

        REQUIRE(C.nnz() == 10);
        REQUIRE_THAT(C.data() != 0.0, AllTrue());
    }

    SECTION("1-norm") {
        REQUIRE_THAT(C.norm(), WithinAbs(11.1, tol));
    }

    SECTION("Frobenius norm") {
        double expect = 8.638286867197685;  // computed in MATLAB and numpy
        REQUIRE_THAT(C.fronorm(), WithinAbs(expect, tol));
    }
}


TEST_CASE("Canonical format", "[CSCMatrix][COOMatrix]")
{
    std::vector<csint> indptr_expect  = {  0,               3,                 6,        8,  10};
    std::vector<csint> indices_expect = {  0,   1,     3,   1,     2,   3,     0,   2,   1,   3};
    std::vector<double> data_expect   = {4.5, 3.1, 103.5, 2.9, 101.7, 0.4, 103.2, 3.0, 0.9, 1.0};

    COOMatrix A = (
        davis_example_small()  // unsorted matrix
        .insert(0, 2, 100.0)   // insert duplicates
        .insert(3, 0, 100.0)
        .insert(2, 1, 100.0)
        .insert(0, 1, 0.0)     // insert zero entries
        .insert(2, 2, 0.0)
        .insert(3, 1, 0.0)
    );

    REQUIRE(A.nnz() == 16);

    // Convert to canonical format
    CSCMatrix C = A.tocsc();  // as member function

    // Duplicates summed
    REQUIRE(C.nnz() == 10);
    REQUIRE_THAT(C(0, 2), WithinAbs(103.2, tol));
    REQUIRE_THAT(C(3, 0), WithinAbs(103.5, tol));
    REQUIRE_THAT(C(2, 1), WithinAbs(101.7, tol));

    // No non-zeros
    REQUIRE_THAT(C.data() != 0.0, AllTrue());

    // Sorted entries
    REQUIRE(C.indptr() == indptr_expect);
    REQUIRE(C.indices() == indices_expect);
    REQUIRE(C.data() == data_expect);

    // Flags set
    REQUIRE(C.has_sorted_indices());
    REQUIRE(C.has_canonical_format());
    REQUIRE_FALSE(C.is_symmetric());

    SECTION("Constructor") {
        CSCMatrix B {A};
        REQUIRE(C.indptr() == B.indptr());
        REQUIRE(C.indices() == B.indices());
        REQUIRE(C.data() == B.data());
    }

    SECTION("Indexing") {
        std::vector<csint> indptr = C.indptr();
        std::vector<csint> indices = C.indices();
        std::vector<double> data = C.data();
        csint N = C.shape()[1];

        for (csint j = 0; j < N; j++) {
            for (csint p = indptr[j]; p < indptr[j+1]; p++) {
                REQUIRE(C(indices[p], j) == data[p]);
            }
        }
    }
}


TEST_CASE("Indexing", "[CSCMatrix][operator()]")
{
    COOMatrix A = davis_example_small();  // non-canonical
    auto [M, N] = A.shape();

    // Define the expected matrix
    std::vector<double> dense_column_major = {
        4.5, 3.1, 0.0, 3.5,
        0.0, 2.9, 1.7, 0.4,
        3.2, 0.0, 3.0, 0.0,
        0.0, 0.9, 0.0, 1.0
    };
    
    // NOTE that comparing to the dense matrix is *behavioral* testing. But
    // actually we'd like to check the precise structure of the matrix to ensure
    // that we have found/inserted elements correctly.

    // std::vector<csint> indptr_expect = {0, 3, 6, 8, 10};

    // Non-canonical form
    // std::vector<csint> indices_expect = {  1,   3,   0,   1,   3,   2,   2,   0,   3,   1};
    // std::vector<double> data_expect   = {3.1, 3.5, 4.5, 2.9, 0.4, 1.7, 3.0, 3.2, 1.0, 0.9};

    // Canonical form
    // std::vector<csint> indices_expect = {  0,   1,   3,   1,   2,   3,   0,   2,   1,   3};
    // std::vector<double> data_expect   = {4.5, 3.1, 3.5, 2.9, 1.7, 0.4, 3.2, 3.0, 0.9, 1.0};

    bool is_canonical = GENERATE(true, false);
    CAPTURE(is_canonical);

    bool is_sorted = GENERATE(true, false);
    CAPTURE(is_sorted);

    bool is_const = GENERATE(true, false);
    CAPTURE(is_const);

    SECTION("Without duplicates") {
        // do nothing
    }

    SECTION("With a duplicate") {
        A.insert(3, 1, 56.0);
        dense_column_major[3 + 1 * M] = 56.4;  // expected value
    }

    SECTION("With multiple duplicates") {
        A.insert(3, 1, 56.0).insert(3, 1, 2.0).insert(3, 1, 7.2);
        dense_column_major[3 + 1 * M] = (0.4 + 56.0 + 2.0 + 7.2);
    }

    CSCMatrix C = is_canonical ? A.tocsc() : (is_sorted ? A.compress().sort() : A.compress());

    if (is_const) {
        // Create a const reference to force use of the const version
        const CSCMatrix& const_C = C;
        check_sparse_eq_dense(const_C, dense_column_major, C.shape(), tol);
    } else {
        check_sparse_eq_dense(C, dense_column_major, C.shape(), tol);
    }
}


TEST_CASE("Item Comparison Operators", "[CSCMatrix][operator==]")
{
    CSCMatrix A = davis_example_small().tocsc();

    // vs. double
    CHECK(A(0, 0) == 4.5);
    CHECK(A(0, 0) >= 4.0);
    CHECK(A(0, 0) <= 5.0);
    CHECK(A(0, 0) > 4.0);
    CHECK(A(0, 0) < 5.0);

    // vs. ItemProxy (another element)
    CHECK(A(0, 0) == A(0, 0));
    CHECK(A(0, 0) >= A(1, 0));
    CHECK(A(1, 0) <= A(0, 0));
    CHECK(A(0, 0) > A(1, 0));
    CHECK(A(1, 0) < A(0, 0));
}


TEST_CASE("Compound Assignment Operators", "[CSCMatrix][operator+=]")
{
    CSCMatrix A = davis_example_small().tocsc();

    // A(0, 0) = 4.5;

    SECTION("Plus") {
        A(0, 0) += 0.5;
        CHECK_THAT(A(0, 0), WithinAbs(5.0, tol));
    }

    SECTION("Minus") {
        A(0, 0) -= 0.5;
        CHECK_THAT(A(0, 0), WithinAbs(4.0, tol));
    }

    SECTION("Times") {
        A(0, 0) *= 2.0;
        CHECK_THAT(A(0, 0), WithinAbs(9.0, tol));
    }

    SECTION("Divide") {
        A(0, 0) /= 2.0;
        CHECK_THAT(A(0, 0), WithinAbs(2.25, tol));
    }

    SECTION("Divide by Zero") {
        REQUIRE_THROWS_WITH(A(0, 0) /= 0.0, "Division by zero");
    }

    SECTION("Pre-increment") {
        CHECK_THAT(A(0, 0), WithinAbs(4.5, tol));
        CHECK_THAT(++A(0, 0), WithinAbs(5.5, tol));  // changed *before* compare
        CHECK_THAT(A(0, 0), WithinAbs(5.5, tol));
    }

    SECTION("Post-increment") {
        CHECK_THAT(A(0, 0), WithinAbs(4.5, tol));
        CHECK_THAT(A(0, 0)++, WithinAbs(4.5, tol));  // changed *after* compare
        CHECK_THAT(A(0, 0), WithinAbs(5.5, tol));
    }

    SECTION("Pre-decrement") {
        CHECK_THAT(A(0, 0), WithinAbs(4.5, tol));
        CHECK_THAT(--A(0, 0), WithinAbs(3.5, tol));  // changed *before* compare
        CHECK_THAT(A(0, 0), WithinAbs(3.5, tol));
    }

    SECTION("Post-decrement") {
        CHECK_THAT(A(0, 0), WithinAbs(4.5, tol));
        CHECK_THAT(A(0, 0)--, WithinAbs(4.5, tol));  // changed *after* compare
        CHECK_THAT(A(0, 0), WithinAbs(3.5, tol));
    }
}


TEST_CASE("Exercise 2.2: Conversion to COOMatrix") {
    CSCMatrix C = davis_example_small().compress();  // non-canonical

    auto convert_test = [](const COOMatrix& A) {
        // Columns are sorted, but not rows
        std::vector<csint>  expect_i = {  1,   3,   0,   1,   3,   2,   2,   0,   3,   1};
        std::vector<csint>  expect_j = {  0,   0,   0,   1,   1,   1,   2,   2,   3,   3};
        std::vector<double> expect_v = {3.1, 3.5, 4.5, 2.9, 0.4, 1.7, 3.0, 3.2, 1.0, 0.9};

        REQUIRE(A.nnz() == 10);
        REQUIRE(A.nzmax() >= 10);
        REQUIRE(A.shape() == Shape{4, 4});
        REQUIRE(A.row() == expect_i);
        REQUIRE(A.col() == expect_j);
        REQUIRE(A.data() == expect_v);
    };

    SECTION("As constructor") {
        COOMatrix A {C};  // via constructor
        convert_test(A);
    }

    SECTION("As function") {
        COOMatrix A = C.tocoo();  // via member function
        convert_test(A);
    }
}


TEST_CASE("Exercise 2.13: is_symmetric.", "[ex2.13]")
{
    std::vector<csint>  i = {0, 1, 2};
    std::vector<csint>  j = {0, 1, 2};
    std::vector<double> v = {1, 2, 3};

    SECTION("Diagonal matrix") {
        CSCMatrix A = COOMatrix(v, i, j).tocsc();
        REQUIRE(A.is_symmetric());
    }

    SECTION("Non-symmetric matrix with off-diagonals") {
        CSCMatrix A = COOMatrix(v, i, j)
                       .insert(0, 1, 1.0)
                       .tocsc();
        REQUIRE_FALSE(A.is_symmetric());
    }

    SECTION("Symmetric matrix with off-diagonals") {
        CSCMatrix A = COOMatrix(v, i, j)
                       .insert(0, 1, 1.0)
                       .insert(1, 0, 1.0)
                       .tocsc();
        REQUIRE(A.is_symmetric());
    }
}


// See demo2 and demo3
TEST_CASE("is_triangular")
{
    const CSCMatrix A = davis_example_chol();
    auto [M, N] = A.shape();

    SECTION("Symmetric, Non-triangular matrix") {
        REQUIRE(A.is_symmetric());
        REQUIRE(A.is_triangular() == 0);
    }

    SECTION("Lower triangular matrix") {
        REQUIRE(A.band(-M, 0).is_triangular() == -1);
    }

    SECTION("Upper triangular matrix") {
        REQUIRE(A.band(0, N).is_triangular() == 1);
    }
}



TEST_CASE("Matrix permutation", "[permute]")
{
    CSCMatrix A = davis_example_small().compress();

    SECTION("No-op") {
        std::vector<csint> p = {0, 1, 2, 3};
        std::vector<csint> q = {0, 1, 2, 3};

        CSCMatrix C = A.permute(inv_permute(p), q);

        compare_matrices(C, A);
        compare_matrices(A.permute_rows(p), A);
        compare_matrices(A.permute_cols(q), A);
    }

    SECTION("Row permutation") {
        std::vector<csint> p = {1, 0, 2, 3};
        std::vector<csint> q = {0, 1, 2, 3};

        // See Davis pp 7-8, Eqn (2.1)
        CSCMatrix expect = COOMatrix(
            std::vector<double> {3.0,  3.1,  1.0,  3.2,  2.9,  3.5,  0.4,  0.9,  4.5,  1.7},
            std::vector<csint>  {2,    0,    3,    1,    0,    3,    3,    0,    1,    2},
            std::vector<csint>  {2,    0,    3,    2,    1,    0,    1,    3,    0,    1}
        ).tocsc();

        CSCMatrix C = A.permute(inv_permute(p), q);

        compare_matrices(C, expect);
        compare_matrices(A.permute_rows(inv_permute(p)), expect);
    }

    SECTION("Column permutation") {
        std::vector<csint> p = {0, 1, 2, 3};
        std::vector<csint> q = {1, 0, 2, 3};

        // See Davis pp 7-8, Eqn (2.1)
        CSCMatrix expect = COOMatrix(
            std::vector<double> {3.0,  3.1,  1.0,  3.2,  2.9,  3.5,  0.4,  0.9,  4.5,  1.7},
            std::vector<csint>  {2,    1,    3,    0,    1,    3,    3,    1,    0,    2},
            std::vector<csint>  {2,    1,    3,    2,    0,    1,    0,    3,    1,    0}
        ).tocsc();

        CSCMatrix C = A.permute(inv_permute(p), q);

        compare_matrices(C, expect);
        compare_matrices(A.permute_cols(q), expect);
    }

    SECTION("Both row and column permutation") {
        std::vector<csint> p = {3, 0, 2, 1};
        std::vector<csint> q = {2, 1, 3, 0};

        // See Davis pp 7-8, Eqn (2.1)
        CSCMatrix expect = COOMatrix(
            std::vector<double> {3.0,  3.1,  1.0,  3.2,  2.9,  3.5,  0.4,  0.9,  4.5,  1.7},
            std::vector<csint>  {2,    3,    0,    1,    3,    0,    0,    3,    1,    2},
            std::vector<csint>  {0,    3,    2,    0,    1,    3,    1,    2,    3,    1}
        ).tocsc();

        std::vector<csint> p_inv = inv_permute(p);
        CSCMatrix C = A.permute(p_inv, q);

        compare_matrices(C, expect);
        compare_matrices(A.permute_rows(p_inv).permute_cols(q), expect);

        SECTION("Symbolic permutation") {
            CSCMatrix Cs = A.permute(inv_permute(p), q, false);  // no values
            CSCMatrix Cs2 = A.permute_rows(p_inv, false).permute_cols(q, false);
            CHECK(Cs.data().empty());
            CHECK(Cs2.data().empty());
            compare_matrices(Cs, expect, false);
            compare_matrices(Cs2, expect, false);
        }
    }

    SECTION("Symperm") {
        // Define a symmetric matrix by zero-ing out below-diagonal entries in A
        A.assign(1, 0, 0.0)
         .assign(2, 1, 0.0)
         .assign(3, 0, 0.0)
         .assign(3, 1, 0.0)
         .dropzeros();

        std::vector<csint> p = {3, 0, 2, 1};

        // See Davis pp 7-8, Eqn (2.1)
        CSCMatrix expect = COOMatrix(
            std::vector<double> {3.0,  1.0,  3.2,  2.9,  0.9,  4.5},
            std::vector<csint>  {2,    0,    1,    3,    0,    1},
            std::vector<csint>  {2,    0,    2,    3,    3,    1}
        ).tocsc();

        CSCMatrix C = A.symperm(inv_permute(p));

        compare_matrices(C, expect);

        SECTION("Symbolic permutation") {
            CSCMatrix Cs = A.symperm(inv_permute(p), false);  // no values
            CHECK(Cs.data().empty());
            compare_matrices(Cs, expect, false);
        }
    }
}


TEST_CASE("Exercise 2.26: permuted transpose", "[ex2.26][permute_transpose]")
{
    CSCMatrix A = davis_example_small().compress();

    SECTION("Non-permuted transpose") {
        std::vector<csint> p = {0, 1, 2, 3};
        std::vector<csint> q = {0, 1, 2, 3};

        CSCMatrix expect = A.T();
        CSCMatrix C = A.permute_transpose(inv_permute(p), inv_permute(q));

        compare_matrices(C, expect);
    }

    SECTION("Row-permuted transpose") {
        std::vector<csint> p = {3, 0, 1, 2};
        std::vector<csint> q = {0, 1, 2, 3};

        // See Davis pp 7-8, Eqn (2.1)
        CSCMatrix expect = COOMatrix(
            std::vector<double> {3.0,  3.1,  1.0,  3.2,  2.9,  3.5,  0.4,  0.9,  4.5,  1.7},
            std::vector<csint>  {3,    2,    0,    1,    2,    0,    0,    2,    1,    3},
            std::vector<csint>  {2,    0,    3,    2,    1,    0,    1,    3,    0,    1}
        ).tocsc().T();

        CSCMatrix C = A.permute_transpose(inv_permute(p), inv_permute(q));

        compare_matrices(C, expect);
    }

    SECTION("Column-permuted transpose") {
        std::vector<csint> p = {0, 1, 2, 3};
        std::vector<csint> q = {3, 0, 1, 2};

        // See Davis pp 7-8, Eqn (2.1)
        CSCMatrix expect = COOMatrix(
            std::vector<double> {3.0,  3.1,  1.0,  3.2,  2.9,  3.5,  0.4,  0.9,  4.5,  1.7},
            std::vector<csint>  {2,    1,    3,    0,    1,    3,    3,    1,    0,    2},
            std::vector<csint>  {3,    1,    0,    3,    2,    1,    2,    0,    1,    2}
        ).tocsc().T();

        CSCMatrix C = A.permute_transpose(inv_permute(p), inv_permute(q));

        compare_matrices(C, expect);
    }

    SECTION("Permuted transpose") {
        std::vector<csint> p = {3, 0, 2, 1};
        std::vector<csint> q = {2, 1, 3, 0};

        // See Davis pp 7-8, Eqn (2.1)
        CSCMatrix expect = COOMatrix(
            std::vector<double> {3.0,  3.1,  1.0,  3.2,  2.9,  3.5,  0.4,  0.9,  4.5,  1.7},
            std::vector<csint>  {2,    3,    0,    1,    3,    0,    0,    3,    1,    2},
            std::vector<csint>  {0,    3,    2,    0,    1,    3,    1,    2,    3,    1}
        ).tocsc().T();

        CSCMatrix C = A.permute_transpose(inv_permute(p), inv_permute(q));

        compare_matrices(C, expect);

        SECTION("Symbolic permutation") {
            CSCMatrix Cs = A.permute_transpose(inv_permute(p), inv_permute(q), false);
            CHECK(Cs.data().empty());
            compare_matrices(Cs, expect, false);
        }
    }
}


TEST_CASE("Exercise 2.15: Band function", "[ex2.15][band]")
{
    csint N = 6;
    csint nnz = N*N;

    CSCMatrix A {std::vector<double>(nnz, 1), {N, N}};  // (N, N) of ones

    SECTION("Main diagonal") {
        int kl = 0,
            ku = 0;

        COOMatrix Ab = A.band(kl, ku).tocoo();

        std::vector<csint> expect_rows = {0, 1, 2, 3, 4, 5};
        std::vector<csint> expect_cols = {0, 1, 2, 3, 4, 5};
        std::vector<double> expect_data(expect_rows.size(), 1);

        CHECK(Ab.nnz() == N);
        CHECK(Ab.row() == expect_rows);
        CHECK(Ab.col() == expect_cols);
        REQUIRE(Ab.data() == expect_data);
    }

    SECTION("Arbitrary diagonals") {
        int kl = -3,
            ku = 2;

        // CSCMatrix Ab = A.band(kl, ku);
        // std::cout << Ab;
        COOMatrix Ab = A.band(kl, ku).tocoo();

        std::vector<csint> expect_rows = {0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5};
        std::vector<csint> expect_cols = {0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5};
        std::vector<double> expect_data(expect_rows.size(), 1);

        CHECK(Ab.nnz() == 27);
        CHECK(Ab.row() == expect_rows);
        CHECK(Ab.col() == expect_cols);
        REQUIRE(Ab.data() == expect_data);
    }
}


TEST_CASE("Exercise 2.16: CSC to/from dense", "[ex2.16][fromdense][todense]")
{
    COOMatrix A = davis_example_small();
    CSCMatrix C = A.compress();  // non-canonical

    std::vector<double> dense_column_major = {
        4.5, 3.1, 0.0, 3.5,
        0.0, 2.9, 1.7, 0.4,
        3.2, 0.0, 3.0, 0.0,
        0.0, 0.9, 0.0, 1.0
    };

    std::vector<double> dense_row_major = {
        4.5, 0.0, 3.2, 0.0,
        3.1, 2.9, 0.0, 0.9,
        0.0, 1.7, 3.0, 0.0,
        3.5, 0.4, 0.0, 1.0
    };

    SECTION("From Dense Column-major") {
        CSCMatrix B {dense_column_major, {4, 4}, 'F'};
        compare_matrices(B, C);
    }

    SECTION("From Dense Row-major") {
        CSCMatrix B {dense_row_major, {4, 4}, 'C'};
        compare_matrices(B, C);
    }

    SECTION("To Dense Column-major") {
        REQUIRE(A.tocsc().to_dense_vector() == dense_column_major);  // canonical form
        REQUIRE(C.to_dense_vector() == dense_column_major);          // non-canonical form
    }

    SECTION("To Dense Row-major") {
        REQUIRE(A.tocsc().to_dense_vector('C') == dense_row_major);  // canonical form
        REQUIRE(C.to_dense_vector('C') == dense_row_major);          // non-canonical form
    }
}

// Create a dummy class that builds an invalid matrix
class TestCSCMatrix {
    CSCMatrix& test_matrix_;  // matrix to test

public:
    TestCSCMatrix(CSCMatrix& A) : test_matrix_(A) {}

    void corrupt_p_size(csint wrong_size) {
        test_matrix_.p_.resize(wrong_size);
    }

    void corrupt_p_front(csint wrong_value) {
        if (!test_matrix_.p_.empty()) {
            test_matrix_.p_.front() = wrong_value;
        }
    }

    void corrupt_p_back(csint wrong_value) {
        if (!test_matrix_.p_.empty()) {
            test_matrix_.p_.back() = wrong_value;
        }
    }

    void corrupt_value_size() {
        // make test_matrix_.v_ larger
        test_matrix_.v_.resize(test_matrix_.i_.size() + 56);
    }

    void empty_values() {
        test_matrix_.v_.clear();
    }
};


// "cs_ok"
TEST_CASE("Exercise 2.12: Validity check", "[ex2.12][is_valid]")
{
    CSCMatrix A = davis_example_small().compress();
    TestCSCMatrix A_test_helper(A);

    constexpr bool SORTED = true;
    constexpr bool VALUES = true;

    // Create canonical matrix
    SECTION("Canonical") {
        REQUIRE(A.is_valid(!SORTED, !VALUES));
        REQUIRE(A.to_canonical().is_valid());
    }

    // Create corrupted matrix
    SECTION("Wrong number of columns (p_.size() != N+1)") {
        A_test_helper.corrupt_p_size(56);

        REQUIRE_THROWS_WITH(A.is_valid(!SORTED, !VALUES),
            "Number of columns inconsistent!");
    }

    SECTION("First column index not zero (p_.front() != 0)") {
        A_test_helper.corrupt_p_front(1);

        REQUIRE_THROWS_WITH(A.is_valid(!SORTED, !VALUES),
            "First column index should be 0!");
    }

    SECTION("Last column count inconsistent (p_.back() != nnz())") {
        A_test_helper.corrupt_p_back(56);

        REQUIRE_THROWS_WITH(A.is_valid(!SORTED, !VALUES),
            "Column counts inconsistent!");
    }

    SECTION("Mismatch between indices and values sizes") {
        A_test_helper.corrupt_value_size();  // makes v_.size() != i_.size()

        REQUIRE_THROWS_WITH(A.is_valid(!SORTED, VALUES),
            "Indices and values sizes inconsistent!");
    }

    SECTION("Empty values vector (v_ empty)") {
        A_test_helper.empty_values();

        REQUIRE_THROWS_WITH(A.is_valid(!SORTED, VALUES),
            "No values!");
    }

    SECTION("Sorted") {
        REQUIRE_THROWS_WITH(A.is_valid(), "Columns not sorted!");
        REQUIRE_THROWS_WITH(A.is_valid(SORTED, !VALUES), "Columns not sorted!");
        REQUIRE(A.sort().is_valid(SORTED, !VALUES));
        REQUIRE(A.sort().is_valid());  // no non-zeros
    }

    SECTION("Explicit Non-zeros") {
        A = davis_example_small().insert(0, 1, 0.0).compress();

        REQUIRE_THROWS_WITH(A.is_valid(!SORTED), "Explicit zeros!");
        REQUIRE_THROWS_WITH(A.sort().is_valid(), "Explicit zeros!");
    }

    SECTION("Duplicate Entry") {
        A = davis_example_small().insert(1, 1, 1.0).compress();

        // Un-sorted columns will fail before duplicates are checked
        REQUIRE_THROWS_WITH(A.is_valid(), "Columns not sorted!");
        REQUIRE_THROWS_WITH(A.sort().is_valid(), "Duplicate entries exist!");
    }
}


// "hcat" and "vcat"
TEST_CASE("Exercise 2.22: Concatentation", "[ex2.22][hstack][vstack]")
{
    CSCMatrix E = E_mat();
    CSCMatrix A = A_mat();

    SECTION("Horizontal concatenation") {
        CSCMatrix expect = COOMatrix(
            std::vector<double> {1, -2, 1, 1, 2, 4, -2, 1, -6, 7,  1, 2},
            std::vector<csint>  {0,  1, 1, 2, 0, 1,  2, 0,  1, 2,  0, 2},
            std::vector<csint>  {0,  0, 1, 2, 3, 3,  3, 4,  4, 4,  5, 5}
        ).tocsc();

        CSCMatrix C = hstack(E, A);
        compare_matrices(C, expect);
    }

    SECTION("Vertical concatenation") {
        CSCMatrix expect = COOMatrix(
            std::vector<double> {1, -2, 1, 1, 2, 4, -2, 1, -6, 7,  1, 2},
            std::vector<csint>  {0,  1, 1, 2, 3, 4,  5, 3,  4, 5,  3, 5},
            std::vector<csint>  {0,  0, 1, 2, 0, 0,  0, 1,  1, 1,  2, 2}
        ).tocsc();

        CSCMatrix C = vstack(E, A);
        compare_matrices(C, expect);
    }
}


// slicing with contiguous indices
TEST_CASE("Exercise 2.23: Slicing", "[ex2.23][slice]")
{
    CSCMatrix A = davis_example_small().tocsc();
    auto [M, N] = A.shape();

    SECTION("Row slicing") {
        CSCMatrix expect = COOMatrix(
            std::vector<double> {3.1, 2.9, 1.7, 3.0, 0.9},
            std::vector<csint>  {  0,   0,   1,   1,   0},
            std::vector<csint>  {  0,   1,   1,   2,   3}
        ).tocsc();

        CSCMatrix C = A.slice(1, 3, 0, A.shape()[1]);
        compare_matrices(C, expect);
    }

    SECTION("Column slicing") {
        CSCMatrix expect = COOMatrix(
            std::vector<double> {2.9, 1.7, 0.4, 3.2, 3.0},
            std::vector<csint>  {  1,   2,   3,   0,   2},
            std::vector<csint>  {  0,   0,   0,   1,   1}
        ).tocsc();

        CSCMatrix C = A.slice(0, A.shape()[0], 1, 3);
        compare_matrices(C, expect);
    }

    SECTION("Row and column slicing") {
        CSCMatrix expect = COOMatrix(
            std::vector<double> {2.9, 1.7, 3.0, 0.9},
            std::vector<csint>  {  0,   1,   1,   0},
            std::vector<csint>  {  0,   0,   1,   2}
        ).tocsc();

        CSCMatrix C = A.slice(1, 3, 1, 4);
        compare_matrices(C, expect);
    }

    SECTION("Empty row") {
        CSCMatrix expect {{M, 0}, 0};
        CSCMatrix C = A.slice(0, M, 0, 0);
        compare_matrices(C, expect);
    }

    SECTION("Empty column") {
        CSCMatrix expect {{0, N}, 0};
        CSCMatrix C = A.slice(0, 0, 0, N);
        compare_matrices(C, expect);
    }
}


// indexing with (possibly) non-contiguous indices
TEST_CASE("Exercise 2.24: Non-contiguous indexing", "[ex2.24][index]")
{
    CSCMatrix A = davis_example_small().tocsc();

    SECTION("Indexing without duplicates") {
        CSCMatrix C = A.index({2, 0}, {0, 3, 2});

        CSCMatrix expect = COOMatrix(
            std::vector<double> {4.5, 3.2, 3.0},
            std::vector<csint>  {  1,   1,   0},
            std::vector<csint>  {  0,   2,   2}
        ).tocsc();

        compare_matrices(C, expect);
    }

    SECTION("Indexing with duplicate rows") {
        CSCMatrix C = A.index({2, 0, 1, 1}, {0, 3, 2});

        CSCMatrix expect = COOMatrix(
            std::vector<double> {4.5, 3.1, 3.1, 0.9, 0.9, 3.2, 3.0},
            std::vector<csint>  {  1,   2,   3,   2,   3,   1,   0},
            std::vector<csint>  {  0,   0,   0,   1,   1,   2,   2}
        ).tocsc();

        compare_matrices(C, expect);
    }

    SECTION("Indexing with duplicate columns") {
        CSCMatrix C = A.index({2, 0}, {0, 3, 2, 0});

        CSCMatrix expect = COOMatrix(
            std::vector<double> {4.5, 3.2, 3.0, 4.5},
            std::vector<csint>  {  1,   1,   0,   1},
            std::vector<csint>  {  0,   2,   2,   3}
        ).tocsc();

        compare_matrices(C, expect);
    }
}


TEST_CASE("Exercise 2.25: Indexing for single assignment.", "[ex2.25][assign]")
{
    auto test_assignment = [](
        CSCMatrix& A_in,
        const csint i,
        const csint j,
        const double v,
        const bool is_existing
    )
    {
        // Test both ways of assigning to a single element
        enum AssignmentMethod { AssignMethod, OperatorMethod };
        std::vector<AssignmentMethod> methods_to_test = { AssignMethod, OperatorMethod };

        for (const auto& method : methods_to_test) {
            CSCMatrix A = A_in;  // Copy the original matrix for each test

            csint nnz = A.nnz();
            double original_value = A(i, j); // Capture original value before assignment

            if (method == AssignMethod) {
                A.assign(i, j, v);
            } else {
                A(i, j) = v;
            }

            if (is_existing) {
                CHECK(A.nnz() == nnz);
            } else {
                CHECK(A.nnz() == nnz + 1);
            }

            CAPTURE(
                i, j, v,
                "Method tested: ", method == AssignMethod ? "assign()" : "operator=()",
                "Original value: ", original_value,
                A.has_sorted_indices(),
                A.has_canonical_format()
            );

            // The actual test that we set the value correctly
            REQUIRE(A(i, j) == v);

            // set_item_ should turn off this flag if the value is 0.0
            if (v == 0.0) {
                REQUIRE_FALSE(A.has_canonical_format());
            }
        }
    };

    SECTION("Canonical format") {
        CSCMatrix A = davis_example_small().tocsc();

        SECTION("Re-assign existing element") {
            test_assignment(A, 2, 1, 56.0, true);
        }

        SECTION("Add a new element") {
            test_assignment(A, 0, 1, 56.0, false);
        }
    }

    SECTION("Non-canonical format") {
        CSCMatrix A = davis_example_small().compress();

        SECTION("Re-assign existing element") {
            test_assignment(A, 2, 1, 56.0, true);
        }

        SECTION("Add a new element") {
            test_assignment(A, 0, 1, 56.0, false);
        }
    }

    SECTION("Assign an explicit zero value") {
        CSCMatrix A = davis_example_small().tocsc();
        test_assignment(A, 2, 1, 0.0, true);
    }

    SECTION("Assign to an item that has duplicate entries.") {
        CSCMatrix A = davis_example_small()
                        .insert(3, 1, 56.0)
                        .insert(3, 1,  7.3)
                        .insert(3, 1,  0.2)
                        .compress();  // don't remove duplicates

        SECTION("Not sorted") {
            CHECK_FALSE(A.has_sorted_indices());
            CHECK_FALSE(A.has_canonical_format());
        }

        SECTION("Sorted") {
            A.sort();  // sort in-place
            CHECK(A.has_sorted_indices());
            CHECK_FALSE(A.has_canonical_format());  // still has dups
        }

        REQUIRE(A.nnz() == 13);                      // 10 + 3 duplicates
        REQUIRE(A(3, 1) == (0.4 + 56 + 7.3 + 0.2));  // A(3, 1) == 0.4
        test_assignment(A, 3, 1, 99.0, true);        // works on a copy

        // Check that the duplicates are removed
        A(3, 1) = 99.0;
        A.dropzeros();
        REQUIRE(A.nnz() == 10);  // 10 - 3 duplicates
    }

    SECTION("Multiple assignment") {
        CSCMatrix A = davis_example_small().tocsc();

        std::vector<csint> rows = {2, 0};
        std::vector<csint> cols = {0, 3, 2};

        SECTION("Dense assignment") {
            std::vector<double> vals = {100, 101, 102, 103, 104, 105};

            A.assign(rows, cols, vals);

            for (csint i = 0; i < rows.size(); i++) {
                for (csint j = 0; j < cols.size(); j++) {
                    double A_val = A(rows[i], cols[j]);
                    double expect_val = vals[i + j * rows.size()];
                    CAPTURE(i, j, A_val, expect_val);
                    REQUIRE(A_val == expect_val);
                }
            }
        }

        SECTION("Sparse assignment") {
            const CSCMatrix C = CSCMatrix(
                std::vector<double> {100, 101, 102, 103, 104, 105},
                std::vector<csint> {0, 1, 0, 1, 0, 1},
                std::vector<csint> {0, 2, 4, 6},
                Shape{2, 3}
            );

            A.assign(rows, cols, C);

            for (csint i = 0; i < rows.size(); i++) {
                for (csint j = 0; j < cols.size(); j++) {
                    double A_val = A(rows[i], cols[j]);
                    double expect_val = C(i, j);
                    CAPTURE(i, j, A_val, expect_val);
                    REQUIRE(A_val == expect_val);
                }
            }
        }
    }
}


TEST_CASE("Exercise 2.29: Adding empty rows and columns to a CSCMatrix.", "[ex2.29][add_empty]")
{
    const CSCMatrix A = davis_example_small().tocsc();
    CSCMatrix C = A;
    int k = 3;  // number of rows/columns to add

    SECTION("Add empty rows to top") {
        C.add_empty_top(k);

        std::vector<csint> expect_indices = A.indices();
        for (auto& x : expect_indices) {
            x += k;
        }

        REQUIRE(C.nnz() == A.nnz());
        REQUIRE(C.shape()[0] == A.shape()[0] + k);
        REQUIRE(C.shape()[1] == A.shape()[1]);
        REQUIRE(C.indptr() == A.indptr());
        REQUIRE(C.indices() == expect_indices);
    }

    SECTION("Add empty rows to bottom") {
        C.add_empty_bottom(k);

        REQUIRE(C.nnz() == A.nnz());
        REQUIRE(C.shape()[0] == A.shape()[0] + k);
        REQUIRE(C.shape()[1] == A.shape()[1]);
        REQUIRE(C.indptr() == A.indptr());
        REQUIRE(C.indices() == A.indices());
    }

    SECTION("Add empty columns to left") {
        C.add_empty_left(k);

        std::vector<csint> expect_indptr(k, 0);
        expect_indptr.insert(
            expect_indptr.end(),
            A.indptr().begin(),
            A.indptr().end()
        );

        REQUIRE(C.nnz() == A.nnz());
        REQUIRE(C.shape()[0] == A.shape()[0]);
        REQUIRE(C.shape()[1] == A.shape()[1] + k);
        REQUIRE(C.indptr() == expect_indptr);
        REQUIRE(C.indices() == A.indices());
    }

    SECTION("Add empty columns to right") {
        C.add_empty_right(k);

        std::vector<csint> expect_indptr = A.indptr();
        std::vector<csint> nnzs(k, A.nnz());
        expect_indptr.insert(
            expect_indptr.end(),
            nnzs.begin(),
            nnzs.end()
        );

        REQUIRE(C.nnz() == A.nnz());
        REQUIRE(C.shape()[0] == A.shape()[0]);
        REQUIRE(C.shape()[1] == A.shape()[1] + k);
        REQUIRE(C.indptr() == expect_indptr);
        REQUIRE(C.indices() == A.indices());
    }
}


TEST_CASE("Sum the rows and columns of a matrix", "[sum_rows][sum_cols]")
{
    CSCMatrix A = davis_example_small().tocsc();

    SECTION("Sum the rows") {
        std::vector<double> expect = {7.7, 6.9, 4.7, 4.9};

        std::vector<double> sums = A.sum_rows();

        REQUIRE(sums.size() == expect.size());
        REQUIRE(sums == expect);
    }

    SECTION("Sum the columns") {
        std::vector<double> expect = {11.1,  5.0,  6.2,  1.9};

        std::vector<double> sums = A.sum_cols();

        REQUIRE(sums.size() == expect.size());
        REQUIRE(sums == expect);
    }
}


}  // namespace cs

/*==============================================================================
 *============================================================================*/
