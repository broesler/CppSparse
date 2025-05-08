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

#include <sstream>
#include <string>
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

    SECTION ("CSCMatrix printing") {
        std::stringstream s;

        SECTION("Print short") {
            std::string expect =
                "<Compressed Sparse Column matrix\n"
                "        with 10 stored elements and shape (4, 4)>\n";

            C.print(s);  // default verbose=false

            REQUIRE(s.str() == expect);
        }

        SECTION("Print verbose") {
            std::string expect =
                "<Compressed Sparse Column matrix\n"
                "        with 10 stored elements and shape (4, 4)>\n"
                "(1, 0): 3.1\n"
                "(3, 0): 3.5\n"
                "(0, 0): 4.5\n"
                "(1, 1): 2.9\n"
                "(3, 1): 0.4\n"
                "(2, 1): 1.7\n"
                "(2, 2): 3\n"
                "(0, 2): 3.2\n"
                "(3, 3): 1\n"
                "(1, 3): 0.9\n";

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

    SECTION("Indexing: unsorted, without duplicates") {
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

    SECTION("Indexing: unsorted, with a duplicate") {
        const CSCMatrix C = A.assign(3, 3, 56.0).compress();

        // NOTE "double& operator()" function is being called when we are
        // trying to compare the value. Not sure why.
        // A: The non-const version is called when C is non-const. If C is
        // const, then the const version is called.
        REQUIRE_THAT(C(3, 3), WithinAbs(57.0, tol));
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
                    REQUIRE(C(i, j) == C_T(j, i));
                }
            }
        };

        SECTION("Square matrix M == N") {
            transpose_test(C);  // shape = {4, 4}
        }

        SECTION("Non-square matrix M < N") {
            transpose_test(A.assign(0, 4, 1.6).compress()); // shape = {4, 5}
        }

        SECTION("Non-square matrix M > N") {
            transpose_test(A.assign(4, 0, 1.6).compress()); // shape = {5, 4}
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
        C = A.assign(0, 4, 1.6).compress();  // {4, 5}

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
        C = A.assign(0, 2, 100.0)
             .assign(3, 0, 100.0)
             .assign(2, 1, 100.0)
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
        // Assign explicit zeros
        C = davis_example_small()
            .assign(0, 1, 0.0)
            .assign(2, 1, 0.0)
            .assign(3, 1, 0.0)
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

    SECTION("Exercise 2.2: Conversion to COOMatrix") {
        auto convert_test = [](const COOMatrix& B) {
            // Columns are sorted, but not rows
            std::vector<csint>  expect_i = {  1,   3,   0,   1,   3,   2,   2,   0,   3,   1};
            std::vector<csint>  expect_j = {  0,   0,   0,   1,   1,   1,   2,   2,   3,   3};
            std::vector<double> expect_v = {3.1, 3.5, 4.5, 2.9, 0.4, 1.7, 3.0, 3.2, 1.0, 0.9};

            REQUIRE(B.nnz() == 10);
            REQUIRE(B.nzmax() >= 10);
            REQUIRE(B.shape() == Shape{4, 4});
            REQUIRE(B.row() == expect_i);
            REQUIRE(B.column() == expect_j);
            REQUIRE(B.data() == expect_v);
        };

        SECTION("As constructor") {
            COOMatrix B {C};  // via constructor
            convert_test(B);
        }

        SECTION("As function") {
            COOMatrix B = C.tocoo();  // via member function
            convert_test(B);
        }
    }

    // (inverse)
    SECTION("Exercise 2.16: Conversion to dense array in column-major format") {
        // Column-major order
        std::vector<double> expect = {
            4.5, 3.1, 0.0, 3.5,
            0.0, 2.9, 1.7, 0.4,
            3.2, 0.0, 3.0, 0.0,
            0.0, 0.9, 0.0, 1.0
        };

        REQUIRE(A.tocsc().to_dense_vector() == expect);  // canonical form
        REQUIRE(C.to_dense_vector() == expect);          // non-canonical form
    }

    SECTION("Conversion to dense array in row-major format") {
        // Row-major order
        std::vector<double> expect = {
            4.5, 0.0, 3.2, 0.0,
            3.1, 2.9, 0.0, 0.9,
            0.0, 1.7, 3.0, 0.0,
            3.5, 0.4, 0.0, 1.0
        };

        REQUIRE(A.tocsc().to_dense_vector('C') == expect);  // canonical form
        REQUIRE(C.to_dense_vector('C') == expect);          // non-canonical form
    }
}


TEST_CASE("Canonical format", "[CSCMatrix][COOMatrix]")
{
    std::vector<csint> indptr_expect  = {  0,               3,                 6,        8,  10};
    std::vector<csint> indices_expect = {  0,   1,     3,   1,     2,   3,     0,   2,   1,   3};
    std::vector<double> data_expect   = {4.5, 3.1, 103.5, 2.9, 101.7, 0.4, 103.2, 3.0, 0.9, 1.0};

    COOMatrix A = (
        davis_example_small()  // unsorted matrix
        .assign(0, 2, 100.0)   // assign duplicates
        .assign(3, 0, 100.0)
        .assign(2, 1, 100.0)
        .assign(0, 1, 0.0)     // assign zero entries
        .assign(2, 2, 0.0)
        .assign(3, 1, 0.0)
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


}  // namespace cs

/*==============================================================================
 *============================================================================*/
