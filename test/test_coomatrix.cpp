/*==============================================================================
 *     File: test_coomatrix.cpp
 *  Created: 2025-05-08 11:38
 *   Author: Bernie Roesler
 *
 *  Description: Test COOMatrix constructors and other basic functions.
 *
 *============================================================================*/

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "csparse.h"
#include "test_helpers.h"

namespace cs {


TEST_CASE("COOMatrix Constructors", "[COOMatrix][constructor]")
{
    SECTION("Empty constructor") {
        COOMatrix A;

        CHECK(A.nnz() == 0);
        CHECK(A.nzmax() == 0);
        CHECK(A.shape() == Shape{0, 0});
    }

    SECTION("Make new from given shape") {
        COOMatrix A{{56, 37}};
        CHECK(A.nnz() == 0);
        CHECK(A.nzmax() == 0);
        CHECK(A.shape() == Shape{56, 37});
    }

    SECTION("Allocate new from shape and nzmax") {
        int nzmax = 1e4;
        COOMatrix A{{56, 37}, nzmax};
        CHECK(A.nnz() == 0);
        CHECK(A.nzmax() >= nzmax);
        CHECK(A.shape() == Shape{56, 37});
    }

    SECTION("From (v, i, j) literals") {
        // See Davis pp 7-8, Eqn (2.1)
        std::vector<csint>  i{2,    1,    3,    0,    1,    3,    3,    1,    0,    2};
        std::vector<csint>  j{2,    0,    3,    2,    1,    0,    1,    3,    0,    1};
        std::vector<double> v{3.0,  3.1,  1.0,  3.2,  2.9,  3.5,  0.4,  0.9,  4.5,  1.7};
        COOMatrix A{v, i, j};

        CHECK(A.nnz() == 10);
        CHECK(A.nzmax() >= 10);
        CHECK(A.shape() == Shape{4, 4});
        CHECK(A.row() == i);
        CHECK(A.col() == j);
        CHECK(A.data() == v);
    }
}


TEST_CASE("COOMatrix methods", "[COOMatrix][methods]")
{
    auto A = davis_example_small();

    SECTION("Printing") {
        std::stringstream s;
        std::string expect;

        SECTION("Print short") {
            expect =
                "<C++Sparse COOrdinate Sparse matrix\n"
                "        with 10 stored elements and shape (4, 4)>\n";

            A.print(s);  // default verbose=false
        }

        SECTION("Print verbose") {
            expect =
                "<C++Sparse COOrdinate Sparse matrix\n"
                "        with 10 stored elements and shape (4, 4)>\n"
                "(2, 2):  3\n"
                "(1, 0):  3.1\n"
                "(3, 3):  1\n"
                "(0, 2):  3.2\n"
                "(1, 1):  2.9\n"
                "(3, 0):  3.5\n"
                "(3, 1):  0.4\n"
                "(1, 3):  0.9\n"
                "(0, 0):  4.5\n"
                "(2, 1):  1.7\n";

            SECTION("Print from function") {
                A.print(s, true);
            }

            SECTION("Print from operator<< overload") {
                s << A;
            }
        }

        REQUIRE(s.str() == expect);

        // Clear the stringstream to prevent memory leaks
        s.str("");
        s.clear();
    }

    SECTION("Insert an existing element to create a duplicate") {
        A.insert(3, 3, 56.0);

        REQUIRE(A.nnz() == 11);
        REQUIRE(A.nzmax() >= 11);
        REQUIRE(A.shape() == Shape{4, 4});
        // REQUIRE_THAT(A(3, 3), WithinAbs(57.0, tol));
    }

    SECTION("Insert a new element that changes the dimensions") {
        A.insert(4, 3, 69.0);

        REQUIRE(A.nnz() == 11);
        REQUIRE(A.nzmax() >= 11);
        REQUIRE(A.shape() == Shape{5, 4});
        // REQUIRE_THAT(A(4, 3), WithinAbs(69.0, tol));
    }

    SECTION("Exercise 2.5: Insert a dense submatrix") {
        std::vector<csint> rows{2, 3, 4};
        std::vector<csint> cols{4, 5, 6};
        std::vector<double> vals{1, 2, 3, 4, 5, 6, 7, 8, 9};

        A.insert(rows, cols, vals);

        REQUIRE(A.nnz() == 19);
        REQUIRE(A.nzmax() >= 19);
        REQUIRE(A.shape() == Shape{5, 7});
    }

    SECTION("Exercise 2.6: Tranpose") {
        auto A_T = A.transpose();
        auto A_TT = A.T();

        REQUIRE(A_T.row() == A.col());
        REQUIRE(A_T.col() == A.row());
        REQUIRE(A_T.row() == A_TT.row());
        REQUIRE(A_T.col() == A_TT.col());
        REQUIRE(&A != &A_T);
    }

    SECTION("Read from a file") {
        auto F = COOMatrix::from_file("./data/t1");

        REQUIRE(A.row() == F.row());
        REQUIRE(A.col() == F.col());
        REQUIRE(A.data() == F.data());
    }

    SECTION("Conversion to dense array: Column-major") {
        std::vector<double> expect{
            4.5, 3.1, 0.0, 3.5,
            0.0, 2.9, 1.7, 0.4,
            3.2, 0.0, 3.0, 0.0,
            0.0, 0.9, 0.0, 1.0
        };

        REQUIRE(A.to_dense_vector() == expect);
        REQUIRE(A.to_dense_vector(DenseOrder::ColMajor) == expect);
    }

    SECTION("Conversion to dense array: Row-major") {
        std::vector<double> expect{
            4.5, 0.0, 3.2, 0.0,
            3.1, 2.9, 0.0, 0.9,
            0.0, 1.7, 3.0, 0.0,
            3.5, 0.4, 0.0, 1.0
        };

        REQUIRE(A.to_dense_vector(DenseOrder::RowMajor) == expect);
    }

    SECTION("Generate random matrix") {
        double density = 0.25;
        csint M = 5, N = 10;
        unsigned int seed = 56;  // seed for reproducibility
        auto A = COOMatrix::random(M, N, density, seed);

        REQUIRE(A.shape() == Shape{M, N});
        REQUIRE(A.nnz() == (csint)(density * M * N));
    }
}


}  // namespace cs

/*==============================================================================
 *============================================================================*/
