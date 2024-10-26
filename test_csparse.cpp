/*==============================================================================
 *     File: test_csparse.cpp
 *  Created: 2024-10-01 21:07
 *   Author: Bernie Roesler
 *
 *  Description: Basic test of my CSparse implementation.
 *
 *============================================================================*/

#define CATCH_CONFIG_MAIN  // tell the compiler to define `main()`

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "csparse.h"

using namespace std;
using Catch::Matchers::Equals;
using Catch::Matchers::WithinAbs;

// constexpr double tol = 1e-16;


TEST_CASE("Test COOMatrix Constructors", "[COOMatrix]")
{
    SECTION("Empty constructor") {
        COOMatrix A;

        REQUIRE(A.nnz() == 0);
        REQUIRE(A.nzmax() == 0);
        REQUIRE(A.shape() == std::array<csint, 2>{0, 0});
    }

    SECTION("Make new from given shape") {
        COOMatrix A(56, 37);
        REQUIRE(A.nnz() == 0);
        REQUIRE(A.nzmax() == 0);
        REQUIRE(A.shape() == std::array<csint, 2>{56, 37});
    }

    SECTION("Allocate new from shape and nzmax") {
        int nzmax = 1e4;
        COOMatrix A(56, 37, nzmax);
        REQUIRE(A.nnz() == nzmax);
        REQUIRE(A.nzmax() >= nzmax);
        REQUIRE(A.shape() == std::array<csint, 2>{56, 37});
    }

}


TEST_CASE("COOMatrix from (v, i, j) literals.", "[COOMatrix]")
{
    // See Davis pp 7-8, Eqn (2.1)
    std::vector<csint>  i = {2,    1,    3,    0,    1,    3,    3,    1,    0,    2};
    std::vector<csint>  j = {2,    0,    3,    2,    1,    0,    1,    3,    0,    1};
    std::vector<double> v = {3.0,  3.1,  1.0,  3.2,  2.9,  3.5,  0.4,  0.9,  4.5,  1.7};
    COOMatrix A(v, i, j);

    SECTION("Test attributes") {
        REQUIRE(A.nnz() == 10);
        REQUIRE(A.nzmax() >= 10);
        REQUIRE(A.shape() == std::array<csint, 2>{4, 4});
        REQUIRE(A.row() == i);
        REQUIRE(A.column() == j);
        REQUIRE(A.data() == v);
    }

    SECTION("Test printing") {
        std::stringstream s;

        SECTION("Print short") {
            std::string expect =
                "<COOrdinate Sparse matrix\n"
                "        with 10 stored elements and shape (4, 4)>\n";

            A.print(s);  // default verbose=false

            REQUIRE(s.str() == expect);
        }

        SECTION("Print verbose") {
            std::string expect =
                "<COOrdinate Sparse matrix\n"
                "        with 10 stored elements and shape (4, 4)>\n"
                "(2, 2): 3\n"
                "(1, 0): 3.1\n"
                "(3, 3): 1\n"
                "(0, 2): 3.2\n"
                "(1, 1): 2.9\n"
                "(3, 0): 3.5\n"
                "(3, 1): 0.4\n"
                "(1, 3): 0.9\n"
                "(0, 0): 4.5\n"
                "(2, 1): 1.7\n";

            SECTION("Print from function") {
                A.print(s, true);
                REQUIRE(s.str() == expect);
            }

            SECTION("Print from operator<< overload") {
                s << A;
                REQUIRE(s.str() == expect);
            }
        }
    }

    SECTION("Assign an existing element") {
        A.assign(3, 3, 56.0);

        REQUIRE(A.nnz() == 11);
        REQUIRE(A.nzmax() >= 11);
        REQUIRE(A.shape() == std::array<csint, 2>{4, 4});
        // TODO implement a private "search" function to ensure it exists?
    }

    SECTION("Assign a new element that changes the dimensions") {
        A.assign(4, 3, 69.0);

        REQUIRE(A.nnz() == 11);
        REQUIRE(A.nzmax() >= 11);
        REQUIRE(A.shape() == std::array<csint, 2>{5, 4});
    }

    SECTION("Tranpose") {
        COOMatrix A_T = A.T();  // copy

        REQUIRE(A_T.row() == j);
        REQUIRE(A_T.column() == i);
        REQUIRE(&A != &A_T);
    }

    SECTION("Read from a file") {
        std::ifstream fp("./data/t1");
        COOMatrix F(fp);

        // TODO implement A == F? Doesn't really work since we don't make any
        // guarantees on the order of the elements. It would essentially just
        // convert them to CSCMatrix, sort columns, and then compare.
        REQUIRE(A.row() == F.row());
        REQUIRE(A.column() == F.column());
        REQUIRE(A.data() == F.data());
    }

    // TODO move these tests to a separate TEST_CASE
    SECTION("Test conversion to CSCMatrix") {
        CSCMatrix C = A.tocsc();

        // cout << "C = \n" << C;
        SECTION("Test attributes") {
            std::vector<csint> indptr_expect  = {  0,             3,             6,        8,  10};
            std::vector<csint> indices_expect = {  1,   3,   0,   1,   3,   2,   2,   0,   3,   1};
            std::vector<double> data_expect   = {3.1, 3.5, 4.5, 2.9, 0.4, 1.7, 3.0, 3.2, 1.0, 0.9};

            REQUIRE(C.nnz() == 10);
            REQUIRE(C.nzmax() >= 10);
            REQUIRE(C.shape() == std::array<csint, 2>{4, 4});
            REQUIRE(C.indptr() == indptr_expect);
            REQUIRE(C.indices() == indices_expect);
            REQUIRE(C.data() == data_expect);
        }

        // SECTION("Transpose") {
        //     cout << "C.T = \n" << C.T();
        // }

        SECTION("Sum duplicates") {
            A.assign(0, 2, 100.0);
            A.assign(3, 0, 100.0);
            A.assign(2, 1, 100.0);
            C = A.tocsc().sum_duplicates();

            // TODO implement indexing into CSCMatrix
            // REQUIRE_THAT(C(0, 2), WithinAbs(103.2, tol));
            // REQUIRE_THAT(C(3, 0), WithinAbs(103.5, tol));
            // REQUIRE_THAT(C(2, 1), WithinAbs(101.7, tol));
        }

        SECTION("Test droptol") {
            C = COOMatrix(v, i, j).tocsc().droptol(2.0);

            // TODO write custom Matcher to assert all C >= 2.0
        }

        SECTION("Test dropzeros") {
            // TODO have assign return "*this" so we can chain assignments
            // Test dropzeros
            // C = COOMatrix(v, i, j)
            //     .assign(0, 1, 0.0)
            //     .assign(2, 1, 0.0)
            //     .assign(3, 1, 0.0)
            //     .tocsc();

            // Assign explicit zeros
            A.assign(0, 1, 0.0);
            A.assign(2, 1, 0.0);
            A.assign(3, 1, 0.0);
            C = A.tocsc();

            // cout << "C with zeros = \n" << C;
            C.dropzeros();
            // cout << "C without zeros = \n" << C;
            // TODO assert that C.data() does not contain any zeros.
        }
    }

    // TODO test whether transpose, droptol, etc. change the original if we do
    // an assignment

}


/*==============================================================================
 *============================================================================*/
