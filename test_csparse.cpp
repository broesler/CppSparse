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

// #include <compare>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "csparse.h"

using namespace std;
using Catch::Matchers::AllTrue;
using Catch::Matchers::WithinAbs;

constexpr double tol = 1e-16;


// TODO figure out how to use the "spaceship" operator<=> to define all
// of the comparisons in one fell swoop? 
// A: May only work if we define a wrapper class on std::vector and define the
//    operator within the class vs. scalars.

/** Return a boolean vector comparing each individual element.
 *
 * @param vec   a vector of doubles.
 * @param c     the value against which to compare
 * @return out  a vector whose elements are vec[i] <=> c.
 */
// std::vector<bool> operator<=>(const std::vector<double>& vec, const double c)
// {
//     std::vector<bool> out(vec.size());

//     for (auto const& v : vec) {
//         if (v < c) {
//             out.push_back(std::strong_ordering::less);
//         } else if (v > c) {
//             out.push_back(std::strong_ordering::greater);
//         } else {
//             out.push_back(std::strong_ordering::equal);
//         }
//     }

//     return out;
// }


/** Return a boolean vector comparing each individual element.
 *
 * @param vec   a vector of doubles
 * @param c     the value against which to compare
 * @param comp  the comparison function for elements of the vector and scalar
 *
 * @return out  a vector whose elements are vec[i] <=> c.
 */
std::vector<bool> compare_vec(
    const std::vector<double>& vec,
    const double c,
    std::function<bool(double, double)> comp
    )
{
    std::vector<bool> out;
    out.reserve(vec.size());
    for (const auto& v : vec) {
        out.push_back(comp(v, c));
    }
    return out;
}


// Create the comparison operators by passing the single comparison function to
// our vector comparison function.
std::vector<bool> operator>=(const std::vector<double>& vec, const double c)
{
    return compare_vec(vec, c, std::greater_equal<double>());
}


std::vector<bool> operator!=(const std::vector<double>& vec, const double c)
{
    return compare_vec(vec, c, std::not_equal_to<double>());
}


/*------------------------------------------------------------------------------
 *         Test Utilities
 *----------------------------------------------------------------------------*/
TEST_CASE("Test vector ops", "[vector]")
{
    std::vector<double> a = {1, 2, 3};

    SECTION("Scale a vector") {
        std::vector<double> expect = {2, 4, 6};

        REQUIRE((2 * a) == expect);
        REQUIRE((a * 2) == expect);
    }

    SECTION("Add two vectors") {
        std::vector<double> b = {4, 5, 6};

        REQUIRE((a + b) == std::vector<double>{5, 7, 9});
    }
}


TEST_CASE("Test vector permutations", "[vector]")
{
    std::vector<double> b = {0, 1, 2, 3, 4};
    std::vector<csint> p = {2, 0, 1, 4, 3};

    REQUIRE(pvec(p, b) == std::vector<double>{2, 0, 1, 4, 3});
    REQUIRE(ipvec(p, b) == std::vector<double>{1, 2, 0, 4, 3});
    REQUIRE(inv_permute(p) == std::vector<csint>{1, 2, 0, 4, 3});
}


/*------------------------------------------------------------------------------
 *         Test Matrix Functions 
 *----------------------------------------------------------------------------*/
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
                A.print(s, true);  // FIXME memory leak?
                REQUIRE(s.str() == expect);
            }

            SECTION("Print from operator<< overload") {
                s << A;  // FIXME memory leak?
                REQUIRE(s.str() == expect);
            }
        }

        // Clear the stringstream to prevent memory leaks
        s.str("");
        s.clear();
    }

    SECTION("Assign an existing element to create a duplicate") {
        A.assign(3, 3, 56.0);

        REQUIRE(A.nnz() == 11);
        REQUIRE(A.nzmax() >= 11);
        REQUIRE(A.shape() == std::array<csint, 2>{4, 4});
        // REQUIRE_THAT(A(3, 3), WithinAbs(57.0, tol));
    }

    SECTION("Assign a new element that changes the dimensions") {
        A.assign(4, 3, 69.0);

        REQUIRE(A.nnz() == 11);
        REQUIRE(A.nzmax() >= 11);
        REQUIRE(A.shape() == std::array<csint, 2>{5, 4});
        // REQUIRE_THAT(A(4, 3), WithinAbs(69.0, tol));
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

        SECTION ("Test CSCMatrix printing") {
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
                    C.print(s, true);  // FIXME memory leak?
                    REQUIRE(s.str() == expect);
                }

                SECTION("Print from operator<< overload") {
                    s << C;  // FIXME memory leak?
                    REQUIRE(s.str() == expect);
                }
            }

            // Clear the stringstream to prevent memory leaks
            s.str("");
            s.clear();
        }

        SECTION("Test indexing: no duplicates") {
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

        SECTION("Test indexing: with a duplicate") {
            C = A.assign(3, 3, 56.0).tocsc();

            REQUIRE_THAT(C(3, 3), WithinAbs(57.0, tol));
        }

        // Test the transpose -> use indexing to test A(i, j) == A(j, i)
        SECTION("Transpose") {
            CSCMatrix C_T = C.T();

            csint M, N;
            std::tie(M, N) = C.shape();

            REQUIRE(C.nnz() == C_T.nnz());
            REQUIRE(M == C_T.shape()[1]);
            REQUIRE(N == C_T.shape()[0]);

            for (csint i = 0; i < M; i++) {
                for (csint j = 0; j < N; j++) {
                    REQUIRE(C(i, j) == C_T(j, i));
                }
            }
        }

        SECTION("Sum duplicates") {
            C = A.assign(0, 2, 100.0)
                 .assign(3, 0, 100.0)
                 .assign(2, 1, 100.0)
                 .tocsc()
                 .sum_duplicates();

            REQUIRE_THAT(C(0, 2), WithinAbs(103.2, tol));
            REQUIRE_THAT(C(3, 0), WithinAbs(103.5, tol));
            REQUIRE_THAT(C(2, 1), WithinAbs(101.7, tol));
        }

        SECTION("Test droptol") {
            C = COOMatrix(v, i, j).tocsc().droptol(2.0);

            REQUIRE(C.nnz() == 6);
            REQUIRE(C.shape() == std::array<csint, 2>{4, 4});
            REQUIRE_THAT(C.data() >= 2.0, AllTrue());
        }

        SECTION("Test dropzeros") {
            // Assign explicit zeros
            C = COOMatrix(v, i, j)
                .assign(0, 1, 0.0)
                .assign(2, 1, 0.0)
                .assign(3, 1, 0.0)
                .tocsc();

            REQUIRE(C.nnz() == 13);

            C.dropzeros();

            REQUIRE(C.nnz() == 10);
            REQUIRE_THAT(C.data() != 0.0, AllTrue());
        }

        SECTION("Test 1-norm") {
            REQUIRE_THAT(C.norm(), WithinAbs(11.1, tol));
        }
    }

    // TODO test whether transpose, droptol, etc. change the original if we do
    // an assignment
}


TEST_CASE("Matrix-vector multiply + addition.", "[math]")
{
    std::vector<csint>  i = {0, 1, 2};
    std::vector<csint>  j = {0, 1, 2};
    std::vector<double> v = {1, 2, 3};
    CSCMatrix A = COOMatrix(v, i, j).tocsc();

    std::vector<double> x = {1, 2, 3};
    std::vector<double> y = {9, 6, 1};

    REQUIRE(gaxpy(A, x, y) == std::vector<double>{10, 10, 10});
    REQUIRE((A * x) == std::vector<double>{1, 4, 9});
    REQUIRE((A * x + y) == std::vector<double>{10, 10, 10});
    // REQUIRE((A * x - y) == std::vector<double>{-8, -2, -8});
    // REQUIRE(-y == std::vector<double>{-9, -6, -1});
}


TEST_CASE("Matrix-matrix multiply.", "[math]")
{
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
    ).tocsc();

    CSCMatrix B = COOMatrix(
        std::vector<double> {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},  // vals
        std::vector<csint>  {0, 0, 0, 1, 1, 1, 2, 2, 2,  3,  3,  3},  // rows
        std::vector<csint>  {0, 1, 2, 0, 1, 2, 0, 1, 2,  0,  1,  2}   // cols
    ).tocsc();

    CSCMatrix expect = COOMatrix(
        std::vector<double> {70, 80, 90, 158, 184, 210},  // vals
        std::vector<csint>  { 0,  0,  0,   1,   1,   1},  // rows
        std::vector<csint>  { 0,  1,  2,   0,   1,   2}   // cols
    ).tocsc();

    CSCMatrix C = A * B;  // FIXME heap overflow
    csint M, N;
    std::tie(M, N) = C.shape();

    REQUIRE(M == A.shape()[0]);
    REQUIRE(N == B.shape()[1]);

    for (csint i = 0; i < M; i++) {
        for (csint j = 0; j < N; j++) {
            REQUIRE(C(i, j) == expect(i, j));
        }
    }
}


TEST_CASE("Scaling by a constant", "[math]")
{
    std::vector<csint> i{{0, 0, 0, 1, 1, 1}};  // rows
    std::vector<csint> j{{0, 1, 2, 0, 1, 2}};  // cols

    CSCMatrix A = COOMatrix(
        std::vector<double> {1, 2, 3, 4, 5, 6},
        i, j
    ).tocsc();

    CSCMatrix expect = COOMatrix(
        std::vector<double> {0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
        i, j
    ).tocsc();

    csint M, N;
    std::tie(M, N) = A.shape();

    // Test operator overloading
    CSCMatrix C = 0.1 * A;

    for (csint i = 0; i < M; i++) {
        for (csint j = 0; j < N; j++) {
            REQUIRE_THAT(C(i, j), WithinAbs(expect(i, j), tol));
        }
    }
}


TEST_CASE("Matrix-matrix addition.", "[math]")
{
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
    ).tocsc();

    CSCMatrix B = COOMatrix(
        std::vector<double> {1, 1, 1, 1, 1, 1},
        i, j
    ).tocsc();

    CSCMatrix expect = COOMatrix(
        std::vector<double> {9.1, 9.2, 9.3, 9.4, 9.5, 9.6},
        i, j
    ).tocsc();

    csint M, N;
    std::tie(M, N) = A.shape();

    // Test function definition
    CSCMatrix Cf = add(A, B, 0.1, 9.0);
    
    // Test operator overloading
    CSCMatrix C = 0.1 * A + 9.0 * B;
    // cout << "C = \n" << C << endl;  // FIXME infinite loop?

    // TODO rewrite these element-tests to compare the entire matrix, so that
    // when we have a failure, we can see the indices
    for (csint i = 0; i < M; i++) {
        for (csint j = 0; j < N; j++) {
            REQUIRE(Cf(i, j) == expect(i, j));
            REQUIRE(C(i, j) == expect(i, j));
        }
    }
}


TEST_CASE("Test matrix permutation", "[permute]")
{
    // Matrix with 1, 2, 3, 4 on the diagonal
    std::vector<csint>  i = {0, 1, 2, 3};
    std::vector<csint>  j = {0, 1, 2, 3};
    std::vector<double> v = {1, 2, 3, 4};
    CSCMatrix A = COOMatrix(v, i, j).tocsc();

    SECTION("Test no-op") {
        std::vector<csint> p = {0, 1, 2, 3};
        std::vector<csint> q = {0, 1, 2, 3};

        std::vector<csint> p_inv = inv_permute(p);

        CSCMatrix C = A.permute(p_inv, q);

        REQUIRE(A.shape() == C.shape());

        csint M, N;
        std::tie(M, N) = A.shape();

        for (csint i = 0; i < M; i++) {
            for (csint j = 0; j < N; j++) {
                REQUIRE(C(i, j) == A(i, j));
            }
        }
    }

    SECTION("Test actual permutation") {
        std::vector<csint> p = {3, 0, 2, 1};
        std::vector<csint> q = {2, 1, 0, 3};

        std::vector<csint> p_inv = inv_permute(p);
        std::vector<csint> q_inv = inv_permute(q);

        CSCMatrix expect = COOMatrix(v, p_inv, q_inv).tocsc();
        CSCMatrix C = A.permute(p_inv, q);

        REQUIRE(A.shape() == C.shape());

        csint M, N;
        std::tie(M, N) = A.shape();

        for (csint i = 0; i < M; i++) {
            for (csint j = 0; j < N; j++) {
                REQUIRE(C(i, j) == expect(i, j));
            }
        }
    }
}


/*==============================================================================
 *============================================================================*/
