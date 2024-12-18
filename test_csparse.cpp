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
#include <ranges>

#include "csparse.h"

using namespace std;
using Catch::Matchers::AllTrue;
using Catch::Matchers::WithinAbs;

constexpr double tol = 1e-14;


COOMatrix davis_21_coo() 
{
    // See Davis pp 7-8, Eqn (2.1)
    std::vector<csint>  i = {2,    1,    3,    0,    1,    3,    3,    1,    0,    2};
    std::vector<csint>  j = {2,    0,    3,    2,    1,    0,    1,    3,    0,    1};
    std::vector<double> v = {3.0,  3.1,  1.0,  3.2,  2.9,  3.5,  0.4,  0.9,  4.5,  1.7};
    return COOMatrix {v, i, j};
}

// See: Strang, p 25
// E = [[ 1, 0, 0],
//      [-2, 1, 0],
//      [ 0, 0, 1]]
//
// A = [[ 2, 1, 1],
//      [ 4,-6, 0],
//      [-2, 7, 2]]

// Build matrices with sorted columns
CSCMatrix E_mat() 
{
    return COOMatrix(
        std::vector<double> {1, -2, 1, 1},  // vals
        std::vector<csint>  {0,  1, 1, 2},  // rows
        std::vector<csint>  {0,  0, 1, 2}   // cols
    ).tocsc();
}

CSCMatrix A_mat() 
{
    return COOMatrix(
        std::vector<double> {2, 4, -2, 1, -6, 7, 1, 2},  // vals
        std::vector<csint>  {0, 1,  2, 0,  1, 2, 0, 2},  // rows
        std::vector<csint>  {0, 0,  0, 1,  1, 1, 2, 2}   // cols
    ).tocsc();
}


/** Compare two matrices for equality.
 *
 * @note This function expects the matrices to be in canonical form.
 *
 * @param C       the matrix to test
 * @param expect  the expected matrix
 */
auto matrix_compare(const CSCMatrix& C, const CSCMatrix& expect)
{
    REQUIRE(C.has_canonical_format());
    REQUIRE(expect.has_canonical_format());
    CHECK(C.nnz() == expect.nnz());
    CHECK(C.shape() == expect.shape());
    CHECK(C.indptr() == expect.indptr());
    CHECK(C.indices() == expect.indices());
    REQUIRE(C.data() == expect.data());
}


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


std::vector<bool> is_close(
    const std::vector<double>& a,
    const std::vector<double>& b,
    const double tol=1e-15
    )
{
    assert(a.size() == b.size());

    std::vector<bool> out(a.size());
    for (int i = 0; i < a.size(); i++) {
        out[i] = (std::fabs(a[i] - b[i]) < tol);
    }

    return out;
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


TEST_CASE("Test cumsum", "[vector]")
{
    std::vector<csint> a = {1, 1, 1, 1};
    std::vector<csint> c = cumsum(a);
    std::vector<csint> expect = {0, 1, 2, 3, 4};
    REQUIRE(c == expect);
    REQUIRE(a == expect);  // result also copied into input!!
    REQUIRE(&a != &c);
}


TEST_CASE("Test vector permutations", "[vector]")
{
    std::vector<double> b = {0, 1, 2, 3, 4};
    std::vector<csint> p = {2, 0, 1, 4, 3};

    REQUIRE(pvec(p, b) == std::vector<double>{2, 0, 1, 4, 3});
    REQUIRE(ipvec(p, b) == std::vector<double>{1, 2, 0, 4, 3});
    REQUIRE(inv_permute(p) == std::vector<csint>{1, 2, 0, 4, 3});
}


TEST_CASE("Test argsort.", "[vector]")
{
    SECTION("Test vector of doubles") {
        std::vector<double> v = {5.6, 6.9, 42.0, 1.7, 9.0};
        REQUIRE(argsort(v) == std::vector<csint> {3, 0, 1, 4, 2});
    }

    SECTION("Test vector of ints") {
        std::vector<int> v = {5, 6, 42, 1, 9};
        REQUIRE(argsort(v) == std::vector<csint> {3, 0, 1, 4, 2});
    }
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

    // Exercise 2.5
    SECTION("Assign a dense submatrix") {
        std::vector<csint> rows = {2, 3, 4};
        std::vector<csint> cols = {4, 5, 6};
        std::vector<double> vals = {1, 2, 3, 4, 5, 6, 7, 8, 9};

        A.assign(rows, cols, vals);

        REQUIRE(A.nnz() == 19);
        REQUIRE(A.nzmax() >= 19);
        REQUIRE(A.shape() == std::array<csint, 2>{5, 7});
        // cout << "A = " << endl << A;  // rows sorted
        // cout << "A.compress() = " << endl << A.compress();  // cols sorted
    }

    SECTION("Tranpose") {
        COOMatrix A_T = A.transpose();
        COOMatrix A_TT = A.T();

        REQUIRE(A_T.row() == j);
        REQUIRE(A_T.column() == i);
        REQUIRE(A_T.row() == A_TT.row());
        REQUIRE(A_T.column() == A_TT.column());
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
}


TEST_CASE("Test CSCMatrix", "[CSCMatrix]")
{
    COOMatrix A = davis_21_coo();
    CSCMatrix C = A.compress();

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

    SECTION("Test indexing") {
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
        C = A.assign(3, 3, 56.0).compress();

        REQUIRE_THAT(C(3, 3), WithinAbs(57.0, tol));
    }

    // Test the transpose -> use indexing to test A(i, j) == A(j, i)
    SECTION("Transpose") {
        // lambda to test on M == N, M < N, M > N
        auto transpose_test = [](CSCMatrix C) {
            CSCMatrix C_T = C.transpose();

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
        };

        SECTION("Test square matrix M == N") {
            transpose_test(C);  // shape = {4, 4}
        }

        SECTION("Test non-square matrix M < N") {
            transpose_test(A.assign(0, 4, 1.6).compress()); // shape = {4, 5}
        }

        SECTION("Test non-square matrix M > N") {
            transpose_test(A.assign(4, 0, 1.6).compress()); // shape = {5, 4}
        }
    }

    SECTION("Sort rows/columns") {
        // Test on non-square matrix M != N
        C = A.assign(0, 4, 1.6).compress();  // {4, 5}

        auto sort_test = [](const CSCMatrix& Cs) {
            std::array<csint, 2> shape_expect = {4, 5};
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

    SECTION("Test droptol") {
        C = davis_21_coo().compress().droptol(2.0);

        REQUIRE(C.nnz() == 6);
        REQUIRE(C.shape() == std::array<csint, 2>{4, 4});
        REQUIRE_THAT(C.data() >= 2.0, AllTrue());
    }

    SECTION("Test dropzeros") {
        // Assign explicit zeros
        C = davis_21_coo()
            .assign(0, 1, 0.0)
            .assign(2, 1, 0.0)
            .assign(3, 1, 0.0)
            .compress();

        REQUIRE(C.nnz() == 13);

        C.dropzeros();

        REQUIRE(C.nnz() == 10);
        REQUIRE_THAT(C.data() != 0.0, AllTrue());
    }

    SECTION("Test 1-norm") {
        REQUIRE_THAT(C.norm(), WithinAbs(11.1, tol));
    }

    // Exercise 2.2
    SECTION("Test Conversion to COOMatrix") {
        auto convert_test = [](const COOMatrix& B) {
            // Columns are sorted, but not rows
            std::vector<csint>  expect_i = {  1,   3,   0,   1,   3,   2,   2,   0,   3,   1};
            std::vector<csint>  expect_j = {  0,   0,   0,   1,   1,   1,   2,   2,   3,   3};
            std::vector<double> expect_v = {3.1, 3.5, 4.5, 2.9, 0.4, 1.7, 3.0, 3.2, 1.0, 0.9};

            REQUIRE(B.nnz() == 10);
            REQUIRE(B.nzmax() >= 10);
            REQUIRE(B.shape() == std::array<csint, 2>{4, 4});
            REQUIRE(B.row() == expect_i);
            REQUIRE(B.column() == expect_j);
            REQUIRE(B.data() == expect_v);
        };

        SECTION("As constructor") {
            COOMatrix B(C);  // via constructor
            convert_test(B);
        }

        SECTION("As function") {
            COOMatrix B = C.tocoo();  // via member function
            convert_test(B);
        }
    }
}

// TODO test whether transpose, droptol, etc. change the original if we do
// an assignment


TEST_CASE("Test canonical format", "[CSCMatrix][COOMatrix]")
{
    std::vector<csint> indptr_expect  = {  0,               3,                 6,        8,  10};
    std::vector<csint> indices_expect = {  0,   1,     3,   1,     2,   3,     0,   2,   1,   3};
    std::vector<double> data_expect   = {4.5, 3.1, 103.5, 2.9, 101.7, 0.4, 103.2, 3.0, 0.9, 1.0};

    COOMatrix A = (
        davis_21_coo()        // unsorted matrix
        .assign(0, 2, 100.0)  // assign duplicates
        .assign(3, 0, 100.0)
        .assign(2, 1, 100.0)
        .assign(0, 1, 0.0)    // assign zero entries
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

    SECTION("Test constructor") {
        CSCMatrix B(A);
        REQUIRE(C.indptr() == B.indptr());
        REQUIRE(C.indices() == B.indices());
        REQUIRE(C.data() == B.data());
    }

    SECTION("Test indexing") {
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


// Exercise 2.13
TEST_CASE("Test is_symmetric.") {
    std::vector<csint>  i = {0, 1, 2};
    std::vector<csint>  j = {0, 1, 2};
    std::vector<double> v = {1, 2, 3};

    SECTION("Test diagonal matrix") {
        CSCMatrix A = COOMatrix(v, i, j).tocsc();
        REQUIRE(A.is_symmetric());
    }

    SECTION("Test non-symmetric matrix with off-diagonals") {
        CSCMatrix A = COOMatrix(v, i, j)
                       .assign(0, 1, 1.0)
                       .tocsc();
        REQUIRE_FALSE(A.is_symmetric());
    }

    SECTION("Test symmetric matrix with off-diagonals") {
        CSCMatrix A = COOMatrix(v, i, j)
                       .assign(0, 1, 1.0)
                       .assign(1, 0, 1.0)
                       .tocsc();
        REQUIRE(A.is_symmetric());
    }
}


/*------------------------------------------------------------------------------
 *          Math Operations
 *----------------------------------------------------------------------------*/
TEST_CASE("Matrix-(dense) vector multiply + addition.", "[math]")
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
        REQUIRE_THAT(is_close(gaxpy(A, x, zero),   expect_Ax,   tol), AllTrue());
        REQUIRE_THAT(is_close(gaxpy(A, x, y),      expect_Axpy, tol), AllTrue());
        REQUIRE_THAT(is_close(gatxpy(A.T(), x, y), expect_Axpy, tol), AllTrue());
        REQUIRE_THAT(is_close(A.dot(x),            expect_Ax,   tol), AllTrue());
        REQUIRE_THAT(is_close((A * x),             expect_Ax,   tol), AllTrue());
        REQUIRE_THAT(is_close((A * x + y),         expect_Axpy, tol), AllTrue());
    };

    SECTION("Test a non-square matrix.") {
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

    SECTION("Test a symmetric (diagonal) matrix.") {
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
        REQUIRE_THAT(is_close(sym_gaxpy(A, x, y),  expect_Axpy, tol), AllTrue());
    }

    SECTION("Test an arbitrary non-symmetric matrix.") {
        COOMatrix Ac = davis_21_coo();
        CSCMatrix A = Ac.compress();

        std::vector<double> x = {1, 2, 3, 4};
        std::vector<double> y = {1, 1, 1, 1};

        // A @ x + y
        std::vector<double> expect_Ax   = {14.1, 12.5, 12.4,  8.3};
        std::vector<double> expect_Axpy = {15.1, 13.5, 13.4,  9.3};

        multiply_test(A, x, y, expect_Ax, expect_Axpy);

        // Test COOMatrix
        REQUIRE_THAT(is_close(Ac.dot(x), expect_Ax, tol), AllTrue());
        REQUIRE_THAT(is_close((Ac * x),  expect_Ax, tol), AllTrue());
    }

    SECTION("Test an arbitrary symmetric matrix.") {
        // See Davis pp 7-8, Eqn (2.1)
        std::vector<csint>  i = {  0,   1,   3,   0,   1,   2,   1,   2,   0,   3};
        std::vector<csint>  j = {  0,   0,   0,   1,   1,   1,   2,   2,   3,   3};
        std::vector<double> v = {4.5, 3.1, 3.5, 3.1, 2.9, 1.7, 1.7, 3.0, 3.5, 1.0};
        CSCMatrix A = COOMatrix(v, i, j).compress();

        std::vector<double> x = {1, 2, 3, 4};
        std::vector<double> y = {1, 1, 1, 1};

        // A @ x + y
        std::vector<double> expect_Axpy = {25.7, 15.0, 13.4,  8.5};

        REQUIRE_THAT(is_close(sym_gaxpy(A, x, y), expect_Axpy, tol), AllTrue());
    }
}


TEST_CASE("Matrix-matrix multiply.", "[math]")
{
    SECTION("Test square matrices") {
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
            csint M, N;
            std::tie(M, N) = C.shape();

            REQUIRE(M == E.shape()[0]);
            REQUIRE(N == A.shape()[1]);

            for (csint i = 0; i < M; i++) {
                for (csint j = 0; j < N; j++) {
                    REQUIRE_THAT(C(i, j), WithinAbs(expect(i, j), tol));
                }
            }
        };

        SECTION("Test CSCMatrix::dot (aka cs_multiply)") {
            CSCMatrix C = E * A;
            multiply_test(C, E, A, expect);
        }

        SECTION("Test dot_2x two-pass multiply") {
            CSCMatrix C = E.dot_2x(A);
            multiply_test(C, E, A, expect);
        }
    }

    SECTION("Test arbitrary size matrices") {
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

        CSCMatrix C = A * B;
        csint M, N;
        std::tie(M, N) = C.shape();

        REQUIRE(M == A.shape()[0]);
        REQUIRE(N == B.shape()[1]);

        for (csint i = 0; i < M; i++) {
            for (csint j = 0; j < N; j++) {
                REQUIRE_THAT(C(i, j), WithinAbs(expect(i, j), tol));
            }
        }
    }
}


// Exercise 2.18
TEST_CASE("Sparse Vector-Vector Multiply", "[math]")
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

    SECTION("Test unsorted indices") {
        CHECK_THAT(x.T() * y, WithinAbs(expect, tol));
        REQUIRE_THAT(x.vecdot(y), WithinAbs(expect, tol));
    }

    SECTION("Test sorted indices") {
        x.sort();
        y.sort();

        CHECK_THAT(x.T() * y, WithinAbs(expect, tol));
        REQUIRE_THAT(x.vecdot(y), WithinAbs(expect, tol));
    }
}


TEST_CASE("Scaling by a constant", "[math]")
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


// Exercise 2.4
TEST_CASE("Scale rows and columns", "[math]")
{
    CSCMatrix A = davis_21_coo().compress();

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

    csint M, N;
    std::tie(M, N) = A.shape();

    for (csint i = 0; i < M; i++) {
        for (csint j = 0; j < N; j++) {
            REQUIRE_THAT(RAC(i, j), WithinAbs(expect_RAC(i, j), tol));
        }
    }
}


TEST_CASE("Matrix-matrix addition.", "[math]")
{
    SECTION("Test non-square matrices") {
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

        CSCMatrix expect = COOMatrix(
            std::vector<double> {9.1, 9.2, 9.3, 9.4, 9.5, 9.6},
            i, j
        ).compress();

        csint M, N;
        std::tie(M, N) = A.shape();

        // Test function definition
        CSCMatrix Cf = add_scaled(A, B, 0.1, 9.0);

        // Test operator overloading
        CSCMatrix C = 0.1 * A + 9.0 * B;
        // cout << "C = \n" << C << endl;

        // TODO rewrite these element-tests to compare the entire matrix, so that
        // when we have a failure, we can see the indices
        for (csint i = 0; i < M; i++) {
            for (csint j = 0; j < N; j++) {
                REQUIRE(Cf(i, j) == expect(i, j));
                REQUIRE(C(i, j) == expect(i, j));
            }
        }
    }

    SECTION("Test sparse column vectors") {
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

        csint M, N;
        std::tie(M, N) = a.shape();

        SECTION("Test operator") {
            CSCMatrix C = a + b;

            REQUIRE(C.shape() == a.shape());

            for (csint i = 0; i < M; i++) {
                for (csint j = 0; j < N; j++) {
                    REQUIRE_THAT(C(i, j), WithinAbs(expect(i, j), tol));
                }
            }
        }

        // Exercise 2.21
        SECTION("Test saxpy") {
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
}


TEST_CASE("Test matrix permutation", "[permute]")
{
    // Matrix with 1, 2, 3, 4 on the diagonal
    std::vector<csint>  rows = {0, 1, 2, 3};
    std::vector<double> vals = {1, 2, 3, 4};
    CSCMatrix A = COOMatrix(vals, rows, rows).compress();

    SECTION("Test no-op") {
        std::vector<csint> p = {0, 1, 2, 3};
        std::vector<csint> q = {0, 1, 2, 3};

        std::vector<csint> p_inv = inv_permute(p);

        CSCMatrix C = A.permute(p_inv, q);

        REQUIRE(C.nnz() == A.nnz());
        REQUIRE(C.shape() == A.shape());

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

        CSCMatrix expect = COOMatrix(vals, p_inv, q_inv).compress();

        CSCMatrix C = A.permute(p_inv, q);

        REQUIRE(C.nnz() == A.nnz());
        REQUIRE(C.shape() == A.shape());

        csint M, N;
        std::tie(M, N) = A.shape();

        for (csint i = 0; i < M; i++) {
            for (csint j = 0; j < N; j++) {
                REQUIRE(C(i, j) == expect(i, j));
            }
        }
    }

    SECTION("Test symperm") {
        // Test actual permutation
        std::vector<csint> p = {3, 0, 2, 1};
        std::vector<double> expect_v = {4, 1, 3, 2};

        CSCMatrix expect = COOMatrix(expect_v, rows, rows).compress();

        std::vector<csint> p_inv = inv_permute(p);

        CSCMatrix C = A.symperm(p_inv);

        REQUIRE(C.nnz() == A.nnz());
        REQUIRE(C.shape() == A.shape());

        csint N = p.size();
        for (csint i = 0; i < N; i++) {
            for (csint j = i; j < N; j++) {
                REQUIRE(C(i, j) == expect(i, j));
            }
        }
    }
}


// Exercise 2.15
TEST_CASE("Test band function")
{
    csint N = 6;
    csint nnz = N*N;

    // CSCMatrix A = ones((N, N)); // TODO
    std::vector<csint> r(N);
    std::iota(r.begin(), r.end(), 0);  // sequence to repeat

    std::vector<csint> rows;
    rows.reserve(nnz);

    std::vector<csint> cols;
    cols.reserve(nnz);

    std::vector<double> vals(nnz, 1);

    // Repeat {0, 1, 2, 3, 0, 1, 2, 3, ...}
    for (csint i = 0; i < N; i++) {
        for (auto& x : r) {
            rows.push_back(x);
        }
    }

    // Repeat {0, 0, 0, 1, 1, 1, 2, 2, 2, ...}
    for (auto& x : r) {
        for (csint i = 0; i < N; i++) {
            cols.push_back(x);
        }
    }

    CSCMatrix A = COOMatrix(vals, rows, cols).tocsc();

    SECTION("Test main diagonal") {
        int kl = 0,
            ku = 0;

        COOMatrix Ab = A.band(kl, ku).tocoo();

        std::vector<csint> expect_rows = {0, 1, 2, 3, 4, 5};
        std::vector<csint> expect_cols = {0, 1, 2, 3, 4, 5};
        std::vector<double> expect_data(expect_rows.size(), 1);

        CHECK(Ab.nnz() == N);
        CHECK(Ab.row() == expect_rows);
        CHECK(Ab.column() == expect_cols);
        REQUIRE(Ab.data() == expect_data);
    }

    SECTION("Test arbitrary diagonals") {
        int kl = -3,
            ku = 2;

        // CSCMatrix Ab = A.band(kl, ku);
        // cout << Ab;
        COOMatrix Ab = A.band(kl, ku).tocoo();

        std::vector<csint> expect_rows = {0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5};
        std::vector<csint> expect_cols = {0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5};
        std::vector<double> expect_data(expect_rows.size(), 1);

        CHECK(Ab.nnz() == 27);
        CHECK(Ab.row() == expect_rows);
        CHECK(Ab.column() == expect_cols);
        REQUIRE(Ab.data() == expect_data);
    }
}


// Exercise 2.16
TEST_CASE("Test CSC from dense column-major")
{
    std::vector<double> dense_mat = {
        4.5, 3.1, 0.0, 3.5,
        0.0, 2.9, 1.7, 0.4,
        3.2, 0.0, 3.0, 0.0,
        0.0, 0.9, 0.0, 1.0
    };

    CSCMatrix A {dense_mat, 4, 4};

    CSCMatrix expect_A = davis_21_coo().tocsc();

    CHECK(A.nnz() == expect_A.nnz());
    CHECK(A.indptr() == expect_A.indptr());
    CHECK(A.indices() == expect_A.indices());
    REQUIRE(A.data() == expect_A.data());
}


// Exercise 2.12 "cs_ok"
TEST_CASE("Test validity check")
{
    COOMatrix C = davis_21_coo();
    CSCMatrix A = davis_21_coo().compress();

    constexpr bool SORTED = true;
    constexpr bool VALUES = true;

    REQUIRE(A.is_valid());
    REQUIRE_FALSE(A.is_valid(SORTED));
    REQUIRE_FALSE(A.is_valid(SORTED));

    REQUIRE(A.sort().is_valid(SORTED));
    REQUIRE(A.sort().is_valid(SORTED, VALUES));  // no non-zeros

    // Add explicit non-zeros
    A = C.assign(0, 1, 0.0).compress();
    REQUIRE_FALSE(A.is_valid(!SORTED, VALUES));
}


// Exercise 2.22 "hcat" and "vcat"
TEST_CASE("Test concatentation")
{
    CSCMatrix E = E_mat();
    CSCMatrix A = A_mat();

    SECTION("Test horizontal concatenation") {
        CSCMatrix expect = COOMatrix(
            std::vector<double> {1, -2, 1, 1, 2, 4, -2, 1, -6, 7,  1, 2},
            std::vector<csint>  {0,  1, 1, 2, 0, 1,  2, 0,  1, 2,  0, 2},
            std::vector<csint>  {0,  0, 1, 2, 3, 3,  3, 4,  4, 4,  5, 5}
        ).tocsc();

        CSCMatrix C = hstack(E, A);
        matrix_compare(C, expect);
    }

    SECTION("Test vertical concatenation") {
        CSCMatrix expect = COOMatrix(
            std::vector<double> {1, -2, 1, 1, 2, 4, -2, 1, -6, 7,  1, 2},
            std::vector<csint>  {0,  1, 1, 2, 3, 4,  5, 3,  4, 5,  3, 5},
            std::vector<csint>  {0,  0, 1, 2, 0, 0,  0, 1,  1, 1,  2, 2}
        ).tocsc();

        CSCMatrix C = vstack(E, A);
        matrix_compare(C, expect);
    }
}


// Exercise 2.23 slicing with contiguous indices
TEST_CASE("Test slicing")
{
    CSCMatrix A = davis_21_coo().tocsc();

    SECTION("Test row slicing") {
        CSCMatrix expect = COOMatrix(
            std::vector<double> {3.1, 2.9, 1.7, 3.0, 0.9},
            std::vector<csint>  {  0,   0,   1,   1,   0},
            std::vector<csint>  {  0,   1,   1,   2,   3}
        ).tocsc();

        CSCMatrix C = A.slice(1, 3, 0, A.shape()[1]);
        matrix_compare(C, expect);
    }

    SECTION("Test column slicing") {
        CSCMatrix expect = COOMatrix(
            std::vector<double> {2.9, 1.7, 0.4, 3.2, 3.0},
            std::vector<csint>  {  1,   2,   3,   0,   2},
            std::vector<csint>  {  0,   0,   0,   1,   1}
        ).tocsc();

        CSCMatrix C = A.slice(0, A.shape()[0], 1, 3);
        matrix_compare(C, expect);
    }

    SECTION("Test row and column slicing") {
        CSCMatrix expect = COOMatrix(
            std::vector<double> {2.9, 1.7, 3.0, 0.9},
            std::vector<csint>  {  0,   1,   1,   0},
            std::vector<csint>  {  0,   0,   1,   2}
        ).tocsc();

        CSCMatrix C = A.slice(1, 3, 1, 4);
        matrix_compare(C, expect);
    }
}

    }
}

/*==============================================================================
 *============================================================================*/
