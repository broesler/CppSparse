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

#include <algorithm>  // reverse
#include <iostream>
#include <fstream>
#include <map>
#include <numeric>    // iota
#include <random>
#include <string>
#include <sstream>
#include <vector>

#include "csparse.h"

using namespace cs;

using Catch::Matchers::AllTrue;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::UnorderedEquals;

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
auto compare_canonical(
    const CSCMatrix& C,
	const CSCMatrix& expect,
	bool values=true,
	double tol=1e-14
)
{
    REQUIRE(C.has_canonical_format());
    REQUIRE(expect.has_canonical_format());
    CHECK(C.nnz() == expect.nnz());
    CHECK(C.shape() == expect.shape());
    CHECK(C.indptr() == expect.indptr());
    CHECK(C.indices() == expect.indices());
    if (values) {
        for (csint p = 0; p < C.nnz(); p++) {
            REQUIRE_THAT(C.data()[p], WithinAbs(expect.data()[p], tol));
        }
    }
}


/** Compare two matrices for equality.
 *
 * @note This function does not require the matrices to be in canonical form.
 *
 * @param C       the matrix to test
 * @param expect  the expected matrix
 */
auto compare_noncanonical(
    const CSCMatrix& C,
	const CSCMatrix& expect,
    bool values=true,
	double tol=1e-14
)
{
    REQUIRE(C.nnz() == expect.nnz());
    REQUIRE(C.shape() == expect.shape());

    auto [M, N] = C.shape();

    if (values) {
        // Need to check all elements of the matrix because operator() combines
        // duplicate entries, whereas just going through the non-zeros of one
        // matrix does not combine those duplicates.
        for (csint i = 0; i < M; i++) {
            for (csint j = 0; j < N; j++) {
                REQUIRE_THAT(C(i, j), WithinAbs(expect(i, j), tol));
            }
        }
    }
}


auto compare_matrices(
    const CSCMatrix& C,
	const CSCMatrix& expect,
	bool values=true,
	double tol=1e-14
)
{
    if (C.has_canonical_format() && expect.has_canonical_format()) {
        compare_canonical(C, expect, values, tol);
    } else {
        compare_noncanonical(C, expect, values, tol);
    }
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
    const double tol=1e-14
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

    SECTION("Negate a vector") {
        REQUIRE(-a == std::vector<double>{-1, -2, -3});
    }

    SECTION("Subtract two vectors") {
        std::vector<double> b = {4, 5, 6};

        REQUIRE((a - b) == std::vector<double>{-3, -3, -3});
    }
}


TEST_CASE("Test cumsum", "[vector]")
{
    std::vector<csint> a = {1, 1, 1, 1};
    std::vector<csint> c = cumsum(a);
    std::vector<csint> expect = {0, 1, 2, 3, 4};
    REQUIRE(c == expect);
    REQUIRE(&a != &c);
}


TEST_CASE("Test vector permutations", "[vector]")
{
    std::vector<double> b = {0, 1, 2, 3, 4};
    std::vector<csint> p = {2, 0, 1, 4, 3};

    REQUIRE(pvec(p, b) == std::vector<double>{2, 0, 1, 4, 3});
    REQUIRE(ipvec(p, b) == std::vector<double>{1, 2, 0, 4, 3});
    REQUIRE(inv_permute(p) == std::vector<csint>{1, 2, 0, 4, 3});
    REQUIRE(pvec(inv_permute(p), b) == ipvec(p, b));
    REQUIRE(ipvec(inv_permute(p), b) == pvec(p, b));
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
        REQUIRE(A.shape() == Shape{0, 0});
    }

    SECTION("Make new from given shape") {
        COOMatrix A({56, 37});
        REQUIRE(A.nnz() == 0);
        REQUIRE(A.nzmax() == 0);
        REQUIRE(A.shape() == Shape{56, 37});
    }

    SECTION("Allocate new from shape and nzmax") {
        int nzmax = 1e4;
        COOMatrix A({56, 37}, nzmax);
        REQUIRE(A.nnz() == 0);
        REQUIRE(A.nzmax() >= nzmax);
        REQUIRE(A.shape() == Shape{56, 37});
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
        REQUIRE(A.shape() == Shape{4, 4});
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
        REQUIRE(A.shape() == Shape{4, 4});
        // REQUIRE_THAT(A(3, 3), WithinAbs(57.0, tol));
    }

    SECTION("Assign a new element that changes the dimensions") {
        A.assign(4, 3, 69.0);

        REQUIRE(A.nnz() == 11);
        REQUIRE(A.nzmax() >= 11);
        REQUIRE(A.shape() == Shape{5, 4});
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
        REQUIRE(A.shape() == Shape{5, 7});
        // std::cout << "A = " << std::endl << A;  // rows sorted
        // std::cout << "A.compress() = " << std::endl << A.compress();  // cols sorted
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

        REQUIRE(A.row() == F.row());
        REQUIRE(A.column() == F.column());
        REQUIRE(A.data() == F.data());
    }

    SECTION("Conversion to dense array: Column-major") {
        std::vector<double> expect = {
            4.5, 3.1, 0.0, 3.5,
            0.0, 2.9, 1.7, 0.4,
            3.2, 0.0, 3.0, 0.0,
            0.0, 0.9, 0.0, 1.0
        };

        REQUIRE(A.to_dense_vector() == expect);
        REQUIRE(A.to_dense_vector('F') == expect);
    }

    SECTION("Conversion to dense array: Row-major") {
        std::vector<double> expect = {
            4.5, 0.0, 3.2, 0.0,
            3.1, 2.9, 0.0, 0.9,
            0.0, 1.7, 3.0, 0.0,
            3.5, 0.4, 0.0, 1.0
        };

        REQUIRE(A.to_dense_vector('C') == expect);
    }

    SECTION("Generate random matrix") {
        double density = 0.25;
        csint M = 5, N = 10;
        unsigned int seed = 56;  // seed for reproducibility
        COOMatrix A = COOMatrix::random(M, N, density, seed);

        REQUIRE(A.shape() == Shape{M, N});
        REQUIRE(A.nnz() == (csint)(density * M * N));
    }
}


TEST_CASE("Test CSCMatrix", "[CSCMatrix]")
{
    COOMatrix A = davis_21_coo();
    CSCMatrix C = A.compress();  // unsorted columns

    // std::cout << "C = \n" << C;
    SECTION("Test attributes") {
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

    SECTION("Test indexing: unsorted, without duplicates") {
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

    SECTION("Test indexing: unsorted, with a duplicate") {
        const CSCMatrix C = A.assign(3, 3, 56.0).compress();

        // NOTE "double& operator()" function is being called when we are
        // trying to compare the value. Not sure why.
        // A: The non-const version is called when C is non-const. If C is
        // const, then the const version is called.
        REQUIRE_THAT(C(3, 3), WithinAbs(57.0, tol));
    }

    // Test the transpose -> use indexing to test A(i, j) == A(j, i)
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

        SECTION("Test square matrix M == N") {
            transpose_test(C);  // shape = {4, 4}
        }

        SECTION("Test non-square matrix M < N") {
            transpose_test(A.assign(0, 4, 1.6).compress()); // shape = {4, 5}
        }

        SECTION("Test non-square matrix M > N") {
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
        // Test on non-square matrix M != N
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
        REQUIRE(C.shape() == Shape{4, 4});
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

    SECTION("Test Frobenius norm") {
        double expect = 8.638286867197685;  // computed in MATLAB and numpy
        REQUIRE_THAT(C.fronorm(), WithinAbs(expect, tol));
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
            REQUIRE(B.shape() == Shape{4, 4});
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

    // Exercise 2.16 (inverse)
    SECTION("Test conversion to dense array in column-major format") {
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

    SECTION("Test conversion to dense array in row-major format") {
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
        REQUIRE_THAT(is_close(A.gaxpy(x, zero),   expect_Ax,   tol), AllTrue());
        REQUIRE_THAT(is_close(A.gaxpy(x, y),      expect_Axpy, tol), AllTrue());
        REQUIRE_THAT(is_close(A.T().gatxpy(x, y), expect_Axpy, tol), AllTrue());
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
        REQUIRE_THAT(is_close(A.sym_gaxpy(x, y),  expect_Axpy, tol), AllTrue());
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

        REQUIRE_THAT(is_close(A.sym_gaxpy(x, y), expect_Axpy, tol), AllTrue());
    }
}


// Exercise 2.27
TEST_CASE("Matrix-(dense) matrix multiply + addition.")
{
    CSCMatrix A = davis_21_coo().compress();

    SECTION("Test identity op") {
        std::vector<double> I = {
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        };

        std::vector<double> Z(16, 0);

        CSCMatrix expect = A;

        compare_matrices(CSCMatrix(A.gaxpy_col(I, Z), {4, 4}), expect);
        compare_matrices(CSCMatrix(A.T().gatxpy_col(I, Z), {4, 4}), expect);
    }

    SECTION("Test arbitrary square matrix in column-major format") {
        std::vector<double> A_dense = A.to_dense_vector();

        // A.T @ A + A in column-major format
        std::vector<double> expect = {
            46.61, 13.49, 14.4 ,  9.79,
            10.39, 14.36,  6.8 ,  3.41,
            17.6 ,  5.1 , 22.24,  0.0 ,
             6.29,  3.91,  0.0 ,  2.81
        };

        std::vector<double> C_col = A.T().gaxpy_col(A_dense, A_dense);
        std::vector<double> C_block = A.T().gaxpy_block(A_dense, A_dense);
        std::vector<double> CT_col = A.gatxpy_col(A_dense, A_dense);
        std::vector<double> CT_block = A.gatxpy_block(A_dense, A_dense);

        REQUIRE_THAT(is_close(C_col, expect, tol), AllTrue());
        REQUIRE_THAT(is_close(C_block, expect, tol), AllTrue());
        REQUIRE_THAT(is_close(CT_col, expect, tol), AllTrue());
        REQUIRE_THAT(is_close(CT_block, expect, tol), AllTrue());
    }

    SECTION("Test arbitrary square matrix in row-major format") {
        std::vector<double> A_dense = A.to_dense_vector('C');

        // A.T @ A + A in row-major format
        std::vector<double> expect = {
            46.61, 10.39, 17.6 ,  6.29,
            13.49, 14.36,  5.1 ,  3.91,
            14.4 ,  6.8 , 22.24,  0.0 ,
             9.79,  3.41,  0.0 ,  2.81
        };

        std::vector<double> C = A.T().gaxpy_row(A_dense, A_dense);
        std::vector<double> CT = A.gatxpy_row(A_dense, A_dense);

        REQUIRE_THAT(is_close(C, expect, tol), AllTrue());
        REQUIRE_THAT(is_close(CT, expect, tol), AllTrue());
    }

    SECTION("Test non-square matrix in column-major format.") {
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

        REQUIRE_THAT(is_close(Ab.gaxpy_col(Ac_dense, A_dense), expect, tol),
                     AllTrue());
        REQUIRE_THAT(is_close(Ab.gaxpy_block(Ac_dense, A_dense), expect, tol),
                     AllTrue());
        REQUIRE_THAT(is_close(Ab.T().gatxpy_col(Ac_dense, A_dense), expect, tol),
                     AllTrue());
        REQUIRE_THAT(is_close(Ab.T().gatxpy_block(Ac_dense, A_dense), expect, tol),
                     AllTrue());
    }

    SECTION("Test non-square matrix in row-major format.") {
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

        REQUIRE_THAT(is_close(Ab.gaxpy_row(Ac_dense, A_dense), expect, tol),
                     AllTrue());
        REQUIRE_THAT(is_close(Ab.T().gatxpy_row(Ac_dense, A_dense), expect, tol),
                     AllTrue());
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
            auto [M, N] = C.shape();

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
        auto [M, N] = C.shape();

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

    auto [M, N] = A.shape();

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

    auto [M, N] = A.shape();

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

        SECTION("Addition") {
            CSCMatrix expect = COOMatrix(
                std::vector<double> {9.1, 9.2, 9.3, 9.4, 9.5, 9.6},
                i, j
            ).compress();

            // Test function definition
            CSCMatrix Cf = add_scaled(A, B, 0.1, 9.0);

            // Test operator overloading
            CSCMatrix C = 0.1 * A + 9.0 * B;

            compare_matrices(C, expect);
            compare_matrices(Cf, expect);
        }

        SECTION("Subtraction") {
            CSCMatrix expect = COOMatrix(
                std::vector<double> {0, 1, 2, 3, 4, 5},
                i, j
            ).compress();

            // Test function definition
            CSCMatrix Cf = add_scaled(A, B, 1.0, -1.0);

            // Test operator overloading
            CSCMatrix C = A - B;

            compare_matrices(C, expect);
            compare_matrices(Cf, expect);
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

        auto [M, N] = a.shape();

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
    CSCMatrix A = davis_21_coo().compress();

    SECTION("Test no-op") {
        std::vector<csint> p = {0, 1, 2, 3};
        std::vector<csint> q = {0, 1, 2, 3};

        CSCMatrix C = A.permute(inv_permute(p), q);

        compare_matrices(C, A);
        compare_matrices(A.permute_rows(p), A);
        compare_matrices(A.permute_cols(q), A);
    }

    SECTION("Test row permutation") {
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

    SECTION("Test column permutation") {
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

    SECTION("Test both row and column permutation") {
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

    SECTION("Test symperm") {
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

    // Exercise 2.26
    SECTION("Test non-permuted transpose") {
        std::vector<csint> p = {0, 1, 2, 3};
        std::vector<csint> q = {0, 1, 2, 3};

        CSCMatrix expect = A.T();
        CSCMatrix C = A.permute_transpose(inv_permute(p), inv_permute(q));

        compare_matrices(C, expect);
    }

    SECTION("Test row-permuted transpose") {
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

    SECTION("Test column-permuted transpose") {
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

    SECTION("Test permuted transpose") {
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


// Exercise 2.15
TEST_CASE("Test band function")
{
    csint N = 6;
    csint nnz = N*N;

    CSCMatrix A(std::vector<double>(nnz, 1), {N, N});  // (N, N) of ones

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
        // std::cout << Ab;
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
TEST_CASE("Test CSC from dense")
{
    CSCMatrix expect_A = davis_21_coo().tocsc();

    SECTION("Column-major") {
        std::vector<double> dense_mat = {
            4.5, 3.1, 0.0, 3.5,
            0.0, 2.9, 1.7, 0.4,
            3.2, 0.0, 3.0, 0.0,
            0.0, 0.9, 0.0, 1.0
        };
        CSCMatrix A {dense_mat, {4, 4}, 'F'};
        compare_matrices(A, expect_A);
    }

    SECTION("Row-major") {
        std::vector<double> dense_mat = {
            4.5, 0.0, 3.2, 0.0,
            3.1, 2.9, 0.0, 0.9,
            0.0, 1.7, 3.0, 0.0,
            3.5, 0.4, 0.0, 1.0
        };
        CSCMatrix A {dense_mat, {4, 4}, 'C'};
        compare_matrices(A, expect_A);
    }
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
        compare_matrices(C, expect);
    }

    SECTION("Test vertical concatenation") {
        CSCMatrix expect = COOMatrix(
            std::vector<double> {1, -2, 1, 1, 2, 4, -2, 1, -6, 7,  1, 2},
            std::vector<csint>  {0,  1, 1, 2, 3, 4,  5, 3,  4, 5,  3, 5},
            std::vector<csint>  {0,  0, 1, 2, 0, 0,  0, 1,  1, 1,  2, 2}
        ).tocsc();

        CSCMatrix C = vstack(E, A);
        compare_matrices(C, expect);
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
        compare_matrices(C, expect);
    }

    SECTION("Test column slicing") {
        CSCMatrix expect = COOMatrix(
            std::vector<double> {2.9, 1.7, 0.4, 3.2, 3.0},
            std::vector<csint>  {  1,   2,   3,   0,   2},
            std::vector<csint>  {  0,   0,   0,   1,   1}
        ).tocsc();

        CSCMatrix C = A.slice(0, A.shape()[0], 1, 3);
        compare_matrices(C, expect);
    }

    SECTION("Test row and column slicing") {
        CSCMatrix expect = COOMatrix(
            std::vector<double> {2.9, 1.7, 3.0, 0.9},
            std::vector<csint>  {  0,   1,   1,   0},
            std::vector<csint>  {  0,   0,   1,   2}
        ).tocsc();

        CSCMatrix C = A.slice(1, 3, 1, 4);
        compare_matrices(C, expect);
    }
}


// Exercise 2.24 indexing with (possibly) non-contiguous indices
TEST_CASE("Test non-contiguous indexing")
{
    CSCMatrix A = davis_21_coo().tocsc();

    SECTION("Test indexing without duplicates") {
        CSCMatrix C = A.index({2, 0}, {0, 3, 2});

        CSCMatrix expect = COOMatrix(
            std::vector<double> {4.5, 3.2, 3.0},
            std::vector<csint>  {  1,   1,   0},
            std::vector<csint>  {  0,   2,   2}
        ).tocsc();

        compare_matrices(C, expect);
    }

    SECTION("Test indexing with duplicate rows") {
        CSCMatrix C = A.index({2, 0, 1, 1}, {0, 3, 2});

        CSCMatrix expect = COOMatrix(
            std::vector<double> {4.5, 3.1, 3.1, 0.9, 0.9, 3.2, 3.0},
            std::vector<csint>  {  1,   2,   3,   2,   3,   1,   0},
            std::vector<csint>  {  0,   0,   0,   1,   1,   2,   2}
        ).tocsc();

        compare_matrices(C, expect);
    }

    SECTION("Test indexing with duplicate columns") {
        CSCMatrix C = A.index({2, 0}, {0, 3, 2, 0});

        CSCMatrix expect = COOMatrix(
            std::vector<double> {4.5, 3.2, 3.0, 4.5},
            std::vector<csint>  {  1,   1,   0,   1},
            std::vector<csint>  {  0,   2,   2,   3}
        ).tocsc();

        compare_matrices(C, expect);
    }
}


// Exercise 2.25 indexing for assignment
TEST_CASE("Test indexing for single assignment.")
{
    auto test_assignment = [](
        CSCMatrix& A,
        const csint i,
        const csint j,
        const double v,
        const bool is_existing
    )
    {
        csint nnz = A.nnz();

        A.assign(i, j, v);

        if (is_existing) {
            CHECK(A.nnz() == nnz);
        } else {
            CHECK(A.nnz() == nnz + 1);
        }
        REQUIRE(A(i, j) == v);
    };

    SECTION("Canonical format") {
        CSCMatrix A = davis_21_coo().tocsc();

        SECTION("Re-assign existing element") {
            test_assignment(A, 2, 1, 56.0, true);
        }

        SECTION("Add a new element") {
            test_assignment(A, 0, 1, 56.0, false);
        }
    }

    SECTION("Non-canonical format") {
        CSCMatrix A = davis_21_coo().compress();

        SECTION("Re-assign existing element") {
            test_assignment(A, 2, 1, 56.0, true);
        }

        SECTION("Add a new element") {
            test_assignment(A, 0, 1, 56.0, false);
        }
    }

    SECTION("Test multiple assignment") {
        CSCMatrix A = davis_21_coo().tocsc();

        std::vector<csint> rows = {2, 0};
        std::vector<csint> cols = {0, 3, 2};

        SECTION("Dense assignment") {
            std::vector<double> vals = {100, 101, 102, 103, 104, 105};

            A.assign(rows, cols, vals);

            for (csint i = 0; i < rows.size(); i++) {
                for (csint j = 0; j < cols.size(); j++) {
                    REQUIRE(A(rows[i], cols[j]) == vals[i + j * rows.size()]);
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
                    REQUIRE(A(rows[i], cols[j]) == C(i, j));
                }
            }
        }
    }
}


// Exercise 2.29
TEST_CASE("Test adding empty rows and columns to a CSCMatrix.")
{
    CSCMatrix A = davis_21_coo().tocsc();
    int k = 3;  // number of rows/columns to add

    SECTION("Add empty rows to top") {
        CSCMatrix C = A.add_empty_top(k);

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
        CSCMatrix C = A.add_empty_bottom(k);

        REQUIRE(C.nnz() == A.nnz());
        REQUIRE(C.shape()[0] == A.shape()[0] + k);
        REQUIRE(C.shape()[1] == A.shape()[1]);
        REQUIRE(C.indptr() == A.indptr());
        REQUIRE(C.indices() == A.indices());
    }

    SECTION("Add empty columns to left") {
        CSCMatrix C = A.add_empty_left(k);

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
        CSCMatrix C = A.add_empty_right(k);

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


TEST_CASE("Sum the rows and columns of a matrix")
{
    CSCMatrix A = davis_21_coo().tocsc();

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

/*------------------------------------------------------------------------------
 *          Matrix Solutions
 *----------------------------------------------------------------------------*/
TEST_CASE("Test triangular solve with dense RHS")
{
    const CSCMatrix L = COOMatrix(
        std::vector<double> {1, 2, 3, 4, 5, 6},
        std::vector<csint>  {0, 1, 1, 2, 2, 2},
        std::vector<csint>  {0, 0, 1, 0, 1, 2}
    ).tocsc();

    const CSCMatrix U = L.T();

    const std::vector<double> expect = {1, 1, 1};

    SECTION("Forward solve L x = b") {
        const std::vector<double> b = {1, 5, 15};  // row sums of L
        const std::vector<double> x = lsolve(L, b);

        REQUIRE(x.size() == expect.size());
        REQUIRE_THAT(is_close(x, expect, tol), AllTrue());
    }

    SECTION("Backsolve L.T x = b") {
        const std::vector<double> b = {7, 8, 6};  // row sums of L.T == col sums of L
        const std::vector<double> x = ltsolve(L, b);

        REQUIRE(x.size() == expect.size());
        REQUIRE_THAT(is_close(x, expect, tol), AllTrue());
    }

    SECTION("Backsolve U x = b") {
        const std::vector<double> b = {7, 8, 6};  // row sums of L.T == col sums of L
        const std::vector<double> x = usolve(U, b);

        REQUIRE(x.size() == expect.size());
        REQUIRE_THAT(is_close(x, expect, tol), AllTrue());
    }

    SECTION("Forward solve U.T x = b") {
        const std::vector<double> b = {1, 5, 15};  // row sums of L
        const std::vector<double> x = utsolve(U, b);

        REQUIRE(x.size() == expect.size());
        REQUIRE_THAT(is_close(x, expect, tol), AllTrue());
    }
}


TEST_CASE("Reachability and DFS")
{
    csint N = 14;  // size of L

    // Define a lower-triangular matrix L with arbitrary non-zeros
    std::vector<csint> rows = {2, 3, 4, 6, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13};
    std::vector<csint> cols = {0, 1, 2, 1, 2, 4, 1, 3, 5, 5, 6, 7,  6,  9,  8, 10,  8,  9, 10, 11,  9, 12};

    // Add the diagonals
    std::vector<csint> diags(N);
    std::iota(diags.begin(), diags.end(), 0);
    rows.insert(rows.end(), diags.begin(), diags.end());
    cols.insert(cols.end(), diags.begin(), diags.end());

    // All values are 1
    std::vector<double> vals(rows.size(), 1);

    CSCMatrix L = COOMatrix(vals, rows, cols).tocsc();
    CSCMatrix U = L.T();

    // Define the rhs matrix B
    CSCMatrix B(Shape {N, 1});

    SECTION("dfs from a single node") {
        // Assign non-zeros to rows 3 and 5 in column 0
        csint j = 3;
        B.assign(j, 0, 1.0);
        std::vector<csint> expect = {13, 12, 11, 8, 3};  // reversed in stack

        std::vector<bool> marked(N, false);
        std::vector<csint> xi;  // do not initialize!
        xi.reserve(N);

        xi = dfs(L, j, marked, xi);

        REQUIRE(xi == expect);
    }

    SECTION("Reachability from a single node") {
        // Assign non-zeros to rows 3 and 5 in column 0
        B.assign(3, 0, 1.0);
        std::vector<csint> expect = {3, 8, 11, 12, 13};

        std::vector<csint> xi = reach(L, B, 0);

        REQUIRE(xi == expect);
    }

    SECTION("Reachability from multiple nodes") {
        // Assign non-zeros to rows 3 and 5 in column 0
        B.assign(3, 0, 1.0).assign(5, 0, 1.0).to_canonical();
        std::vector<csint> expect = {5, 9, 10, 3, 8, 11, 12, 13};

        std::vector<csint> xi = reach(L, B, 0);

        REQUIRE(xi == expect);
    }

    SECTION("spsolve Lx = b with dense RHS") {
        // Create RHS from sums of rows of L, so that x == ones(N)
        std::vector<double> b = {1., 1., 2., 2., 2., 1., 2., 3., 4., 4., 3., 3., 5., 3.};
        for (int i = 0; i < N; i++) {
            B.assign(i, 0, b[i]);
        }
        std::vector<double> expect(N, 1.0);

        // Use structured bindings to unpack the result
        auto [xi, x] = spsolve(L, B, 0);

        REQUIRE(x == expect);
    }

    SECTION("spsolve Lx = b with sparse RHS") {
        // RHS is just B with non-zeros in the first column
        B.assign(3, 0, 1.0);

        std::vector<double> expect = { 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  1.,  0.,  0.};

        // Use structured bindings to unpack the result
        auto [xi, x] = spsolve(L, B, 0);

        REQUIRE(x == expect);
    }

    SECTION("spsolve Ux = b with sparse RHS") {
        // RHS is just B with non-zeros in the first column
        B.assign(3, 0, 1.0);

        std::vector<double> expect = {0., -1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.};

        auto [xi, x] = spsolve(U, B, 0, false);

        REQUIRE(x == expect);
    }
}


TEST_CASE("Permuted triangular solvers")
{
    // >>> L.toarray()
    // === array([[1, 0, 0, 0, 0, 0],
    //            [2, 2, 0, 0, 0, 0],
    //            [3, 3, 3, 0, 0, 0],
    //            [4, 4, 4, 4, 0, 0],
    //            [5, 5, 5, 5, 5, 0],
    //            [6, 6, 6, 6, 6, 6]])
    // >>> (P @ L).toarray().astype(int)
    // === array([[ 6,  6,  6,  6,  6, *6],
    //            [ 4,  4,  4, *4,  0,  0],
    //            [*1,  0,  0,  0,  0,  0],
    //            [ 2, *2,  0,  0,  0,  0],
    //            [ 5,  5,  5,  5, *5,  0],
    //            [ 3,  3, *3,  0,  0,  0]])
    //
    // Starred elements are the diagonals of the un-permuted matrix

    // Create full matrix with row numbers as values
    const std::vector<double> row_vals = {1, 2, 3, 4, 5, 6};
    const csint N = row_vals.size();

    std::vector<double> A_vals;
    A_vals.reserve(N * N);

    for (csint i = 0; i < N; i++) {
        A_vals.insert(A_vals.end(), row_vals.begin(), row_vals.end());
    }

    const CSCMatrix A = CSCMatrix(A_vals, {N, N});

    // Un-permuted matrices
    const CSCMatrix L = A.band(-N, 0);
    const CSCMatrix U = A.band(0, N);

    // TODO I am curious what happens when there is more than one singleton row
    // at a time in find_tri_permutation(). Try removing a few entries from each
    // matrix to make them more sparse and see if the permutation order is still
    // correct.

    const std::vector<csint> p = {5, 3, 0, 1, 4, 2};
    const std::vector<csint> q = {1, 4, 0, 2, 5, 3};

    // Permute the rows (non-canonical form works too)
    const CSCMatrix PL = L.permute_rows(inv_permute(p)).to_canonical();
    const CSCMatrix PU = U.permute_rows(inv_permute(p)).to_canonical();

    // Permute the columns (non-canonical form works too)
    const CSCMatrix LQ = L.permute_cols(p).to_canonical();
    const CSCMatrix UQ = U.permute_cols(p).to_canonical();

    // Permute both rows and columns
    const CSCMatrix PLQ = L.permute(inv_permute(p), q).to_canonical();
    const CSCMatrix PUQ = U.permute(inv_permute(p), q).to_canonical();

    SECTION("Find diagonals of permuted L") {
        std::vector<csint> expect = {2, 8, 14, 16, 19, 20};
        std::vector<csint> p_diags = find_lower_diagonals(PL);
        CHECK(p_diags == expect);

        // Check that we can get the inverse permutation
        std::vector<csint> p_inv = inv_permute(p);  // {2, 3, 5, 1, 4, 0};
        std::vector<csint> diags;
        for (const auto& p : p_diags) {
            diags.push_back(PL.indices()[p]);
        }
        REQUIRE(diags == p_inv);
    }

    SECTION("Find diagonals of permuted U") {
        std::vector<csint> expect = {0, 2, 5, 6, 13, 15};
        std::vector<csint> p_diags = find_upper_diagonals(PU);
        CHECK(p_diags == expect);

        // Check that we can get the inverse permutation
        std::vector<csint> p_inv = inv_permute(p);  // {2, 3, 5, 1, 4, 0};
        std::vector<csint> diags;
        for (const auto& p : p_diags) {
            diags.push_back(PU.indices()[p]);
        }
        REQUIRE(diags == p_inv);
    }

    SECTION("Find diagonals of non-triangular matrix") {
        const CSCMatrix A = davis_21_coo().tocsc();
        REQUIRE_THROWS(find_lower_diagonals(A));
        REQUIRE_THROWS(find_upper_diagonals(A));
        REQUIRE_THROWS(find_tri_permutation(A));
    }

    SECTION("Find permutation vectors of permuted L") {
        std::vector<csint> expect_p = inv_permute(p);
        std::vector<csint> expect_q = inv_permute(q);

        auto [p_inv, q_inv, p_diags] = find_tri_permutation(PLQ);

        CHECK(p_inv == expect_p);
        CHECK(q_inv == expect_q);
        compare_matrices(L, PLQ.permute(inv_permute(p_inv), q_inv));
        compare_matrices(PLQ, L.permute(p_inv, inv_permute(q_inv)));
    }

    SECTION("Find permutation vectors of permuted U") {
        std::vector<csint> expect_p = inv_permute(p);
        std::vector<csint> expect_q = inv_permute(q);

        // NOTE returns *reversed* vectors for an upper triangular matrix!!
        auto [p_inv, q_inv, p_diags] = find_tri_permutation(PUQ);
        std::reverse(p_inv.begin(), p_inv.end());
        std::reverse(q_inv.begin(), q_inv.end());

        CHECK(p_inv == expect_p);
        CHECK(q_inv == expect_q);
        compare_matrices(U, PUQ.permute(inv_permute(p_inv), q_inv));
        compare_matrices(PUQ, U.permute(p_inv, inv_permute(q_inv)));
    }

    SECTION("Permuted P L x = b, with unknown P") {
        // Create RHS for Lx = b
        // Set b s.t. x == {1, 2, 3, 4, 5, 6} to see output permutation
        const std::vector<double> b = { 1,  6, 18, 40, 75, 126};
        const std::vector<double> expect = {1, 2, 3, 4, 5, 6};

        // Solve Lx = b
        const std::vector<double> x = lsolve(L, b);
        CHECK_THAT(is_close(x, expect, tol), AllTrue());

        // Solve PLx = b
        const std::vector<double> xp = lsolve_rows(PL, b);

        REQUIRE_THAT(is_close(xp, expect, tol), AllTrue());
    }

    SECTION("Permuted L Q x = b, with unknown Q") {
        // Create RHS for Lx = b
        // Set b s.t. x == {1, 2, 3, 4, 5, 6} to see output permutation
        const std::vector<double> b = { 1,  6, 18, 40, 75, 126};
        const std::vector<double> expect = {1, 2, 3, 4, 5, 6};

        // Solve L Q.T x = b
        const std::vector<double> xp = lsolve_cols(LQ, b);

        REQUIRE_THAT(is_close(xp, expect, tol), AllTrue());
    }

    SECTION("Permuted P U x = b, with unknown P") {
        // Create RHS for Ux = b
        // Set b s.t. x == {1, 2, 3, 4, 5, 6} to see output permutation
        const std::vector<double> b = {21, 40, 54, 60, 55, 36};
        const std::vector<double> expect = {1, 2, 3, 4, 5, 6};

        // Solve Ux = b (un-permuted)
        const std::vector<double> x = usolve(U, b);
        CHECK_THAT(is_close(x, expect, tol), AllTrue());

        // Solve PUx = b
        const std::vector<double> xp = usolve_rows(PU, b);
        REQUIRE_THAT(is_close(xp, expect, tol), AllTrue());
    }

    SECTION("Permuted U Q x = b, with unknown Q") {
        // Create RHS for Ux = b
        // Set b s.t. x == {1, 2, 3, 4, 5, 6} to see output permutation
        const std::vector<double> b = {21, 40, 54, 60, 55, 36};
        const std::vector<double> expect = {1, 2, 3, 4, 5, 6};

        // Solve U Q.T x = b
        const std::vector<double> xp = usolve_cols(UQ, b);

        REQUIRE_THAT(is_close(xp, expect, tol), AllTrue());
    }

    SECTION("Permuted P L Q x = b, with unknown P and Q") {
        // Create RHS for Lx = b
        // Set b s.t. x == {1, 2, 3, 4, 5, 6} to see output permutation
        const std::vector<double> b = { 1,  6, 18, 40, 75, 126};
        const std::vector<double> expect = {1, 2, 3, 4, 5, 6};

        // Solve P L Q x = b
        const std::vector<double> xt = tri_solve_perm(PLQ, b);
        REQUIRE_THAT(is_close(xt, expect, tol), AllTrue());
    }

    SECTION("Permuted P U Q x = b, with unknown P and Q") {
        // Create RHS for Ux = b
        // Set b s.t. x == {1, 2, 3, 4, 5, 6} to see output permutation
        const std::vector<double> b = {21, 40, 54, 60, 55, 36};
        const std::vector<double> expect = {1, 2, 3, 4, 5, 6};

        // Solve P U Q x = b
        std::vector<double> xp = tri_solve_perm(PUQ, b, true);
        REQUIRE_THAT(is_close(xp, expect, tol), AllTrue());
    }
}


TEST_CASE("Cholesky decomposition")
{
    // Define the test matrix A (See Davis, Figure 4.2, p 39)
    csint N = 11;
    std::vector<csint> rows = {5, 6, 2, 7, 9, 10, 5, 9, 7, 10, 8, 9, 10, 9, 10, 10};
    std::vector<csint> cols = {0, 0, 1, 1, 2,  2, 3, 3, 4,  4, 5, 5,  6, 7,  7,  9};

    csint N_offdiag = rows.size();

    // Include diagonals
    std::vector<csint> diags(N);
    std::iota(diags.begin(), diags.end(), 0);

    rows.insert(rows.end(), diags.begin(), diags.end());
    cols.insert(cols.end(), diags.begin(), diags.end());

    std::vector<double> vals(rows.size(), 1);

    // Make A positive definite by increasing the diagonal
    for (csint i = 0; i < N; i++) {
        vals[i + N_offdiag] = 10.0 + i;
    }

    CSCMatrix L = COOMatrix(vals, rows, cols).tocsc();
    CSCMatrix A = L + L.T().band(1, N);

    CHECK(A.is_symmetric());
    // CHECK(A.has_canonical_format());

    SECTION("Elimination Tree") {
        std::vector<csint> expect_A = {5, 2, 7, 5, 7, 6, 8, 9, 9, 10, -1};
        std::vector<csint> expect_ATA = {3, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1};
        REQUIRE(etree(A) == expect_A);
        REQUIRE(etree(A, true) == expect_ATA);
        REQUIRE(etree(A.T() * A) == etree(A, true));
    }

    SECTION("Reachability of Elimination Tree") {
        // Map defines the row subtrees.
        // See Davis Figure 4.4, p 40.
        std::map<csint, std::vector<csint>> expect_map = {
            {0, {}},
            {1, {}},
            {2, {1}},
            {3, {}},
            {4, {}},
            {5, {3, 0}},
            {6, {0, 5}},
            {7, {4, 1, 2}},
            {8, {5, 6}},
            {9, {3, 5, 6, 8, 2, 7}},
            {10, {6, 8, 4, 2, 7, 9}},
        };

        std::vector<csint> parent = etree(A);

        for (const auto& [key, expect] : expect_map) {
            std::vector<csint> xi = ereach(A, key, parent);
            // REQUIRE_THAT(xi, UnorderedEquals(expect));
            REQUIRE(xi == expect);  // in exact order
        }
    }

    SECTION("Post-order of Elimination Tree") {
        std::vector<csint> parent = etree(A);
        std::vector<csint> expect = {1, 2, 4, 7, 0, 3, 5, 6, 8, 9, 10};
        std::vector<csint> postorder = post(parent);
        REQUIRE(postorder == expect);
    }

    SECTION("Reachability of Post-ordered Elimination Tree") {
        // Map defines the post-ordered row subtrees.
        // See Davis Figure 4.8, p 49.
        std::map<csint, std::vector<csint>> expect_map = {
            {0, {}},
            {1, {0}},
            {2, {}},
            {3, {0, 1, 2}},
            {4, {}},
            {5, {}},
            {6, {4, 5}},
            {7, {4, 6}},
            {8, {6, 7}},
            {9, {1, 3, 5, 6, 7, 8}},
            {10, {1, 2, 3, 7, 8, 9}},
        };

        // Post-order A and recompute the elimination tree
        std::vector<csint> parent = etree(A);
        std::vector<csint> p = post(parent);

        // NOTE that we cannot just permute parent, as the post-ordering
        // is not a permutation of the elimination tree.
        A = A.permute(inv_permute(p), p).to_canonical();
        parent = etree(A);

        for (const auto& [key, expect] : expect_map) {
            std::vector<csint> xi = ereach_post(A, key, parent);
            CHECK(xi == expect);
        }
    }

    SECTION("First descendants and levels") {
        std::vector<csint> expect_firsts = {4, 0, 0, 5, 2, 4, 4, 0, 4, 0, 0};
        std::vector<csint> expect_levels = {5, 4, 3, 5, 3, 4, 3, 2, 2, 1, 0};
        std::vector<csint> parent = etree(A);
        auto [firsts, levels] = firstdesc(parent, post(parent));
        REQUIRE(firsts == expect_firsts);
        REQUIRE(levels == expect_levels);
    }

    SECTION("Rowcounts of L") {
        std::vector<csint> expect = {1, 1, 2, 1, 1, 3, 3, 4, 3, 7, 7};
        REQUIRE(chol_rowcounts(A) == expect);
    }

    SECTION("Column counts of L") {
        std::vector<csint> expect = {3, 3, 4, 3, 3, 4, 4, 3, 3, 2, 1};
        REQUIRE(chol_colcounts(A) == expect);
    }

    SECTION("Column counts of L from A^T A") {
        std::vector<csint> expect = {7, 6, 8, 8, 7, 6, 5, 4, 3, 2, 1};
        REQUIRE(chol_colcounts(A.T() * A) == expect);
        REQUIRE(chol_colcounts(A, true) == expect);
    }

    SECTION("Symbolic factorization") {
        SymbolicChol S = schol(A, AMDOrder::Natural);

        std::vector<csint> expect_p_inv(A.shape()[1]);
        std::iota(expect_p_inv.begin(), expect_p_inv.end(), 0);

        auto c = chol_colcounts(A);
        csint expect_nnz = std::accumulate(c.begin(), c.end(), 0);

        REQUIRE(S.p_inv == expect_p_inv);
        REQUIRE(S.parent == etree(A));
        REQUIRE(S.cp == cumsum(chol_colcounts(A)));
        REQUIRE(S.cp.back() == expect_nnz);
        REQUIRE(S.lnz == expect_nnz);
    }

    SECTION("Numeric factorization of non-positive definite matrix") {
        // Decrease the diagonal to make A non-positive definite
        for (csint i = 0; i < N; i++) {
            A.assign(i, i, 1.0);
        }
        SymbolicChol S = schol(A, AMDOrder::Natural);
        CHECK_THROWS(chol(A, S));  // A is not positive definite
    }

    SECTION("Numeric factorization") {
        SymbolicChol S = schol(A, AMDOrder::Natural);

        // should be no permutation with AMDOrder::Natural
        std::vector<csint> expect_p_inv(A.shape()[1]);
        std::iota(expect_p_inv.begin(), expect_p_inv.end(), 0);
        CHECK(S.p_inv == expect_p_inv);

        // Now compute the numeric factorization
        CSCMatrix L = chol(A, S);

        // Check that the factorization is correct
        CSCMatrix LLT = (L * L.T()).droptol().to_canonical();

        compare_matrices(LLT, A);

        SECTION("Update Cholesky") {
            // Create a random vector with the sparsity of a column of L
            csint k = 3;  // arbitrary column index
            std::default_random_engine rng(56);
            std::uniform_real_distribution<double> unif(0.0, 1.0);

            COOMatrix w({L.shape()[0], 1});

            for (csint p = L.indptr()[k]; p < L.indptr()[k + 1]; p++) {
                w.assign(L.indices()[p], 0, unif(rng));
            }

            CSCMatrix W = w.tocsc();  // for arithmetic operations

            // Update the input matrix for testing
            CSCMatrix A_up = (A + W * W.T()).to_canonical();

            // Update the factorization in-place
            CSCMatrix L_up = chol_update(L.to_canonical(), 1, W, S.parent);

            CSCMatrix LLT_up = (L_up * L_up.T()).droptol().to_canonical();
            CHECK(LLT_up.nnz() == A_up.nnz());

            compare_matrices(LLT_up, A_up);
        }
    }

    SECTION("Exercise 4.9: Use post-ordering with natural ordering") {
        // Compute the symbolic factorization with postordering
        bool use_postorder = true;
        SymbolicChol S = schol(A, AMDOrder::Natural, use_postorder);
        CSCMatrix L = chol(A, S);
        CSCMatrix LLT = (L * L.T()).droptol().to_canonical();

        // The factorization will be postordered!
        CSCMatrix expect_A = A.permute(S.p_inv, inv_permute(S.p_inv));

        compare_matrices(LLT, expect_A);
    }

    SECTION("Exercise 4.1: etree and counts from ereach") {
        std::vector<csint> expect_parent = etree(A);
        std::vector<csint> expect_rowcounts = chol_rowcounts(A);
        std::vector<csint> expect_colcounts = chol_colcounts(A);

        auto [parent, rowcounts, colcounts] = chol_etree_counts(A);

        CHECK(parent == expect_parent);
        CHECK(rowcounts == expect_rowcounts);
        REQUIRE(colcounts == expect_colcounts);
    }

    SECTION("Exercise 4.3: Solve Lx = b") {
        // Compute the numeric factorization
        SymbolicChol S = schol(A);
        CSCMatrix L = chol(A, S);

        // TODO zero-out a few rows of expect to make it "sparse"
        // Create RHS for Lx = b
        std::vector<double> expect(N);
        std::iota(expect.begin(), expect.end(), 1);
        const std::vector<double> b_vals = L * expect;

        // Create the sparse RHS matrix
        CSCMatrix b(b_vals, {N, 1});

        // Solve Lx = b
        auto [xi, x] = chol_lsolve(L, b, S.parent);

        CHECK_THAT(is_close(x, expect, tol), AllTrue());

        // Solve Lx = b, inferring parent from L
        auto [xi_s, x_s] = chol_lsolve(L, b);

        CHECK(xi == xi_s);
        REQUIRE_THAT(is_close(x_s, expect, tol), AllTrue());
    }

    SECTION("Exercise 4.4: Solve L^T x = b") {
        // Compute the numeric factorization
        SymbolicChol S = schol(A);
        CSCMatrix L = chol(A, S);

        // Create RHS for Lx = b
        std::vector<double> expect(N);
        std::iota(expect.begin(), expect.end(), 1);
        const std::vector<double> b_vals = L.T() * expect;

        // Create the sparse RHS matrix
        CSCMatrix b(b_vals, {N, 1});

        // Solve Lx = b
        auto [xi, x] = chol_ltsolve(L, b, S.parent);

        CHECK_THAT(is_close(x, expect, tol), AllTrue());

        // Solve Lx = b, inferring parent from L
        auto [xi_s, x_s] = chol_ltsolve(L, b);

        CHECK(xi == xi_s);
        REQUIRE_THAT(is_close(x_s, expect, tol), AllTrue());
    }

    SECTION("Exercise 4.6: etree height") {
        std::vector<csint> parent = etree(A);
        REQUIRE(etree_height(parent) == 6);
    }

    SECTION("Exercise 4.10: Symbolic Cholesky") {
        SymbolicChol S = schol(A);
        CSCMatrix L = chol(A, S);  // numeric factorization
        CSCMatrix Ls = symbolic_cholesky(A, S);

        CHECK(Ls.nnz() == L.nnz());
        CHECK(Ls.shape() == L.shape());
        CHECK(Ls.indptr() == L.indptr());
        CHECK(Ls.indices() == L.indices());
        CHECK(Ls.data().size() == L.data().size());  // allocation only
    }

    SECTION("Exercise 4.11: Left-looking Cholesky") {
        SymbolicChol S = schol(A, AMDOrder::Natural);
        CSCMatrix L = symbolic_cholesky(A, S);

        // Compute the numeric factorization using the non-zero pattern
        L = leftchol(A, S, L);

        CSCMatrix LLT = (L * L.T()).droptol().to_canonical();

        compare_matrices(LLT, A);
    }

    SECTION("Exercise 4.12: Up-looking Cholesky with Pattern") {
        SymbolicChol S = schol(A, AMDOrder::Natural);
        CSCMatrix L = symbolic_cholesky(A, S);

        // Compute the numeric factorization using the non-zero pattern
        L = rechol(A, S, L);

        CSCMatrix LLT = (L * L.T()).droptol().to_canonical();

        compare_matrices(LLT, A);
    }

    SECTION("Exercise 4.13: Incomplete Cholesky") {
        // Compute the incomplete Cholesky factorization with no fill-in
        double drop_tol = 1e-2;
        CSCMatrix Li = ichol(A, ICholMethod::ICT, drop_tol);

        // Compute the complete Cholesky factorization for comparison
        CSCMatrix L = chol(A, schol(A));

        CSCMatrix LLT = (Li * Li.T()).droptol().to_canonical();

        // std::cout << "Li:" << std::endl;
        // Li.print_dense();
        // std::cout << "LLT:" << std::endl;
        // LLT.print_dense();
        // std::cout << "A:" << std::endl;
        // A.print_dense();

        CHECK(Li.nnz() <= L.nnz());
        CHECK(LLT.nnz() >= A.nnz());
        // NOTE the ICT method uses the column counts from `schol`, so any
        // entries that are dropped will be exactly 0 in L.
        // REQUIRE_THAT(Li.data() >= drop_tol, AllTrue());

        REQUIRE((LLT - A).fronorm() / A.fronorm() < drop_tol);
    }
}


TEST_CASE("Householder Reflection")
{
    SECTION("Unit x") {
        std::vector<double> x = {1, 0, 0};

        std::vector<double> expect_v = {1, 0, 0};
        double expect_beta = 0.0;
        double expect_s = 1.0;

        Householder H = house(x);

        CHECK_THAT(is_close(H.v, expect_v, tol), AllTrue());
        CHECK_THAT(H.beta, WithinAbs(expect_beta, tol));
        CHECK_THAT(H.s, WithinAbs(expect_s, tol));

        // Apply the reflection
        CSCMatrix V = COOMatrix(H.v, {0, 1, 2}, {0, 0, 0}).tocsc();
        std::vector<double> Hx = happly(V, 0, H.beta, x);

        REQUIRE_THAT(is_close(Hx, x, tol), AllTrue());
    }

    SECTION("Negative unit x") {
        std::vector<double> x = {-1, 0, 0};

        std::vector<double> expect_v = {1, 0, 0};
        double expect_beta = 0.0;
        double expect_s = -1.0;

        Householder H = house(x);

        CHECK_THAT(is_close(H.v, expect_v, tol), AllTrue());
        CHECK_THAT(H.beta, WithinAbs(expect_beta, tol));
        CHECK_THAT(H.s, WithinAbs(expect_s, tol));

        // Apply the reflection
        CSCMatrix V = COOMatrix(H.v, {0, 1, 2}, {0, 0, 0}).tocsc();
        std::vector<double> Hx = happly(V, 0, H.beta, x);

        REQUIRE_THAT(is_close(Hx, x, tol), AllTrue());
    }

    SECTION("Arbitrary x, x[0] > 0") {
        std::vector<double> x = {3, 4};  // norm(x) == 5

        // These are the *unscaled* values from Octave
        // std::vector<double> expect_v = {8, 4};
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

        CHECK_THAT(is_close(H.v, expect_v, tol), AllTrue());
        CHECK_THAT(H.beta, WithinAbs(expect_beta, tol));
        CHECK_THAT(H.s, WithinAbs(expect_s, tol));

        // Apply the vector
        // Hx = [±norm(x), 0, 0]
        std::vector<double> expect = {-5, 0}; // LAPACK
        // std::vector<double> expect = {5, 0};  // Davis

        // Use column 0 of V to apply the Householder reflection
        CSCMatrix V = COOMatrix(H.v, {0, 1}, {0, 0}).tocsc();
        std::vector<double> Hx = happly(V, 0, H.beta, x);

        REQUIRE_THAT(is_close(Hx, expect, tol), AllTrue());
    }

    SECTION("Arbitrary x, x[0] < 0") {
        std::vector<double> x = {-3, 4};  // norm(x) == 5

        // These are the values from python's scipy.linalg.qr (via LAPACK):
        std::vector<double> expect_v {1, -0.5};
        double expect_beta = 1.6;
        double expect_s = 5;

        Householder H = house(x);

        CHECK_THAT(is_close(H.v, expect_v, tol), AllTrue());
        CHECK_THAT(H.beta, WithinAbs(expect_beta, tol));
        CHECK_THAT(H.s, WithinAbs(expect_s, tol));

        // Apply the vector
        std::vector<double> expect = {5, 0}; // LAPACK or Davis

        CSCMatrix V = COOMatrix(H.v, {0, 1}, {0, 0}).tocsc();
        std::vector<double> Hx = happly(V, 0, H.beta, x);

        REQUIRE_THAT(is_close(Hx, expect, tol), AllTrue());
    }
}


TEST_CASE("QR factorization of the Identity Matrix")
{
    csint N = 8;
    std::vector<csint> rows(N);
    std::iota(rows.begin(), rows.end(), 0);
    std::vector<double> vals(N, 1.0);

    CSCMatrix I = COOMatrix(vals, rows, rows).tocsc();

    SymbolicQR S = sqr(I);

    SECTION("Symbolic factorization") {
        std::vector<csint> expect_identity = {0, 1, 2, 3, 4, 5, 6, 7};
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

        compare_matrices(res.V, I);
        CHECK_THAT(is_close(res.beta, expect_beta, tol), AllTrue());
        compare_matrices(res.R, I);
    }
}


TEST_CASE("QR Decomposition of Square, Non-symmetric A")
{
    csint N = 8;  // number of rows and columns

    // Define the test matrix A (See Davis, Figure 5.1, p 74)
    std::vector<csint> rows = {0, 1, 2, 3, 4, 5, 6,
                               3, 6, 1, 6, 0, 2, 5, 7, 4, 7, 0, 1, 3, 7, 5, 6};
    std::vector<csint> cols = {0, 1, 2, 3, 4, 5, 6,
                               0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7};

    // Label the diagonal elements 1..7, skipping the 8th
    std::vector<double> vals(rows.size(), 1.0);
    std::iota(vals.begin(), vals.begin() + 7, 1.0);

    // Non-canonical format for testing
    CSCMatrix A = COOMatrix(vals, rows, cols).compress();

    // NOTE we need to include A[7, 7] even though it is numerically zero!
    A.assign(7, 7, 0.0);

    // See etree in Figure 5.1, p 74
    std::vector<csint> parent = {3, 2, 3, 6, 5, 6, 7, -1};

    std::vector<csint> expect_leftmost = {0, 1, 2, 0, 4, 4, 1, 4};
    std::vector<csint> expect_p_inv = {0, 1, 3, 7, 4, 5, 2, 6};  // cs_qr MATLAB

    SECTION("find_leftmost") {
        CHECK(find_leftmost(A) == expect_leftmost);
    }

    SECTION("vcount") {
        SymbolicQR S;
        S.parent.assign(parent.begin(), parent.end());
        S.leftmost = find_leftmost(A);
        vcount(A, S);

        CHECK(S.p_inv == expect_p_inv);
        CHECK(S.vnz == 16);
        CHECK(S.m2 == N);
    }

    SECTION("Symbolic factorization") {
        std::vector<csint> expect_q = {0, 1, 2, 3, 4, 5, 6, 7};  // natural
        std::vector<csint> expect_parent = parent;

        SymbolicQR S = sqr(A);

        CHECK(S.p_inv == expect_p_inv);
        CHECK(S.q == expect_q);
        CHECK(S.parent == expect_parent);
        CHECK(S.leftmost == expect_leftmost);
        CHECK(S.m2 == N);
        CHECK(S.vnz == 16);  // manual counts Figure 5.1, p 74
        CHECK(S.rnz == 24);
    }

    SECTION("Numeric factorization") {
        // Expected values circularly computed from python computation with
        // csparse.qr, where we know Q @ R produces the correct result. This
        // isn't a great test, but at least lets us know if we break anything
        // from this point forward.
        CSCMatrix expect_V {
            {1.                , 0.                , 0.                , 0.                , 0.                , 0.                , 0.                , 0.,
             0.                , 1.                , 0.                , 0.                , 0.                , 0.                , 0.                , 0.,
             0.                , 0.2360679774997897, 1.                , 0.                , 0.                , 0.                , 0.                , 0.,
             0.                , 0.                , 0.8619788607068875, 1.                , 0.                , 0.                , 0.                , 0.,
             0.                , 0.                , 0.                , 0.                , 1.                , 0.                , 0.                , 0.,
             0.                , 0.                , 0.                , 0.                , 0.0980762113533159, 1.                , 0.                , 0.,
             0.                , 0.                , 0.                , 0.                , 0.0980762113533159, 0.0592952558196218, 1.                , 0.,
             0.4142135623730951, 0.                , 0.                , 0.9329077440557915, 0.                , 0.                , 0.8441594335316119, 1.
            },
            {8, 8},
            'C'  // row-major order
        };

        std::vector<double> expect_beta {
            1.7071067811865472,
            1.8944271909999157,
            1.1474419561548972,
            1.0693375245281538,
            1.9622504486493761,
            1.992992782143569 ,
            1.1678115068791026,
            0.
        };

        CSCMatrix expect_R {
            {-1.4142135623730951,  0.                ,  0.                , -3.5355339059327378,  0.                ,  0.                , -1.414213562373095 ,  0.                ,
              0.                , -2.23606797749979  , -1.341640786499874 ,  0.                ,  0.                ,  0.                , -4.024922359499621 , -0.4472135954999579,
              0.                ,  0.                , -3.03315017762062  , -0.9890707100936806,  0.                ,  0.                , -0.8571946154145236, -0.1318760946791574,
              0.                ,  0.                ,  0.                , -2.1264381322847794,  0.                ,  0.                ,  0.3987071498033972,  0.061339561508215 ,
              0.                ,  0.                ,  0.                ,  0.                , -5.196152422706632 , -2.309401076758503 , -0.1924500897298752, -0.1924500897298752,
              0.                ,  0.                ,  0.                ,  0.                ,  0.                , -5.715476066494082 , -0.0972019739199674, -0.9720197391996737,
              0.                ,  0.                ,  0.                ,  0.                ,  0.                ,  0.                , -5.818914395248401 , -0.8474056139492476,
              0.                ,  0.                ,  0.                ,  0.                ,  0.                ,  0.                ,  0.                ,  0.2808744717175516
            },
            {8, 8},
            'C' // row-major order
        };

        // ---------- Factor the matrix
        SymbolicQR S = sqr(A);
        QRResult res = qr(A, S);

        compare_matrices(res.V, expect_V);
        CHECK_THAT(is_close(res.beta, expect_beta, tol), AllTrue());
        compare_matrices(res.R, expect_R);
    }
}


// TODO test QR factorization of a non-square matrix M > N, and M < N

/*==============================================================================
 *============================================================================*/
