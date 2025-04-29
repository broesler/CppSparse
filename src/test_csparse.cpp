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
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>  // reverse
#include <cmath>
#include <iomanip>  // setprecision, scientific, etc.
#include <iostream>
#include <fstream>
#include <map>
#include <new>         // bad_alloc
#include <numeric>    // iota
#include <optional>   // nullopt
#include <random>
#include <string>
#include <sstream>
#include <vector>
#include <utility>  // as_const

#include "csparse.h"

using namespace cs;

using Catch::Approx;
using Catch::Matchers::AllTrue;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::UnorderedEquals;
using Catch::Matchers::Equals;

constexpr double tol = 1e-14;


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
 *         Utilities
 *----------------------------------------------------------------------------*/
TEST_CASE("Vector ops", "[vector]")
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


TEST_CASE("Cumsum", "[vector]")
{
    std::vector<csint> a = {1, 1, 1, 1};
    std::vector<csint> c = cumsum(a);
    std::vector<csint> expect = {0, 1, 2, 3, 4};
    REQUIRE(c == expect);
    REQUIRE(&a != &c);
}


TEST_CASE("Vector permutations", "[vector]")
{
    std::vector<double> b = {0, 1, 2, 3, 4};
    std::vector<csint> p = {2, 0, 1, 4, 3};

    REQUIRE(pvec(p, b) == std::vector<double>{2, 0, 1, 4, 3});
    REQUIRE(ipvec(p, b) == std::vector<double>{1, 2, 0, 4, 3});
    REQUIRE(inv_permute(p) == std::vector<csint>{1, 2, 0, 4, 3});
    REQUIRE(pvec(inv_permute(p), b) == ipvec(p, b));
    REQUIRE(ipvec(inv_permute(p), b) == pvec(p, b));
}


TEST_CASE("Random permutation", "[randperm]")
{
    csint N = 10;
    csint seed = 0;
    std::vector<csint> expect_p = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    SECTION("Identity permutation") {
        seed = 0;
    }

    SECTION("Reverse permutation") {
        seed = -1;
        std::reverse(expect_p.begin(), expect_p.end());
    }

    SECTION("Arbitrary permutation") {
        seed = 565656;
        std::default_random_engine rng(seed);
        std::shuffle(expect_p.begin(), expect_p.end(), rng);
    }

    std::vector<csint> p = randperm(N, seed);
    std::vector<csint> p_inv = inv_permute(p);

    REQUIRE(p == expect_p);

    // Check that the inverse permutation is correct
    for (csint i = 0; i < N; i++) {
        REQUIRE(p[p_inv[i]] == i);
        REQUIRE(p_inv[p[i]] == i);
    }
}


TEST_CASE("Argsort", "[vector]")
{
    SECTION("Vector of doubles") {
        std::vector<double> v = {5.6, 6.9, 42.0, 1.7, 9.0};
        REQUIRE(argsort(v) == std::vector<csint> {3, 0, 1, 4, 2});
    }

    SECTION("Vector of ints") {
        std::vector<int> v = {5, 6, 42, 1, 9};
        REQUIRE(argsort(v) == std::vector<csint> {3, 0, 1, 4, 2});
    }
}


TEST_CASE("Vector norms", "[vector]")
{
    std::vector<double> v = {3, 4};

    SECTION("L0 norm") {
        REQUIRE(norm(v, 0) == 2);
    }

    SECTION("L1 norm") {
        REQUIRE(norm(v, 1) == 7);
    }

    SECTION("L2 norm") {
        REQUIRE(norm(v, 2) == Approx(5.0));
    }

    SECTION("LPI norm") {
        double pi = 4 * atan(1.0);  // pi = 3.14159...
        REQUIRE(norm(v, pi) == Approx(4.457284396597481));
    }

    SECTION("Linf norm") {
        REQUIRE(norm(v, INFINITY) == 4);
    }
}


/*------------------------------------------------------------------------------
 *         Matrix Functions
 *----------------------------------------------------------------------------*/
TEST_CASE("COOMatrix Constructors", "[COOMatrix]")
{
    SECTION("Empty constructor") {
        COOMatrix A;

        REQUIRE(A.nnz() == 0);
        REQUIRE(A.nzmax() == 0);
        REQUIRE(A.shape() == Shape{0, 0});
    }

    SECTION("Make new from given shape") {
        COOMatrix A {{56, 37}};
        REQUIRE(A.nnz() == 0);
        REQUIRE(A.nzmax() == 0);
        REQUIRE(A.shape() == Shape{56, 37});
    }

    SECTION("Allocate new from shape and nzmax") {
        int nzmax = 1e4;
        COOMatrix A {{56, 37}, nzmax};
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
    COOMatrix A {v, i, j};

    SECTION("Attributes") {
        REQUIRE(A.nnz() == 10);
        REQUIRE(A.nzmax() >= 10);
        REQUIRE(A.shape() == Shape{4, 4});
        REQUIRE(A.row() == i);
        REQUIRE(A.column() == j);
        REQUIRE(A.data() == v);
    }

    SECTION("Printing") {
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

    SECTION("Exercise 2.5: Assign a dense submatrix") {
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
        COOMatrix F {fp};

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


TEST_CASE("CSCMatrix", "[CSCMatrix]")
{
    COOMatrix A = davis_example_small();
    CSCMatrix C = A.compress();  // unsorted columns

    // std::cout << "C = \n" << C;
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
        davis_example_small()        // unsorted matrix
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


TEST_CASE("Exercise 2.13: Is_symmetric.") {
    std::vector<csint>  i = {0, 1, 2};
    std::vector<csint>  j = {0, 1, 2};
    std::vector<double> v = {1, 2, 3};

    SECTION("Diagonal matrix") {
        CSCMatrix A = COOMatrix(v, i, j).tocsc();
        REQUIRE(A.is_symmetric());
    }

    SECTION("Non-symmetric matrix with off-diagonals") {
        CSCMatrix A = COOMatrix(v, i, j)
                       .assign(0, 1, 1.0)
                       .tocsc();
        REQUIRE_FALSE(A.is_symmetric());
    }

    SECTION("Symmetric matrix with off-diagonals") {
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
        REQUIRE_THAT(is_close(sym_gaxpy(A, x, y),  expect_Axpy, tol), AllTrue());
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
        REQUIRE_THAT(is_close(Ac.dot(x), expect_Ax, tol), AllTrue());
        REQUIRE_THAT(is_close((Ac * x),  expect_Ax, tol), AllTrue());
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

        REQUIRE_THAT(is_close(sym_gaxpy(A, x, y), expect_Axpy, tol), AllTrue());
    }
}


TEST_CASE("Exercise 2.27: Matrix-(dense) matrix multiply + addition.")
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

        compare_matrices(CSCMatrix(gaxpy_col(A, I, Z), {4, 4}), expect);
        compare_matrices(CSCMatrix(gatxpy_col(A.T(), I, Z), {4, 4}), expect);
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

        REQUIRE_THAT(is_close(C_col, expect, tol), AllTrue());
        REQUIRE_THAT(is_close(C_block, expect, tol), AllTrue());
        REQUIRE_THAT(is_close(CT_col, expect, tol), AllTrue());
        REQUIRE_THAT(is_close(CT_block, expect, tol), AllTrue());
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

        REQUIRE_THAT(is_close(C, expect, tol), AllTrue());
        REQUIRE_THAT(is_close(CT, expect, tol), AllTrue());
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

        REQUIRE_THAT(is_close(gaxpy_col(Ab, Ac_dense, A_dense), expect, tol),
                     AllTrue());
        REQUIRE_THAT(is_close(gaxpy_block(Ab, Ac_dense, A_dense), expect, tol),
                     AllTrue());
        REQUIRE_THAT(is_close(gatxpy_col(Ab.T(), Ac_dense, A_dense), expect, tol),
                     AllTrue());
        REQUIRE_THAT(is_close(gatxpy_block(Ab.T(), Ac_dense, A_dense), expect, tol),
                     AllTrue());
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

        REQUIRE_THAT(is_close(gaxpy_row(Ab, Ac_dense, A_dense), expect, tol),
                     AllTrue());
        REQUIRE_THAT(is_close(gatxpy_row(Ab.T(), Ac_dense, A_dense), expect, tol),
                     AllTrue());
    }
}


TEST_CASE("Matrix-matrix multiply.", "[math][dot]")
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

            compare_matrices(C, expect);
        }

        SECTION("M > N") {
            CSCMatrix CT = B.T() * A.T();
            auto [N, M] = CT.shape();

            REQUIRE(M == A.shape()[0]);
            REQUIRE(N == B.shape()[1]);

            compare_matrices(CT, expect.T());
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
        compare_matrices(C, expect, false);
    }
}


TEST_CASE("Exercise 2.18: Sparse Vector-Vector Multiply", "[math]")
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

    // Operator overloading
    CSCMatrix C = 0.1 * A;

    for (csint i = 0; i < M; i++) {
        for (csint j = 0; j < N; j++) {
            REQUIRE_THAT(C(i, j), WithinAbs(expect(i, j), tol));
        }
    }
}


TEST_CASE("Exercise 2.4: Scale rows and columns", "[math]")
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


TEST_CASE("Matrix-matrix addition.", "[math][add_scaled]")
{
    SECTION("Non-square matrices") {
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

            compare_matrices(C, expect);
            compare_matrices(Cf, expect);
            compare_matrices(Cm, expect);
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

            compare_matrices(C, expect);
            compare_matrices(Cf, expect);
        }

        SECTION("Symbolic Addition") {
            CSCMatrix As = COOMatrix(std::vector<double> {}, i, j).tocsc();
            CSCMatrix expect = COOMatrix(std::vector<double> {}, i, j).tocsc();

            CSCMatrix Cs = add_scaled(As, B, 1.0, 1.0);

            compare_matrices(Cs, expect, false);  // don't compare values
        }
    }

    SECTION("Sparse column vectors") {
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

    SECTION("Exercise 2.26: Non-permuted transpose") {
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


TEST_CASE("Exercise 2.15: Band function")
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
        CHECK(Ab.column() == expect_cols);
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
        CHECK(Ab.column() == expect_cols);
        REQUIRE(Ab.data() == expect_data);
    }
}


TEST_CASE("Exercise 2.16: CSC from dense")
{
    CSCMatrix expect_A = davis_example_small().tocsc();

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

// Create a dummy class that builds an invalid matrix
namespace cs {

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

}  // namespace cs


// "cs_ok"
TEST_CASE("Exercise 2.12: Validity check")
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
        A = davis_example_small().assign(0, 1, 0.0).compress();

        REQUIRE_THROWS_WITH(A.is_valid(!SORTED), "Explicit zeros!");
        REQUIRE_THROWS_WITH(A.sort().is_valid(), "Explicit zeros!");
    }

    SECTION("Duplicate Entry") {
        A = davis_example_small().assign(1, 1, 1.0).compress();

        // Un-sorted columns will fail before duplicates are checked
        REQUIRE_THROWS_WITH(A.is_valid(), "Columns not sorted!");
        REQUIRE_THROWS_WITH(A.sort().is_valid(), "Duplicate entries exist!");
    }
}


// "hcat" and "vcat"
TEST_CASE("Exercise 2.22: Concatentation")
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
TEST_CASE("Exercise 2.23: Slicing")
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
TEST_CASE("Exercise 2.24: Non-contiguous indexing")
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


// indexing for assignment
TEST_CASE("Exercise 2.25: Indexing for single assignment.")
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

    SECTION("Multiple assignment") {
        CSCMatrix A = davis_example_small().tocsc();

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


TEST_CASE("Exercise 2.29: Adding empty rows and columns to a CSCMatrix.")
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


TEST_CASE("Sum the rows and columns of a matrix")
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

/*------------------------------------------------------------------------------
 *          Matrix Solutions
 *----------------------------------------------------------------------------*/
TEST_CASE("Triangular solve with dense RHS")
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
    CSCMatrix B {Shape {N, 1}};

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

        auto [xi, x] = spsolve(U, B, 0, std::nullopt, false);

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
        const CSCMatrix A = davis_example_small().tocsc();
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


/*------------------------------------------------------------------------------
 *         Chapter 4: Cholesky Factorization
 *----------------------------------------------------------------------------*/
TEST_CASE("Cholesky Factorization")
{
    // Define the test matrix A (See Davis, Figure 4.2, p 39)
    CSCMatrix A = davis_example_chol();
    csint N = A.shape()[1];

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

    SECTION("Symbolic analysis") {
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
        // CSparse only uses AMDOrder::Natural and AMDOrder::APlusAT for
        // Cholesky factorization in cs_chol.m (see Test/test3.m)
        AMDOrder order = AMDOrder::Natural;

        // should be no permutation with AMDOrder::Natural
        std::vector<csint> expect_p(N);
        std::iota(expect_p.begin(), expect_p.end(), 0);

        SECTION("Natural") {}

        SECTION("APlusAT ordering") {
            order = AMDOrder::APlusAT;
            // MATLAB [L, p] = cs_chol(A) -> order = 1
            expect_p = {1, 4, 6, 8, 0, 3, 5, 2, 9, 10, 7};
        }

        SymbolicChol S = schol(A, order);

        std::vector<csint> expect_p_inv = inv_permute(expect_p);
        CHECK(S.p_inv == expect_p_inv);

        // Now compute the numeric factorization
        CSCMatrix L = chol(A, S);

        CHECK(L.has_sorted_indices());
        CHECK(L._test_sorted());

        // Check that the factorization is correct
        CSCMatrix LLT = (L * L.T()).droptol().to_canonical();

        // Permute the input matrix to match LL^T = P^T A P
        CSCMatrix PAPT = A.permute(S.p_inv, inv_permute(S.p_inv)).to_canonical();

        compare_matrices(LLT, PAPT);
    }

    SECTION("Update Cholesky") {
        SymbolicChol S = schol(A, AMDOrder::Natural);
        CSCMatrix L = chol(A, S);

        // Create a random vector with the sparsity of a column of L
        csint k = 3;  // arbitrary column index
        std::default_random_engine rng(56);
        std::uniform_real_distribution<double> unif(0.0, 1.0);

        COOMatrix w {{L.shape()[0], 1}};

        for (csint p = L.indptr()[k]; p < L.indptr()[k + 1]; p++) {
            w.assign(L.indices()[p], 0, unif(rng));
        }

        CSCMatrix W = w.tocsc();  // for arithmetic operations

        // Update the input matrix for testing
        CSCMatrix A_up = (A + W * W.T()).to_canonical();

        // Update the factorization in-place
        CSCMatrix L_up = chol_update(L.to_canonical(), true, W, S.parent);

        CSCMatrix LLT_up = (L_up * L_up.T()).droptol().to_canonical();
        CHECK(LLT_up.nnz() == A_up.nnz());

        compare_matrices(LLT_up, A_up);
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
        AMDOrder order;

        SECTION("Natural") {
            order = AMDOrder::Natural;
        }

        SECTION("APlusAT") {
            order = AMDOrder::APlusAT;
        }

        // Compute the numeric factorization
        SymbolicChol S = schol(A, order);
        CSCMatrix L = chol(A, S);

        // Create RHS for Lx = b
        std::vector<double> expect(N);
        std::iota(expect.begin(), expect.end(), 1);
        // zero-out a few rows of expect to make it "sparse"
        for (const auto& i : {2, 4, 7, 9}) {
            expect[i] = 0;
        }

        const std::vector<double> b_vals = L * expect;

        // Create the sparse RHS matrix
        CSCMatrix b {b_vals, {N, 1}};

        // Solve Lx = b
        auto [xi, x] = chol_lsolve(L, b, S.parent);

        CHECK_THAT(is_close(x, expect, tol), AllTrue());

        // Solve Lx = b, inferring parent from L
        auto [xi_s, x_s] = chol_lsolve(L, b);

        CHECK(xi == xi_s);
        REQUIRE_THAT(is_close(x_s, expect, tol), AllTrue());
    }

    SECTION("Exercise 4.4: Solve L^T x = b") {
        AMDOrder order;

        SECTION("Natural") {
            order = AMDOrder::Natural;
        }

        SECTION("APlusAT") {
            order = AMDOrder::APlusAT;
        }

        // Compute the numeric factorization
        SymbolicChol S = schol(A, order);
        CSCMatrix L = chol(A, S);

        // Create RHS for Lx = b
        std::vector<double> expect(N);
        std::iota(expect.begin(), expect.end(), 1);
        // zero-out a few rows of expect to make it "sparse"
        for (const auto& i : {2, 4, 7, 9}) {
            expect[i] = 0;
        }

        const std::vector<double> b_vals = L.T() * expect;

        // Create the sparse RHS matrix
        CSCMatrix b {b_vals, {N, 1}};

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

    SECTION("Exercise 4.9: Use post-ordering with natural ordering") {
        // Compute the symbolic factorization with postordering
        bool use_postorder = true;

        AMDOrder order = AMDOrder::Natural;

        SECTION("Natural") {}

        SECTION("APlusAT") {
            order = AMDOrder::APlusAT;
        }

        SymbolicChol S = schol(A, order, use_postorder);
        CSCMatrix L = chol(A, S);
        CSCMatrix LLT = (L * L.T()).droptol().to_canonical();

        // The factorization will be postordered!
        CSCMatrix PAPT = A.permute(S.p_inv, inv_permute(S.p_inv)).to_canonical();

        compare_matrices(LLT, PAPT);
    }

    SECTION("Exercise 4.10: Symbolic Cholesky") {
        AMDOrder order;

        SECTION("Natural") {
            order = AMDOrder::Natural;
        }

        SECTION("APlusAT") {
            order = AMDOrder::APlusAT;
        }

        SymbolicChol S = schol(A, order);
        CSCMatrix L = chol(A, S);  // numeric factorization
        CSCMatrix Ls = symbolic_cholesky(A, S);

        CHECK(Ls.nnz() == L.nnz());
        CHECK(Ls.shape() == L.shape());
        CHECK(Ls.indptr() == L.indptr());
        CHECK(Ls.indices() == L.indices());
        CHECK(Ls.data().size() == L.data().size());  // allocation only
    }

    SECTION("Exercise 4.11: Left-looking Cholesky") {
        AMDOrder order;

        SECTION("Natural") {
            order = AMDOrder::Natural;
        }

        SECTION("APlusAT") {
            order = AMDOrder::APlusAT;
        }

        SymbolicChol S = schol(A, order);
        CSCMatrix L = symbolic_cholesky(A, S);

        // Compute the numeric factorization using the non-zero pattern
        L = leftchol(A, S, L);

        CSCMatrix LLT = (L * L.T()).droptol().to_canonical();
        CSCMatrix PAPT = A.permute(S.p_inv, inv_permute(S.p_inv)).to_canonical();

        compare_matrices(LLT, PAPT);
    }

    SECTION("Exercise 4.12: Up-looking Cholesky with Pattern") {
        AMDOrder order;

        SECTION("Natural") {
            order = AMDOrder::Natural;
        }

        SECTION("APlusAT") {
            order = AMDOrder::APlusAT;
        }

        SymbolicChol S = schol(A, order);
        CSCMatrix L = symbolic_cholesky(A, S);

        // Compute the numeric factorization using the non-zero pattern
        L = rechol(A, S, L);

        CSCMatrix LLT = (L * L.T()).droptol().to_canonical();
        CSCMatrix PAPT = A.permute(S.p_inv, inv_permute(S.p_inv)).to_canonical();

        compare_matrices(LLT, PAPT);
    }

    SECTION("Exercise 4.13: Incomplete Cholesky") {
        const SymbolicChol S = schol(A, AMDOrder::Natural);

        SECTION("IC0: No Fill") {
            // Compute the incomplete Cholesky factorization with no fill-in.
            const CSCMatrix L = ichol_nofill(A, S);

            // Compute the complete Cholesky factorization for comparison
            const CSCMatrix Lf = chol(A, schol(A));

            // std::cout << "Lf:" << std::endl;
            // Lf.print_dense();
            // std::cout << "L:" << std::endl;
            // L.print_dense();

            // std::cout << "A:" << std::endl;
            // A.print_dense();

            // L is lower triangular with the same sparsity pattern as A
            const CSCMatrix A_tril = std::as_const(A).band(-N, 0);

            csint fill_in = 6;  // shown in book example
            CHECK(L._test_sorted());
            CHECK(L.nnz() == Lf.nnz() - fill_in);  // fill-in is 6
            CHECK(L.nnz() == A_tril.nnz());
            CHECK(L.indptr() == A_tril.indptr());
            CHECK(L.indices() == A_tril.indices());

            // Test norm just on non-zero pattern of A
            // MATLAB >> norm(A - (Lf * Lf') * spones(A), "fro") / norm(A, "fro")

            const CSCMatrix LLT = (L * L.T()).droptol().to_canonical();
            const CSCMatrix LLT_Anz = LLT.fkeep(
                [A](csint i, csint j, double x) {
                    return std::as_const(A)(i, j) != 0.0;
                }
            );
            const CSCMatrix AmLLT = (A - LLT).droptol(tol).to_canonical();

            // std::cout << "A - LLT:" << std::endl;
            // AmLLT.print_dense();

            CHECK(LLT_Anz.nnz() == A.nnz());
            CHECK(AmLLT.is_symmetric());
            CHECK(AmLLT.nnz() == fill_in);

            double nz_norm = (A - LLT_Anz).fronorm() / A.fronorm();
            double norm = AmLLT.fronorm() / A.fronorm();  // total norm

            // std::cout << "nz_norm: " << std::format("{:6.4g}", nz_norm) << std::endl;
            // std::cout << "   norm: " << std::format("{:6.4g}", norm) << std::endl;

            CHECK_THAT(nz_norm, WithinAbs(0.0, 1e-15));
            CHECK(norm > nz_norm * 1e10);  // hack number
        }

        SECTION("ICT: Threshold") {
            SECTION("Full Cholesky (drop_tol = 0)") {
                // Should match full decomposition
                double drop_tol = 0.0;
                const CSCMatrix L = icholt(A, S, drop_tol);
                const CSCMatrix Lf = chol(A, S);
                compare_matrices(L, Lf);
            }

            SECTION("Drop all non-diagonal entries (drop_tol = inf)") {
                double drop_tol = 1.0;
                const CSCMatrix L = icholt(A, S, drop_tol);
                const CSCMatrix LLT = (L * L.T()).droptol().to_canonical();
                CHECK(L.nnz() == N);
                for (csint k = 0; k < N; k++) {
                    CHECK(LLT(k, k) == Approx(A(k, k)));
                }
            }

            SECTION("Arbitrary Tolerance") {
                // Compute the incomplete Cholesky factorization with a threshold
                double drop_tol = 0.005;
                const CSCMatrix L = icholt(A, S, drop_tol);

                // Compute the complete Cholesky factorization for comparison
                const CSCMatrix Lf = chol(A, S);
                const CSCMatrix LLT = (L * L.T()).droptol().to_canonical();

                // std::cout << "Lf:" << std::endl;
                // Lf.print_dense();
                // std::cout << "L:" << std::endl;
                // L.print_dense();

                // std::cout << "LLT:" << std::endl;
                // LLT.print_dense();
                // std::cout << "A:" << std::endl;
                // A.print_dense();

                // std::cout << "A - LLT:" << std::endl;
                // (A - LLT).print_dense();

                csint expect_drops = 6;  // rel_drop_tol = 0.005;

                CHECK(L._test_sorted());
                CHECK(L.nnz() == Lf.nnz() - expect_drops);

                // Only true for absolute drop tolerance
                // for (const auto& x : L.data()) {
                //     CHECK(std::fabs(x) > drop_tol);
                // }

                // Test the norm just on the pattern of A
                const CSCMatrix LLT_Anz = LLT.fkeep(
                    [A](csint i, csint j, double x) {
                        return std::as_const(A)(i, j) != 0.0;
                    }
                );

                const CSCMatrix AmLLTnz = (A - LLT_Anz).droptol(tol).to_canonical();
                const CSCMatrix AmLLT = (A - LLT).droptol(tol).to_canonical();

                CHECK(LLT_Anz.nnz() == A.nnz());
                CHECK(AmLLT.is_symmetric());
                CHECK(AmLLT.nnz() == expect_drops);

                double nz_norm = AmLLTnz.fronorm() / A.fronorm();
                double norm = (A - LLT).fronorm() / A.fronorm();

                // Only really true for realistically small drop_tol
                CHECK_THAT(nz_norm, WithinAbs(0.0, 1e-15));
                CHECK(norm < drop_tol);  // not always true!
            }
        }
    }
}  // cholesky


// -----------------------------------------------------------------------------
//         Chapter 5: QR Factorization
// -----------------------------------------------------------------------------
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

    AMDOrder order = GENERATE(
        AMDOrder::Natural,
        AMDOrder::APlusAT,
        AMDOrder::ATANoDenseRows,
        AMDOrder::ATA
    );

    SymbolicQR S = sqr(I, order);

    SECTION("Symbolic analysis") {
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
    CSCMatrix A = davis_example_qr();

    // See etree in Figure 5.1, p 74
    std::vector<csint> parent = {3, 2, 3, 6, 5, 6, 7, -1};

    std::vector<csint> expect_leftmost = {0, 1, 2, 0, 4, 4, 1, 4};
    std::vector<csint> expect_p_inv = {0, 1, 3, 7, 4, 5, 2, 6};  // cs_qr MATLAB

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
        std::vector<csint> expect_q = {0, 1, 2, 3, 4, 5, 6, 7};  // natural
        std::vector<csint> expect_parent = parent;
        csint expect_Vnz = 16;  // manual counts Figure 5.1, p 74
        csint expect_Rnz = 24;

        AMDOrder order = AMDOrder::Natural;

        SECTION("Natural") {}

        // CSparse only uses 2 possible orders for QR factorization:
        // MATLAB [V, beta, p, R] = cs_qr(A); -> order = 0
        //        [V, beta, p, R, q] = cs_qr(A); -> order = 3
        SECTION("ATA") {
            order = AMDOrder::ATA;
            expect_p_inv = {0, 2, 7, 1, 4, 5, 3, 6};
            expect_q = {0, 3, 1, 2, 4, 5, 7, 6};
            expect_parent = {1, 3, 3, 6, 5, 6, 7, -1};
            expect_leftmost = {0, 2, 1, 0, 4, 4, 2, 4};  // C = A[:, q]
            expect_Rnz = 23;
        }

        SymbolicQR S = sqr(A, order);  // natural ordering

        CHECK(S.p_inv == expect_p_inv);
        CHECK(S.q == expect_q);
        CHECK(S.parent == expect_parent);
        CHECK(S.leftmost == expect_leftmost);
        CHECK(S.m2 == N);
        CHECK(S.vnz == expect_Vnz);  // manual counts Figure 5.1, p 74
        REQUIRE(S.rnz == expect_Rnz);
    }

    SECTION("Numeric factorization") {
        // Create the identity matrix for testing
        std::vector<csint> rows(N);
        std::iota(rows.begin(), rows.end(), 0);
        std::vector<double> vals(N, 1.0);
        CSCMatrix I = COOMatrix(vals, rows, rows).tocsc();

        // ---------- Factor the matrix
        SymbolicQR S = sqr(A);
        QRResult res = qr(A, S);

        CSCMatrix QT = apply_qtleft(res.V, res.beta, S.p_inv, I);
        CSCMatrix QR = (QT.T() * res.R).droptol().to_canonical();
        compare_matrices(QR, A);

        SECTION("Exercise 5.1: Symbolic factorization") {
            QRResult sym_res = symbolic_qr(A, S);

            CHECK(sym_res.V.indptr() == res.V.indptr());
            CHECK(sym_res.V.indices() == res.V.indices());
            CHECK(sym_res.V.data().size() == res.V.data().size());  // allocation only
            CHECK(sym_res.beta.empty());
            CHECK(sym_res.R.indptr() == res.R.indptr());
            CHECK(sym_res.R.indices() == res.R.indices());
            REQUIRE(sym_res.R.data().size() == res.R.data().size());  // allocation only
        }

        SECTION("Exercise 5.3: Re-QR factorization") {
            res = symbolic_qr(A, S);

            // Compute the numeric factorization using the symbolic result
            reqr(A, S, res);

            CSCMatrix QT = apply_qtleft(res.V, res.beta, S.p_inv, I);
            CSCMatrix QR = (QT.T() * res.R).droptol().to_canonical();
            compare_matrices(QR, A);
        }

        SECTION("Exercise 5.5: Use post-ordering with natural ordering") {
            // Compute the symbolic factorization with postordering
            bool use_postorder = true;
            SymbolicQR S = sqr(A, AMDOrder::Natural, use_postorder);
            QRResult res = qr(A, S);

            // The postordering of this matrix *is* the natural ordering.
            // TODO Find an example with a different postorder for testing
            CSCMatrix QT = apply_qtleft(res.V, res.beta, S.p_inv, I);
            CSCMatrix QR = (QT.T() * res.R).droptol().to_canonical();
            compare_matrices(QR, A);
        }
    }
}


TEST_CASE("QR factorization of overdetermined matrix M > N")
{
    // Define the test matrix A (See Davis, Figure 5.1, p 74)
    // except remove the last 2 columns
    csint M = 8;
    csint N = 5;
    CSCMatrix A = davis_example_qr().slice(0, M, 0, N);

    CHECK(A.shape() == Shape {M, N});

    // See etree in Figure 5.1, p 74
    std::vector<csint> parent = {3, 2, 3, -1, -1};

    std::vector<csint> expect_leftmost = {0, 1, 2, 0, 4, 4, 1, 4};
    std::vector<csint> expect_p_inv = {0, 1, 3, 5, 4, 6, 2, 7};

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
        std::vector<csint> expect_q = {0, 1, 2, 3, 4};  // natural
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

    SECTION("Numeric factorization") {
        SymbolicQR S = sqr(A);
        QRResult res = qr(A, S);

        // Expected values from scipy.linalg.qr
        CSCMatrix expect_V {
            {1.                , 0.                , 0.                , 0.                , 0.                ,
             0.                , 1.                , 0.                , 0.                , 0.                ,
             0.                , 0.2360679774997897, 1.                , 0.                , 0.                ,
             0.                , 0.                , 0.8619788607068873, 1.                , 0.                ,
             0.                , 0.                , 0.                , 0.                , 1.                ,
             0.4142135623730951, 0.                , 0.                , 0.9329077440557915, 0.                ,
             0.                , 0.                , 0.                , 0.                , 0.0980762113533159,
             0.                , 0.                , 0.                , 0.                , 0.0980762113533159
            },
            {M, N},
            'C'  // row-major order
        };

        std::vector<double> expect_beta {
            1.7071067811865472,
            1.8944271909999157,
            1.1474419561548972,
            1.0693375245281538,
            1.9622504486493761
        };

        CSCMatrix expect_R {
            {-1.4142135623730951,  0.                ,  0.                , -3.5355339059327378,  0.                ,
              0.                , -2.23606797749979  , -1.341640786499874 ,  0.                ,  0.                ,
              0.                ,  0.                , -3.0331501776206204, -0.9890707100936804,  0.                ,
              0.                ,  0.                ,  0.                , -2.1264381322847794,  0.                ,
              0.                ,  0.                ,  0.                ,  0.                , -5.196152422706632 ,
              0.                ,  0.                ,  0.                ,  0.                ,  0.                ,
              0.                ,  0.                ,  0.                ,  0.                ,  0.                ,
              0.                ,  0.                ,  0.                ,  0.                ,  0.
            },
            {M, N},
            'C' // row-major order
        };

        compare_matrices(res.V, expect_V);
        CHECK_THAT(is_close(res.beta, expect_beta, tol), AllTrue());
        compare_matrices(res.R, expect_R);

        SECTION("Exercise 5.1: Symbolic factorization") {
            QRResult sym_res = symbolic_qr(A, S);

            CHECK(sym_res.V.indptr() == expect_V.indptr());
            CHECK_THAT(sym_res.V.indices(), UnorderedEquals(expect_V.indices()));
            CHECK(sym_res.V.data().size() == expect_V.data().size());
            CHECK(sym_res.beta.empty());
            CHECK(sym_res.R.indptr() == expect_R.indptr());
            CHECK_THAT(sym_res.R.indices(), UnorderedEquals(expect_R.indices()));
            REQUIRE(sym_res.R.data().size() == expect_R.data().size());
        }

        SECTION("Exercise 5.3: Re-QR factorization") {
            res = symbolic_qr(A, S);

            // Compute the numeric factorization using the symbolic result
            reqr(A, S, res);

            compare_matrices(res.V, expect_V);
            CHECK_THAT(is_close(res.beta, expect_beta, tol), AllTrue());
            compare_matrices(res.R, expect_R);
        }
    }
}


TEST_CASE("QR factorization of an underdetermined matrix M < N", "[under]")
{
    // NOTE As written, when M < N, the cs::qr code computes a QR factorization
    // that results in V size (N, N), and R size (N, N). The actual sizes should
    // be V (M, M) and R (M, N). We currently just slice the result to get the
    // desired sizes.

    // Define the test matrix A (See Davis, Figure 5.1, p 74)
    // except remove the last 2 columns
    csint M = 5;
    csint N = 8;
    CSCMatrix A = davis_example_qr().slice(0, M, 0, N);
    CHECK(A.shape() == Shape {M, N});

    // See etree in Figure 5.1, p 74
    std::vector<csint> parent = {3, 2, 3, 6, 5, -1, -1, -1};

    std::vector<csint> expect_leftmost = {0, 1, 2, 0, 4};
    std::vector<csint> expect_p_inv = {0, 1, 2, 3, 4, 5, 6, 7};  // natural

    SECTION("find_leftmost") {
        REQUIRE(find_leftmost(A) == expect_leftmost);
    }

    SECTION("vcount") {
        SymbolicQR S;
        S.parent.assign(parent.begin(), parent.end());
        S.leftmost = find_leftmost(A);
        vcount(A, S);

        CHECK(S.p_inv == expect_p_inv);
        CHECK(S.vnz == 9);
        REQUIRE(S.m2 == N);  // extra rows added!
    }

    SECTION("Symbolic analysis") {
        std::vector<csint> expect_q = {0, 1, 2, 3, 4, 5, 6, 7};  // natural
        std::vector<csint> expect_parent = parent;

        SymbolicQR S = sqr(A);

        CHECK(S.p_inv == expect_p_inv);
        CHECK(S.q == expect_q);
        CHECK(S.parent == expect_parent);
        CHECK(S.leftmost == expect_leftmost);
        CHECK(S.m2 == N);  // extra rows added!
        CHECK(S.vnz == 9);
        REQUIRE(S.rnz == 16);
    }

    SECTION("Numeric factorization") {
        SymbolicQR S = sqr(A);
        QRResult res = qr(A, S);

        // Expected values from scipy.linalg.qr
        CSCMatrix expect_V {
            {1.                , 0.                , 0.                , 0.                , 0.                ,
             0.                , 1.                , 0.                , 0.                , 0.                ,
             0.                , 0.                , 1.                , 0.                , 0.                ,
             0.4142135623730951, 0.                , 0.                , 1.                , 0.                ,
             0.                , 0.                , 0.                , 0.                , 1.
            },
            {M, M},
            'C'  // row-major order
        };

        std::vector<double> expect_beta(M);  // (M,)
        expect_beta[0] = 1.7071067811865472;

        CSCMatrix expect_R {
            {-1.4142135623730951,  0.                ,  0.                , -3.5355339059327378,  0.                ,  0.                , -1.414213562373095     ,  0.                ,
              0.                ,  2.                ,  1.                ,  0.                ,  0.                ,  0.                ,  1.                    ,  0.                ,
              0.                ,  0.                ,  3.                ,  1.                ,  0.                ,  0.                ,  0.                    ,  0.                ,
              0.                ,  0.                ,  0.                ,  2.1213203435596424,  0.                ,  0.                , -4.7442685329306630e-17,  0.                ,
              0.                ,  0.                ,  0.                ,  0.                ,  5.                ,  1.                ,  0.                    ,  0.
            },
            {M, N},
            'C' // row-major order
        };

        // std::cout << "V:" << std::endl;
        // res.V.print_dense();
        // std::cout << "beta:" << res.beta << std::endl;
        // std::cout << "R:" << std::endl;
        // res.R.print_dense();

        compare_matrices(res.V, expect_V);
        CHECK_THAT(is_close(res.beta, expect_beta, tol), AllTrue());
        compare_matrices(res.R, expect_R);
    }
}


// -----------------------------------------------------------------------------
//         Chapter 6: LU Factorization
// -----------------------------------------------------------------------------
/** Define a helper function to test LU decomposition */
LUResult lu_test(const CSCMatrix& A, AMDOrder order=AMDOrder::Natural)
{
    SymbolicLU S = slu(A, order);
    LUResult res = lu(A, S);
    CSCMatrix LU = (res.L * res.U).droptol().to_canonical();
    CSCMatrix PAQ = A.permute(res.p_inv, res.q).to_canonical();
    compare_matrices(LU, PAQ);
    return res;
}


TEST_CASE("LU Factorization of Square Matrix", "[lu]")
{
    const CSCMatrix A = davis_example_qr(10);
    auto [M, N] = A.shape();

    std::vector<csint> expect_q = {0, 1, 2, 3, 4, 5, 6, 7};

    SECTION("Symbolic Factorization") {
        AMDOrder order = AMDOrder::Natural;
        csint expect_lnz;

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


TEST_CASE("Numeric LU Factorization of Square Matrix", "[lu_numeric]")
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


TEST_CASE("Solve Ax = b with LU")
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

    CHECK_THAT(is_close(x, x_ov, tol), AllTrue());
    REQUIRE_THAT(is_close(x, expect, tol), AllTrue());
}


TEST_CASE("Exercise 6.1: Solve A^T x = b with LU")
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

    REQUIRE_THAT(is_close(x, x_ov, tol), AllTrue());
    REQUIRE_THAT(is_close(x, expect, tol), AllTrue());
}


TEST_CASE("Exercise 6.3: Column Pivoting in LU", "[ex6.3]")
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
        compare_matrices(LU, PAQ);
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
            A = A.dropzeros();

            expect_q = {0, 1, 2, 4, 5, 6, 7, 3};
        }

        SECTION("Multiple zero columns") {
            // Remove a column to test the column pivoting
            for (const auto& k : {2, 3, 5}) {
                for (csint i = 0; i < M; i++) {
                    A(i, k) = 0.0;
                }
            }
            A = A.dropzeros();

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


TEST_CASE("Exercise 6.4: relu", "[ex6.4]")
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
    compare_matrices(B, PBp);  // LU should match the un-permuted B
    compare_matrices(LU, B.to_canonical());
}


TEST_CASE("Exercise 6.5: LU for square, singular matrices", "[ex6.5]")
{
    CSCMatrix A = davis_example_qr(10).to_canonical();
    auto [M, N] = A.shape();

    CSCMatrix B = A;  // create a copy to edit

    SECTION("Single pair of linearly dependent columns") {
        // Create a singular matrix by setting column 3 = 2 * column 5
        for (csint i = 0; i < M; i++) {
            B(i, 3) = 2 * B(i, 5);
        }
    }

    SECTION("Two pairs of linearly dependent columns") {
        for (csint i = 0; i < M; i++) {
            B(i, 3) = 2 * B(i, 5);
            B(i, 2) = 3 * B(i, 4);
        }
    }

    SECTION("Single pair of linearly dependent rows") {
        // Create a singular matrix by setting row 3 = 2 * row 5
        for (csint j = 0; j < N; j++) {
            B(3, j) = 2 * B(5, j);
        }
    }

    SECTION("Two pairs of linearly dependent rows") {
        for (csint j = 0; j < N; j++) {
            B(3, j) = 2 * B(5, j);
            B(2, j) = 3 * B(4, j);
        }
    }

    SECTION("Single zero column") {
        for (csint i = 0; i < M; i++) {
            B(i, 3) = 0.0;
        }

        SECTION("Structural") {
            B = B.dropzeros();
        }
    }

    SECTION("Multiple zero columns") {
        for (csint i = 0; i < M; i++) {
            for (const auto& j : {2, 3, 4}) {
                B(i, j) = 0.0;
            }
        }

        SECTION("Structural") {
            B = B.dropzeros();
        }
    }

    SECTION("Single zero row") {
        for (csint j = 0; j < N; j++) {
            B(3, j) = 0.0;
        }

        SECTION("Structural") {
            B = B.dropzeros();
        }
    }

    SECTION("Multiple zero rows") {
        for (const auto& i : {2, 3, 4}) {
            for (csint j = 0; j < N; j++) {
                B(i, j) = 0.0;
            }
        }

        SECTION("Structural") {
            B = B.dropzeros();
        }
    }

    lu_test(B);
}


TEST_CASE("Exercise 6.6: LU Factorization of Rectangular Matrices", "[ex6.6]")
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


TEST_CASE("Exercise 6.7: Crout's method LU Factorization", "[ex6.7]")
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

        compare_matrices(LU, PA);
    }
}


TEST_CASE("Exercise 6.11: lu_realloc", "[ex6.11]")
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


TEST_CASE("Exercise 6.13: Incomplete LU Decomposition", "[ex6.13]")
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

            compare_matrices(res.L, ires.L);
            compare_matrices(res.U, ires.U);
            compare_matrices(iLU, A);
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
            CHECK_THAT(ires.L.data() >= drop_tol, AllTrue());
            CHECK_THAT(ires.U.data() >= drop_tol, AllTrue());
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


TEST_CASE("Exercise 6.15: 1-norm condition number estimate", "[ex6.15]")
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
        REQUIRE(est_norm == Approx(expect));
    }

    SECTION("Estimate condition number of A") {
        double kappa = cond1est(A);
        double expect = 2.422875115852452;  // MATLAB and python calcs

        REQUIRE(kappa == Approx(expect));
    }
}


/*------------------------------------------------------------------------------
 *          Chapter 7: Fill-Reducing Orderings
 *----------------------------------------------------------------------------*/
TEST_CASE("Approximate Minimum Degree (AMD)", "[amd]")
{
    const CSCMatrix A = davis_example_amd();

    auto [M, N] = A.shape();

    // Test build_graph for each value of AMDOrder
    SECTION("Build Graph") {
        // Number of entries required for a dense row
        csint dense = N + 1;  // force keeping all rows

        CSCMatrix expect;
        AMDOrder order = AMDOrder::Natural;
        bool values = false;

        SECTION("Natural") {
            order = AMDOrder::Natural;
            expect = CSCMatrix {{}, A.indices(), A.indptr(), A.shape()};
        }

        SECTION("A + A^T") {
            order = AMDOrder::APlusAT;
            expect = A + A.transpose(values);
        }

        SECTION("A^T A (no dense)") {
            order = AMDOrder::ATANoDenseRows;
            expect = A.transpose(values) * A;
        }

        SECTION("A^T A") {
            order = AMDOrder::ATA;
            expect = A.transpose(values) * A;
        }

        // Remove diagonal elements function
        auto remove_diagonals = [](CSCMatrix& A) {
            A.fkeep([] (csint i, csint j, double v) { return i != j; });
        };

        remove_diagonals(expect);

        const CSCMatrix C = build_graph(A, order, dense);

        CHECK(C.data().empty());
        compare_matrices(C, expect, values);
    }

    SECTION("AMD ordering") {
        AMDOrder order = AMDOrder::Natural;
        std::vector<csint> expect_p;

        SECTION("Natural") {
            order = AMDOrder::Natural;
            // ordering should be the same as the original
            expect_p.resize(N);
            std::iota(expect_p.begin(), expect_p.end(), 0);
        }

        SECTION("A + A^T") {
            order = AMDOrder::APlusAT;
            expect_p = {0, 5, 9, 7, 3, 2, 4, 6, 8, 1};  // MATLAB cs_amd
        }

        SECTION("A^T A (no dense)") {
            order = AMDOrder::ATANoDenseRows;
            expect_p = {0, 3, 4, 5, 7, 8, 9, 1, 2, 6};
        }

        SECTION("A^T A") {
            order = AMDOrder::ATA;
            expect_p = {0, 3, 4, 5, 7, 8, 9, 1, 2, 6};
        }

        std::vector<csint> p = amd(A, order);
        CHECK(p == expect_p);
    }
}


TEST_CASE("Maximum Matching", "[maxmatch]")
{
    // TODO Test on non-symmetric matrices M > N and M < N
    CSCMatrix A = davis_example_amd();
    auto [M, N] = A.shape();

    csint seed = 0;

    csint expect_rank = M;
    std::vector<csint> expect_jmatch(M);
    std::vector<csint> expect_imatch(N);

    SECTION("Square, Symmetric Full rank") {
        // Matrix is full rank, so should be identity permutation
        expect_rank = 10;
        std::iota(expect_jmatch.begin(), expect_jmatch.end(), 0);
        std::iota(expect_imatch.begin(), expect_imatch.end(), 0);
    }

    SECTION("Square, Rank-deficient (zero rows)") {
        // Zero-out some rows
        for (csint i = 2; i < 5; i++) {
            for (csint j = 0; j < N; j++) {
                A(i, j) = 0.0;
            }
        }
        A = A.to_canonical();

        expect_rank = 7;
        expect_jmatch = {0, 1, -1, -1, -1,  2, 3, 6,  4,  7};
        expect_imatch = {0, 1,  5,  6,  8, -1, 7, 9, -1, -1};
    }

    SECTION("Square, Rank-deficient (zero columns)") {
        // Zero-out some columns
        for (csint j = 2; j < 5; j++) {
            for (csint i = 0; i < M; i++) {
                A(i, j) = 0.0;
            }
        }
        A = A.to_canonical();

        expect_rank = 7;
        expect_jmatch = {0, 1,  5,  6,  8, -1, 7, 9, -1, -1};
        expect_imatch = {0, 1, -1, -1, -1,  2, 3, 6,  4,  7};
    }

    SECTION("M < N, Full Row Rank") {
        // Slice some rows
        A = A.slice(0, M - 3, 0, N);

        expect_rank = 7;
        expect_jmatch = {0, 1, 2, 3, 4, 5, 6};
        expect_imatch = {0, 1, 2, 3, 4, 5, 6, -1, -1, -1};
    }

    SECTION("M > N, Full Column Rank") {
        // Slice some columns
        A = A.slice(0, M, 0, N - 3);

        expect_rank = 7;
        expect_jmatch = {0, 1, 2, 3, 4, 5, 6, -1, -1, -1};
        expect_imatch = {0, 1, 2, 3, 4, 5, 6};
    }

    SECTION("M < N, Row Rank-Deficient") {
        // Slice some rows
        A = A.slice(0, M - 3, 0, N);

        // Zero out some rows
        for (csint i = 2; i < 5; i++) {
            for (csint j = 0; j < N; j++) {
                A(i, j) = 0.0;
            }
        }
        A = A.to_canonical();

        expect_rank = 4;
        expect_jmatch = {0, 1, -1, -1, -1,  2,  3};
        expect_imatch = {0, 1,  5,  6, -1, -1, -1, -1, -1, -1};
    }

    SECTION("M > N, Column Rank-Deficient") {
        // Slice some columns
        A = A.slice(0, M, 0, N - 3);

        // Zero out some columns
        for (csint j = 2; j < 5; j++) {
            for (csint i = 0; i < M; i++) {
                A(i, j) = 0.0;
            }
        }
        A = A.to_canonical();

        expect_rank = 4;
        expect_jmatch = {0, 1,  5,  6, -1, -1, -1, -1, -1, -1};
        expect_imatch = {0, 1, -1, -1, -1,  2,  3};
    }

    MaxMatch res = maxtrans(A, seed);

    // Count number of non-negative entries in jmatch
    csint row_rank = std::accumulate(
        res.jmatch.begin(), res.jmatch.end(), 0,
        [](csint sum, csint j) { return sum + (j >= 0); }
    );

    csint col_rank = std::accumulate(
        res.imatch.begin(), res.imatch.end(), 0,
        [](csint sum, csint i) { return sum + (i >= 0); }
    );

    csint sprank = std::min(row_rank, col_rank);

    CHECK(sprank == expect_rank);
    CHECK(res.jmatch == expect_jmatch);
    CHECK(res.imatch == expect_imatch);
}


TEST_CASE("Strongly Connected Components", "[scc]")
{
    CSCMatrix A = davis_example_amd();
    auto [M, N] = A.shape();

    csint expect_Nb = 0;
    std::vector<csint> expect_p(N);
    std::vector<csint> expect_r(N);

    SECTION("Full Rank") {
        expect_Nb = 1;  // full rank
        expect_p = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        expect_r = {0, 10};
    }

    SECTION("Zero Rows") {
        // Zero out some rows
        for (csint i = 2; i < 5; i++) {
            for (csint j = 0; j < N; j++) {
                A(i, j) = 0.0;
            }
        }
        A = A.to_canonical();

        expect_Nb = 4;  // test with cs_scc
        expect_p = {0, 1, 5, 6, 7, 8, 9, 2, 3, 4};
        expect_r = {0, 7, 8, 9, 10};
    }

    SCCResult D = scc(A);

    CHECK(D.p == expect_p);
    CHECK(D.r == expect_r);
    CHECK(D.Nb == expect_Nb);
}


TEST_CASE("Dulmage-Mendelsohn Permutation", "[dmperm]")
{
    CSCMatrix A = davis_example_amd();
    auto [M, N] = A.shape();

    csint seed = 0;

    csint expect_Nb = 0;
    std::vector<csint> expect_p(N),
                       expect_q(N),
                       expect_r(N),
                       expect_s(N);
    std::array<csint, 5> expect_cc;
    std::array<csint, 5> expect_rr;

    SECTION("Full Rank") {
        // MATLAB results
        expect_Nb = 1;  // full rank
        expect_p = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        expect_q = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        expect_r = {0, 10};
        expect_s = {0, 10};
        expect_cc = {0, 0,  0, 10, 10};
        expect_rr = {0, 0, 10, 10, 10};
    }

    SECTION("Zero Rows") {
        // Zero out some rows
        for (csint i = 2; i < 5; i++) {
            for (csint j = 0; j < N; j++) {
                A(i, j) = 0.0;
            }
        }
        A = A.to_canonical();

        expect_Nb = 2;  // test with MATLAB dmperm
        expect_p = {0, 1, 5, 6, 8, 7, 9, 2, 3, 4};
        expect_q = {5, 8, 9, 0, 1, 2, 3, 4, 6, 7};
        expect_r = {0,  7, 10};
        expect_s = {0, 10, 10};
        expect_cc = {0, 3, 10, 10, 10};
        expect_rr = {0, 7,  7,  7, 10};
    }

    DMPermResult D = dmperm(A, seed);

    // std::cout << "p: " << D.p << std::endl;
    // std::cout << "q: " << D.q << std::endl;
    // std::cout << "r: " << D.r << std::endl;
    // std::cout << "s: " << D.s << std::endl;
    // std::cout << "Nb: " << D.Nb << std::endl;
    // std::cout << "rr: " << D.rr << std::endl;
    // std::cout << "cc: " << D.cc << std::endl;

    CHECK(D.p == expect_p);
    CHECK(D.q == expect_q);
    CHECK(D.r == expect_r);
    CHECK(D.s == expect_s);
    CHECK(D.Nb == expect_Nb);
    CHECK(D.rr == expect_rr);
    CHECK(D.cc == expect_cc);
}

/*==============================================================================
 *============================================================================*/
