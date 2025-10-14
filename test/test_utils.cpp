/*==============================================================================
 *     File: test_utils.cpp
 *  Created: 2025-05-08 11:01
 *   Author: Bernie Roesler
 *
 *  Description: Test CSparse utility functions
 *
 *============================================================================*/

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include <algorithm>  // reverse
#include <random>
#include <vector>

#include "csparse.h"
#include "test_helpers.h"

using Catch::Matchers::WithinAbs;

namespace cs {


TEST_CASE("Vector Operators", "[vector][ops]")
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


TEST_CASE("Vector permutations", "[vector][perm]")
{
    std::vector<double> b = {0, 1, 2, 3, 4};
    std::vector<csint> p = {2, 0, 1, 4, 3};

    REQUIRE(pvec(p, b) == std::vector<double>{2, 0, 1, 4, 3});
    REQUIRE(ipvec(p, b) == std::vector<double>{1, 2, 0, 4, 3});
    REQUIRE(inv_permute(p) == std::vector<csint>{1, 2, 0, 4, 3});
    REQUIRE(pvec(inv_permute(p), b) == ipvec(p, b));
    REQUIRE(ipvec(inv_permute(p), b) == pvec(p, b));
}


TEST_CASE("Random permutation", "[vector][randperm]")
{
    csint N = 10;
    csint seed;
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

    REQUIRE(p == expect_p);
}


TEST_CASE("Vector norms", "[vector][norm]")
{
    std::vector<double> v = {3, -4};

    SECTION("L0 norm") {
        REQUIRE(norm(v, 0) == 2);
    }

    SECTION("L1 norm") {
        REQUIRE(norm(v, 1) == 7);
    }

    SECTION("L2 norm") {
        REQUIRE_THAT(norm(v, 2), WithinAbs(5.0, tol));
    }

    SECTION("LPI norm") {
        double pi = 4 * atan(1.0);  // pi = 3.14159...
        REQUIRE_THAT(norm(v, pi), WithinAbs(4.457284396597481, tol));
    }

    SECTION("Linf norm") {
        REQUIRE(norm(v, INFINITY) == 4);
    }
}


}  // namespace cs

/*==============================================================================
 *============================================================================*/
