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
#include "Vector.h"
#include "test_helpers.h"

using Catch::Matchers::WithinAbs;

namespace cs {


TEST_CASE("Vector Operators", "[vector][ops]")
{
    VectorD a{1, 2, 3};

    SECTION("Scale a vector") {
        VectorD expect{2, 4, 6};

        REQUIRE((2 * a) == expect);
        REQUIRE((a * 2) == expect);
    }

    SECTION("Add two vectors") {
        VectorD b{4, 5, 6};

        REQUIRE((a + b) == VectorD{5, 7, 9});
    }

    SECTION("Negate a vector") {
        REQUIRE(-a == VectorD{-1, -2, -3});
    }

    SECTION("Subtract two vectors") {
        VectorD b{4, 5, 6};

        REQUIRE((a - b) == VectorD{-3, -3, -3});
    }
}


TEST_CASE("Vector permutations", "[vector][perm]")
{
    VectorD b{0, 1, 2, 3, 4};
    VectorI p{2, 0, 1, 4, 3};

    REQUIRE(pvec(p, b) == VectorD{2, 0, 1, 4, 3});
    REQUIRE(ipvec(p, b) == VectorD{1, 2, 0, 4, 3});
    REQUIRE(inv_permute(p) == VectorI{1, 2, 0, 4, 3});
    REQUIRE(pvec(inv_permute(p), b) == ipvec(p, b));
    REQUIRE(ipvec(inv_permute(p), b) == pvec(p, b));
}


TEST_CASE("Random permutation", "[vector][randperm]")
{
    csint N = 10;
    csint seed;
    VectorI expect_p{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    SECTION("Identity permutation") {
        seed = 0;
    }

    SECTION("Reverse permutation") {
        seed = -1;
        std::ranges::reverse(expect_p);
    }

    SECTION("Arbitrary permutation") {
        seed = 565656;
        std::default_random_engine rng(seed);
        std::ranges::shuffle(expect_p, rng);
    }

    auto p = randperm(N, seed);

    REQUIRE(p == expect_p);
}


TEST_CASE("Vector norms", "[vector][norm]")
{
    VectorD v{3, -4};

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
