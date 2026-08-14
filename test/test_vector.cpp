/*==============================================================================
 *     File: test_vector.cpp
 *  Created: 2026-08-12 21:52
 *   Author: Bernie Roesler
 *
 *  Description: Test CSparse Vector and VectorView classes
 *
 *============================================================================*/


#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "Vector.h"


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


}  // namespace cs

/*==============================================================================
 *============================================================================*/
