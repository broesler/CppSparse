/*==============================================================================
 *     File: test/test_vector_view.cpp
 *  Created: 2026-08-14 21:06
 *   Author: Bernie Roesler
 *
 *  Description: Test CSparse VectorView class
 *
 *============================================================================*/


#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include <ranges>  // from_range

#include "Vector.hpp"
#include "VectorView.hpp"


namespace cs {

TEST_CASE("VectorView Construction", "[vectorview][basic]")
{
    SECTION("Default constructor") {
        VectorViewD a;
        REQUIRE(a.size() == 0);
        REQUIRE(a.empty());
    }

    SECTION("Vector constructor") {
        VectorD x{1, 2, 3};
        VectorViewD a(x);
        REQUIRE(a.size() == 3);
        REQUIRE(!a.empty());
        for (size_t i = 0; i < a.size(); ++i) {
            CAPTURE(i);
            REQUIRE(a[i] == static_cast<double>(i + 1));
        }
        REQUIRE(a.front() == 1);
        REQUIRE(a.back() == 3);
    }
}


TEST_CASE("Vector Operators", "[vectorview][ops]")
{
    VectorD a{1, 2, 3};

    SECTION("operator==") {
        VectorViewD a_view(a);
        VectorViewD b_view(a);
        REQUIRE(a_view == b_view);
        REQUIRE_FALSE(a_view != b_view);
    }

    SECTION("operator+=") {
        VectorD b{4, 5, 6};
        VectorViewD a_view(a);
        VectorViewD b_view(b);

        b_view += a_view;

        VectorD expect {5, 7, 9};
        VectorViewD expect_view(b);

        REQUIRE(b_view == expect_view);
    }

    SECTION("operator-=") {
        VectorD b{4, 6, 8};
        VectorViewD a_view(a);
        VectorViewD b_view(b);

        b_view -= a_view;

        VectorD expect {3, 4, 5};
        VectorViewD expect_view(b);

        REQUIRE(b_view == expect_view);
    }

    SECTION("operator*=") {
        VectorD b{4, 5, 6};
        VectorViewD a_view(a);
        VectorViewD b_view(b);

        b_view *= a_view;

        VectorD expect {4, 10, 18};
        VectorViewD expect_view(b);

        REQUIRE(b_view == expect_view);
    }

    SECTION("operator/=") {
        VectorD b{4, 5, 6};
        VectorViewD a_view(a);
        VectorViewD b_view(b);

        b_view /= a_view;

        VectorD expect {4, 2.5, 2};
        VectorViewD expect_view(b);

        REQUIRE(b_view == expect_view);
    }
}


}  // namespace cs

/*==============================================================================
 *============================================================================*/
