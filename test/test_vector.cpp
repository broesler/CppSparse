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

#include <ranges>  // from_range

#include "Vector.hpp"


namespace cs {

TEST_CASE("Vector Checks", "[vector][basic]")
{
    SECTION("Default constructor") {
        VectorD a;
        REQUIRE(a.size() == 0);
        REQUIRE(a.empty());
    }

    SECTION("Initializer list constructor") {
        VectorD a{1, 2, 3};
        REQUIRE(a.size() == 3);
        REQUIRE(!a.empty());
        for (size_t i = 0; i < a.size(); ++i) {
            CAPTURE(i);
            REQUIRE(a[i] == static_cast<double>(i + 1));
            REQUIRE(a.at(i) == static_cast<double>(i + 1));
        }
        REQUIRE_THROWS(a.at(3));
        REQUIRE(a.front() == 1);
        REQUIRE(a.back() == 3);
    }

    SECTION("Vector from std::vector") {
        std::vector<double> vec{4, 5, 6};
        VectorD a(vec);
        REQUIRE(a.size() == 3);
        for (size_t i = 0; i < a.size(); ++i) {
            CAPTURE(i);
            REQUIRE(a[i] == vec[i]);
        }
    }

    SECTION("Vector from std::span") {
        std::vector<double> vec{7, 8, 9};
        std::span<const double> vec_view(vec);
        VectorD a(vec_view.begin(), vec_view.end());
        VectorD b(std::from_range, vec_view);
        REQUIRE(a.size() == 3);
        REQUIRE(b.size() == 3);
        for (size_t i = 0; i < a.size(); ++i) {
            CAPTURE(i);
            REQUIRE(a[i] == vec[i]);
            REQUIRE(b[i] == vec[i]);
        }
    }
}


TEST_CASE("Vector Operators", "[vector][ops]")
{
    VectorD a{1, 2, 3};

    SECTION("Add two vectors") {
        VectorD b{4, 5, 6};
        VectorD expect {5, 7, 9};
        REQUIRE((a + b) == expect);
    }

    SECTION("Negate a vector") {
        VectorD expect{-1, -2, -3};
        REQUIRE(-a == expect);
    }

    SECTION("Subtract two vectors") {
        VectorD b{4, 6, 8};
        REQUIRE((a - b) == VectorD{-3, -4, -5});
    }

    SECTION("Multiply two vectors") {
        VectorD b{4, 5, 6};
        REQUIRE((a * b) == VectorD{4, 10, 18});
    }

    SECTION("Divide two vectors") {
        VectorD b{4, 5, 6};
        REQUIRE((a / b) == VectorD{0.25, 0.4, 0.5});
    }

    SECTION("Add a scalar to a vector") {
        VectorD expect{2, 3, 4};
        REQUIRE((a + 1) == expect);
        REQUIRE((1 + a) == expect);
    }

    SECTION("Multiply a vector by a scalar") {
        VectorD expect{2, 4, 6};
        REQUIRE((a * 2) == expect);
        REQUIRE((2 * a) == expect);
    }

    SECTION("Subtract a scalar from a vector") {
        VectorD expect{0, 1, 2};
        REQUIRE((a - 1) == expect);
        REQUIRE((1 - a) == -expect);
    }

    SECTION("Divide a vector by a scalar") {
        VectorD expect{0.5, 1, 1.5};
        REQUIRE((a / 2) == expect);
        REQUIRE((2 / a) == VectorD{2, 1, 2.0/3.0});
    }

    SECTION("Min, max, sum, mean") {
        REQUIRE(a.min() == 1);
        REQUIRE(a.max() == 3);
        REQUIRE(a.sum() == 6);
        REQUIRE(a.mean() == 2);
    }
}


}  // namespace cs

/*==============================================================================
 *============================================================================*/
