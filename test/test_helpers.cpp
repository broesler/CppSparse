/*==============================================================================
 *     File: test_helpers.cpp
 *  Created: 2025-05-08 10:06
 *   Author: Bernie Roesler
 *
 *  Description: Helpers for testing the csparse library.
 *
 *============================================================================*/

#include <catch2/matchers/catch_matchers_all.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>       // fabs
#include <functional>  // function, greater_equal, not_equal_to
#include <vector>

#include "csparse.h"
#include "test_helpers.h"

using Catch::Matchers::WithinAbs;


namespace cs {


void check_canonical_allclose(
    const CSCMatrix& A,
	const CSCMatrix& expect,
	bool values,
	double tol
)
{
    REQUIRE(A.has_canonical_format());
    REQUIRE(expect.has_canonical_format());
    CHECK(A.nnz() == expect.nnz());
    CHECK(A.shape() == expect.shape());
    CHECK(A.indptr() == expect.indptr());
    CHECK(A.indices() == expect.indices());
    if (values) {
        for (csint p = 0; p < A.nnz(); p++) {
            CAPTURE(p);
            CHECK_THAT(A.data()[p], WithinAbs(expect.data()[p], tol));
        }
    }
}


void check_noncanonical_allclose(
    const CSCMatrix& A,
	const CSCMatrix& expect,
    bool values,
	double tol
)
{
    REQUIRE(A.nnz() == expect.nnz());
    REQUIRE(A.shape() == expect.shape());

    auto [M, N] = A.shape();

    if (values) {
        // Need to check all elements of the matrix because operator() combines
        // duplicate entries, whereas just going through the non-zeros of one
        // matrix does not combine those duplicates.
        for (csint i = 0; i < M; i++) {
            for (csint j = 0; j < N; j++) {
                // Capture the values for comparison on failure
                double A_val = A(i, j);
                double expect_val = expect(i, j);
                CAPTURE(i, j, A_val, expect_val);
                REQUIRE_THAT(A_val, WithinAbs(expect_val, tol));
            }
        }
    }
}


void check_sparse_allclose(
    const CSCMatrix& C,
	const CSCMatrix& expect,
	bool values,
	double tol
)
{
    if (C.has_canonical_format() && expect.has_canonical_format()) {
        check_canonical_allclose(C, expect, values, tol);
    } else {
        check_noncanonical_allclose(C, expect, values, tol);
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
    const double tol
)
{
    assert(a.size() == b.size());

    std::vector<bool> out(a.size());
    for (int i = 0; i < a.size(); i++) {
        out[i] = (std::fabs(a[i] - b[i]) < tol);
    }

    return out;
}

}  // namespace cs

/*==============================================================================
 *============================================================================*/
