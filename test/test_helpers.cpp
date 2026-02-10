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
        for (csint p = 0; p < A.nnz(); ++p) {
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

    if (values) {
        // Need to check all elements of the matrix because operator() combines
        // duplicate entries, whereas just going through the non-zeros of one
        // matrix does not combine those duplicates.
        for (auto i : A.row_range()) {
            for (auto j : A.column_range()) {
                // Capture the values for comparison on failure
                auto A_val = A(i, j);
                auto expect_val = expect(i, j);
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


void check_all_greater_equal(const std::vector<double>& vec, const double c)
{
    return check_all_compare(vec, c, std::greater_equal<double>());
}


void check_all_not_equal(const std::vector<double>& vec, const double c)
{
    return check_all_compare(vec, c, std::not_equal_to<double>());
}


void check_vectors_allclose(
    const std::vector<double>& a,
    const std::vector<double>& b,
    const double tol
)
{
    REQUIRE(a.size() == b.size());
    for (int i = 0; i < std::ssize(a); ++i) {
        CAPTURE(i, a[i], b[i]);
        CHECK_THAT(a[i], WithinAbs(b[i], tol));
    }
}

}  // namespace cs

/*==============================================================================
 *============================================================================*/
