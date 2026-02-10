//==============================================================================
//     File: test_helpers.h
//  Created: 2025-05-08 10:06
//   Author: Bernie Roesler
//
//  Description: Helpers for testing the csparse library.
//
//==============================================================================

#ifndef _CSPARSE_TEST_HELPERS_H_
#define _CSPARSE_TEST_HELPERS_H_

#include <functional>  // function
#include <vector>

#include "types.h"

using Catch::Matchers::WithinAbs;

namespace cs {


// General comparison tolerance
constexpr double tol = 1e-14;


/** Check that a sparse matrix is equal to a dense matrix.
 *
 * We use a template function so that the function can be used with both
 * const and non-const matrices.
 *
 * @param A       a sparse matrix
 * @param expect  the expected dense matrix in column-major format
 * @param shape   shape of the expected matrix
 * @param tol     tolerance for comparison
 */
template <typename MatrixT>
void check_sparse_eq_dense(
    MatrixT& A,
    const std::vector<double>& expect,
    Shape shape,
    double tol=1e-14
)
{
    const auto [M, N] = shape;
    REQUIRE(A.shape() == shape);
    REQUIRE(static_cast<csint>(expect.size()) == M * N);

    // Check all elements
    for (csint i = 0; i < M; i++) {
        for (csint j = 0; j < N; j++) {
            // Capture the values for comparison on failure
            double sparse_val = A(i, j);
            double dense_val = expect[i + j * M];
            CAPTURE(i, j, sparse_val, dense_val);
            CHECK_THAT(sparse_val, WithinAbs(dense_val, tol));
        }
    }
}


/** Compare two matrices for equality.
 *
 * @note This function expects the matrices to be in canonical form.
 *
 * @param A       the matrix to test
 * @param expect  the expected matrix
 */
void check_canonical_allclose(
    const CSCMatrix& A,
	const CSCMatrix& expect,
	bool values=true,
	double tol=1e-14
);


/** Compare two matrices for equality.
 *
 * @note This function does not require the matrices to be in canonical form.
 *
 * @param A       the matrix to test
 * @param expect  the expected matrix
 */
void check_noncanonical_allclose(
    const CSCMatrix& A,
	const CSCMatrix& expect,
    bool values=true,
	double tol=1e-14
);


/** Compare two matrices for equality.
 *
 * @note This function does not require the matrices to be in canonical form.
 *
 * If both matrices are in canonical form, then the canonical comparison is
 * used, which is faster.
 *
 * @param A       the matrix to test
 * @param expect  the expected matrix
 */
void check_sparse_allclose(
    const CSCMatrix& A,
	const CSCMatrix& expect,
	bool values=true,
	double tol=1e-14
);


/** Check that all elements of a vector compare to a double.
 *
 * @param vec  a vector of doubles
 * @param c    the double to compare to
 */
template <typename T, typename Compare>
void check_all_compare(
    const std::vector<T>& vec,
    const T& c,
    Compare comp
)
{
    for (size_t i = 0; i < vec.size(); i++) {
        CAPTURE(i, vec[i], c);
        CHECK(comp(vec[i], c));
    }
}


/** Check that all elements of a vector are greater than or equal to a double.
 *
 * @param vec  a vector of doubles
 * @param c    the double to compare to
 */
void check_all_greater_equal(const std::vector<double>& vec, const double c);


/** Check that all elements of a vector are not equal to a double.
 *
 * @param vec  a vector of doubles
 * @param c    the double to compare to
 */
void check_all_not_equal(const std::vector<double>& vec, const double c);


/** Check that all elements of two vectors are within a given tolerance.
 *
 * @param a     a vector of doubles
 * @param b     a vector of doubles
 * @param tol   the tolerance for comparison
 */
void check_vectors_allclose(
    const std::vector<double>& a,
    const std::vector<double>& b,
    const double tol=1e-14
);


}  // namespace cs

#endif  // _CSPARSE_TEST_HELPERS_H_

//==============================================================================
//==============================================================================
