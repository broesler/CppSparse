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

namespace cs {


// General comparison tolerance
constexpr double tol = 1e-14;


/** Compare two matrices for equality.
 *
 * @note This function expects the matrices to be in canonical form.
 *
 * @param C       the matrix to test
 * @param expect  the expected matrix
 */
void compare_canonical(
    const CSCMatrix& C,
	const CSCMatrix& expect,
	bool values=true,
	double tol=1e-14
);


/** Compare two matrices for equality.
 *
 * @note This function does not require the matrices to be in canonical form.
 *
 * @param C       the matrix to test
 * @param expect  the expected matrix
 */
void compare_noncanonical(
    const CSCMatrix& C,
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
 * @param C       the matrix to test
 * @param expect  the expected matrix
 */
void compare_matrices(
    const CSCMatrix& C,
	const CSCMatrix& expect,
	bool values=true,
	double tol=1e-14
);


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
);


/** Create the comparison operators by passing the single comparison function to
 * our vector comparison function.
 */
std::vector<bool> operator>=(const std::vector<double>& vec, const double c);
std::vector<bool> operator!=(const std::vector<double>& vec, const double c);


/** Return a boolean vector comparing each individual element.
 *
 * @param a     a vector of doubles
 * @param b     a vector of doubles
 * @param tol   the tolerance for comparison
 *
 * @return out  a vector whose elements are |a[i] - b[i]| < tol.
 */
std::vector<bool> is_close(
    const std::vector<double>& a,
    const std::vector<double>& b,
    const double tol=1e-14
);


}  // namespace cs

#endif  // _CSPARSE_TEST_HELPERS_H_

//==============================================================================
//==============================================================================
