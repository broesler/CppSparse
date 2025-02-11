//==============================================================================
//    File: utils.h
// Created: 2024-11-02 17:29
//  Author: Bernie Roesler
//
//  Description: Utility functions for CSparse++.
//
//==============================================================================

#ifndef _CSPARSE_UTILS_H_
#define _CSPARSE_UTILS_H_

#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>  // std::iota, inner_product, accumulate, partial_sum
#include <string>
#include <vector>

#include "types.h"

namespace cs {

std::vector<double> operator+(
    const std::vector<double>& a,
    const std::vector<double>& b
);

std::vector<double> operator-(
    const std::vector<double>& a,
    const std::vector<double>& b
);

std::vector<double> operator-(const std::vector<double>& a);

std::vector<double> operator*(const double c, const std::vector<double>& x);
std::vector<double> operator*(const std::vector<double>& x, const double c);
std::vector<double>& operator*=(std::vector<double>& x, const double c);

/** Compute the inverse (or transpose) of a permutation vector.
 *
 * @note This function is named `cs_pinv` in CSparse, but we have changed the
 * name to avoid conflict with similarly named variables, and the well-known
 * Matlab funvtion to compute the pseudo-inverse of a matrix.
 *
 * @param p  permutation vector
 *
 * @return pinv  inverse permutation vector
 */
std::vector<csint> inv_permute(const std::vector<csint>& p);

/** Compute the cumulative sum of a vector, starting with 0.
 *
 * @param w  a reference to a vector of length N.
 *
 * @return p  the cumulative sum of `w`, of length N + 1.
 */
std::vector<csint> cumsum(const std::vector<csint>& w);


/*------------------------------------------------------------------------------
 *          Vector Permutations
 *----------------------------------------------------------------------------*/
/** Compute \f$ x = Pb \f$ where P is a permutation matrix, represented as
 * a vector.
 *
 * @param p  permutation vector, where `p[k] = i` means `p_{ki} = 1`.
 * @param b  vector of data to permute
 *
 * @return x  `x = Pb` the permuted vector, like `x = p(b)` in Matlab.
 */
template <typename T>
std::vector<T> pvec(
    const std::vector<csint>& p,
    const std::vector<T>& b
    )
{
    std::vector<T> x(b.size());

    for (csint k = 0; k < b.size(); k++)
        x[k] = b[p[k]];

    return x;
}


/** Compute \f$ x = P^T b = P^{-1} b \f$ where P is a permutation matrix,
 * represented as a vector.
 *
 * @param p  permutation vector, where `p[k] = i` means `p_{ki} = 1`.
 * @param b  vector of data to permute
 *
 * @return x  `x = Pb` the permuted vector, like `x = p(b)` in Matlab.
 */
template <typename T>
std::vector<T> ipvec(
    const std::vector<csint>& p,
    const std::vector<T>& b
    )
{
    std::vector<T> x(b.size());

    for (csint k = 0; k < b.size(); k++)
        x[p[k]] = b[k];

    return x;
}


/*------------------------------------------------------------------------------
 *         Declare Template Functions
 *----------------------------------------------------------------------------*/
/** Print a std::vector. */
template <typename T>
void print_vec(
    const std::vector<T>& vec,
    std::ostream& os=std::cout,
    const std::string end="\n"
)
{
    os << "[";
    for (int i = 0; i < vec.size(); i++) {
        os << vec[i];
        if (i < vec.size() - 1) {
            os << ", ";
        }
    }
    os << "]" << end;
}


/** Print a std::vector to an output stream. */
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
{
    print_vec(vec, os, "");
    return os;
}


/** Sort the indices of a vector. */
template <typename T>
std::vector<csint> argsort(const std::vector<T>& vec)
{
    std::vector<csint> idx(vec.size());
    std::iota(idx.begin(), idx.end(), 0);

    // Sort the indices by referencing the vector
    std::sort(
        idx.begin(),
        idx.end(),
        [&vec](csint i, csint j) { return vec[i] < vec[j]; }
    );

    return idx;
}


/** Print a matrix in dense format.
 *
 * @param A  a dense matrix
 * @param M, N  the number of rows and columns
 * @param os  the output stream
 * @param order  the order to print the matrix ('C' for row-major or 'F' for
 *        column-major)
 * 
 * @return os  the output stream
 */
template <typename T>
void print_dense_vec(
    const std::vector<T>& A,
    const csint M,
    const csint N,
    const char order='F',
    std::ostream& os=std::cout
)
{
    const std::string indent(3, ' ');
    for (csint i = 0; i < M; i++) {
        os << indent;  // indent
        for (csint j = 0; j < N; j++) {
            if (order == 'F') {
                os << std::setw(6) << std::setprecision(4) << A[i + j*M] << indent;  // print in column-major order
            } else {
                os << std::setw(6) << std::setprecision(4) << A[i*N + j] << indent;  // print in row-major order
            }
        }
        os << std::endl;
    }
}


} // namespace cs

#endif  // _UTILS_H_

//==============================================================================
//==============================================================================
