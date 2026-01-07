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

#include <algorithm>  // sort
#include <iostream>
#include <numeric>    // iota
#include <span>
#include <string>
#include <span>
#include <vector>

#include "types.h"

namespace cs {

std::vector<double> operator+(
    std::span<const double> a,
    std::span<const double> b
);

std::vector<double> operator-(
    std::span<const double> a,
    std::span<const double> b
);

std::vector<double> operator-(const std::vector<double>& a);

std::span<double> operator+=(
    std::span<double> a,
    std::span<const double> b
);

std::vector<double> operator*(const double c, const std::vector<double>& x);
std::vector<double> operator*(const std::vector<double>& x, const double c);
std::vector<double>& operator*=(std::vector<double>& x, const double c);


/** Compute the norm of a vector.
 *
 * @param x  the vector
 * @param ord  the order of the norm
 *
 * @return norm  the norm of the vector
 */
double norm(std::span<const double> x, const double ord=2.0);

/*------------------------------------------------------------------------------
 *          Vector Permutations
 *----------------------------------------------------------------------------*/
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


/** Compute \f$ x = Pb \f$ where P is a permutation matrix, represented as
 * a vector.
 *
 * @param p  permutation vector, where `p[k] = i` means `p_{ki} = 1`.
 * @param b  vector of data to permute
 *
 * @return x  `x = Pb` the permuted vector, like `x = b(p)` in MATLAB.
 */
template <typename T>
void pvec(std::span<const csint> p, std::span<const T> b, std::span<T> x)
{
    for (size_t k = 0; k < b.size(); k++)
        x[k] = b[p[k]];
}


/** Compute \f$ x = Pb \f$ where P is a permutation matrix, represented as
 * a vector.
 *
 * @param p  permutation vector, where `p[k] = i` means `p_{ki} = 1`.
 * @param b  vector of data to permute
 *
 * @return x  `x = Pb` the permuted vector, like `x = b(p)` in MATLAB.
 */
template <typename T>
std::vector<T> pvec(const std::vector<csint>& p, const std::vector<T>& b)
{
    std::vector<T> x(b.size());
    pvec<T>(p, b, x);  // pass in workspace
    return x;
}


/** Compute \f$ x = P^T b = P^{-1} b \f$ where P is a permutation matrix,
 * represented as a vector.
 *
 * @param p  permutation vector, where `p[k] = i` means `p_{ki} = 1`.
 * @param b  vector of data to permute
 * @param x[out]  `x = P^T b` the permuted vector, like `x(p) = b` in MATLAB.
 */
template <typename T>
void ipvec(std::span<const csint> p, std::span<const T> b, std::span<T> x)
{
    for (size_t k = 0; k < b.size(); k++)
        x[p[k]] = b[k];
}


/** Compute \f$ x = P^T b = P^{-1} b \f$ where P is a permutation matrix,
 * represented as a vector.
 *
 * @param p  permutation vector, where `p[k] = i` means `p_{ki} = 1`.
 * @param b  vector of data to permute
 *
 * @return x  `x = P^T b` the permuted vector, like `x(p) = b` in MATLAB.
 */
template <typename T>
std::vector<T> ipvec(const std::vector<csint>& p, const std::vector<T>& b)
{
    std::vector<T> x(b.size());
    ipvec<T>(p, b, x);  // pass in workspace
    return x;
}


/** Create a random permutation of integers [0, N-1].
 *
 * @param N  the size of the permutation
 * @param seed  the seed for the random number generator. If `seed` is 0, no
 *        permutation is applied. If `seed` is -1, the permutation is the
 *        reverse of the identity. Otherwise, a random permutation is generated.
 *
 * @return p  the random permutation vector
 */
std::vector<csint> randperm(csint N, csint seed=0);


/*------------------------------------------------------------------------------
 *         Printing
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
    for (size_t i = 0; i < vec.size(); i++) {
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


/** Print a std::array to an output stream. */
template <typename T, std::size_t N>
std::ostream& operator<<(std::ostream& os, const std::array<T, N>& arr)
{
    os << "{";
    for (std::size_t i = 0; i < N; ++i) {
        os << arr[i];
        if (i < N - 1) {
            os << ", ";
        }
    }
    os << "}";
    return os;
}


} // namespace cs

#endif  // _UTILS_H_

//==============================================================================
//==============================================================================
