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


/** Compute the cumulative sum of a vector, starting with 0.
 *
 * @param w  a reference to a vector of length N.
 *
 * @return p  the cumulative sum of `w`, of length N + 1.
 */
std::vector<csint> cumsum(const std::vector<csint>& w);


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
std::vector<T> pvec(const std::vector<csint>& p, const std::vector<T>& b)
{
    std::vector<T> x(b.size());

    for (size_t k = 0; k < b.size(); k++)
        x[k] = b[p[k]];

    return x;
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

    for (size_t k = 0; k < b.size(); k++)
        x[p[k]] = b[k];

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


/** Print a matrix in dense format.
 *
 * @param A  a dense matrix
 * @param M, N  the number of rows and columns
 * @param order  the order to print the matrix ('C' for row-major or 'F' for
 *        column-major)
 * @param precision  the number of significant digits to print
 * @param suppress  print small numbers as 0 if true
 * @param os  the output stream
 * 
 * @return os  the output stream
 */
void print_dense_vec(
    const std::vector<double>& A,
    csint M,
    csint N,
    char order='F',
    int precision=4,
    bool suppress=true,
    std::ostream& os=std::cout
);


} // namespace cs

#endif  // _UTILS_H_

//==============================================================================
//==============================================================================
