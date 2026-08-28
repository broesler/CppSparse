//==============================================================================
//    File: utils.h
// Created: 2024-11-02 17:29
//  Author: Bernie Roesler
//
//  Description: Utility functions for CSparse++.
//
//==============================================================================

#pragma once

#include "types.hpp"

#include <ranges>       // views::transform
#include <span>
#include <type_traits>  // decay_t, conditional_t
#include <vector>

namespace cs {

/** Compute the norm of a vector.
 *
 * @param x  the vector
 * @param ord  the order of the norm
 *
 * @return norm  the norm of the vector
 */
double norm(cVectorViewD x, double ord=2.0);


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
std::vector<csint> inv_permute(std::span<const csint> p);


/** Compute \f$ x = Pb \f$ where P is a permutation matrix, represented as
 * a vector.
 *
 * @param p  permutation vector, where `p[k] = i` means `p_{ki} = 1`.
 * @param b  vector of data to permute
 *
 * @return x  `x = Pb` the permuted vector, like `x = b(p)` in MATLAB.
 */
template <std::ranges::contiguous_range PRange,
          std::ranges::contiguous_range InRange,
          std::ranges::contiguous_range OutRange>
requires std::same_as<std::ranges::range_value_t<PRange>, csint>
    && std::same_as<
        std::ranges::range_value_t<InRange>,
        std::ranges::range_value_t<OutRange>
    >
void pvec(const PRange& p, const InRange& b, OutRange& x)
{
    for (size_t k = 0; k < p.size(); k++) {
        x[k] = b[p[k]];
    }
}


/** Compute \f$ x = Pb \f$ where P is a permutation matrix, represented as
 * a vector.
 *
 * @param p  permutation vector, where `p[k] = i` means `p_{ki} = 1`.
 * @param b  vector of data to permute
 *
 * @return x  `x = Pb` the permuted vector, like `x = b(p)` in MATLAB.
 */
template <std::ranges::contiguous_range PRange,
          std::ranges::contiguous_range InRange>
requires std::same_as<std::ranges::range_value_t<PRange>, csint>
auto pvec(const PRange& p, const InRange& b)
{
    using T = std::ranges::range_value_t<InRange>;
    using OutputType = std::conditional_t<
        DenseVector<InRange>, Vector<T>, std::vector<T>
    >;

    OutputType x(p.size());
    pvec(p, b, x);  // pass in workspace
    return x;
}


/** Compute \f$ x = P^T b = P^{-1} b \f$ where P is a permutation matrix,
 * represented as a vector.
 *
 * @param p  permutation vector, where `p[k] = i` means `p_{ki} = 1`.
 * @param b  vector of data to permute
 * @param x[out]  `x = P^T b` the permuted vector, like `x(p) = b` in MATLAB.
 */
template <std::ranges::contiguous_range PRange,
          std::ranges::contiguous_range InRange,
          std::ranges::contiguous_range OutRange>
requires std::same_as<std::ranges::range_value_t<PRange>, csint>
    && std::same_as<
        std::ranges::range_value_t<InRange>,
        std::ranges::range_value_t<OutRange>
    >
void ipvec(const PRange& p, const InRange& b, OutRange& x)
{
    for (size_t k = 0; k < p.size(); k++) {
        x[p[k]] = b[k];
    }
}


/** Compute \f$ x = P^T b = P^{-1} b \f$ where P is a permutation matrix,
 * represented as a vector.
 *
 * @param p  permutation vector, where `p[k] = i` means `p_{ki} = 1`.
 * @param b  vector of data to permute
 *
 * @return x  `x = P^T b` the permuted vector, like `x(p) = b` in MATLAB.
 */
template <std::ranges::contiguous_range PRange,
          std::ranges::contiguous_range InRange>
requires std::same_as<std::ranges::range_value_t<PRange>, csint>
auto ipvec(const PRange& p, const InRange& b)
{
    using T = std::ranges::range_value_t<InRange>;
    using OutputType = std::conditional_t<
        DenseVector<InRange>, Vector<T>, std::vector<T>
    >;

    OutputType x(p.size());
    ipvec(p, b, x);  // pass in workspace
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


} // namespace cs


//==============================================================================
//==============================================================================
