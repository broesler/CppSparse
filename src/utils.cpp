/*==============================================================================
 *     File: utils.cpp
 *  Created: 2024-11-02 17:32
 *   Author: Bernie Roesler
 *
 *  Description: Utility functions.
 *
 *============================================================================*/

#include <algorithm>  // max_element
#include <cassert>
#include <cmath>      // isfinite
#include <format>
#include <limits>     // numeric_limits
#include <numeric>    // accumulate
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include "utils.h"

namespace cs {

/*------------------------------------------------------------------------------
 *         Vector Operators 
 *----------------------------------------------------------------------------*/
/** Vector-vector addition */
std::vector<double> operator+(
    std::span<const double> a,
    std::span<const double> b
)
{
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            std::format(
                "Vector size mismatch for addition: size a = {}, size b = {}",
                a.size(), b.size()
            )
        );
    }

    std::vector<double> out(a.size());

    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] + b[i];
    }

    return out;
}

/** Unary minus operator for a vector */
std::vector<double> operator-(std::span<const double> a)
{
    std::vector<double> out(a.size());

    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = -a[i];
    }

    return out;
}


/** Vector-vector subtraction */
std::vector<double> operator-(
    std::span<const double> a,
    std::span<const double> b
)
{
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            std::format(
                "Vector size mismatch for subtraction: size a = {}, size b = {}",
                a.size(), b.size()
            )
        );
    }

    std::vector<double> out(a.size());

    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] - b[i];
    }

    return out;
}


/** Scale a vector by a scalar */
std::vector<double> operator*(const double c, std::span<const double> vec)
{
    std::vector<double> out(vec.begin(), vec.end());
    for (auto& x : out) {
        x *= c;
    }
    return out;
}


std::vector<double> operator*(std::span<const double> vec, const double c)
{
    return c * vec;
}


std::span<double> operator*=(std::span<double> vec, const double c)
{
    for (auto& x : vec) {
        x *= c;
    }
    return vec;
}


std::span<double> operator+=(
    std::span<double> a,
    std::span<const double> b
)
{
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            std::format(
                "Vector size mismatch for addition: size a = {}, size b = {}",
                a.size(), b.size()
            )
        );
    }

    for (size_t i = 0; i < a.size(); ++i) {
        a[i] += b[i];
    }

    return a;
}


/*------------------------------------------------------------------------------
 *          Vector Functions
 *----------------------------------------------------------------------------*/

std::vector<csint> inv_permute(std::span<const csint> p)
{
    std::vector<csint> out(p.size());

    for (size_t k = 0; k < p.size(); ++k)
        out[p[k]] = k;

    return out;
}


double norm(std::span<const double> x, const double ord)
{
    if (x.empty()) {
        return 0.0;
    }

    if (ord == std::numeric_limits<double>::infinity()) {
        // infinity norm
        return std::fabs(*std::max_element(
            x.begin(), x.end(),
            [](double a, double b) { return std::fabs(a) < std::fabs(b); }
        ));
    } else if (ord == 0) {
        return std::count_if(
            x.begin(), x.end(),
            [](double val) {
                return std::fabs(val) > std::numeric_limits<double>::epsilon(); 
            }
        );
    } else if (ord == 1) {
        return std::transform_reduce(
            x.begin(),
            x.end(),
            0.0,
            std::plus<double>(),
            [](double val) { return std::fabs(val); }
        );
    } else if (ord == 2) {
        return std::sqrt(
            std::transform_reduce(
                x.begin(),
                x.end(),
                0.0,
                std::plus<double>(),
                [](double val) { return val * val; }
            )
        );
    } else {
        return std::pow(
            std::transform_reduce(
                x.begin(),
                x.end(),
                0.0,
                std::plus<double>(),
                [ord](double val) { return std::pow(std::fabs(val), ord); }
            ),
            1.0 / ord
        );
    }
}


std::vector<csint> randperm(csint N, csint seed)
{
    std::vector<csint> res(N);
    std::iota(res.begin(), res.end(), 0);  // itentity permutation

    if (seed == 0) {
        return res;
    } else if (seed == -1) {
        std::ranges::reverse(res);
        return res;
    } else {
        if (seed < 0) {
            throw std::invalid_argument("Seed must be non-negative.");
        }
        std::default_random_engine rng(seed);
        std::shuffle(res.begin(), res.end(), rng);
        return res;
    }
}


} // namespace cs

/*==============================================================================
 *============================================================================*/
