/*==============================================================================
 *     File: utils.cpp
 *  Created: 2024-11-02 17:32
 *   Author: Bernie Roesler
 *
 *  Description: Utility functions.
 *
 *============================================================================*/

#include <algorithm>  // fold_left, count_if
#include <cmath>      // isfinite
#include <format>
#include <limits>     // numeric_limits
#include <random>
#include <ranges>     // views::transform
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
std::vector<double> operator*(double c, std::span<const double> vec)
{
    std::vector<double> out(vec.begin(), vec.end());
    for (auto& x : out) {
        x *= c;
    }
    return out;
}


std::vector<double> operator*(std::span<const double> vec, double c)
{
    return c * vec;
}


std::span<double> operator*=(std::span<double> vec, double c)
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


double norm(std::span<const double> x, double ord)
{
    if (x.empty()) {
        return 0.0;
    }

    auto abs_view = x | std::views::transform([](double val) { return std::fabs(val); });

    if (ord == std::numeric_limits<double>::infinity()) {
        // infinity norm: max(|x_i|)
        return std::ranges::max(abs_view);
    } else if (ord == 0) {
        // Zero "norm": number of non-zero entries
        return std::ranges::count_if(
            abs_view, [](double v) { return v > std::numeric_limits<double>::epsilon(); }
        );
    } else if (ord == 1) {
        // One norm: ∑|x_i|
        return std::ranges::fold_left(abs_view, 0.0, std::plus<>());
    } else if (ord == 2) {
        // Two norm: sqrt(∑|x_i|^2)
        auto sqr_view = x | std::views::transform([](double val) { return val * val; });
        auto sum_sqr = std::ranges::fold_left(sqr_view, 0.0, std::plus<>());
        return std::sqrt(sum_sqr);
    } else {
        // General p-norm: (∑|x_i|^p)^(1/p)
        auto pow_view = x | std::views::transform(
            [ord](double val) { return std::pow(std::fabs(val), ord); }
        );
        auto sum_pow = std::ranges::fold_left(pow_view, 0.0, std::plus<>());
        return std::pow(sum_pow, 1.0 / ord);
    }
}


std::vector<csint> randperm(csint N, csint seed)
{
    std::vector<csint> res(N);
    std::ranges::iota(res, 0);  // itentity permutation

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
