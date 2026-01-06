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
#include <fstream>
#include <iomanip>    // setw, fixed, setprecision
#include <iostream>
#include <limits>     // numeric_limits
#include <numeric>    // accumulate
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils.h"

namespace cs {

/*------------------------------------------------------------------------------
 *         Vector Operators 
 *----------------------------------------------------------------------------*/
/** Vector-vector addition */
std::vector<double> operator+(
    const std::vector<double>& a,
    const std::vector<double>& b
)
{
    assert(a.size() == b.size());

    std::vector<double> out(a.size());

    for (size_t i = 0; i < a.size(); i++) {
        out[i] = a[i] + b[i];
    }

    return out;
}

/** Unary minus operator for a vector */
std::vector<double> operator-(const std::vector<double>& a)
{
    std::vector<double> out(a.size());

    for (size_t i = 0; i < a.size(); i++) {
        out[i] = -a[i];
    }

    return out;
}


/** Vector-vector subtraction */
std::vector<double> operator-(
    const std::vector<double>& a,
    const std::vector<double>& b
    )
{
    assert(a.size() == b.size());

    std::vector<double> out(a.size());

    for (size_t i = 0; i < a.size(); i++) {
        out[i] = a[i] - b[i];
    }

    return out;
}


/** Scale a vector by a scalar */
std::vector<double> operator*(const double c, const std::vector<double>& vec)
{
    std::vector<double> out(vec);
    for (auto& x : out) {
        x *= c;
    }
    return out;
}


std::vector<double> operator*(const std::vector<double>& vec, const double c)
{
    return c * vec;
}


std::vector<double>& operator*=(std::vector<double>& vec, const double c)
{
    for (auto& x : vec) {
        x *= c;
    }
    return vec;
}


std::vector<double>& operator+=(
    std::vector<double>& a,
    const std::vector<double>& b
)
{
    assert(a.size() == b.size());

    for (size_t i = 0; i < a.size(); i++) {
        a[i] += b[i];
    }

    return a;
}


/*------------------------------------------------------------------------------
 *          Vector Functions
 *----------------------------------------------------------------------------*/

std::vector<csint> inv_permute(const std::vector<csint>& p)
{
    std::vector<csint> out(p.size());

    for (size_t k = 0; k < p.size(); k++)
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
        // for (const auto& val : x) {
        //     res += (val != 0);
        // }
        return std::count_if(
            x.begin(), x.end(),
            [](double val) {
                return std::fabs(val) > std::numeric_limits<double>::epsilon(); 
            }
        );
    } else if (ord == 1) {
        // for (const auto& val : x) {
        //     res += std::fabs(val);
        // }
        return std::accumulate(
            x.begin(), x.end(), 0.0,
            [](double sum, double val) { return sum + std::fabs(val); }
        );
    } else if (ord == 2) {
        // for (const auto& val : x) {
        //     res += val * val;
        // }
        // res = std::sqrt(res);
        return std::sqrt(
            std::accumulate(
                x.begin(), x.end(), 0.0,
                [](double sum, double val) { return sum + val * val; }
            )
        );
    } else {
        // for (const auto& val : x) {
        //     res += std::pow(std::fabs(val), ord);
        // }
        // res = std::pow(res, 1.0 / ord);
        return std::pow(
            std::accumulate(
                x.begin(), x.end(), 0.0,
                [ord](double sum, double val) {
                    return sum + std::pow(std::fabs(val), ord); 
                }
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
        std::reverse(res.begin(), res.end());
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
