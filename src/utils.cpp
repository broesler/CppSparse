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
#include <cmath>
#include <format>
#include <fstream>
#include <iomanip>    // setw, fixed, setprecision
#include <iostream>
#include <limits>     // numeric_limits
#include <numeric>    // partial_sum, accumulate
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

    for (csint i = 0; i < a.size(); i++) {
        out[i] = a[i] + b[i];
    }

    return out;
}

/** Unary minus operator for a vector */
std::vector<double> operator-(const std::vector<double>& a)
{
    std::vector<double> out(a.size());

    for (csint i = 0; i < a.size(); i++) {
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

    for (csint i = 0; i < a.size(); i++) {
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


/*------------------------------------------------------------------------------
 *          Vector Functions
 *----------------------------------------------------------------------------*/

std::vector<csint> inv_permute(const std::vector<csint>& p)
{
    std::vector<csint> out(p.size());

    for (csint k = 0; k < p.size(); k++)
        out[p[k]] = k;

    return out;
}


std::vector<csint> cumsum(const std::vector<csint>& w)
{
    std::vector<csint> out(w.size() + 1);

    // Row pointers are the cumulative sum of the counts, starting with 0
    std::partial_sum(w.begin(), w.end(), out.begin() + 1);

    return out;
}


double norm(std::span<const double> x, const double ord)
{
    if (x.empty()) {
        return 0.0;
    }

    if (ord == std::numeric_limits<double>::infinity()) {
        // infinity norm
        return *std::max_element(
            x.begin(), x.end(),
            [](double a, double b) { return std::fabs(a) < std::fabs(b); }
        );
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


/*------------------------------------------------------------------------------
 *         Printing 
 *----------------------------------------------------------------------------*/
void print_dense_vec(
    const std::vector<double>& A,
    csint M,
    csint N,
    char order,
    int precision,
    bool suppress,
    std::ostream& os
)
{
    if (A.size() != (M * N)) {
        throw std::runtime_error("Matrix size does not match dimensions!");
    }

    // Determine whether to use scientific notation
    double abs_max = 0.0;
    for (const auto& val : A) {
        if (std::isfinite(val)) {
            abs_max = std::max(abs_max, std::fabs(val));
        }
    }

    bool use_scientific = !suppress || (abs_max < 1e-4 || abs_max > 1e4);

    // Compute column width
    int width = use_scientific ? (9 + precision) : (6 + precision);
    width = std::max(width, 5);  // enough for "nan", "-inf", etc.

    constexpr double suppress_tol = 1e-10;

    const std::string indent(1, ' ');

    for (csint i = 0; i < M; i++) {
        os << indent;
        for (csint j = 0; j < N; j++) {
            csint idx = (order == 'F') ? (i + j*M) : (i*N + j);
            double val = A[idx];

            if (val == 0.0 || (suppress && std::fabs(val) < suppress_tol)) {
                os << std::format("{:>{}}", "0", width);
            } else {
                // bool is_integer = std::abs(val - std::round(val)) < suppress_tol;
                // bool print_integer = is_integer && !use_scientific;
                os << std::setw(width)
                   // << std::setprecision(print_integer ? 0 : precision)
                   << std::setprecision(precision)
                   << (use_scientific ? std::scientific : std::fixed)
                   << val;
            }
        }
        os << std::endl;
    }
}


} // namespace cs

/*==============================================================================
 *============================================================================*/
