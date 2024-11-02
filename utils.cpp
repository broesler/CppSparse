/*==============================================================================
 *     File: utils.cpp
 *  Created: 2024-11-02 17:32
 *   Author: Bernie Roesler
 *
 *  Description: Utility functions.
 *
 *============================================================================*/

#include "csparse.h"


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

// TODO operator- for unary vector and vector-vector

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


/*==============================================================================
 *============================================================================*/
