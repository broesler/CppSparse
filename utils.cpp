/*==============================================================================
 *     File: utils.cpp
 *  Created: 2024-11-02 17:32
 *   Author: Bernie Roesler
 *
 *  Description: Utility functions.
 *
 *============================================================================*/

#include <numeric>

#include "csparse.h"


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
std::vector<double> pvec(
    const std::vector<csint> p,
    const std::vector<double> b
    )
{
    std::vector<double> x(b.size());

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
std::vector<double> ipvec(
    const std::vector<csint> p,
    const std::vector<double> b
    )
{
    std::vector<double> x(b.size());

    for (csint k = 0; k < b.size(); k++)
        x[p[k]] = b[k];

    return x;
}


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
std::vector<csint> inv_permute(const std::vector<csint> p)
{
    std::vector<csint> out(p.size());

    for (csint k = 0; k < p.size(); k++)
        out[p[k]] = k;

    return out;
}


/** Compute the cumulative sum of a vector, starting with 0, and also copy the
 * result back into the vector.
 *
 * @param w  a reference to a vector (typically a "workspace")
 *
 * @return p  the cumulative sum of `w`.
 */
std::vector<csint> cumsum(std::vector<csint>& w)
{
    std::vector<csint> out(w.size() + 1);

    // Row pointers are the cumulative sum of the counts, starting with 0
    std::partial_sum(w.begin(), w.end(), out.begin() + 1);

    // Also copy the cumulative sum back into the workspace for iteration
    w = out;

    return out;
}


/** Print a std::vector. */
template <typename T>
void print_vec(const std::vector<T>& vec)
{
    std::cout << "[";
    for (int i = 0; i < vec.size(); i++) {
        std::cout << vec[i];
        if (i < vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}


/*==============================================================================
 *============================================================================*/
