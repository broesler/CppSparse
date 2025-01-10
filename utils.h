//==============================================================================
//    File: utils.h
// Created: 2024-11-02 17:29
//  Author: Bernie Roesler
//
//  Description: Utility functions for CSparse++.
//
//==============================================================================

#ifndef _UTILS_H_
#define _UTILS_H_

#include <numeric>  // std::iota

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

std::vector<double> pvec(const std::vector<csint>& p, const std::vector<double>& b);
std::vector<double> ipvec(const std::vector<csint>& p, const std::vector<double>& b);

std::vector<csint> inv_permute(const std::vector<csint>& p);

std::vector<csint> cumsum(const std::vector<csint>& w);

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


#endif

//==============================================================================
//==============================================================================
