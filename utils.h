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

std::vector<double> operator+(
    const std::vector<double>&,
    const std::vector<double>&
);

std::vector<double> operator*(const double, const std::vector<double>&);
std::vector<double> operator*(const std::vector<double>&, const double);
std::vector<double>& operator*=(std::vector<double>&, const double);


#endif

//==============================================================================
//==============================================================================
