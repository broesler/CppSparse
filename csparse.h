//==============================================================================
//    File: csparse.h
// Created: 2024-10-01 21:08
//  Author: Bernie Roesler
//
//  Description: The header file for the CSparse++ package with definitions of
//    the matrix classes and associated functions.
//
//==============================================================================

#ifndef _CSPARSE_H_
#define _CSPARSE_H_

#include <vector>

typedef uint64_t csint;


class COOMatrix
{
    csint nzmax_;            // maximum number of entries
    csint M_;                // number of rows
    csint N_;                // number of columns
    std::vector<csint> p_;   // column pointers (CSC size n+1) or column indices (triplet size nzmax)
    std::vector<csint> i_;   // row indices, size nzmax
    std::vector<double> x_;  // numerical values, size nzmax
    csint nz_;               // number of entries

    public:
        COOMatrix();

        csint nnz();  // number of non-zeros
        std::array<csint, 2> shape(); // the dimensions of the matrix
};


#endif

//==============================================================================
//==============================================================================
