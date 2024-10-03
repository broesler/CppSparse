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

#include <array>
#include <cstdlib>
#include <iostream>
#include <vector>

typedef std::size_t csint;


class COOMatrix
{
    std::vector<double> v_;  // numerical values, size nzmax
    std::vector<csint> i_;   // row indices, size nzmax
    std::vector<csint> j_;   // column pointers (CSC size n+1) or column indices (triplet size nzmax)
    csint nnz_ = 0;          // number of entries
    csint M_ = 0;            // number of rows
    csint N_ = 0;            // number of columns
    csint nzmax_ = 0;        // maximum number of entries

    public:
        // Constructors
        COOMatrix();  // NOTE need default since we have others

        // Do not need other "Rule of Five" since we have no pointers
        // COOMatrix(const COOMatrix&);
        // COOMatrix& operator=(COOMatrix);
        // COOMatrix(COOMatrix&&);
        // ~COOMatrix();
        // friend void swap(COOMatrix&, COOMatrix&);

        COOMatrix(
            const std::vector<double>&,
            const std::vector<csint>&,
            const std::vector<csint>&
        );

        // Accessors
        csint nnz();                   // number of non-zeros
        csint nzmax();                 // maximum number of non-zeros
        std::array<csint, 2> shape();  // the dimensions of the matrix

        // Other
        void print(
            bool verbose=false,
            csint threshold=1000,
            std::ostream& os=std::cout
        );
};


#endif

//==============================================================================
//==============================================================================
