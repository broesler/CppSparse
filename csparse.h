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
    std::vector<csint> j_;   // column indices (triplet size nzmax)
    csint nnz_ = 0;          // number of entries
    csint M_ = 0;            // number of rows
    csint N_ = 0;            // number of columns
    csint nzmax_ = 0;        // maximum number of entries

    inline void print_elems_(std::ostream& os, csint start, csint end) const
    {
        for (csint k = start; k < end; k++)
            os << "(" << i_[k] << ", " << j_[k] << "): " << v_[k] << std::endl;
    }

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
        csint nnz() const;                   // number of non-zeros
        csint nzmax() const;                 // maximum number of non-zeros
        std::array<csint, 2> shape() const;  // the dimensions of the matrix

        void assign(csint, csint, double);  // assign an element of the matrix
        // double operator()(csint, csint) const;

        // Other
        void print(
            std::ostream& os=std::cout,
            bool verbose=false,
            csint threshold=1000
        ) const;
};

// Operator overloads
std::ostream& operator<<(std::ostream&, const COOMatrix&);

#endif

//==============================================================================
//==============================================================================
