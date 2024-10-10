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
    // Private members
    std::vector<double> v_;  // numerical values, size nzmax (auto doubles)
    std::vector<csint> i_;   // row indices, size nzmax
    std::vector<csint> j_;   // column indices, size nzmax
    csint M_ = 0;            // number of rows
    csint N_ = 0;            // number of columns

    inline void print_elems_(std::ostream& os, csint start, csint end) const
    {
        for (csint k = start; k < end; k++)
            os << "(" << i_[k] << ", " << j_[k] << "): " << v_[k] << std::endl;
    }

    public:
        // ---------- Constructors
        COOMatrix();  // NOTE need default since we have others

        // Do not need other "Rule of Five" since we have no pointers
        // COOMatrix(const COOMatrix&);
        // COOMatrix& operator=(COOMatrix);
        // COOMatrix(COOMatrix&&);
        // ~COOMatrix();
        // friend void swap(COOMatrix&, COOMatrix&);

        // Provide data and coordinates as vectors
        COOMatrix(
            const std::vector<double>&,
            const std::vector<csint>&,
            const std::vector<csint>&
        );

        COOMatrix(csint, csint, csint nzmax=0);  // allocate dims + nzmax
        COOMatrix(std::istream& fp);             // from file

        // ---------- Accessors
        csint nnz() const;                   // number of non-zeros
        csint nzmax() const;                 // maximum number of non-zeros
        std::array<csint, 2> shape() const;  // the dimensions of the matrix

        const std::vector<csint>& row() const;     // indices and data
        const std::vector<csint>& column() const;
        const std::vector<double>& data() const;

        void assign(csint, csint, double);  // assign an element of the matrix
        // double operator()(csint, csint) const;  // throw runtime error?

        // ---------- Math Operations
        COOMatrix T() const;  // transpose a copy

        // ---------- Other
        void print(
            std::ostream& os=std::cout,
            bool verbose=false,
            csint threshold=1000
        ) const;
};

// Operator overloads
std::ostream& operator<<(std::ostream&, const COOMatrix&);


// class CSCMatrix
// {
//     // Private members
//     std::vector<double> v_;  // numerical values, size nzmax
//     std::vector<csint> i_;   // row indices, size nzmax
//     std::vector<csint> p_;   // column pointers (CSC size N_);
//     csint nnz_ = 0;          // number of entries
//     csint M_ = 0;            // number of rows
//     csint N_ = 0;            // number of columns

//     // FIXME
//     // inline void print_elems_(std::ostream& os, csint start, csint end) const
//     // {
//     //     for (csint k = start; k < end; k++)
//     //         os << "(" << i_[k] << ", " << j_[k] << "): " << v_[k] << std::endl;
//     // }

//     public:
//         // ---------- Constructors
//         CSCMatrix();

//         // Provide data and coordinates as vectors
//         CSCMatrix(
//             const std::vector<double>&,
//             const std::vector<csint>&,
//             const std::vector<csint>&
//         );

//         CSCMatrix(csint, csint, csint nzmax=0);  // allocate dims + nzmax
//         CSCMatrix(std::istream& fp);             // from file

//         // ---------- Accessors
//         csint nnz() const;                   // number of non-zeros
//         std::array<csint, 2> shape() const;  // the dimensions of the matrix

//         const std::vector<csint>& row() const;     // indices and data
//         const std::vector<csint>& column() const;
//         const std::vector<double>& data() const;

//         void assign(csint, csint, double);  // assign an element of the matrix
//         // double operator()(csint, csint) const;  // throw runtime error?

//         // ---------- Math Operations
//         CSCMatrix T() const;  // transpose a copy

//         // ---------- Other
//         void print(
//             std::ostream& os=std::cout,
//             bool verbose=false,
//             csint threshold=1000
//         ) const;
// };



#endif

//==============================================================================
//==============================================================================
