//==============================================================================
//    File: csc.h
// Created: 2024-10-09 20:57
//  Author: Bernie Roesler
//
//  Description: The header file for the CSparse++ package with definitions of
//    the matrix classes and associated functions.
//
//==============================================================================

#ifndef _CSC_H_
#define _CSC_H_


class CSCMatrix
{
    // Private members
    std::vector<double> v_;  // numerical values, size nzmax
    std::vector<csint> i_;   // row indices, size nzmax
    std::vector<csint> p_;   // column pointers (CSC size N_);
    csint M_ = 0;            // number of rows
    csint N_ = 0;            // number of columns

    inline void print_elems_(std::ostream& os, csint start, csint end) const
    {
        csint n = 0;  // number of elements printed
        for (csint j = 0; j <= N_; j++) {
            for (csint p = p_[j]; p < p_[j + 1]; p++) {
                if ((n >= start) && (n < end)) {
                    os << "(" << i_[p] << ", " << j << "): " << v_[p] << std::endl;
                }
                n++;
            }
        }
    }

    public:
        // ---------- Constructors
        CSCMatrix();

        // Provide data, coordinates, and shsape as vectors
        CSCMatrix(
            const std::vector<double>&,
            const std::vector<csint>&,
            const std::vector<csint>&,
            const std::array<csint, 2>&
        );

        CSCMatrix(csint, csint, csint nzmax=0);  // allocate dims + nzmax

        // ---------- Accessors
        csint nnz() const;                   // number of non-zeros
        csint nzmax() const;                 // maximum number of non-zeros
        std::array<csint, 2> shape() const;  // the dimensions of the matrix

        const std::vector<csint>& indices() const;     // indices and data
        const std::vector<csint>& indptr() const;
        const std::vector<double>& data() const;

        // ---------- Math Operations
        CSCMatrix T() const;  // transpose a copy
        CSCMatrix& sum_duplicates();

        // ---------- Other
        void print(
            std::ostream& os=std::cout,
            bool verbose=false,
            csint threshold=1000
        ) const;
};


std::ostream& operator<<(std::ostream&, const CSCMatrix&);


#endif

//==============================================================================
//==============================================================================
