//==============================================================================
//    File: coo.h
// Created: 2024-10-01 21:08
//  Author: Bernie Roesler
//
//  Description: The header file for the CSparse++ package with definitions of
//    the matrix classes and associated functions.
//
//==============================================================================

#ifndef _COO_H_
#define _COO_H_


class COOMatrix
{
    // Private members
    static constexpr std::string_view format_desc_ = "COOrdinate Sparse";
    std::vector<double> v_;  // numerical values, size nzmax (auto doubles)
    std::vector<csint> i_;   // row indices, size nzmax
    std::vector<csint> j_;   // column indices, size nzmax
    csint M_ = 0;            // number of rows
    csint N_ = 0;            // number of columns

    void print_elems_(std::ostream&, csint, csint) const;

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
            const std::vector<csint>&,
            const std::array<csint, 2>& shape={0, 0}
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

        COOMatrix& assign(csint, csint, double);  // assign an element of the matrix

        // ---------- Format Conversions
        CSCMatrix tocsc() const;  // convert to CSC format

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


#endif

//==============================================================================
//==============================================================================
