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

namespace cs {

class COOMatrix
{
    // Private members
    static constexpr std::string_view format_desc_ = "COOrdinate Sparse";
    std::vector<double> v_;  // numerical values, size nzmax (auto doubles)
    std::vector<csint> i_;   // row indices, size nzmax
    std::vector<csint> j_;   // column indices, size nzmax
    csint M_ = 0;            // number of rows
    csint N_ = 0;            // number of columns

    void print_elems_(std::ostream& os, const csint start, const csint end) const;

    public:
        friend class CSCMatrix;

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
            const std::vector<double>& vals,
            const std::vector<csint>& rows,
            const std::vector<csint>& cols
        );

        COOMatrix(csint M, csint N, csint nzmax=0);  // allocate dims + nzmax
        COOMatrix(std::istream& fp);                 // from file

        COOMatrix(const CSCMatrix& A);               // Exercise 2.2, Matlab's find

        // See Exercise 2.27 performance testing
        static COOMatrix random(csint M, csint N, double density=0.1,
                                unsigned int seed=0);

        // ---------- Accessors
        csint nnz() const;                   // number of non-zeros
        csint nzmax() const;                 // maximum number of non-zeros
        Shape shape() const;  // the dimensions of the matrix

        const std::vector<csint>& row() const;     // indices and data
        const std::vector<csint>& column() const;
        const std::vector<double>& data() const;

        COOMatrix& assign(csint i, csint j, double v);  // assign an element
        COOMatrix& assign(
            std::vector<csint> i,
            std::vector<csint> j,
            std::vector<double> v
        );  // assign a dense submatrix (Exercise 2.5)

        // ---------- Format Conversions
        CSCMatrix compress() const;  // raw CSC format (no sorting, duplicates)
        CSCMatrix tocsc() const;     // canonical CSC format, Exercise 2.9

        std::vector<double> toarray(const char order='F') const;

        // ---------- Math Operations
        COOMatrix transpose() const;  // transpose a copy, Exercise 2.6
        COOMatrix T() const;

        std::vector<double> dot(const std::vector<double>& x) const;

        // ---------- Other
        void print(
            std::ostream& os=std::cout,
            const bool verbose=false,
            const csint threshold=1000
        ) const;
};

// Operator overloads
std::ostream& operator<<(std::ostream& os, const COOMatrix& A);

// Exercise 2.10
std::vector<double> operator*(const COOMatrix& A, const std::vector<double>& x);

}  // namespace cs

#endif  // _COO_H_

//==============================================================================
//==============================================================================
