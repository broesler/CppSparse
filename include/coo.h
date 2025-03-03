//==============================================================================
//    File: coo.h
// Created: 2024-10-01 21:08
//  Author: Bernie Roesler
//
//  Description: The header file for the CSparse++ package with definitions of
//    the matrix classes and associated functions.
//
//==============================================================================

#ifndef _CSPARSE_COO_H_
#define _CSPARSE_COO_H_

#include <iostream>
#include <string>
#include <string_view>
#include <sstream>
#include <vector>

#include "types.h"


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

    /** Print elements of the matrix between `start` and `end`.
     *
     * @param ss          the output string stream
     * @param start, end  print the all elements where `p ∈ [start, end]`,
     *        counting column-wise.
     */
    void write_elems_(std::stringstream& ss, csint start, csint end) const;

    public:
        friend class CSCMatrix;

        //----------------------------------------------------------------------
        //        Constructors
        //----------------------------------------------------------------------
        COOMatrix();  // NOTE need default since we have others

        // Do not need other "Rule of Five" since we have no pointers
        // COOMatrix(const COOMatrix&);
        // COOMatrix& operator=(COOMatrix);
        // COOMatrix(COOMatrix&&);
        // ~COOMatrix();
        // friend void swap(COOMatrix&, COOMatrix&);

        /** Construct a COOMatrix from arrays of values and coordinates.
         *
         * The entries are *not* sorted in any order, and duplicates are allowed. Any
         * duplicates will be summed.
         *
         * The matrix shape `(M, N)` will be inferred from the maximum indices given.
         *
         * @param vals  the values of the entries in the matrix
         * @param rows, cols  the non-negative integer row and column indices of
         *        the values
         * @param shape  the dimensions of the matrix
         *
         * @return a new COOMatrix object
         */
        COOMatrix(
            const std::vector<double>& vals,
            const std::vector<csint>& rows,
            const std::vector<csint>& cols,
            const Shape shape=Shape{0, 0}
        );

        /** Allocate a COOMatrix for a given shape and number of non-zeros.
         *
         * @param shape  the dimensions of the matrix
         * @param nzmax  integer capacity of space to reserve for non-zeros
         */
        COOMatrix(const Shape& shape, csint nzmax=0);

        /** Read a COOMatrix matrix from a file.
         *
         * The file is expected to be in "triplet format" `(i, j, v)`, where `(i, j)`
         * are the index coordinates, and `v` is the value to be assigned.
         *
         * @param fp    a reference to the file stream.
         * @throws std::runtime_error if file format is not in triplet format
         */
        COOMatrix(std::istream& fp);                 // from file

        /** Convert a CSCMatrix to a COOMatrix, like Matlab's `find`.
         *
         * @see CSCMatrix::tocoo(), cs_find (Davis, Exercise 2.2)
         *
         * @param A a CSCMatrix.
         * @return C the equivalent matrix in triplet form.
         */
        COOMatrix(const CSCMatrix& A);               // Exercise 2.2, Matlab's find

        /** Create a random sparse matrix.
         *
         * See: Exercise 2.27 performance testing
         *
         * @param M, N  the dimensions of the matrix
         * @param density  the fraction of non-zero elements
         * @param seed  the random seed
         *
         * @return a random sparse matrix
         */
        static COOMatrix random(csint M, csint N, double density=0.1,
                                unsigned int seed=0);

        //----------------------------------------------------------------------
        //        Setters and Getters
        //----------------------------------------------------------------------
        csint nnz() const;    // number of non-zeros
        csint nzmax() const;  // maximum number of non-zeros
        Shape shape() const;  // the dimensions of the matrix

        const std::vector<csint>& row() const;     // indices and data
        const std::vector<csint>& column() const;
        const std::vector<double>& data() const;

        /** Assign a value to a pair of indices.
         *
         * Note that there is no argument checking other than for positive indices.
         * Assigning to an index that is outside of the dimensions of the matrix will
         * just increase the size of the matrix accordingly.
         *
         * Duplicate entries are also allowed to ease incremental construction of
         * matrices from files, or, e.g., finite element applications. Duplicates will be
         * summed upon compression to sparse column/row form.
         *
         * @param i, j  integer indices of the matrix
         * @param v     the value to be assigned
         *
         * @return A    a reference to itself for method chaining.
         *
         * @see cs_entry Davis p 12.
         */
        COOMatrix& assign(csint i, csint j, double v);

        /** Assign a dense submatrix to vectors of indices.
         *
         * See: Davis, Exercise 2.5.
         *
         * @param i, j  vectors of integer indices of length `N`.
         * @param v     dense submatrix of size `N`-by-`N`, in column-major order.
         *
         * @return A    a reference to itself for method chaining.
         *
         * @see cs_entry Davis p 12.
         */
        COOMatrix& assign(
            std::vector<csint> i,
            std::vector<csint> j,
            std::vector<double> v
        );

        //----------------------------------------------------------------------
        //        Format Conversions
        //----------------------------------------------------------------------
        /** Convert a coordinate format matrix to a compressed sparse column matrix.
         *
         * The columns are not guaranteed to be sorted, and duplicates are allowed.
         *
         * @return a copy of the `COOMatrix` in CSC format.
         */
        CSCMatrix compress() const;

        /** Create a canonical format CSCMatrix from a COOMatrix.
         *
         * See: Davis, Exercise 2.9
         */
        CSCMatrix tocsc() const;

        /** Convert the matrix to a dense array.
         *
         * The array is in column-major order, like Fortran.
         *
         * @param order  the order of the array, either 'C' or 'F' for row-major or
         *        column-major order.
         *
         * @return a copy of the matrix as a dense array.
         */
        std::vector<double> to_dense_vector(const char order='F') const;

        //----------------------------------------------------------------------
        //        Math Operations
        //----------------------------------------------------------------------
        /** Transpose the matrix as a copy.
         *
         * See: Davis, Exercise 2.6.
         *
         * @return new COOMatrix object with transposed rows and columns.
         */
        COOMatrix transpose() const;  // transpose a copy, Exercise 2.6
        COOMatrix T() const;

        /** Multiply a COOMatrix by a dense vector.
         *
         * See: Davis, Exercise 2.10.
         *
         * @param x  the dense vector to multiply by.
         *
         * @return y  the result of the matrix-vector multiplication.
         */
        std::vector<double> dot(const std::vector<double>& x) const;

        //----------------------------------------------------------------------
        //        Other
        //----------------------------------------------------------------------
        /** Convert the matrix to a string.
         *
         * @param verbose     if True, print all non-zeros and their coordinates
         * @param threshold   if `nnz > threshold`, print only the first and last
         *        3 entries in the matrix. Otherwise, print all entries.
         */
        std::string to_string(
            bool verbose=false,
            csint threshold=1000
        ) const;

        /** Print the matrix.
         *
         * @param os          the output stream, defaults to std::cout
         * @param verbose     if True, print all non-zeros and their coordinates
         * @param threshold   if `nnz > threshold`, print only the first and last
         *        3 entries in the matrix. Otherwise, print all entries.
         */
        void print(
            std::ostream& os=std::cout,
            bool verbose=false,
            csint threshold=1000
        ) const;

};  // class COOMatrix


//------------------------------------------------------------------------------
//        Operator Overloads
//------------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& os, const COOMatrix& A);

// Exercise 2.10
std::vector<double> operator*(const COOMatrix& A, const std::vector<double>& x);


}  // namespace cs

#endif  // _COO_H_

//==============================================================================
//==============================================================================
