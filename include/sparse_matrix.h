//==============================================================================
//     File: sparse_matrix.h
//  Created: 2025-05-08 20:41
//   Author: Bernie Roesler
//
//  Description: Header file for the abstract SparseMatrix class.
//
//==============================================================================

#ifndef _CSPARSE_SPARSE_MATRIX_H_
#define _CSPARSE_SPARSE_MATRIX_H_

#include <iostream>
#include <span>
#include <vector>

#include "types.h"
#include "utils.h"  // print_dense_vec

namespace cs {

class SparseMatrix {
protected:
    /** Return the format description of the matrix. */
    virtual std::string_view get_format_desc_() const = 0;

    /** Make the format string for printing one element of the matrix.
     *
     * The element will be printed as: "(i, u): v" where `i` is the row index,
     * `u` is the column index, and `v` is the value of the element. This
     * function sets the format specifiers for `std::format` depending on the
     * values of the entire matrix, so that the output is consistent.
     *
     * @return a string describing the format of a single element.
     */
    virtual std::string make_format_string_() const;

    /** Print elements of the matrix between `start` and `end`.
     *
     * @param ss          the output string stream
     * @param start, end  print the all elements where `p ∈ [start, end]`,
     *        counting column-wise.
     */
    virtual void write_elems_(std::stringstream& ss, csint start, csint end) const = 0;

public:
    /// Virtual destructor: essential for base classes when using polymorphism.
    virtual ~SparseMatrix();

    virtual csint nnz() const = 0;    // number of non-zeros
    virtual csint nzmax() const = 0;  // maximum number of non-zeros
    virtual Shape shape() const = 0;  // the dimensions of the matrix

    virtual const std::vector<double>& data() const = 0;  // numerical values

    /// Matrix-vector right-multiply (see cs_multiply)
    virtual std::vector<double> dot(std::span<const double> x) const = 0;

    /// Convert the matrix to a dense array.
    ///
    /// The array is in column-major order, like Fortran.
    ///
    /// @param order  the order of the array, either 'C' or 'F' for row-major or
    ///        column-major order.
    ///
    /// @return a copy of the matrix as a dense array.
    virtual std::vector<double> to_dense_vector(const char order='F') const = 0;

    // -------------------------------------------------------------------------
    //         Printing
    // -------------------------------------------------------------------------
    ///  Convert the matrix to a string.
    /// 
    /// @param verbose     if True, print all non-zeros and their coordinates
    /// @param threshold   if `nnz > threshold`, print only the first and last
    ///        3 entries in the matrix. Otherwise, print all entries.
    virtual std::string to_string(
        bool verbose=false,
        csint threshold=100
    ) const;

    ///  Print the matrix in dense format.
    /// 
    /// @param precision  the number of decimal places to print.
    /// @param suppress  if true, small values will be printed as "0".
    /// @param os  a reference to the output stream.
    /// 
    /// @return os  a reference to the output stream.
    virtual void print_dense(
        int precision=4,
        bool suppress=true,
        std::ostream& os=std::cout
    ) const;

    ///  Print the matrix.
    /// 
    /// @param os          the output stream, defaults to std::cout
    /// @param verbose     if True, print all non-zeros and their coordinates
    /// @param threshold   if `nz > threshold`, print only the first and last
    ///        3 entries in the matrix. Otherwise, print all entries.
    /// 
    virtual void print(
        std::ostream& os=std::cout,
        bool verbose=false,
        csint threshold=100
    ) const
    {
        os << to_string(verbose, threshold) << std::endl;
    }
};  // class SparseMatrix


// Exercise 2.10
inline std::vector<double> operator*(
    const SparseMatrix& A,
    std::span<const double> x
)
{
    return A.dot(x); 
}

// Exercise 2.10 (overload for exact match with vector inputs)
inline std::vector<double> operator*(
    const SparseMatrix& A,
    const std::vector<double>& x
)
{
    return A.dot(x); 
}



// inline since it's defined in the header
inline std::ostream& operator<<(std::ostream& os, const SparseMatrix& A)
{
    A.print(os, true);  // verbose printing assumed
    return os;
}


}  // namespace cs

#endif  // _CSPARSE_SPARSE_MATRIX_H_

//==============================================================================
//==============================================================================
