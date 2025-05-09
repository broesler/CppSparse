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
#include <memory>    // unique_ptr
#include <vector>

#include "types.h"
#include "utils.h"  // print_dense_vec

namespace cs {

class SparseMatrix {
public:
    /// Virtual destructor: essential for base classes when using polymorphism.
    virtual ~SparseMatrix() = default;

    virtual csint nnz() const = 0;    // number of non-zeros
    virtual csint nzmax() const = 0;  // maximum number of non-zeros
    virtual Shape shape() const = 0;  // the dimensions of the matrix

    /// Assign a value to a specific element in the matrix.
    /// 
    /// This function takes O(log M) time if the columns are sorted, and O(M) time
    /// if they are not.
    /// 
    /// See: Davis, Exercise 2.25 assign by index.
    /// 
    /// @param i, j the row and column indices of the element to access.
    /// @param v the value to be assigned.
    /// 
    /// @return a reference to itself for method chaining.
    virtual SparseMatrix& assign(csint i, csint j, double v) = 0;

    /// Assign a dense matrix to the CSCMatrix at the specified locations.
    /// 
    /// See: Davis, Excercises 2.5, and Exercise 2.25.
    /// 
    /// @param rows, cols the row and column indices of the elements to access.
    /// @param C the dense matrix to be assigned.
    /// 
    /// @return a reference to itself for method chaining.
    virtual SparseMatrix& assign(
        const std::vector<csint>& i,
        const std::vector<csint>& j,
        const std::vector<double>& C  // dense column-major
    ) = 0;

    /// Matrix-vector right-multiply (see cs_multiply)
    virtual std::vector<double> dot(const std::vector<double>& x) const = 0;

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
    ) const
    {
        auto [M, N] = shape();
        print_dense_vec(to_dense_vector('F'), M, N, 'F', precision, suppress, os);
    }

    ///  Convert the matrix to a string.
    /// 
    /// @param verbose     if True, print all non-zeros and their coordinates
    /// @param threshold   if `nnz > threshold`, print only the first and last
    ///        3 entries in the matrix. Otherwise, print all entries.
    virtual std::string to_string(
        bool verbose=false,
        csint threshold=1000
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
        csint threshold=1000
    ) const
    {
        os << to_string(verbose, threshold) << std::endl;
    }
};  // class SparseMatrix


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
