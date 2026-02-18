//==============================================================================
//     File: sparse_matrix.h
//  Created: 2025-05-08 20:41
//   Author: Bernie Roesler
//
//  Description: Header file for the abstract SparseMatrix class.
//
//==============================================================================

#pragma once

#include <functional>
#include <iostream>
#include <ranges>
#include <span>
#include <vector>

#include "types.h"

namespace cs {

using ElemFunc = std::function<void(csint i, csint j, double v)>;


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
     * @param out         the output string
     * @param start, end  print the `kth` element(s) for `k ∈ [start, end)`.
     */
    virtual void write_elems_(std::string& out, csint start, csint end) const;

public:
    /// Virtual destructor: essential for base classes when using polymorphism,
    /// aka via a function like "auto func(const SparseMatrix& A)".
    virtual ~SparseMatrix() noexcept = default;

    virtual csint nnz() const = 0;    // number of non-zeros
    virtual csint nzmax() const = 0;  // maximum number of non-zeros
    virtual Shape shape() const = 0;  // the dimensions of the matrix

    virtual const std::vector<double>& data() const = 0;  // numerical values

    // Iterator over the non-zero elements of the matrix, as (i, j, v) tuples,
    // taking the `kth` element for `k ∈ [0, nnz())`.
    virtual void for_each_in_range(csint start, csint end, ElemFunc func) const = 0;

    /** Return a range for iterating over the columns.
     *
     * @return a range 0, 1, ..., N-1 where N is the number of columns.
     */
    auto column_range() const {
        const auto [M, N] = shape();
        return std::views::iota(0, N);
    }

    /** Return a range for iterating over the rows.
     *
     * @return a range 0, 1, ..., N-1 where N is the number of columns.
     */
    auto row_range() const {
        const auto [M, N] = shape();
        return std::views::iota(0, M);
    }

    /// Matrix-vector right-multiply (see cs_multiply)
    virtual std::vector<double> dot(std::span<const double> x) const = 0;

    /// Convert the matrix to a dense array.
    ///
    /// The array is in column-major order, like Fortran.
    ///
    /// @param order  the order of the array, either DenseOrder::RowMajor or
    ///        DenseOrder::ColMajor (default).
    ///
    /// @return a copy of the matrix as a dense array.
    virtual std::vector<double> to_dense_vector(
        const DenseOrder order = DenseOrder::ColMajor
    ) const = 0;

    // -------------------------------------------------------------------------
    //         Printing
    // -------------------------------------------------------------------------
    /// Write the matrix to a string.
    /// 
    /// @param out         the output string into which to write.
    /// @param verbose     if True, print all non-zeros and their coordinates.
    /// @param threshold   if `nnz > threshold`, print only the first and last
    ///        3 entries in the matrix. Otherwise, print all entries.
    virtual void format_to(
        std::string& out,
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

};  // class SparseMatrix


// Exercise 2.10
inline auto operator*(const SparseMatrix& A, std::span<const double> x)
{
    return A.dot(x);
}


// Exercise 2.10 (overload for exact match with vector inputs)
inline auto operator*(const SparseMatrix& A, const std::vector<double>& x)
{
    return A.dot(x);
}

}  // namespace cs


// -----------------------------------------------------------------------------
//         Printing Support for C++23
// -----------------------------------------------------------------------------
template<>
struct std::formatter<cs::SparseMatrix> : std::formatter<std::string_view>
{
    // TODO accept a threshold for how many non-zeros to print, e.g. {:100v}
    // Accept {:v} for verbose printing, or just {} for default.
    bool verbose = false;
    cs::csint threshold = 100;

    constexpr auto parse(std::format_parse_context& ctx)
    {
        auto it = ctx.begin();
        if (it == ctx.end()) {
            return it;
        }

        if (*it == 'v') {
            verbose = true;
            ++it;
        }

        if (it != ctx.end() && *it != '}') {
            throw std::format_error("Invalid format args for SparseMatrix.");
        }

        return it;
    }

    auto format(const cs::SparseMatrix& A, std::format_context& ctx) const
    {
        std::string buffer;
        A.format_to(buffer, verbose, threshold);
        return std::formatter<std::string_view>::format(buffer, ctx);
    }
};


namespace cs {

// Overload operator<< for compatibility with C++20 and earlier
inline std::ostream& operator<<(std::ostream& os, const SparseMatrix& A)
{
    return os << std::format("{:v}", A);  // verbose printing assumed
}

}  // namespace cs


//==============================================================================
//==============================================================================
