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
    /// \param out       the output string into which to write.
    /// \param precision  the number of decimal places to print.
    /// \param suppress  if true, small values will be printed as "0".
    ///
    virtual void format_dense_to(
        std::string& out,
        int width=-1,
        int precision=-1,
        char format_spec='\0',
        bool suppress=true
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
    enum class PrintMode {
        Summary,  // print only the summary lines
        Verbose,  // print all non-zeros and their coordinates
        Dense,    // print the matrix in dense format
    };

    PrintMode mode = PrintMode::Summary;
    bool verbose = false;
    int threshold = 100;
    int width = -1;
    int precision = -1;
    char format_spec = '\0';
    bool suppress = true;

    constexpr auto parse(std::format_parse_context& ctx)
    {
        auto it = ctx.begin();

        // std::is_digit is not constexpr, so define our own
        auto is_digit = [](char c) { return c >= '0' && c <= '9'; };
        auto is_alpha = [](char c) {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
        };

        // Check for empty format specifier, e.g. {}
        if (it == ctx.end() || *it == '}') {
            return it;
        }

        // Check for threshold prefix, e.g. {:100v}
        if (is_digit(*it)) {
            threshold = 0;
            while (it != ctx.end() && is_digit(*it)) {
                threshold = threshold * 10 + (*it - '0');
                ++it;
            }
        }

        // If we're at the end, specifier is invalid, e.g. {:100}
        if (it == ctx.end() || *it == '}') {
            throw std::format_error(
                "Invalid format args for SparseMatrix. "
                "Threshold needs to be followed by 'v' for verbose printing."
            );
        }

        if (*it == 'v') {
            mode = PrintMode::Verbose;
            ++it;
        } else if (*it == 'd') {
            mode = PrintMode::Dense;
            ++it;

            // Parse width and precision for dense format, e.g. {:10.4d}
            if (it != ctx.end() && is_digit(*it)) {
                width = 0;
                while (it != ctx.end() && is_digit(*it)) {
                    width = width * 10 + (*it - '0');
                    ++it;
                }
            }

            if (it != ctx.end() && *it == '.') {
                ++it;
                precision = 0;
                while (it != ctx.end() && is_digit(*it)) {
                    precision = precision * 10 + (*it - '0');
                    ++it;
                }
            }

            // Parse format specifier for dense format, e.g. {:10.4de} for
            // scientific notation
            if (it != ctx.end() && is_alpha(*it)) {
                format_spec = *it;
                ++it;
            }

            // Check for '!' to *not* suppress small values, e.g. {:10.4d!}
            if (it != ctx.end() && *it == '!') {
                suppress = false;
                ++it;
            }
        }

        if (it != ctx.end() && *it != '}') {
            throw std::format_error("Invalid format args for SparseMatrix.");
        }

        return it;
    }

    auto format(const cs::SparseMatrix& A, std::format_context& ctx) const
    {
        std::string buffer;
        if (mode == PrintMode::Dense) {
            A.format_dense_to(buffer, width, precision, format_spec, suppress);
        } else {
            A.format_to(buffer, (mode == PrintMode::Verbose), threshold);
        }

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
