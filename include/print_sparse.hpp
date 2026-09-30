//==============================================================================
//     File: print_sparse.hpp
//  Created: 2025-05-08 20:41
//   Author: Bernie Roesler
//
//  Description: Printing Support for C++23 and higher.
//==============================================================================

#pragma once

#include "types.hpp"

#include <algorithm>  // fold_left, max
#include <cmath>      // isfinite, fabs
#include <format>
#include <functional>
#include <iostream>
#include <ranges>
#include <span>
#include <string>
#include <string_view>


namespace cs
{

template <typename T>
inline constexpr std::string_view matrix_format_name = "SparseMatrix";

namespace
{

/// Return max(|x_i|) for x_i in data, ignoring non-finite values.
inline auto get_max_abs_finite(std::span<const double> data)
{
    auto data_view = data
        | std::views::filter([](double v) { return std::isfinite(v); })
        | std::views::transform([](double v) { return std::abs(v); });
    return data_view.empty() ? 0.0 : std::ranges::max(data_view);
}


/// @brief Print elements of the matrix between `start` and `end`.
///
/// The element will be printed as: "(i, j): v" where `i` is the row index,
/// `j` is the column index, and `v` is the value of the element. This
/// function sets the format specifiers for `std::format` depending on the
/// values of the entire matrix, so that the output is consistent.
///
/// @param out         the output string
/// @param start, end  print the `kth` element(s) for `k ∈ [start, end)`.
///
template<PrintableSparseMatrix Matrix>
void write_elems(std::string& out, const Matrix& A, csint start, csint end)
{
    // Compute index width from maximum index
    auto [M, N] = A.shape();
    auto row_width = std::to_string(M - 1).size();
    auto col_width = std::to_string(N - 1).size();

    // Determine whether to use scientific notation
    auto max_abs_val = get_max_abs_finite(A.values());
    bool use_scientific = (max_abs_val < 1e-4 || max_abs_val > 1e4);

    // Leading space aligns for "-" signs
    const auto fmt = use_scientific ? " .4e" : " .4g";
    const auto format_string = std::format(
        "({{0:>{{1}}d}}, {{2:>{{3}}d}}): {{4:{}}}", fmt
    );

    csint k = 0;
    csint total_to_print = end - start;

    // Operate on the non-zero elements of the matrix, as (i, j, v) tuples.
    A.for_each_in_range(
        start,
        end,
        [&](csint i, csint j, double v) {
            std::vformat_to(
                std::back_inserter(out),
                format_string,
                std::make_format_args(i, row_width, j, col_width, v)
            );

            if (++k < total_to_print) {
                out.append("\n");
            }
        }
    );
}

}  // namespace


/// @brief Write the matrix to a string.
///
/// @param out         the output string into which to write.
/// @param verbose     if True, print all non-zeros and their coordinates.
/// @param threshold   if `nnz > threshold`, print only the first and last
///        3 entries in the matrix. Otherwise, print all entries.
template <PrintableSparseMatrix Matrix>
void format_to(
    std::string& out,
    const Matrix& A,
    bool verbose=false,
    csint threshold=100
)
{
    const auto [M, N] = A.shape();
    const auto nz = A.nnz();

    std::format_to(
        std::back_inserter(out),
        "<{} matrix\n        with {} stored elements and shape ({}, {})>",
        matrix_format_name<std::remove_cvref_t<Matrix>>, nz, M, N
    );

    if (verbose) {
        out.append("\n");
        if (nz < threshold) {
            // Print all elements
            write_elems(out, A, 0, nz);
        } else {
            // Print just the first and last Nelems non-zero elements
            constexpr int Nelems = 3;
            write_elems(out, A, 0, Nelems);
            out.append("\n...\n");
            write_elems(out, A, nz - Nelems, nz);
        }
    }
}


/// @brief Print the matrix in dense format.
///
/// @param out       the output string into which to write.
/// @param precision  the number of decimal places to print.
/// @param suppress  if true, small values will be printed as "0".
template <PrintableSparseMatrix Matrix>
void format_dense_to(
    std::string& out,
    const Matrix& A,
    int width=-1,
    int precision=-1,
    char format_spec='\0',
    bool suppress=true
)
{
    const auto order = DenseOrder::ColMajor;  // default Fortran-style column-major order
    const auto A_dense = A.to_dense_vector(order);
    const auto [M, N] = A.shape();

    if (A_dense.size() != static_cast<size_t>(M * N)) {
        throw std::runtime_error("Matrix size does not match dimensions!");
    }

    // Determine whether to use scientific notation (if not specified)
    auto fmt = format_spec;

    if (fmt == '\0') {
        // Use scientific notation if extremum value is very small or very large
        auto max_abs_val = get_max_abs_finite(A.values());
        bool use_scientific = !suppress || (max_abs_val < 1e-4 || max_abs_val > 1e4);
        fmt = use_scientific ? 'e' : 'f';
    }

    const auto p = (precision == -1) ? 4 : precision;

    auto w = width;

    if (w == -1) {
        auto base_w = 4;          // default width e.g. "-1."
        base_w += p;              // add precision "-1.2345"
        if (fmt == 'e') {
            base_w += 4;          // 'e' needs more space for "e+00" part.
        }
        w = std::max(base_w, 4);  // enough for "nan", "-inf", etc.
    }

    // Add column padding
    constexpr auto padding = 4;
    w += padding;

    const auto format_string = std::format("{{:>{}.{}{}}}", w, p, fmt);

    constexpr double suppress_tol = 1e-10;

    for (auto i : std::views::iota(0, M)) {
        out.append(" ");  // indent each row
        for (auto j : std::views::iota(0, N)) {
            csint idx = (order == DenseOrder::ColMajor) ? (i + j*M) : (i*N + j);
            auto val = A_dense[idx];

            if (val == 0.0 || (suppress && std::abs(val) < suppress_tol)) {
                // Print zero with the same width for alignment
                std::format_to(std::back_inserter(out), "{:>{}}", "0", w);
            } else {
                // bool is_integer = std::abs(val - std::round(val)) < suppress_tol;
                // bool print_integer = is_integer && !use_scientific;
                std::vformat_to(
                    std::back_inserter(out),
                    format_string,
                    std::make_format_args(val)
                );
            }
        }
        out.append("\n");
    }
}


namespace detail
{

template <PrintableSparseMatrix Matrix>
struct SparseMatrixFormatter : std::formatter<std::string_view>
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

    auto format(const Matrix& A, std::format_context& ctx) const
    {
        std::string buffer;
        if (mode == PrintMode::Dense) {
            format_dense_to(buffer, A, width, precision, format_spec, suppress);
        } else {
            format_to(buffer, A, (mode == PrintMode::Verbose), threshold);
        }

        return std::formatter<std::string_view>::format(buffer, ctx);
    }
};

}  // namespace detail


// Overload operator<< for compatibility with C++20 and earlier
template <PrintableSparseMatrix Matrix>
inline std::ostream& operator<<(std::ostream& os, const Matrix& A)
{
    return os << std::format("{:v}", A);  // verbose printing assumed
}

}  // namespace cs


//==============================================================================
//==============================================================================
