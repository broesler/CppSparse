/*==============================================================================
 *     File: sparse_matrix.cpp
 *  Created: 2025-05-09 10:17
 *   Author: Bernie Roesler
 *
 *  Description: Implements the abstract SparseMatrix class.
 *
 *============================================================================*/

#include <cmath>      // isfinite, fabs
#include <iomanip>    // setw, setprecision, fixed, scientific
#include <iostream>
#include <format>
#include <string>

#include "sparse_matrix.h"


namespace cs {

void SparseMatrix::format_to(std::string& out, bool verbose, csint threshold) const
{
    const auto [M, N] = shape();
    const auto nz = nnz();

    std::format_to(
        std::back_inserter(out),
        "<{} matrix\n        with {} stored elements and shape ({}, {})>",
        get_format_desc_(), nz, M, N
    );

    if (verbose) {
        out.append("\n");
        if (nz < threshold) {
            // Print all elements
            write_elems_(out, 0, nz);
        } else {
            // Print just the first and last Nelems non-zero elements
            constexpr int Nelems = 3;
            write_elems_(out, 0, Nelems);
            out.append("\n...\n");
            write_elems_(out, nz - Nelems, nz);
        }
    }
}


std::string SparseMatrix::make_format_string_() const
{
    // Determine whether to use scientific notation
    double abs_max = 0.0;
    for (const auto& val : data()) {
        if (std::isfinite(val)) {
            abs_max = std::max(abs_max, std::fabs(val));
        }
    }

    auto use_scientific = (abs_max < 1e-4 || abs_max > 1e4);
    // Leading space aligns for "-" signs
    const auto fmt = use_scientific ? " .4e" : " .4g";

    return std::format("({{0:>{{1}}d}}, {{2:>{{3}}d}}): {{4:{}}}", fmt);
}


void SparseMatrix::write_elems_(std::string& out, csint start, csint end) const
{
    // Compute index width from maximum index
    auto [M, N] = shape();
    auto row_width = std::to_string(M - 1).size();
    auto col_width = std::to_string(N - 1).size();

    const auto format_string = make_format_string_();

    csint k = 0;
    csint total_to_print = end - start;

    // Operate on the non-zero elements of the matrix, as (i, j, v) tuples.
    for_each_in_range(
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


// TODO change print_dense to be compatible with std::print
void SparseMatrix::print_dense(int precision, bool suppress, std::ostream& os) const
{
    const auto order = DenseOrder::ColMajor;  // default Fortran-style column-major order
    const auto A = to_dense_vector(order);
    const auto [M, N] = shape();

    if (A.size() != static_cast<size_t>(M * N)) {
        throw std::runtime_error("Matrix size does not match dimensions!");
    }

    // Determine whether to use scientific notation
    double abs_max = 0.0;
    for (const auto& val : A) {
        if (std::isfinite(val)) {
            abs_max = std::max(abs_max, std::fabs(val));
        }
    }

    const auto use_scientific = !suppress || (abs_max < 1e-4 || abs_max > 1e4);

    // Compute column width
    auto width = use_scientific ? (9 + precision) : (6 + precision);
    width = std::max(width, 5);  // enough for "nan", "-inf", etc.

    constexpr double suppress_tol = 1e-10;

    const std::string indent(1, ' ');

    for (auto i : row_range()) {
        os << indent;
        for (auto j : column_range()) {
            csint idx = (order == DenseOrder::ColMajor) ? (i + j*M) : (i*N + j);
            auto val = A[idx];

            if (val == 0.0 || (suppress && std::fabs(val) < suppress_tol)) {
                os << std::format("{:>{}}", "0", width);
            } else {
                // bool is_integer = std::abs(val - std::round(val)) < suppress_tol;
                // bool print_integer = is_integer && !use_scientific;
                os << std::setw(width)
                   // << std::setprecision(print_integer ? 0 : precision)
                   << std::setprecision(precision)
                   << (use_scientific ? std::scientific : std::fixed)
                   << val;
            }
        }
        os << std::endl;
    }
}


}  // namespace cs


/*==============================================================================
 *============================================================================*/
