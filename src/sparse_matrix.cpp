/*==============================================================================
 *     File: sparse_matrix.cpp
 *  Created: 2025-05-09 10:17
 *   Author: Bernie Roesler
 *
 *  Description: Implements the abstract SparseMatrix class.
 *
 *============================================================================*/

#include <algorithm>  // fold_left, max
#include <cmath>      // isfinite, fabs
#include <format>
#include <ranges>
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


/** Return max(|x_i|) for x_i in data, ignoring non-finite values. */
static inline auto get_max_abs_finite(std::span<const double> data)
{
    auto data_view = data
        | std::views::filter([](double v) { return std::isfinite(v); })
        | std::views::transform([](double v) { return std::abs(v); });
    return data_view.empty() ? 0.0 : std::ranges::max(data_view);
}


void SparseMatrix::write_elems_(std::string& out, csint start, csint end) const
{
    // Compute index width from maximum index
    auto [M, N] = shape();
    auto row_width = std::to_string(M - 1).size();
    auto col_width = std::to_string(N - 1).size();

    // Determine whether to use scientific notation
    auto max_abs_val = get_max_abs_finite(data());
    bool use_scientific = (max_abs_val < 1e-4 || max_abs_val > 1e4);

    // Leading space aligns for "-" signs
    const auto fmt = use_scientific ? " .4e" : " .4g";
    const auto format_string = std::format(
        "({{0:>{{1}}d}}, {{2:>{{3}}d}}): {{4:{}}}", fmt
    );

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


void SparseMatrix::format_dense_to(
    std::string& out,
    int width,
    int precision,
    char format_spec,
    bool suppress
) const
{
    const auto order = DenseOrder::ColMajor;  // default Fortran-style column-major order
    const auto A = to_dense_vector(order);
    const auto [M, N] = shape();

    if (A.size() != static_cast<size_t>(M * N)) {
        throw std::runtime_error("Matrix size does not match dimensions!");
    }

    // Determine whether to use scientific notation (if not specified)
    auto fmt = format_spec;

    if (fmt == '\0') {
        // Use scientific notation if extremum value is very small or very large
        auto max_abs_val = get_max_abs_finite(data());
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

    for (auto i : row_range()) {
        out.append(" ");  // indent each row
        for (auto j : column_range()) {
            csint idx = (order == DenseOrder::ColMajor) ? (i + j*M) : (i*N + j);
            auto val = A[idx];

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


}  // namespace cs


/*==============================================================================
 *============================================================================*/
