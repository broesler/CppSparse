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
#include <sstream>

#include "sparse_matrix.h"


namespace cs {


std::string SparseMatrix::to_string(bool verbose, csint threshold) const
{
    auto [M, N] = shape();
    csint nnz_ = nnz();
    std::stringstream ss;

    ss << std::format(
        "<{} matrix\n"
        "        with {} stored elements and shape ({}, {})>",
        get_format_desc_(), nnz_, M, N);

    if (verbose) {
        ss << std::endl;
        if (nnz_ < threshold) {
            // Print all elements
            write_elems_(ss, 0, nnz_);
        } else {
            // Print just the first and last Nelems non-zero elements
            int Nelems = 3;
            write_elems_(ss, 0, Nelems);
            ss << "\n...\n";
            write_elems_(ss, nnz_ - Nelems, nnz_);
        }
    }

    return ss.str();
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

    bool use_scientific = (abs_max < 1e-4 || abs_max > 1e4);
    // Leading space aligns for "-" signs
    const std::string fmt = use_scientific ? " .4e" : " .4g";

    return std::format("({{0:>{{1}}d}}, {{2:>{{3}}d}}): {{4:{}}}", fmt);
}


void SparseMatrix::print_dense(int precision, bool suppress, std::ostream& os) const
{
    const char order = 'F';  // default to Fortran-style column-major order
    const std::vector<double> A = to_dense_vector(order);
    auto [M, N] = shape();

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

    bool use_scientific = !suppress || (abs_max < 1e-4 || abs_max > 1e4);

    // Compute column width
    int width = use_scientific ? (9 + precision) : (6 + precision);
    width = std::max(width, 5);  // enough for "nan", "-inf", etc.

    constexpr double suppress_tol = 1e-10;

    const std::string indent(1, ' ');

    for (csint i = 0; i < M; i++) {
        os << indent;
        for (csint j = 0; j < N; j++) {
            csint idx = (order == 'F') ? (i + j*M) : (i*N + j);
            double val = A[idx];

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
