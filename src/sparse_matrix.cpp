/*==============================================================================
 *     File: sparse_matrix.cpp
 *  Created: 2025-05-09 10:17
 *   Author: Bernie Roesler
 *
 *  Description: Implements the abstract SparseMatrix class.
 *
 *============================================================================*/

#include <iostream>
#include <format>
#include <sstream>

#include "sparse_matrix.h"


namespace cs {


// default destructor
SparseMatrix::~SparseMatrix() = default;


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

    return "({0:>{1}d}, {2:>{3}d}): {4:" + fmt + "}";
}



}  // namespace cs


/*==============================================================================
 *============================================================================*/
