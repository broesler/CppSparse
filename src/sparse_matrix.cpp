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
            ss << "..." << std::endl;
            write_elems_(ss, nnz_ - Nelems, nnz_);
        }
    }

    return ss.str();
}


}  // namespace cs


/*==============================================================================
 *============================================================================*/
