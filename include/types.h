//==============================================================================
//     File: types.h
//  Created: 2025-01-30 15:46
//   Author: Bernie Roesler
//
//  Description: Define types for the CSparse library.
//
//==============================================================================

#ifndef _CSPARSE_TYPES_H_
#define _CSPARSE_TYPES_H_

namespace cs {

using csint = std::int64_t;
using Shape = std::array<csint, 2>;

// Pre-declare classes for type conversions
class COOMatrix;
class CSCMatrix;
struct Symbolic;

struct CholCounts {
    std::vector<csint> parent, row_counts, col_counts;
};


}  // namespace cs

#endif  // _CSPARSE_TYPES_H_

//==============================================================================
//==============================================================================
