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

using csint = std::int64_t;
using Shape = std::array<csint, 2>;

namespace cs {
    // Pre-declare classes for type conversions
    class COOMatrix;
    class CSCMatrix;
    struct Symbolic;
}  // namespace cs

#endif  // _CS_TYPES_H_

//==============================================================================
//==============================================================================
