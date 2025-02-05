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

// Forward declarations
enum class ICholMethod;

struct CholCounts;
struct TriPerm;
struct SparseSolution;
struct Symbolic;

class COOMatrix;
class CSCMatrix;

}  // namespace cs

#endif  // _CSPARSE_TYPES_H_

//==============================================================================
//==============================================================================
