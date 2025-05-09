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

// Need full enum class definition for default arguments
enum class AMDOrder
{
    Natural,         // Natural ordering (no-op)
    APlusAT,         // Chol: A + A.T
    ATANoDenseRows,  // LU: A.T @ A, but no dense rows
    ATA              // QR: A.T * A
};

// Forward declarations
struct CholCounts;
struct TriPerm;
struct SparseSolution;
struct SymbolicChol;
struct SymbolicQR;
struct SymbolicLU;
struct QRResult;
struct LUResult;
struct MaxMatch;
struct SCCResult;
struct DMPermResult;

class SparseMatrix;
class COOMatrix;
class CSCMatrix;
class TestCSCMatrix;

}  // namespace cs

#endif  // _CSPARSE_TYPES_H_

//==============================================================================
//==============================================================================
