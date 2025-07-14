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

#include <cstdint>
#include <optional>


namespace cs {

using csint = std::int32_t;
using Shape = std::array<csint, 2>;

template <typename T>
using OptionalVectorRef = std::optional<std::reference_wrapper<std::vector<T>>>;

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
struct CholResult;
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

// Internal functions not exposed in the public API
namespace detail {

std::vector<csint> reach_r(const CSCMatrix& A, const CSCMatrix& B);

std::vector<csint>& dfs_r(
    const CSCMatrix& A,
    csint j,
    std::vector<char>& marked,
    std::vector<csint>& xi
);

bool augment_r(
    csint k,
    const CSCMatrix& A,
    std::vector<csint>& jmatch,
    std::vector<csint>& cheap,
    std::vector<csint>& w,
    csint j
);

MaxMatch maxtrans_r(const CSCMatrix& A, csint seed);

}  // namespace detail

}  // namespace cs

#endif  // _CSPARSE_TYPES_H_

//==============================================================================
//==============================================================================
