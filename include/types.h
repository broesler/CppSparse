//==============================================================================
//     File: types.h
//  Created: 2025-01-30 15:46
//   Author: Bernie Roesler
//
//  Description: Define types for the CSparse library.
//
//==============================================================================

#pragma once

#include <cstdint>
#include <format>
#include <span>
#include <string>     // string_view
#include <stdexcept>  // runtime_error
#include <vector>


namespace cs {

using csint = std::int32_t;
using Shape = std::array<csint, 2>;

// Need full enum class definition for default arguments
enum class DenseOrder
{
    RowMajor,  // C-style row-major order
    ColMajor   // Fortran-style column-major order
};

enum class AMDOrder
{
    Natural,         // Natural ordering (no-op)
    APlusAT,         // Chol: A + A.T
    ATANoDenseRows,  // LU: A.T @ A, but no dense rows
    ATA              // QR: A.T * A
};


/** Convert a string to an AMDOrder enum.
 *
 * @param order  the string to convert
 *
 * @return the AMDOrder enum
 */
inline AMDOrder amdorder_from_string(std::string_view order)
{
    if (order == "Natural") { return AMDOrder::Natural; }
    if (order == "APlusAT") { return AMDOrder::APlusAT; }
    if (order == "ATANoDenseRows") { return AMDOrder::ATANoDenseRows; }
    if (order == "ATA") { return AMDOrder::ATA; }
    throw std::runtime_error(std::format("Invalid AMDOrder specified: {}.", order));
}


/** Convert an AMDOrder enum to a string.
 *
 * @param order  the AMDOrder enum to convert
 *
 * @return the string representation of the AMDOrder enum
 */
constexpr std::string_view string_from_amdorder(const AMDOrder order) noexcept
{
    switch (order) {
        case AMDOrder::Natural:         return "Natural";
        case AMDOrder::APlusAT:         return "APlusAT";
        case AMDOrder::ATANoDenseRows:  return "ATANoDenseRows";
        case AMDOrder::ATA:             return "ATA";
    }
    return "Unknown AMDOrder";  // no "default" so compiler catches missing cases
}


// Forward declarations
struct CholCounts;
struct CholResult;
struct TriPerm;
struct SparseSolution;
struct SymbolicChol;
struct SymbolicQR;
struct SymbolicLU;
struct QRResult;
struct QRSolveResult;
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

void dfs_r(
    const CSCMatrix& A,
    csint j,
    std::span<char> marked,
    std::vector<csint>& xi
);

bool augment_r(
    csint k,
    const CSCMatrix& A,
    std::span<csint> jmatch,
    std::span<csint> cheap,
    std::span<csint> w,
    csint j
);

MaxMatch maxtrans_r(const CSCMatrix& A, csint seed);

}  // namespace detail

}  // namespace cs


/** Custom formatter for AMDOrder enum to enable std::format support. */
template <>
struct std::formatter<cs::AMDOrder> : std::formatter<std::string_view>
{
    auto format(const cs::AMDOrder order, auto& ctx) const
    {
        return std::formatter<std::string_view>::format(
            string_from_amdorder(order), ctx
        );
    }
};


namespace cs {

/** Print AMDOrder to a stream */
inline std::ostream& operator<<(std::ostream& os, const AMDOrder& order)
{
    return os << std::format("{}", order);
}

}  // namespace cs

//==============================================================================
//==============================================================================
