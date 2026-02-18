/*==============================================================================
 *     File: coo.cpp
 *  Created: 2024-10-01 21:07
 *   Author: Bernie Roesler
 *
 *  Description: Implements the sparse matrix classes
 *
 *============================================================================*/

#include <algorithm>      // max_element
#include <cassert>
#include <iostream>
#include <fstream>
#include <format>
#include <numeric>        // partial_sum
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <sstream>
#include <unordered_set>
#include <vector>

#include "coo.h"
#include "csc.h"

namespace cs {

/*------------------------------------------------------------------------------
 *     Constructors
 *----------------------------------------------------------------------------*/

COOMatrix::COOMatrix(
    std::span<const double> vals,
    std::span<const csint> rows,
    std::span<const csint> cols,
    const Shape& shape
) : v_(vals.begin(), vals.end()),
    i_(rows.begin(), rows.end()),
    j_(cols.begin(), cols.end())
{
    // Initialize M and N
    if (shape[0] > 0) {
        M_ = shape[0];
    } else {
        if (i_.empty()) {
            M_ = 0;  // no rows
        } else {
            // Infer from the given indices
            M_ = *std::max_element(i_.cbegin(), i_.cend()) + 1;
        }
    }

    if (shape[1] > 0) {
        N_ = shape[1];
    } else {
        if (j_.empty()) {
            N_ = 0;  // no columns
        } else {
            // Infer from the given indices
            N_ = *std::max_element(j_.cbegin(), j_.cend()) + 1;
        }
    }

    // Check that all vectors are the same size
    if (i_.size() != j_.size()) {
        throw std::runtime_error(
            std::format(
                "Index vectors must be the same size: "
                "row size = {}, column size = {}",
                i_.size(), j_.size()
            )
        );
    }

    // Allow v_ to be empty for symbolic computation
    if (!v_.empty() && (v_.size() != i_.size())) {
        throw std::runtime_error(
            std::format(
                "Value vector size must match index vector sizes: "
                "value size = {}, index size = {}",
                v_.size(), i_.size()
            )
        );
    }

    assert(M_ >= 0 && N_ >= 0);

    // Check for any i or j out of bounds
    if (shape[0] && !i_.empty()) {  // shape was given as input, not inferred
        assert(M_ == shape[0]);
        auto max_i = *std::max_element(i_.cbegin(), i_.cend());
        if (max_i >= M_) {
            throw std::runtime_error(
                std::format("Row index out of bounds: {} >= {}", max_i, M_)
            );
        }
    }

    if (shape[1] && !j_.empty()) {  // shape was given as input, not inferred
        assert(N_ == shape[1]);
        auto max_j = *std::max_element(j_.cbegin(), j_.cend());
        if (max_j >= N_) {
            throw std::runtime_error(
                std::format("Column index out of bounds: {} >= {}", max_j, N_)
            );
        }
    }
}


COOMatrix::COOMatrix(const Shape& shape, csint nzmax) : M_{shape[0]}, N_{shape[1]} 
{
    v_.reserve(nzmax);
    i_.reserve(nzmax);
    j_.reserve(nzmax);
}


// Exercise 2.2
COOMatrix::COOMatrix(const CSCMatrix& A) : v_(A.nnz()), i_(A.nnz()), j_(A.nnz())
{
    // Get the shape
    M_ = A.M_;
    N_ = A.N_;
    // Get all elements in column order
    csint nz = 0;
    for (auto j : column_range()) {
        for (auto [i, v] : A.column(j)) {
            i_[nz] = i;
            j_[nz] = j;
            v_[nz++] = v;
        }
    }
}


COOMatrix COOMatrix::from_file(const std::string& filename)
{
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error(std::format("Could not open file: {}", filename));
    }

    try {
        return from_stream(file);
    } catch (const std::runtime_error& e) {
        throw std::runtime_error(
            std::format("Error reading file: {}\n{}", filename, e.what())
        );
    }
}


COOMatrix COOMatrix::from_stream(std::istream& fp)
{
    csint i, j;
    double v;

    COOMatrix A;

    while (fp) {
        std::string line;
        std::getline(fp, line);
        if (!line.empty()) {
            std::stringstream ss(line);
            if (!(ss >> i >> j >> v))
                throw std::runtime_error("File is not in (i, j, v) format!");
            else
                A.insert(i, j, v);
        }
    }

    return A;
}


// Exercise 2.27
COOMatrix COOMatrix::random(csint M, csint N, double density, unsigned int seed)
{
    csint nzmax = M * N * density;

    if (seed == 0) {
        seed = std::random_device{}();
    }

    std::default_random_engine rng{seed};
    std::uniform_int_distribution<csint> idx_dist{0, M * N - 1};
    std::uniform_real_distribution<double> value_dist{0.0, 1.0};

    // Create a set of unique random (linear) indices
    std::unordered_set<csint> idx;

    while (std::ssize(idx) < nzmax) {
        idx.insert(idx_dist(rng));
    }

    // Create the (i, j, v) vectors
    std::vector<csint> row_idx(nzmax);
    std::vector<csint> col_idx(nzmax);
    std::vector<double> values(nzmax);

    // Use ranges::transform to fill the vectors
    std::ranges::transform(idx, row_idx.begin(), [N](csint i) { return i / N; });
    std::ranges::transform(idx, col_idx.begin(), [N](csint i) { return i % N; });
    std::ranges::generate(
        values.begin(),
        values.end(),
        [&rng, &value_dist]() { return value_dist(rng); }
    );

    // Build the matrix
    return {values, row_idx, col_idx, {M, N}};
}


/*------------------------------------------------------------------------------
 *         Setters and Getters
 *----------------------------------------------------------------------------*/
csint COOMatrix::nnz() const { return i_.size(); }
csint COOMatrix::nzmax() const { return i_.capacity(); }
Shape COOMatrix::shape() const { return {M_, N_}; }

const std::vector<csint>& COOMatrix::row() const { return i_; }
const std::vector<csint>& COOMatrix::col() const { return j_; }
const std::vector<double>& COOMatrix::data() const { return v_; }


// cs_entry
COOMatrix& COOMatrix::insert(csint i, csint j, double v)
{
    if ((i < 0) || (j < 0)) {
        throw std::invalid_argument(
            std::format(
                "Row and column indices must be non-negative. "
                "Got ({}, {}).",
                i, j
            )
        );
    }

    i_.push_back(i);
    j_.push_back(j);
    v_.push_back(v);

    assert(v_.size() == i_.size());
    assert(v_.size() == j_.size());

    M_ = std::max(M_, i+1);
    N_ = std::max(N_, j+1);

    return *this;
}


// Exercise 2.5
COOMatrix& COOMatrix::insert(
    std::span<const csint> rows,
    std::span<const csint> cols,
    std::span<const double> C
)
{
    if (rows.size() != cols.size()) {
        throw std::invalid_argument("Index vectors must be the same size.");
    }

    csint N = rows.size();
    
    if (std::ssize(C) != N * N) {
        throw std::invalid_argument("Input matrix must be of size N x N.");
    }

    // Track maximum indices
    csint max_row_idx = 0, 
          max_col_idx = 0;

    for (csint i = 0; i < N; ++i) {
        auto row = rows[i];  // cache value
        max_row_idx = std::max(max_row_idx, row);

        for (csint j = 0; j < N; ++j) {
            auto col = cols[j];  // cache value
            max_col_idx = std::max(max_col_idx, col);
            // Insert the indices and value
            i_.push_back(row);
            j_.push_back(col);
            v_.push_back(C[i + j*N]);  // column-major order
        }
    }

    M_ = std::max(M_, max_row_idx + 1);
    N_ = std::max(N_, max_col_idx + 1);

    return *this;
}


/*------------------------------------------------------------------------------
 *          Format Conversions 
 *----------------------------------------------------------------------------*/
CSCMatrix COOMatrix::compress() const 
{
    const bool values = !v_.empty();
    CSCMatrix C{{M_, N_}, nnz(), values};
    std::vector<csint> w(N_);  // workspace

    // Compute number of elements in each column
    for (auto j : j_) {
        w[j]++;
    }

    // Column pointers are the cumulative sum, starting with 0
    std::partial_sum(w.cbegin(), w.cend(), C.p_.begin() + 1);
    w = C.p_;  // copy back into workspace

    for (auto [i, j, v] : elems()) {
        // A(i, j) is the pth entry in the CSC matrix
        auto p = w[j]++;  // "pointer" to the current element's column
        C.i_[p] = i;
        if (values) {
            C.v_[p] = v;
        }
    }

    return C;
}


// Exercise 2.9
CSCMatrix COOMatrix::tocsc() const { return CSCMatrix{*this}; }


std::vector<double> COOMatrix::to_dense_vector(const DenseOrder order) const
{
    std::vector<double> arr(M_ * N_, 0.0);
    csint idx;

    for (auto [i, j, v] : elems()) {
        // Column- vs row-major order
        if (order == DenseOrder::ColMajor) {
            idx = i + j * M_;
        } else if (order == DenseOrder::RowMajor) {
            idx = j + i * N_;
        } else {
            throw std::invalid_argument("Invalid order argument.");
        }

        arr[idx] = v;
    }

    return arr;
}


/*------------------------------------------------------------------------------
 *          Math Operations
 *----------------------------------------------------------------------------*/
// Exercise 2.6
COOMatrix COOMatrix::transpose() const
{
    return {this->v_, this->j_, this->i_, {this->N_, this->M_}};
}


COOMatrix COOMatrix::T() const { return this->transpose(); }


// Exercise 2.10
std::vector<double> COOMatrix::dot(std::span<const double> x) const
{
    if (std::ssize(x) != N_) {
        throw std::invalid_argument(
            std::format(
                "Matrix-vector size mismatch: "
                "vector size = {}, matrix columns = {}",
                x.size(), N_ 
            )
        );
    }

    std::vector<double> out(x.size());

    for (auto [i, j, v] : elems()) {
        out[i] += v * x[j];
    }

    return out;
}


}  // namespace cs

/*==============================================================================
 *============================================================================*/
