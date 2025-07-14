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
#include <cmath>          // abs
#include <iostream>
#include <fstream>
#include <format>
#include <random>
#include <stdexcept>
#include <string>
#include <sstream>
#include <tuple>          // tie
#include <unordered_set>

#include "utils.h"
#include "coo.h"
#include "csc.h"

namespace cs {

/*------------------------------------------------------------------------------
 *     Constructors
 *----------------------------------------------------------------------------*/
COOMatrix::COOMatrix() {};


COOMatrix::COOMatrix(
    const std::vector<double>& vals,
    const std::vector<csint>& rows,
    const std::vector<csint>& cols,
    const Shape shape
) : v_(vals),
    i_(rows),
    j_(cols)
{
    // Initialize M and N
    if (shape[0] > 0) {
        M_ = shape[0];
    } else {
        if (i_.empty()) {
            M_ = 0;  // no rows
        } else {
            // Infer from the given indices
            M_ = *std::max_element(i_.begin(), i_.end()) + 1;
        }
    }

    if (shape[1] > 0) {
        N_ = shape[1];
    } else {
        if (j_.empty()) {
            N_ = 0;  // no columns
        } else {
            // Infer from the given indices
            N_ = *std::max_element(j_.begin(), j_.end()) + 1;
        }
    }

    // Check that all vectors are the same size
    assert(i_.size() == j_.size());

    // Allow v_ to be empty for symbolic computation
    if (!v_.empty()) {
        assert(v_.size() == i_.size());
    }

    assert(M_ >= 0 && N_ >= 0);

    // Check for any i or j out of bounds
    if (shape[0] && !i_.empty()) {  // shape was given as input, not inferred
        assert(M_ == shape[0]);
        csint max_i = *std::max_element(i_.begin(), i_.end());
        if (max_i >= M_) {
            throw std::runtime_error(
                std::format("Row index out of bounds: {} >= {}", max_i, M_)
            );
        }
    }

    if (shape[1] && !j_.empty()) {  // shape was given as input, not inferred
        assert(N_ == shape[1]);
        csint max_j = *std::max_element(j_.begin(), j_.end());
        if (max_j >= N_) {
            throw std::runtime_error(
                std::format("Column index out of bounds: {} >= {}", max_j, N_)
            );
        }
    }
}


COOMatrix::COOMatrix(const Shape& shape, csint nzmax)
    : M_(shape[0]),
      N_(shape[1]) 
{
    v_.reserve(nzmax);
    i_.reserve(nzmax);
    j_.reserve(nzmax);
}


// Exercise 2.2
COOMatrix::COOMatrix(const CSCMatrix& A)
    : v_(A.nnz()),
      i_(A.nnz()),
      j_(A.nnz())
{
    // Get the shape
    M_ = A.M_;
    N_ = A.N_;
    // Get all elements in column order
    csint nz = 0;
    for (csint j = 0; j < N_; j++) {
        for (csint p = A.p_[j]; p < A.p_[j+1]; p++) {
            i_[nz] = A.i_[p];
            j_[nz] = j;
            v_[nz++] = A.v_[p];
        }
    }
}


COOMatrix COOMatrix::from_file(const std::string& filename)
{
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    try {
        return from_stream(file);
    } catch (const std::runtime_error& e) {
        throw std::runtime_error(
            "Error reading file: " + filename + "\n" + e.what()
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

    std::default_random_engine rng(seed);
    std::uniform_int_distribution<csint> idx_dist(0, M * N - 1);
    std::uniform_real_distribution<double> value_dist(0.0, 1.0);

    // Create a set of unique random (linear) indices
    std::unordered_set<csint> idx;

    while (idx.size() < nzmax) {
        idx.insert(idx_dist(rng));
    }

    // Create the (i, j, v) vectors
    std::vector<csint> row_idx(nzmax);
    std::vector<csint> col_idx(nzmax);
    std::vector<double> values(nzmax);

    // Use ranges::transform to fill the vectors
    std::ranges::transform(idx, row_idx.begin(), [N](csint i) { return i / N; });
    std::ranges::transform(idx, col_idx.begin(), [N](csint i) { return i % N; });
    std::ranges::generate(values.begin(), values.end(), [&rng, &value_dist]() { return value_dist(rng); });

    // Build the matrix
    return COOMatrix(values, row_idx, col_idx, {M, N});
}


/*------------------------------------------------------------------------------
 *         Setters and Getters
 *----------------------------------------------------------------------------*/
csint COOMatrix::nnz() const { return i_.size(); }
csint COOMatrix::nzmax() const { return i_.capacity(); }

Shape COOMatrix::shape() const
{
    return Shape {M_, N_};
}

const std::vector<csint>& COOMatrix::row() const { return i_; }
const std::vector<csint>& COOMatrix::col() const { return j_; }
const std::vector<double>& COOMatrix::data() const { return v_; }


// cs_entry
COOMatrix& COOMatrix::insert(csint i, csint j, double v)
{
    assert((i >= 0) && (j >= 0));

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
    const std::vector<csint>& rows,
    const std::vector<csint>& cols,
    const std::vector<double>& C
)
{
    if (rows.size() != cols.size()) {
        throw std::invalid_argument("Index vectors must be the same size.");
    }

    csint N = rows.size();
    
    if (C.size() != N * N) {
        throw std::invalid_argument("Input matrix must be of size N x N.");
    }

    // Track maximum indices
    csint max_row_idx = 0, 
          max_col_idx = 0;

    for (csint i = 0; i < N; i++) {
        csint row = rows[i];  // cache value
        max_row_idx = std::max(max_row_idx, row);

        for (csint j = 0; j < N; j++) {
            csint col = cols[j];  // cache value
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
    csint nnz_ = nnz();
    bool values = !v_.empty();
    CSCMatrix C {{M_, N_}, nnz_, values};
    std::vector<csint> w(N_);  // workspace

    // Compute number of elements in each column
    for (csint k = 0; k < nnz_; k++)
        w[j_[k]]++;

    // Column pointers are the cumulative sum
    C.p_ = cumsum(w);
    w = C.p_;  // copy back into workspace

    for (csint k = 0; k < nnz_; k++) {
        // A(i, j) is the pth entry in the CSC matrix
        csint p = w[j_[k]]++;  // "pointer" to the current element's column
        C.i_[p] = i_[k];
        if (values) {
            C.v_[p] = v_[k];
        }
    }

    return C;
}


// Exercise 2.9
CSCMatrix COOMatrix::tocsc() const { return CSCMatrix(*this); }


std::vector<double> COOMatrix::to_dense_vector(const char order) const
{
    std::vector<double> arr(M_ * N_, 0.0);
    csint idx;

    for (csint k = 0; k < nnz(); k++) {
        // Column- vs row-major order
        if (order == 'F') {
            idx = i_[k] + j_[k] * M_;
        } else if (order == 'C') {
            idx = j_[k] + i_[k] * N_;
        } else {
            throw std::invalid_argument("Invalid order argument. Use 'F' or 'C'.");
        }

        arr[idx] = v_[k];
    }

    return arr;
}


/*------------------------------------------------------------------------------
 *          Math Operations
 *----------------------------------------------------------------------------*/
// Exercise 2.6
COOMatrix COOMatrix::transpose() const
{
    return COOMatrix(this->v_, this->j_, this->i_, {this->N_, this->M_});
}


COOMatrix COOMatrix::T() const { return this->transpose(); }


// Exercise 2.10
std::vector<double> COOMatrix::dot(const std::vector<double>& x) const
{
    assert(N_ == x.size());
    std::vector<double> out(x.size());

    for (csint p = 0; p < nnz(); p++) {
        out[i_[p]] += v_[p] * x[j_[p]];
    }

    return out;
}


/*------------------------------------------------------------------------------
 *         Printing
 *----------------------------------------------------------------------------*/
void COOMatrix::write_elems_(std::stringstream& ss, csint start, csint end) const
{
    // Compute index width from maximum index
    int row_width = std::to_string(M_ - 1).size();
    int col_width = std::to_string(N_ - 1).size();

    const std::string format_string = make_format_string_();

    for (csint k = start; k < end; k++) {
        ss << std::vformat(
            format_string,
            std::make_format_args(
                i_[k], row_width,
                j_[k], col_width,
                v_[k]
            )
        );

        if (k < end - 1) {
            ss << "\n";
        }
    }
}


}  // namespace cs

/*==============================================================================
 *============================================================================*/
