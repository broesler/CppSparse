/*==============================================================================
 *     File: csc.cpp
 *  Created: 2024-10-09 20:58
 *   Author: Bernie Roesler
 *
 *  Description: Implements the compressed sparse column matrix class
 *
 *============================================================================*/

#include <algorithm>   // lower_bound
#include <cassert>
#include <cmath>       // fabs
#include <format>
#include <numeric>     // partial_sum, iota
#include <stdexcept>
#include <sstream>
#include <string>
#include <span>
#include <vector>

#include "utils.h"
#include "csc.h"
#include "coo.h"

namespace cs {

/*------------------------------------------------------------------------------
 *     Constructors
 *----------------------------------------------------------------------------*/

CSCMatrix::CSCMatrix(
    std::span<const double> data,
    std::span<const csint> indices,
    std::span<const csint> indptr,
    const Shape& shape
)
    : v_(data.begin(), data.end()),
      i_(indices.begin(), indices.end()),
      p_(indptr.begin(), indptr.end()),
      M_(shape[0]),
      N_(shape[1])
{}


CSCMatrix::CSCMatrix(const Shape& shape, csint nzmax, bool values)
    : i_(nzmax),
      p_(shape[1] + 1),
      M_(shape[0]),
      N_(shape[1])
{
    if (values) {
        v_.resize(nzmax);
    }
}


// Exercise 2.2, 2.9
CSCMatrix::CSCMatrix(const COOMatrix& A) : CSCMatrix(A.compress())
{
    if (!v_.empty()) {
        sum_duplicates();  // O(N) space, O(nnz) time
        dropzeros();       // O(nnz) time
    }
    sort();            // O(M) space, O(M + N + nnz) time
    has_canonical_format_ = true;
}


// Exercise 2.16
CSCMatrix::CSCMatrix(
    std::span<const double> A,
    const Shape& shape,
    const DenseOrder order
) : M_(shape[0]),
    N_(shape[1])
{
    if (static_cast<csint>(A.size()) != (M_ * N_)) {
        throw std::invalid_argument(
            std::format(
                "Input array size does not match given shape: "
                "array size = {}, shape = ({}, {})",
                A.size(), M_, N_
            )
        );
    }

    // Allocate memory
    v_.reserve(A.size());
    i_.reserve(A.size());
    p_.reserve(N_);

    csint nz = 0;  // count number of non-zeros

    for (auto j : column_range()) {
        p_.push_back(nz);

        for (csint i = 0; i < M_; ++i) {
            auto val = (order == DenseOrder::ColMajor) ? A[i + j * M_] : A[j + i * N_];

            // Only store non-zeros
            if (val != 0.0) {
                i_.push_back(i);
                v_.push_back(val);
                ++nz;
            }
        }
    }

    // Finalize and free unused space
    p_.push_back(nz);
    realloc();

    has_sorted_indices_ = true;    // guaranteed by algorithm
    has_canonical_format_ = true;  // guaranteed by input format
}


void CSCMatrix::realloc(csint nzmax)
{
    csint Z = (nzmax <= 0) ? p_[N_] : nzmax;

    try {
    p_.resize(N_ + 1);  // always contains N_ columns + nz
    i_.resize(Z);
    if (!v_.empty()) {
        v_.resize(Z);
    }
    } catch (const std::bad_alloc& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << "Failed to allocate memory for CSCMatrix." << std::endl;
        throw;  // let calling code handle it
    }

    p_.shrink_to_fit();  // deallocate memory
    i_.shrink_to_fit();
    v_.shrink_to_fit();
}


/*------------------------------------------------------------------------------
 *         Accessors
 *----------------------------------------------------------------------------*/
csint CSCMatrix::nnz() const { return i_.size(); }
csint CSCMatrix::nzmax() const { return i_.capacity(); }
Shape CSCMatrix::shape() const { return Shape{M_, N_}; }

const std::vector<csint>& CSCMatrix::indices() const { return i_; }
const std::vector<csint>& CSCMatrix::indptr() const { return p_; }
const std::vector<double>& CSCMatrix::data() const { return v_; }


// Exercise 2.9
CSCMatrix& CSCMatrix::to_canonical()
{
    sum_duplicates();
    dropzeros();
    sort();
    has_canonical_format_ = true;
    return *this;
}


bool CSCMatrix::has_sorted_indices() const { return has_sorted_indices_; }
bool CSCMatrix::has_canonical_format() const { return has_canonical_format_; }


// Exercise 2.13
bool CSCMatrix::is_symmetric() const
{
    if (M_ != N_) {
        return false;
    }

    for (auto j : column_range()) {
        for (auto [i, v] : column(j)) {
            if (i == j)
                continue;  // skip diagonal

            if (v != (*this)(j, i))
                return false;
        }
    }

    return true;
}


// CSparse demo2 is_sym
csint CSCMatrix::is_triangular() const
{
    if (M_ != N_) {
        return 0;  // not square
    }

    bool is_upper = true;
    bool is_lower = true;

    for (auto j : column_range()) {
        for (auto i : row_indices(j)) {
            if (i > j) {
                is_upper = false;
            } else if (i < j) {
                is_lower = false;
            }
        }
    }

    return (is_upper ? 1 : (is_lower ? -1 : 0));
}


bool CSCMatrix::_test_sorted() const
{
    for (auto j : column_range()) {
        // Check that the column is sorted
        for (csint p = p_[j]; p < p_[j+1] - 1; ++p) {
            if (i_[p] > i_[p+1]) {
                return false;
            }
        }
    }

    return true;
}


std::pair<bool, csint> CSCMatrix::binary_search_(csint i, csint j) const
{
    // Binary search for t <= i
    auto start = i_.cbegin() + p_[j];
    auto end = i_.cbegin() + p_[j+1];
    auto t = std::lower_bound(start, end, i);
    // Check that we actually found the index t == i
    bool found = (t != end && *t == i);
    csint k = std::distance(i_.cbegin(), t);
    return {found, k};
}


CSCMatrix::GetItemResult CSCMatrix::get_item_(csint i, csint j) const
{
    if (i < 0 || i > M_ || j < 0 || j > N_) {
        throw std::out_of_range("Index out of bounds.");
    }

    double v = 0.0;
    bool found = false;
    csint k = -1;

    if (has_sorted_indices_) {
        std::tie(found, k) = binary_search_(i, j);

        if (found) {
            // Get the value
            if (has_canonical_format_) {
                v = v_[k];
            } else {
                // Sum duplicate entries, k points to the first entry
                csint i_size = static_cast<csint>(i_.size());;
                for (csint p = k; p < i_size && i_[p] == i; ++p) {
                    v += v_[p];
                }
            }
        }
    } else {
        // NOTE this code assumes that columns are *not* sorted, and that
        // duplicate entries may exist, so it will search through *every*
        // element in a column.
        for (auto [p, ip, vp] : enum_column(j)) {
            if (ip == i) {
                if (!found) {
                    found = true;
                    k = p;  // store the minimum index of the element
                }
                v += vp;  // sum duplicate entries
            }
        }
    }

    if (!found) {
        k = p_[j];  // insert at the beginning of the column
    }

    return {v, found, k};
}


void CSCMatrix::set_item_(csint i, csint j, double v)
{
    if (i < 0 || i > M_ || j < 0 || j > N_) {
        throw std::out_of_range("Index out of bounds.");
    }

    if (has_sorted_indices_) {
        auto [found, k] = binary_search_(i, j);

        // Check that we actually found the index t == i
        if (found) {
            v_[k] = v;  // update the value

            if (!has_canonical_format_) {
                // Duplicates may exist, so zero them out
                csint i_size = static_cast<csint>(i_.size());
                for (csint p = k + 1; p < i_size && i_[p] == i; ++p) {
                    v_[p] = 0.0;
                }
            }
        } else {
            // Value does not exist, so insert it
            insert_(i, j, v, k);
        }

    } else {
        // Linear search for the element (+ duplicates)
        csint k;
        bool found = false;

        for (auto [p, ip, vp] : enum_column(j)) {
            if (ip == i) {
                if (!found) {
                    k = p;  // store the minimum index of the element
                    found = true;
                } else {
                    vp = 0;  // zero out duplicates
                }
            }
        }

        if (found) {
            v_[k] = v;
        } else {
            // Columns are not sorted, so insert item at the end of the column.
            insert_(i, j, v, p_[j+1]);
        }
    }

    if (v == 0.0) {
        has_canonical_format_ = false;  // explicit zero
    }
}


// Exercise 2.25
CSCMatrix& CSCMatrix::assign(csint i, csint j, double v)
{
    set_item_(i, j, v);
    return *this;
}


// Exercise 2.25
CSCMatrix& CSCMatrix::assign(
    std::span<const csint> rows,
    std::span<const csint> cols,
    std::span<const double> C
)
{
    if (C.size() != rows.size() * cols.size()) {
        throw std::invalid_argument("Input matrix must be of size M x N.");
    }

    csint rows_size = static_cast<csint>(rows.size());
    csint cols_size = static_cast<csint>(cols.size());
    for (csint i = 0; i < rows_size; ++i) {
        for (csint j = 0; j < cols_size; ++j) {
            set_item_(rows[i], cols[j], C[i + j * rows_size]);
        }
    }

    return *this;
}


CSCMatrix& CSCMatrix::assign(
    std::span<const csint> rows,
    std::span<const csint> cols,
    const CSCMatrix& C
)
{
    csint rows_size = static_cast<csint>(rows.size());
    csint cols_size = static_cast<csint>(cols.size());

    if (C.M_ != rows_size) {
        throw std::invalid_argument(
            std::format(
                "rows must have same number of rows as C. Got {} and {}.",
                rows_size, C.M_
            )
        );
    }

    if (C.N_ != cols_size) {
        throw std::invalid_argument(
            std::format(
                "cols must have same number of columns as C. Got {} and {}.",
                cols_size, C.N_
            )
        );
    }

    for (csint j = 0; j < cols_size; ++j) {
        for (csint p = C.p_[j]; p < C.p_[j+1]; ++p) {
            csint i = C.i_[p];
            (*this)(rows[i], cols[j]) = C.v_[p];
        }
    }

    return *this;
}


double& CSCMatrix::insert_(csint i, csint j, double v, csint p)
{
    i_.insert(i_.begin() + p, i);
    v_.insert(v_.begin() + p, v);

    // Increment all subsequent pointers
    for (csint k = j + 1; k < static_cast<csint>(p_.size()); ++k) {
        p_[k]++;
    }

    return v_[p];
}


/*------------------------------------------------------------------------------
 *     Format Operations
 *----------------------------------------------------------------------------*/
// Exercise 2.2
COOMatrix CSCMatrix::tocoo() const { return COOMatrix{*this}; }


// Exercise 2.16
std::vector<double> CSCMatrix::to_dense_vector(const DenseOrder order) const
{
    std::vector<double> A(M_ * N_, 0.0);
    csint idx;

    for (auto j : column_range()) {
        for (auto [i, v] : column(j)) {
            // Column- vs row-major order
            if (order == DenseOrder::ColMajor) {
                idx = i + j * M_;
            } else if (order == DenseOrder::RowMajor) {
                idx = j + i * N_;
            } else {
                throw std::invalid_argument("Invalid order argument.");
            }

            if (v_.empty()) {
                A[idx] = 1.0; // no values, so set to 1.0
            } else if (has_canonical_format_) {
                A[idx] = v;
            } else {
                A[idx] += v;  // account for duplicates
            }
        }
    }

    return A;
}


CSCMatrix CSCMatrix::transpose(bool values) const
{
    std::vector<csint> w(M_);   // workspace
    CSCMatrix C{{N_, M_}, nnz(), values};  // output

    // Compute number of elements in each row
    for (csint p = 0; p < nnz(); ++p)
        w[i_[p]]++;

    // Row pointers are the cumulative sum of the counts, starting with 0.
    std::partial_sum(w.cbegin(), w.cend(), C.p_.begin() + 1);
    w = C.p_;  // copy back into workspace

    for (auto j : column_range()) {
        for (auto [i, v] : column(j)) {
            // place A(i, j) as C(j, i)
            csint q = w[i]++;
            C.i_[q] = j;
            if (values) {
                C.v_[q] = v;
            }
        }
    }

    return C;
}


// Alias for transpose
CSCMatrix CSCMatrix::T() const { return this->transpose(); }


// Exercise 2.7
CSCMatrix CSCMatrix::tsort() const
{
    CSCMatrix C = this->transpose().transpose();
    C.has_sorted_indices_ = true;
    return C;
}


// Exercise 2.8
CSCMatrix& CSCMatrix::qsort()
{
    // Find maximum column size
    csint max_len = 0;
    for (auto j : column_range()) {
        max_len = std::max(max_len, p_[j+1] - p_[j]);
    }

    // Allocate workspaces
    std::vector<csint> w,
                       idx;
    std::vector<double> x;
    w.reserve(max_len);
    x.reserve(max_len);
    idx.reserve(max_len);

    for (auto j : column_range()) {
        // Pointers to the rows
        csint p = p_[j];
        csint len = p_[j+1] - p;

        // resize workspaces
        w.resize(len);
        x.resize(len);
        idx.resize(len);

        // Copy the row indices and values into the workspace
        std::copy_n(i_.cbegin() + p, len, w.begin());
        std::copy_n(v_.cbegin() + p, len, x.begin());
        std::iota(idx.begin(), idx.end(), 0);

        // argsort the rows to get indices
        std::sort(
            idx.begin(),
            idx.end(),
            [&w](csint i, csint j) { return w[i] < w[j]; }
        );

        // Re-assign the values
        for (csint i = 0; i < len; ++i) {
            i_[p + i] = w[idx[i]];
            v_[p + i] = x[idx[i]];
        }
    }

    has_sorted_indices_ = true;

    return *this;
}


// Exercise 2.11
CSCMatrix& CSCMatrix::sort()
{
    // ----- first transpose
    std::vector<csint> w(M_);   // workspace

    bool values = !v_.empty();

    CSCMatrix C{{N_, M_}, nnz(), values};  // intermediate transpose

    // Compute number of elements in each row
    for (csint p = 0; p < nnz(); ++p)
        w[i_[p]]++;

    // Row pointers are the cumulative sum of the counts, starting with 0.
    std::partial_sum(w.cbegin(), w.cend(), C.p_.begin() + 1);
    w = C.p_;  // copy back into workspace

    for (auto j : column_range()) {
        for (auto [i, v] : column(j)) {
            // place A(i, j) as C(j, i)
            csint q = w[i]++;
            C.i_[q] = j;
            if (values) {
                C.v_[q] = v;
            }
        }
    }

    // ----- second transpose
    // Copy column counts to avoid repeat work
    w = p_;

    for (auto j : C.column_range()) {
        for (auto [Ci, Cv] : C.column(j)) {
            // place C(i, j) as A(j, i)
            csint q = w[Ci]++;
            i_[q] = j;
            if (values) {
                v_[q] = Cv;
            }
        }
    }

    has_sorted_indices_ = true;

    return *this;
}


CSCMatrix& CSCMatrix::sum_duplicates()
{
    csint nz = 0;  // count actual number of non-zeros (excluding dups)
    std::vector<csint> w(M_, -1);                      // row i not yet seen

    for (auto j : column_range()) {
        csint q = nz;                                  // column j will start at q
        for (csint p = p_[j]; p < p_[j + 1]; ++p) {
            csint i = i_[p];                          // A(i, j) is nonzero
            if (w[i] >= q) {
                v_[w[i]] += v_[p];                   // A(i, j) is a duplicate
            } else {
                w[i] = nz;                          // record where row i occurs
                i_[nz] = i;                          // keep A(i, j)
                v_[nz++] = v_[p];
            }
        }
        p_[j] = q;                                    // record start of column j
    }

    p_[N_] = nz;                                     // finalize A
    realloc();

    return *this;
}


CSCMatrix& CSCMatrix::fkeep(KeepFunc fk)
{
    csint nz = 0;  // count actual number of non-zeros
    bool values = !v_.empty();

    for (auto j : column_range()) {
        csint p = p_[j];  // get current location of column j
        p_[j] = nz;       // record new location of column j
        for (; p < p_[j+1]; ++p) {
            if (fk(i_[p], j, values ? v_[p] : 1.0)) {
                if (values) {
                    v_[nz] = v_[p];  // keep A(i, j)
                }
                i_[nz++] = i_[p];
            }
        }
    }

    p_[N_] = nz;    // finalize A
    realloc();

    return *this;
};


CSCMatrix CSCMatrix::fkeep(KeepFunc fk) const
{
    CSCMatrix C{*this};
    return C.fkeep(fk);
}


CSCMatrix& CSCMatrix::dropzeros()
{
    return fkeep(
        [] ([[maybe_unused]] csint i, [[maybe_unused]] csint j, double Aij) {
            return (Aij != 0);
        }
    );
}


CSCMatrix& CSCMatrix::droptol(double tol)
{
    return fkeep(
        [tol] ([[maybe_unused]] csint i, [[maybe_unused]] csint j, double Aij) {
            return std::fabs(Aij) > tol; 
        }
    );
}


// Exercise 2.15
CSCMatrix& CSCMatrix::band(const csint kl, const csint ku)
{
    if (kl > ku) {
        throw std::invalid_argument("kl must be less than or equal to ku.");
    }
    return fkeep(
        [=](csint i, csint j, [[maybe_unused]] double Aij) {
            return (i <= (j - kl)) && (i >= (j - ku));
        }
    );
}


CSCMatrix CSCMatrix::band(const csint kl, const csint ku) const
{
    if (kl > ku) {
        throw std::invalid_argument("kl must be less than or equal to ku.");
    }
    CSCMatrix C{*this};
    return C.band(kl, ku);
}


std::vector<double> CSCMatrix::diagonal(csint k) const
{
    auto [M, N] = shape();
    csint K = std::min(M, N) - std::abs(k);

    if (K < 0) {
        return std::vector<double>{};
    }

    std::vector<double> diag(K);

    for (csint i = 0; i < K; ++i) {
        auto [row, col] = (k < 0) ? std::pair{i - k, i} : std::pair{i, i + k};
        diag[i] = get_item_(row, col).value;
    }

    return diag;
}


double CSCMatrix::structural_symmetry() const
{
    if (M_ != N_) {
        return 0.0;
    }

    csint nnz_AAT = 0;
    csint nnz_A = 0;

    for (auto j : column_range()) {
        for (auto i : row_indices(j)) {
            // Count all off-diagonal elements in A
            if (i != j) {
                ++nnz_A;
            }

            // Count paired off-diagonal elements
            if (i < j && get_item_(j, i).found) {
                nnz_AAT += 2;
            }
        }
    }

    return static_cast<double>(nnz_AAT) / nnz_A;
}


/*------------------------------------------------------------------------------
       Math Operations
----------------------------------------------------------------------------*/
template <bool Transpose = false>
static void gaxpy_check_(
    const CSCMatrix& A,
    std::span<const double> X,
    std::span<const double> Y
)
{
    // (M, N) * (N, K) + (M, K) = (M, K) if Transpose = false
    // (N, M) * (M, K) + (N, K) = (N, K) if Transpose = true
    auto [M, N] = A.shape();
    csint NxK = static_cast<csint>((!Transpose) ? X.size() : Y.size());
    csint MxK = static_cast<csint>((!Transpose) ? Y.size() : X.size());

    if (NxK % N != 0) {
        throw std::invalid_argument(
            std::format("{} size is not compatible with A: {} % {} != 0.",
                (!Transpose) ? "X" : "Y", NxK, N)
        );
    }

    if (MxK % M != 0) {
        throw std::invalid_argument(
            std::format("{} size is not compatible with A: {} % {} != 0.",
                (!Transpose) ? "Y" : "X", NxK, N)
        );
    }

    csint Kx = NxK / N;  // number of columns in X
    csint Ky = MxK / M;  // number of columns in Y

    if (Kx != Ky) {
        throw std::invalid_argument(
            std::format("X and Y have different number of columns: {} != {}.", Kx, Ky)
        );
    }
}


std::vector<double> gaxpy(
    const CSCMatrix& A,
    std::span<const double> x,
    std::span<const double> y
)
{
    gaxpy_check_(A, x, y);

    std::vector<double> out(y.begin(), y.end());  // copy the input vector

    for (auto j : A.column_range()) {
        for (auto [i, v] : A.column(j)) {
            out[i] += v * x[j];
        }
    }

    return out;
};


// Exercise 2.1
std::vector<double> gatxpy(
    const CSCMatrix& A,
    std::span<const double> x,
    std::span<const double> y
)
{
    gaxpy_check_<true>(A, x, y);

    std::vector<double> out(y.begin(), y.end());  // copy the input vector

    for (auto j : A.column_range()) {
        for (auto [i, v] : A.column(j)) {
            out[j] += v * x[i];
        }
    }

    return out;
};


// Exercise 2.3
std::vector<double> sym_gaxpy(
    const CSCMatrix& A,
    std::span<const double> x,
    std::span<const double> y
)
{
    auto [M, N] = A.shape();
    if (M != N) {
        throw std::invalid_argument("A must be square.");
    }

    gaxpy_check_(A, x, y);

    std::vector<double> out(y.begin(), y.end());  // copy the input vector

    for (auto j : A.column_range()) {
        for (auto [i, v] : A.column(j)) {
            if (i > j)
                continue;  // skip lower triangular

            // Add the upper triangular elements
            out[i] += v * x[j];

            // If off-diagonal, also add the symmetric element
            if (i < j)
                out[j] += v * x[i];
        }
    }

    return out;
};


// Exercise 2.27
std::vector<double> gaxpy_col(
    const CSCMatrix& A,
    std::span<const double> X,
    std::span<const double> Y
)
{
    gaxpy_check_(A, X, Y);

    auto [M, N] = A.shape();
    std::vector<double> out(Y.begin(), Y.end());  // copy the input matrix

    csint K = X.size() / N;  // number of columns in X

    // For each column of X
    for (csint k = 0; k < K; ++k) {
        // Compute one column of Y (see gaxpy)
        for (auto j : A.column_range()) {
            double x_val = X[j + k * N];  // cache value

            // Only compute if x_val is non-zero
            if (x_val != 0.0) {
                for (auto [i, v] : A.column(j)) {
                    // Indexing in column-major order
                    out[i + k * M] += v * x_val;
                }
            }
        }
    }

    return out;
}


// Exercise 2.27
std::vector<double> gaxpy_block(
    const CSCMatrix& A,
    std::span<const double> X,
    std::span<const double> Y
)
{
    gaxpy_check_(A, X, Y);

    auto [M, N] = A.shape();
    std::vector<double> out(Y.begin(), Y.end());  // copy the input matrix

    csint K = X.size() / N;  // number of columns in X

    const csint BLOCK_SIZE = 32;  // block size for column operations

    // For each column of X
    for (csint k = 0; k < K; ++k) {
        // Take a block of columns
        for (csint j_start = 0; j_start < N; j_start += BLOCK_SIZE) {
            csint j_end = std::min(j_start + BLOCK_SIZE, N);
            // Compute one column of Y (see gaxpy)
            for (csint j = j_start; j < j_end; ++j) {
                double x_val = X[j + k * N];  // cache value

                // Only compute if x_val is non-zero
                if (x_val != 0.0) {
                    for (auto [i, v] : A.column(j)) {
                        // Indexing in column-major order
                        out[i + k * M] += v * x_val;
                    }
                }
            }
        }
    }

    return out;
}


// Exercise 2.27
std::vector<double> gaxpy_row(
    const CSCMatrix& A,
    std::span<const double> X,
    std::span<const double> Y
)
{
    gaxpy_check_(A, X, Y);

    auto [M, N] = A.shape();
    std::vector<double> out(Y.begin(), Y.end());  // copy the input matrix

    csint K = X.size() / N;  // number of columns in X

    // For each column of X
    for (csint k = 0; k < K; ++k) {
        // Compute one column of Y (see gaxpy)
        for (auto j : A.column_range()) {
            double x_val = X[k + j * K];  // cache value (row-major indexing)

            // Only compute if x_val is non-zero
            if (x_val != 0.0) {
                for (auto [i, v] : A.column(j)) {
                    // Indexing in row-major order
                    out[k + i * K] += v * x_val;
                }
            }
        }
    }

    return out;
}


// Exercise 2.28
std::vector<double> gatxpy_col(
    const CSCMatrix& A,
    std::span<const double> X,
    std::span<const double> Y
)
{
    gaxpy_check_<true>(A, X, Y);

    auto [M, N] = A.shape();
    std::vector<double> out(Y.begin(), Y.end());  // copy the input matrix

    csint K = X.size() / M;  // number of columns in X

    // For each column of X
    for (csint k = 0; k < K; ++k) {
        // Compute one column of Y (see gaxpy)
        for (auto j : A.column_range()) {
            for (auto [i, v] : A.column(j)) {
                // Indexing in column-major order
                out[j + k * N] += v * X[i + k * M];
            }
        }
    }

    return out;
}


// Exercise 2.28
std::vector<double> gatxpy_block(
    const CSCMatrix& A,
    std::span<const double> X,
    std::span<const double> Y
)
{
    gaxpy_check_<true>(A, X, Y);

    auto [M, N] = A.shape();
    std::vector<double> out(Y.begin(), Y.end());  // copy the input matrix

    csint K = X.size() / M;  // number of columns in X

    const csint BLOCK_SIZE = 32;  // block size for column operations

    // For each column of X
    for (csint k = 0; k < K; ++k) {
        // Take a block of columns
        for (csint j_start = 0; j_start < N; j_start += BLOCK_SIZE) {
            csint j_end = std::min(j_start + BLOCK_SIZE, N);
            // Compute one column of Y (see gaxpy)
            for (csint j = j_start; j < j_end; ++j) {
                for (auto [i, v] : A.column(j)) {
                    // Indexing in column-major order
                    out[j + k * N] += v * X[i + k * M];
                }
            }
        }
    }

    return out;
}


// Exercise 2.28
std::vector<double> gatxpy_row(
    const CSCMatrix& A,
    std::span<const double> X,
    std::span<const double> Y
)
{
    gaxpy_check_<true>(A, X, Y);

    auto [M, N] = A.shape();
    std::vector<double> out(Y.begin(), Y.end());  // copy the input matrix

    csint K = X.size() / M;  // number of columns in X

    // For each column of X
    for (csint k = 0; k < K; ++k) {
        // Compute one column of Y (see gaxpy)
        for (auto j : A.column_range()) {
            for (auto [i, v] : A.column(j)) {
                // Indexing in row-major order
                out[k + j * K] += v * X[k + i * K];
            }
        }
    }

    return out;
}


// Exercise 2.4
CSCMatrix CSCMatrix::scale(std::span<const double> r, std::span<const double> c) const
{
    if (static_cast<csint>(r.size()) != M_) {
        throw std::invalid_argument(
            std::format(
                "Row scaling vector size must match number of matrix rows."
                "Got {} and {}.",
                r.size(), M_
            )
        );
    }

    if (static_cast<csint>(c.size()) != N_) {
        throw std::invalid_argument(
            std::format(
                "Column scaling vector size must match number of matrix columns."
                "Got {} and {}.",
                c.size(), N_
            )
        );
    }

    CSCMatrix out{*this};

    for (auto j : column_range()) {
        for (auto [p, i] : enum_row_indices(j)) {
            out.v_[p] *= r[i] * c[j];
        }
    }

    return out;
}


std::vector<double> CSCMatrix::dot(std::span<const double> X) const
{
    csint NxK = static_cast<csint>(X.size());

    if (NxK % N_ != 0) {
        throw std::invalid_argument(
            std::format(
                "Input vector size must be a multiple of number of matrix columns."
                "{} % {} != 0.",
                NxK, N_
            )
        );
    }

    csint K = NxK / N_;  // number of columns in X

    std::vector<double> out(M_ * K);

    for (csint k = 0; k < K; ++k) {
        // Compute one column of output
        for (auto j : column_range()) {
            for (auto [i, v] : column(j)) {
                csint idx = i + k * M_;  // column-major input/output order
                csint jdx = j + k * N_;
                out[idx] += v * X[jdx];
            }
        }
    }

    return out;
}


CSCMatrix CSCMatrix::dot(const double c) const
{
    CSCMatrix out{v_, i_, p_, shape()};
    out.v_ *= c;
    return out;
}


CSCMatrix CSCMatrix::dot(const CSCMatrix& B) const
{
    auto [M, Ka] = shape();
    auto [Kb, N] = B.shape();

    if (Ka != Kb) {
        throw std::invalid_argument(
            std::format("Inner matrix dimensions do not agree. Got {} and {}.", Ka, Kb)
        );
    }

    bool values = !v_.empty() && !B.v_.empty();

    // NOTE See Problem 2.20 for how to compute nnz(A*B)
    CSCMatrix C{{M, N}, nnz() + B.nnz(), values};  // output

    // Allocate workspaces
    std::vector<csint> w(M);
    std::vector<double> x;
    if (values) {
        x.resize(M);
    }

    csint nz = 0;  // track total number of non-zeros in C

    bool fs = true;  // Exercise 2.19 -- first call to scatter

    for (csint j = 0; j < N; ++j) {
        if (nz + M > C.nzmax()) {
            C.realloc(2 * C.nzmax() + M);  // double the size of C
        }

        C.p_[j] = nz;  // column j of C starts here

        // Compute x = A @ B[:, j]
        for (auto [Bi, Bv] : B.column(j)) {
            // Compute x += A[:, B.i_[p]] * B.v_[p]
            nz = scatter(Bi, values ? Bv : 1, w, x, j+1, C, nz, fs);
            fs = false;
        }

        // Gather values into the correct locations in C
        if (values) {
            for (csint p = C.p_[j]; p < nz; ++p) {
                C.v_[p] = x[C.i_[p]];
            }
        }
    }

    // Finalize and deallocate unused memory
    C.p_[N] = nz;
    C.realloc();

    return C;
}


// Exercise 2.20
CSCMatrix CSCMatrix::dot_2x(const CSCMatrix& B) const
{
    auto [M, Ka] = shape();
    auto [Kb, N] = B.shape();
    if (Ka != Kb) {
        throw std::invalid_argument(
            std::format("Inner matrix dimensions do not agree. Got {} and {}.", Ka, Kb)
        );
    }

    // Allocate workspace
    std::vector<csint> w(M);

    // Compute nnz(A*B) by counting non-zeros in each column of C
    csint nz_C = 0;

    for (csint j = 0; j < N; ++j) {
        csint mark = j + 1;
        for (csint p = B.p_[j]; p < B.p_[j+1]; ++p) {
            // Scatter, but without x or C
            csint k = B.i_[p];  // B(k, j) is non-zero
            for (csint pa = p_[k]; pa < p_[k+1]; ++pa) {
                csint i = i_[pa];     // A(i, k) is non-zero
                if (w[i] < mark) {
                    w[i] = mark;     // i is new entry in column k
                    ++nz_C;         // count non-zeros in C, but don't compute
                }
            }
        }
    }

    // Allocate the correct size output matrix
    CSCMatrix C{{M, N}, nz_C};

    // Compute the actual multiplication
    std::fill(w.begin(), w.end(), 0);  // reset workspace
    std::vector<double> x(M);

    csint nz = 0;  // track total number of non-zeros in C
    bool fs = true;  // first call to scatter

    for (csint j = 0; j < N; ++j) {
        C.p_[j] = nz;  // column j of C starts here

        // Compute x = A @ B[:, j]
        for (csint p = B.p_[j]; p < B.p_[j+1]; ++p) {
            // Compute x += A[:, B.i_[p]] * B.v_[p]
            nz = scatter(B.i_[p], B.v_[p], w, x, j+1, C, nz, fs);
            fs = false;
        }

        // Gather values into the correct locations in C
        for (csint p = C.p_[j]; p < nz; ++p) {
            C.v_[p] = x[C.i_[p]];
        }
    }

    // Finalize and deallocate unused memory
    C.p_[N] = nz;
    C.realloc();

    return C;
}


// Exercise 2.18
double CSCMatrix::vecdot(const CSCMatrix& y) const
{
    if ((N_ != 1) || (y.N_ != 1)) {
        throw std::invalid_argument("Both inputs must be column vectors.");
    }

    if (M_ != y.M_) {
        throw std::invalid_argument(
            std::format("Vector lengths must match. Got {} and {}.", M_, y.M_)
        );
    }

    double z = 0.0;

    if (has_sorted_indices_ && y.has_sorted_indices_) {
        csint p = 0, q = 0;  // pointer to row index of each vector

        while ((p < nnz()) && (q < y.nnz())) {
            csint i = i_[p];    // row index of each vector
            csint j = y.i_[q];

            if (i == j) {
                z += v_[p++] * y.v_[q++];
            } else if (i < j) {
                ++p;
            } else {  // (j < i)
                ++q;
            }
        }

    } else {  // unsorted indices
        std::vector<double> w(M_);  // workspace

        // Expand this vector
        for (csint p = 0; p < nnz(); ++p) {
            w[i_[p]] = v_[p];
        }

        // Multiply by non-zero entries in y and sum
        for (csint q = 0; q < y.nnz(); ++q) {
            csint i = y.i_[q];
            if (w[i] != 0) {
                z += w[i] * y.v_[q];
            }
        }
    }

    return z;
}


CSCMatrix operator*(const CSCMatrix& A, const double c) { return A.dot(c); }
CSCMatrix operator*(const double c, const CSCMatrix& A) { return A.dot(c); }
CSCMatrix operator*(const CSCMatrix& A, const CSCMatrix& B) { return A.dot(B); }


CSCMatrix add_scaled(
    const CSCMatrix& A,
    const CSCMatrix& B,
    double alpha=1.0,
    double beta=1.0
    )
{
    if (A.shape() != B.shape()) {
        throw std::invalid_argument("Matrix dimensions do not agree.");
    }

    auto [M, N] = A.shape();

    bool values = !A.v_.empty() && !B.v_.empty();

    CSCMatrix C{{M, N}, A.nnz() + B.nnz(), values};  // output

    // Allocate workspaces
    std::vector<csint> w(M);
    std::vector<double> x;
    if (values) {
        x.resize(M);
    }

    csint nz = 0;    // track total number of non-zeros in C
    bool fs = true;  // Exercise 2.19 -- first call to scatter

    for (csint j = 0; j < N; ++j) {
        C.p_[j] = nz;  // column j of C starts here
        nz = A.scatter(j, alpha, w, x, j+1, C, nz, fs);  // alpha * A(:, j)
        fs = false;
        nz = B.scatter(j,  beta, w, x, j+1, C, nz, fs);  //  beta * B(:, j)

        // Gather results into the correct column of C
        // TODO write C.gather(j, x, nz) as member function?
        // nz == C.i_.size() - C.p_[j]
        if (values) {
            for (csint p = C.p_[j]; p < nz; ++p) {
                C.v_[p] = x[C.i_[p]];
            }
        }
    }

    // Finalize and deallocate unused memory
    C.p_[N] = nz;
    C.realloc();

    return C;
}


// Exercise 2.21
void saxpy(
    const CSCMatrix& a,
    const CSCMatrix& b,
    std::span<char> w,
    std::span<double> x
    )
{
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Matrix dimensions do not agree.");
    }

    if ((a.shape()[1] != 1) || (b.shape()[1] != 1)) {
        throw std::invalid_argument("Both inputs must be column vectors.");
    }

    for (auto [i, v] : a.column(0)) {
        w[i] = true;  // mark as non-zero
        x[i] = v;     // copy a into x
    }

    for (auto [i, v] : b.column(0)) {
        if (w[i] == false) {
            w[i] = true;  // mark as non-zero
            x[i] = v;     // copy b into w
        } else {
            x[i] += v;    // add b to x
        }
    }
}


/** Add a matrix B. */
CSCMatrix CSCMatrix::add(const CSCMatrix& B) const
{
    return add_scaled(*this, B, 1.0, 1.0);
}


CSCMatrix CSCMatrix::subtract(const CSCMatrix& B) const
{
    return add_scaled(*this, B, 1.0, -1.0);
}


CSCMatrix operator+(const CSCMatrix& A, const CSCMatrix& B) { return A.add(B); }
CSCMatrix operator-(const CSCMatrix& A, const CSCMatrix& B) { return A.subtract(B); }

CSCMatrix operator-(const CSCMatrix& A) {
    CSCMatrix C = A;
    for (auto& val : C.v_) {
        val = -val;
    }
    return C;
}


csint CSCMatrix::scatter(
    csint j,
    double beta,
    std::span<csint> w,
    std::span<double> x,
    csint mark,
    CSCMatrix& C,
    csint nz,
    bool fs  // Exercise 2.19
) const
{
    // Check if x is passed as a reference
    bool values = !v_.empty() && !x.empty();

    // Exercise 2.19
    if (fs) {
        // If it's the first call, we can just copy the (scaled) column
        for (auto [i, v] : column(j)) {  // A(i, j) is non-zero
            w[i] = mark;          // i is new entry in column j
            C.i_[nz++] = i;       // add i to sparsity pattern of C(:, j)
            if (values) {
                x[i] = beta * v;  // x = beta * A(i, j)
            }
        }
    } else {
        // Original scatter
        // Otherwise, we need to accumulate the values
        for (auto [i, v] : column(j)) {  // A(i, j) is non-zero
            if (w[i] < mark) {
                w[i] = mark;           // i is new entry in column j
                C.i_[nz++] = i;        // add i to pattern of C(:, j)
                if (values) {
                    x[i] = beta * v;   // x = beta * A(i, j)
                }
            } else {
                if (values) {
                    x[i] += beta * v;  // i exists in C(:, j) already
                }
            }
        }
    }

    return nz;
}


void CSCMatrix::scatter(csint k, std::span<double> x) const
{
    if (static_cast<csint>(x.size()) != M_) {
        throw std::invalid_argument(
            std::format(
                "Input vector size must match number of matrix rows."
                "Got {} and {}.",
                x.size(), M_
            )
        );
    }
    
    if (k < 0 || k >= N_) {
        throw std::out_of_range(
            std::format("Column index k = {} is out of range [0, {}).", k, N_)
        );
    }

    for (csint p = p_[k]; p < p_[k+1]; ++p) {
        x[i_[p]] += v_[p];  // accumulate duplicate entries
    }
}


/*------------------------------------------------------------------------------
 *         Permutations 
 *----------------------------------------------------------------------------*/
CSCMatrix CSCMatrix::permute(
    std::span<const csint> p_inv,
    std::span<const csint> q,
    bool values
) const
{
    CSCMatrix C{{M_, N_}, nnz(), values};
    csint nz = 0;

    for (auto k : column_range()) {
        C.p_[k] = nz;                   // column k of C is column q[k] of A
        csint j = q.empty() ? k : q[k];

        for (csint t = p_[j]; t < p_[j+1]; ++t) {
            if (values) {
                C.v_[nz] = v_[t];       // row i of A is row p_inv[i] of C
            }
            csint i = i_[t];
            C.i_[nz++] = p_inv.empty() ? i : p_inv[i];
        }
    }

    C.p_[N_] = nz;

    return C;
}


CSCMatrix CSCMatrix::permute_rows(std::span<const csint> p_inv, bool values) const
{
    return permute(p_inv, {}, values);
}


CSCMatrix CSCMatrix::permute_cols(std::span<const csint> q, bool values) const
{
    return permute({}, q, values);
}


CSCMatrix CSCMatrix::symperm(std::span<const csint> p_inv, bool values) const
{
    if (M_ != N_) {  // matrix must be square. Symmetry not checked.
        throw std::invalid_argument("Matrix must be square.");
    }

    CSCMatrix C{{N_, N_}, nnz(), values};
    std::vector<csint> w(N_);  // workspace for column counts

    // Count entries in each column of C
    for (auto j : column_range()) {
        csint j2 = p_inv.empty() ? j : p_inv[j];  // column j of A is column j2 of C

        for (auto i : row_indices(j)) {
            if (i > j) {
                continue;   // skip lower triangular part of A
            }

            csint i2 = p_inv.empty() ? i : p_inv[i];    // row i of A is row i2 of C
            w[std::max(i2, j2)]++;  // column count of C
        }
    }

    // Row pointers are the cumulative sum of the counts, starting with 0.
    std::partial_sum(w.cbegin(), w.cend(), C.p_.begin() + 1);
    w = C.p_;  // copy back into workspace

    for (auto j : column_range()) {
        csint j2 = p_inv.empty() ? j : p_inv[j];  // column j of A is column j2 of C

        for (auto [i, v] : column(j)) {
            if (i > j) {
                continue;   // skip lower triangular part of A
            }

            csint i2 = p_inv.empty() ? i : p_inv[i];  // row i of A is row i2 of C
            csint q = w[std::max(i2, j2)]++;
            C.i_[q] = std::min(i2, j2);
            if (values) {
                C.v_[q] = v;
            }
        }
    }

    return C;
}


// Exercise 2.26
CSCMatrix CSCMatrix::permute_transpose(
    std::span<const csint> p_inv,
    std::span<const csint> q_inv,
    bool values
) const
{
    std::vector<csint> w(M_);            // workspace
    CSCMatrix C{{N_, M_}, nnz(), values};  // output

    // Compute number of elements in each permuted row (aka column of C)
    for (csint p = 0; p < nnz(); ++p) {
        csint i = i_[p];
        csint idx = p_inv.empty() ? i : p_inv[i];
        w[idx]++;
    }

    std::partial_sum(w.cbegin(), w.cend(), C.p_.begin() + 1);
    w = C.p_;  // copy back into workspace

    // place A(i, j) as C(j, i) (permuted)
    for (auto j : column_range()) {
        for (auto [i, v] : column(j)) {
            csint idx = p_inv.empty() ? i : p_inv[i];
            csint t = w[idx]++;
            C.i_[t] = q_inv.empty() ? j : q_inv[j];
            if (values) {
                C.v_[t] = v;
            }
        }
    }

    return C;
}


double CSCMatrix::norm() const
{
    double the_norm = 0;

    for (auto j : column_range()) {
        double s = 0;

        for (auto v : col_values(j)) {
            s += std::fabs(v);
        }

        the_norm = std::max(the_norm, s);
    }

    return the_norm;
}


double CSCMatrix::fronorm() const
{
    double sumsq = 0;

    // Sum the squares of the entries in v_
    for (const auto& v : v_) {
        sumsq += v * v;
    }

    return std::sqrt(sumsq);
}


// Exercise 2.12
bool CSCMatrix::is_valid(const bool sorted, const bool values) const
{
    // Check number of columns
    if (static_cast<csint>(p_.size()) != (N_ + 1)) {
        throw std::runtime_error("Number of columns inconsistent!");
    }

    if (p_.front() != 0) {
        throw std::runtime_error("First column index should be 0!");
    }

    if (p_.back() != nnz()) {
        throw std::runtime_error("Column counts inconsistent!");
    }

    // Check array sizes
    if (values) {
        if (v_.empty()) {
            throw std::runtime_error("No values!");
        }

        if (i_.size() != v_.size()) {
            throw std::runtime_error("Indices and values sizes inconsistent!");
        }
    }

    // See also: `scipy.sparse._compressed._cs_matrix.check_format` 
    //  for O(1) and O(|A|) checks.
    for (auto j : column_range()) {
        for (auto p : indptr_range(j)) {
            csint i = i_[p];

            if (i < 0 || i > M_) {
                throw std::runtime_error("Invalid row index!");
            }

            if (sorted && (p < (p_[j+1] - 1)) && (i > i_[p+1])) {
                throw std::runtime_error("Columns not sorted!");
            }

            if (values && v_[p] == 0) {
                throw std::runtime_error("Explicit zeros!");
            }

            // Quick check for duplicates, but won't catch all cases
            if ((p < (p_[j+1] - 1)) && (i == i_[p+1])) {
                throw std::runtime_error("Duplicate entries exist!");
            }
        }

        // At this point, matrix is valid up to column j
        // Check for duplicates in the column
        if (sorted) {
            for (csint p = p_[j]; p < p_[j+1]-1; ++p) {
                if (i_[p] == i_[p+1]) {
                    throw std::runtime_error("Duplicate entries exist!");
                }
            }
        }
    }

    return true;
}


// Exercise 2.22
CSCMatrix hstack(const CSCMatrix& A, const CSCMatrix& B)
{
    if (A.M_ != B.M_) {
        throw std::invalid_argument("Matrix row dimensions do not agree.");
    }

    // Copy the first matrix
    CSCMatrix C = A;
    C.N_ += B.N_;
    C.realloc(A.nnz() + B.nnz());

    // Copy the second matrix
    for (auto j : B.column_range()) {
        C.p_[A.N_ + j] = B.p_[j] + A.nnz();
    }

    std::copy(B.i_.cbegin(), B.i_.cend(), C.i_.begin() + A.nnz());
    std::copy(B.v_.cbegin(), B.v_.cend(), C.v_.begin() + A.nnz());

    C.p_[C.N_] = A.nnz() + B.nnz();

    if (!A.has_canonical_format_ || !B.has_canonical_format_) {
        C = C.to_canonical();
    }
    C.has_canonical_format_ = true;

    return C;
}


// Exercise 2.22
CSCMatrix vstack(const CSCMatrix& A, const CSCMatrix& B)
{
    if (A.N_ != B.N_) {
        throw std::invalid_argument("Matrix column dimensions do not agree.");
    }

    CSCMatrix C{{A.M_ + B.M_, A.N_}, A.nnz() + B.nnz()};

    csint nz = 0;

    for (auto j : C.column_range()) {
        C.p_[j] = nz;  // column j of C starts here

        // Copy column j from the first matrix
        for (csint p = A.p_[j]; p < A.p_[j+1]; ++p) {
            C.i_[nz] = A.i_[p];
            C.v_[nz] = A.v_[p];
            ++nz;
        }

        // Copy column j from the second matrix
        for (csint p = B.p_[j]; p < B.p_[j+1]; ++p) {
            C.i_[A.p_[j+1] + p] = A.M_ + B.i_[p];
            C.v_[A.p_[j+1] + p] = B.v_[p];
            ++nz;
        }
    }

    C.p_[C.N_] = nz;

    return C.to_canonical();
}


// Exercise 2.23
CSCMatrix CSCMatrix::slice(
    const csint i_start,
    const csint i_end,
    const csint j_start,
    const csint j_end
    ) const
{
    if ((i_start < 0) || (i_end > M_) || (i_start > i_end)) {
        throw std::out_of_range(
            std::format(
                "Row slice [{}, {}) is out of range [0, {}].",
                i_start, i_end, M_
            )
        );
    }

    if ((j_start < 0) || (j_end > N_) || (j_start > j_end)) {
        throw std::out_of_range(
            std::format(
                "Column slice [{}, {}) is out of range [0, {}].",
                j_start, j_end, N_
            )
        );
    }

    CSCMatrix C{{i_end - i_start, j_end - j_start}, nnz()};

    csint nz = 0;

    for (csint j = j_start; j < j_end; ++j) {
        C.p_[j - j_start] = nz;  // column j of C starts here

        for (auto [i, v] : column(j)) {
            if ((i >= i_start) && (i < i_end)) {
                C.i_[nz] = i - i_start;
                C.v_[nz] = v;
                ++nz;
            }
        }
    }

    C.p_[C.N_] = nz;
    C.realloc();

    if (!has_canonical_format_) {
        C = C.to_canonical();
    }
    C.has_canonical_format_ = true;

    return C;
}


// Exercise 2.24
CSCMatrix CSCMatrix::index(
    std::span<const csint> rows,
    std::span<const csint> cols
    ) const
{
    csint M = rows.size();
    csint N = cols.size();
    CSCMatrix C{{M, N}, nnz()};

    csint nz = 0;

    for (csint j = 0; j < N; ++j) {
        C.p_[j] = nz;  // column j of C starts here

        // Iterate over `rows` and find the corresponding indices in `i_`.
        for (csint k = 0; k < M; ++k) {
            double val = (*this)(rows[k], cols[j]);
            if (val != 0) {
                C.i_[nz] = k;
                C.v_[nz] = val;
                ++nz;
            }
        }
    }

    C.p_[C.N_] = nz;
    C.realloc();
    
    // Canonical format guaranteed by construction
    C.has_canonical_format_ = true;

    return C;
}


// Exercise 2.29
CSCMatrix& CSCMatrix::add_empty_top(const csint k)
{
    M_ += k;

    // Increate all row indices by k
    for (auto& i : i_) {
        i += k;
    }

    return *this;
}


// Exercise 2.29
CSCMatrix& CSCMatrix::add_empty_bottom(const csint k)
{
    M_ += k;
    return *this;
}


// Exercise 2.29
CSCMatrix& CSCMatrix::add_empty_left(const csint k)
{
    N_ += k;
    p_.insert(p_.begin(), k, 0);  // insert k zeros at the beginning
    return *this;
}


// Exercise 2.29
CSCMatrix& CSCMatrix::add_empty_right(const csint k)
{
    N_ += k;
    p_.insert(p_.end(), k, nnz());  // insert k nnz() at the end
    return *this;
}


std::vector<double> CSCMatrix::sum_rows() const
{
    std::vector<double> out(M_, 0.0);

    for (auto j : column_range()) {
        for (auto [i, v] : column(j)) {
            out[i] += v;
        }
    }

    return out;
}


std::vector<double> CSCMatrix::sum_cols() const
{
    std::vector<double> out(N_, 0.0);

    for (auto j : column_range()) {
        for (auto v : col_values(j)) {
            out[j] += v;
        }
    }

    return out;
}


/*------------------------------------------------------------------------------
 *         Printing
 *----------------------------------------------------------------------------*/
void CSCMatrix::write_elems_(std::stringstream& ss, csint start, csint end) const
{
    const std::string format_string = make_format_string_();

    // Compute index width from maximum index
    int row_width = std::to_string(M_ - 1).size();
    int col_width = std::to_string(N_ - 1).size();

    csint n = 0;  // number of elements printed
    for (auto j : column_range()) {
        for (auto [i, v] : column(j)) {
            if ((n >= start) && (n < end)) {
                ss << std::vformat(
                    format_string,
                    std::make_format_args(i, row_width, j, col_width, v)
                );

                if (n < end - 1) {
                    ss << "\n";
                }
            }
            ++n;
        }
    }
}


}  // namespace cs

/*==============================================================================
 *============================================================================*/
