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
#include <functional>  // reference_wrapper
#include <new>         // bad_alloc
#include <numeric>     // iota
#include <optional>    // optional
#include <ranges>      // views::reverse
#include <sstream>
#include <string>
#include <vector>

#include "utils.h"
#include "csc.h"
#include "coo.h"

namespace cs {

/*------------------------------------------------------------------------------
 *     Constructors
 *----------------------------------------------------------------------------*/
CSCMatrix::CSCMatrix() {};


CSCMatrix::CSCMatrix(
    const std::vector<double>& data,
    const std::vector<csint>& indices,
    const std::vector<csint>& indptr,
    const Shape& shape
    )
    : v_(data),
      i_(indices),
      p_(indptr),
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
    } else {
        v_.resize(0);
        v_.shrink_to_fit();
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
    const std::vector<double>& A,
    const Shape& shape,
    const char order
) : M_(shape[0]),
    N_(shape[1])
{
    assert(A.size() == (M_ * N_));  // ensure input is valid

    if (order != 'C' && order != 'F') {
        throw std::invalid_argument("Order must be 'C' or 'F'.");
    }

    // Allocate memory
    v_.reserve(A.size());
    i_.reserve(A.size());
    p_.reserve(N_);

    csint nz = 0;  // count number of non-zeros

    for (csint j = 0; j < N_; j++) {
        p_.push_back(nz);

        double val;
        for (csint i = 0; i < M_; i++) {
            if (order == 'F') {
                val = A[i + j * M_];  // linear index for column-major order
            } else {
                val = A[j + i * N_];  // linear index for row-major order
            }

            // Only store non-zeros
            if (val != 0.0) {
                i_.push_back(i);
                v_.push_back(val);
                nz++;
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
Shape CSCMatrix::shape() const { return Shape {M_, N_}; }

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

    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            csint i = i_[p];

            if (i == j)
                continue;  // skip diagonal

            if (v_[p] != (*this)(j, i))
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

    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            csint i = i_[p];

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
    for (csint j = 0; j < N_; j++) {
        // Check that the column is sorted
        for (csint p = p_[j]; p < p_[j+1] - 1; p++) {
            if (i_[p] > i_[p + 1]) {
                return false;
            }
        }
    }

    return true;
}


const double CSCMatrix::operator()(csint i, csint j) const
{
    // Assert indices are in-bounds
    assert(i >= 0 && i < M_);
    assert(j >= 0 && j < N_);

    if (has_canonical_format_) {
        // Binary search for t <= i
        auto start = i_.begin() + p_[j];
        auto end = i_.begin() + p_[j+1];

        auto t = std::lower_bound(start, end, i);

        // Check that we actually found the index t == i
        if (t != end && *t == i) {
            return v_[std::distance(i_.begin(), t)];
        } else {
            return 0.0;
        }

    } else {
        // NOTE this code assumes that columns are *not* sorted, and that
        // duplicate entries may exist, so it will search through *every*
        // element in a column.
        double out = 0.0;

        for (csint p = p_[j]; p < p_[j+1]; p++) {
            if (i_[p] == i) {
                out += v_[p];  // sum duplicate entries
            }
        }

        return out;
    }
}


double& CSCMatrix::operator()(const csint i, const csint j)
{
    // Assert indices are in-bounds
    assert(i >= 0 && i < M_);
    assert(j >= 0 && j < N_);

    if (has_canonical_format_) {
        // Binary search for t <= i
        auto start = i_.begin() + p_[j];
        auto end = i_.begin() + p_[j+1];

        auto t = std::lower_bound(start, end, i);

        auto k = std::distance(i_.begin(), t);

        // Check that we actually found the index t == i
        if (t != end && *t == i) {
            return v_[k];
        } else {
            // Value does not exist, so add a place-holder here.
            return this->insert(i, j, 0.0, k);
        }

    } else {
        // Linear search for the element
        csint k;
        bool found = false;

        for (csint p = p_[j]; p < p_[j+1]; p++) {
            if (i_[p] == i) {
                if (!found) {
                    k = p;  // store the index of the element
                    found = true;
                } else {
                    v_[k] += v_[p];  // accumulate duplicates here
                    v_[p] = 0;       // zero out duplicates
                }
            }
        }

        if (found) {
            return v_[k];
        } else {
            // Columns are not sorted, so we can just insert the element at the
            // beginning of the column.
            return this->insert(i, j, 0.0, p_[j]);
        }
    }
}


// Exercise 2.25
CSCMatrix& CSCMatrix::assign(csint i, csint j, double v)
{
    (*this)(i, j) = v;
    return *this;
}


// Exercise 2.25
CSCMatrix& CSCMatrix::assign(
    const std::vector<csint>& rows,
    const std::vector<csint>& cols,
    const std::vector<double>& C
    )
{
    assert(C.size() == rows.size() * cols.size());

    for (csint i = 0; i < rows.size(); i++) {
        for (csint j = 0; j < cols.size(); j++) {
            (*this)(rows[i], cols[j]) = C[i + j * rows.size()];
        }
    }

    return *this;
}


CSCMatrix& CSCMatrix::assign(
    const std::vector<csint>& rows,
    const std::vector<csint>& cols,
    const CSCMatrix& C
    )
{
    assert(C.M_ == rows.size());
    assert(C.N_ == cols.size());

    for (csint j = 0; j < cols.size(); j++) {
        for (csint p = C.p_[j]; p < C.p_[j+1]; p++) {
            csint i = C.i_[p];
            (*this)(rows[i], cols[j]) = C.v_[p];
        }
    }

    return *this;
}


double& CSCMatrix::insert(csint i, csint j, double v, csint p)
{
    i_.insert(i_.begin() + p, i);
    v_.insert(v_.begin() + p, v);

    // Increment all subsequent pointers
    for (csint k = j + 1; k < p_.size(); k++) {
        p_[k]++;
    }

    return v_[p];
}


/*------------------------------------------------------------------------------
 *     Format Operations
 *----------------------------------------------------------------------------*/
// Exercise 2.2
COOMatrix CSCMatrix::tocoo() const { return COOMatrix(*this); }


// Exercise 2.16
std::vector<double> CSCMatrix::to_dense_vector(const char order) const
{
    std::vector<double> A(M_ * N_, 0.0);
    csint idx;

    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            // Column- vs row-major order
            if (order == 'F') {
                idx = i_[p] + j * M_;
            } else if (order == 'C') {
                idx = j + i_[p] * N_;
            } else {
                throw std::invalid_argument("Invalid order argument. Use 'F' or 'C'.");
            }

            if (v_.empty()) {
                A[idx] = 1.0; // no values, so set to 1.0
            } else if (has_canonical_format_) {
                A[idx] = v_[p];
            } else {
                A[idx] += v_[p];  // account for duplicates
            }
        }
    }

    return A;
}


CSCMatrix CSCMatrix::transpose(bool values) const
{
    std::vector<csint> w(M_);   // workspace
    CSCMatrix C({N_, M_}, nnz(), values);  // output

    // Compute number of elements in each row
    for (csint p = 0; p < nnz(); p++)
        w[i_[p]]++;

    // Row pointers are the cumulative sum of the counts, starting with 0.
    C.p_ = cumsum(w);
    w = C.p_;  // copy back into workspace

    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            // place A(i, j) as C(j, i)
            csint q = w[i_[p]]++;
            C.i_[q] = j;
            if (values) {
                C.v_[q] = v_[p];
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
    // Allocate workspaces
    std::vector<csint> w(nnz());
    std::vector<double> x(nnz());

    for (csint j = 0; j < N_; j++) {
        // Pointers to the rows
        csint p = p_[j],
              pn = p_[j+1];

        // clear workspaces
        w.clear();
        x.clear();

        // Copy the row indices and values into the workspace
        std::copy(i_.begin() + p, i_.begin() + pn, std::back_inserter(w));
        std::copy(v_.begin() + p, v_.begin() + pn, std::back_inserter(x));

        // argsort the rows to get indices
        std::vector<csint> idx = argsort(w);

        // Re-assign the values
        for (csint i = 0; i < idx.size(); i++) {
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

    CSCMatrix C({N_, M_}, nnz(), values);  // intermediate transpose

    // Compute number of elements in each row
    for (csint p = 0; p < nnz(); p++)
        w[i_[p]]++;

    // Row pointers are the cumulative sum of the counts, starting with 0.
    C.p_ = cumsum(w);
    w = C.p_;  // copy back into workspace

    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            // place A(i, j) as C(j, i)
            csint q = w[i_[p]]++;
            C.i_[q] = j;
            if (values) {
                C.v_[q] = v_[p];
            }
        }
    }

    // ----- second transpose
    // Copy column counts to avoid repeat work
    w = p_;

    for (csint j = 0; j < C.N_; j++) {
        for (csint p = C.p_[j]; p < C.p_[j+1]; p++) {
            // place C(i, j) as A(j, i)
            csint q = w[C.i_[p]]++;
            i_[q] = j;
            if (values) {
                v_[q] = C.v_[p];
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

    for (csint j = 0; j < N_; j++) {
        csint q = nz;                                  // column j will start at q
        for (csint p = p_[j]; p < p_[j + 1]; p++) {
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

    for (csint j = 0; j < N_; j++) {
        csint p = p_[j];  // get current location of column j
        p_[j] = nz;       // record new location of column j
        for (; p < p_[j+1]; p++) {
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
    CSCMatrix C(*this);
    return C.fkeep(fk);
}


CSCMatrix& CSCMatrix::dropzeros()
{
    return fkeep([] (csint i, csint j, double Aij) { return (Aij != 0); });
}


CSCMatrix& CSCMatrix::droptol(double tol)
{
    return fkeep(
        [tol](csint i, csint j, double Aij) {
            return std::fabs(Aij) > tol; 
        }
    );
}


// Exercise 2.15
CSCMatrix& CSCMatrix::band(const csint kl, const csint ku)
{
    assert(kl <= ku);
    return fkeep(
        [=](csint i, csint j, double Aij) {
            return (i <= (j - kl)) && (i >= (j - ku));
        }
    );
}


CSCMatrix CSCMatrix::band(const csint kl, const csint ku) const
{
    assert(kl <= ku);
    CSCMatrix C(*this);
    return C.band(kl, ku);
}


/*------------------------------------------------------------------------------
       Math Operations
----------------------------------------------------------------------------*/
std::vector<double> gaxpy(
    const CSCMatrix& A,
    const std::vector<double>& x,
    const std::vector<double>& y
)
{
    assert(A.M_ == y.size());  // addition
    assert(A.N_ == x.size());  // multiplication

    std::vector<double> out = y;  // copy the input vector

    for (csint j = 0; j < A.N_; j++) {
        for (csint p = A.p_[j]; p < A.p_[j+1]; p++) {
            out[A.i_[p]] += A.v_[p] * x[j];
        }
    }

    return out;
};


// Exercise 2.1
std::vector<double> gatxpy(
    const CSCMatrix& A,
    const std::vector<double>& x,
    const std::vector<double>& y
)
{
    assert(A.M_ == x.size());  // multiplication
    assert(A.N_ == y.size());  // addition

    std::vector<double> out = y;  // copy the input vector

    for (csint j = 0; j < A.N_; j++) {
        for (csint p = A.p_[j]; p < A.p_[j+1]; p++) {
            out[j] += A.v_[p] * x[A.i_[p]];
        }
    }

    return out;
};


// Exercise 2.3
std::vector<double> sym_gaxpy(
    const CSCMatrix& A,
    const std::vector<double>& x,
    const std::vector<double>& y
)
{
    assert(A.M_ == A.N_);  // matrix must be square to be symmetric
    assert(A.N_ == x.size());
    assert(x.size() == y.size());

    std::vector<double> out = y;  // copy the input vector

    for (csint j = 0; j < A.N_; j++) {
        for (csint p = A.p_[j]; p < A.p_[j+1]; p++) {
            csint i = A.i_[p];

            if (i > j)
                continue;  // skip lower triangular

            // Add the upper triangular elements
            out[i] += A.v_[p] * x[j];

            // If off-diagonal, also add the symmetric element
            if (i < j)
                out[j] += A.v_[p] * x[i];
        }
    }

    return out;
};


// Exercise 2.27
std::vector<double> gaxpy_col(
    const CSCMatrix& A,
    const std::vector<double>& X,
    const std::vector<double>& Y
)
{
    assert(X.size() % A.N_ == 0);  // check that X.size() is a multiple of A.N_
    assert(Y.size() == A.M_ * (X.size() / A.N_));

    std::vector<double> out = Y;  // copy the input matrix

    csint K = X.size() / A.N_;  // number of columns in X

    // For each column of X
    for (csint k = 0; k < K; k++) {
        // Compute one column of Y (see gaxpy)
        for (csint j = 0; j < A.N_; j++) {
            double x_val = X[j + k * A.N_];  // cache value

            // Only compute if x_val is non-zero
            if (x_val != 0.0) {
                for (csint p = A.p_[j]; p < A.p_[j+1]; p++) {
                    // Indexing in column-major order
                    out[A.i_[p] + k * A.M_] += A.v_[p] * x_val;
                }
            }
        }
    }

    return out;
}


// Exercise 2.27
std::vector<double> gaxpy_block(
    const CSCMatrix& A,
    const std::vector<double>& X,
    const std::vector<double>& Y
)
{
    assert(X.size() % A.N_ == 0);  // check that X.size() is a multiple of A.N_
    assert(Y.size() == A.M_ * (X.size() / A.N_));

    std::vector<double> out = Y;  // copy the input matrix

    csint K = X.size() / A.N_;  // number of columns in X

    const csint BLOCK_SIZE = 32;  // block size for column operations

    // For each column of X
    for (csint k = 0; k < K; k++) {
        // Take a block of columns
        for (csint j_start = 0; j_start < A.N_; j_start += BLOCK_SIZE) {
            csint j_end = std::min(j_start + BLOCK_SIZE, A.N_);
            // Compute one column of Y (see gaxpy)
            for (csint j = j_start; j < j_end; j++) {
                double x_val = X[j + k * A.N_];  // cache value

                // Only compute if x_val is non-zero
                if (x_val != 0.0) {
                    for (csint p = A.p_[j]; p < A.p_[j+1]; p++) {
                        // Indexing in column-major order
                        out[A.i_[p] + k * A.M_] += A.v_[p] * x_val;
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
    const std::vector<double>& X,
    const std::vector<double>& Y
)
{
    assert(X.size() % A.N_ == 0);  // check that X.size() is a multiple of A.N_
    assert(Y.size() == A.M_ * (X.size() / A.N_));

    std::vector<double> out = Y;  // copy the input matrix

    csint K = X.size() / A.N_;  // number of columns in X

    // For each column of X
    for (csint k = 0; k < K; k++) {
        // Compute one column of Y (see gaxpy)
        for (csint j = 0; j < A.N_; j++) {
            double x_val = X[k + j * K];  // cache value (row-major indexing)

            // Only compute if x_val is non-zero
            if (x_val != 0.0) {
                for (csint p = A.p_[j]; p < A.p_[j+1]; p++) {
                    // Indexing in row-major order
                    out[k + A.i_[p] * K] += A.v_[p] * x_val;
                }
            }
        }
    }

    return out;
}


// Exercise 2.28
std::vector<double> gatxpy_col(
    const CSCMatrix& A,
    const std::vector<double>& X,
    const std::vector<double>& Y
)
{
    assert(X.size() % A.M_ == 0);  // check that X.size() is a multiple of A.M_
    assert(Y.size() == A.N_ * (X.size() / A.M_));

    std::vector<double> out = Y;  // copy the input matrix

    csint K = X.size() / A.M_;  // number of columns in X

    // For each column of X
    for (csint k = 0; k < K; k++) {
        // Compute one column of Y (see gaxpy)
        for (csint j = 0; j < A.N_; j++) {
            for (csint p = A.p_[j]; p < A.p_[j+1]; p++) {
                // Indexing in column-major order
                out[j + k * A.N_] += A.v_[p] * X[A.i_[p] + k * A.M_];
            }
        }
    }

    return out;
}


// Exercise 2.28
std::vector<double> gatxpy_block(
    const CSCMatrix& A,
    const std::vector<double>& X,
    const std::vector<double>& Y
)
{
    assert(X.size() % A.M_ == 0);  // check that X.size() is a multiple of A.N_
    assert(Y.size() == A.N_ * (X.size() / A.M_));

    std::vector<double> out = Y;  // copy the input matrix

    csint K = X.size() / A.M_;  // number of columns in X

    const csint BLOCK_SIZE = 32;  // block size for column operations

    // For each column of X
    for (csint k = 0; k < K; k++) {
        // Take a block of columns
        for (csint j_start = 0; j_start < A.N_; j_start += BLOCK_SIZE) {
            csint j_end = std::min(j_start + BLOCK_SIZE, A.N_);
            // Compute one column of Y (see gaxpy)
            for (csint j = j_start; j < j_end; j++) {
                for (csint p = A.p_[j]; p < A.p_[j+1]; p++) {
                    // Indexing in column-major order
                    out[j + k * A.N_] += A.v_[p] * X[A.i_[p] + k * A.M_];
                }
            }
        }
    }

    return out;
}


// Exercise 2.28
std::vector<double> gatxpy_row(
    const CSCMatrix& A,
    const std::vector<double>& X,
    const std::vector<double>& Y
)
{
    assert(X.size() % A.M_ == 0);  // check that X.size() is a multiple of A.N_
    assert(Y.size() == A.N_ * (X.size() / A.M_));

    std::vector<double> out = Y;  // copy the input matrix

    csint K = X.size() / A.M_;  // number of columns in X

    // For each column of X
    for (csint k = 0; k < K; k++) {
        // Compute one column of Y (see gaxpy)
        for (csint j = 0; j < A.N_; j++) {
            for (csint p = A.p_[j]; p < A.p_[j+1]; p++) {
                // Indexing in row-major order
                out[k + j * K] += A.v_[p] * X[k + A.i_[p] * K];
            }
        }
    }

    return out;
}


// Exercise 2.4
CSCMatrix CSCMatrix::scale(const std::vector<double>& r, const std::vector<double> c) const
{
    assert(r.size() == M_);
    assert(c.size() == N_);

    CSCMatrix out(*this);

    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            out.v_[p] *= r[i_[p]] * c[j];
        }
    }

    return out;
}


std::vector<double> CSCMatrix::dot(const std::vector<double>& x) const
{
    assert(N_ == x.size());

    std::vector<double> out(M_);

    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            out[i_[p]] += v_[p] * x[j];
        }
    }

    return out;
}


CSCMatrix CSCMatrix::dot(const double c) const
{
    CSCMatrix out(v_, i_, p_, shape());
    out.v_ *= c;
    return out;
}


CSCMatrix CSCMatrix::dot(const CSCMatrix& B) const
{
    auto [M, Ka] = shape();
    auto [Kb, N] = B.shape();
    assert(Ka == Kb);

    bool values = !v_.empty() && !B.v_.empty();

    // NOTE See Problem 2.20 for how to compute nnz(A*B)
    CSCMatrix C({M, N}, nnz() + B.nnz(), values);  // output

    // Allocate workspaces
    std::vector<csint> w(M);
    std::vector<double> x;
    if (values) {
        x.resize(M);
    }

    csint nz = 0;  // track total number of non-zeros in C

    bool fs = true;  // Exercise 2.19 -- first call to scatter

    for (csint j = 0; j < N; j++) {
        if (nz + M > C.nzmax()) {
            C.realloc(2 * C.nzmax() + M);  // double the size of C
        }

        C.p_[j] = nz;  // column j of C starts here

        // Compute x = A @ B[:, j]
        for (csint p = B.p_[j]; p < B.p_[j+1]; p++) {
            // Compute x += A[:, B.i_[p]] * B.v_[p]
            nz = scatter(B.i_[p], values ? B.v_[p] : 1, w, x, j+1, C, nz, fs);
            fs = false;
        }

        // Gather values into the correct locations in C
        if (values) {
            for (csint p = C.p_[j]; p < nz; p++) {
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
    assert(Ka == Kb);

    // Allocate workspace
    std::vector<csint> w(M);

    // Compute nnz(A*B) by counting non-zeros in each column of C
    csint nz_C = 0;

    for (csint j = 0; j < N; j++) {
        csint mark = j + 1;
        for (csint p = B.p_[j]; p < B.p_[j+1]; p++) {
            // Scatter, but without x or C
            csint k = B.i_[p];  // B(k, j) is non-zero
            for (csint pa = p_[k]; pa < p_[k+1]; pa++) {
                csint i = i_[pa];     // A(i, k) is non-zero
                if (w[i] < mark) {
                    w[i] = mark;     // i is new entry in column k
                    nz_C++;         // count non-zeros in C, but don't compute
                }
            }
        }
    }

    // Allocate the correct size output matrix
    CSCMatrix C({M, N}, nz_C);

    // Compute the actual multiplication
    std::fill(w.begin(), w.end(), 0);  // reset workspace
    std::vector<double> x(M);

    csint nz = 0;  // track total number of non-zeros in C
    bool fs = true;  // first call to scatter

    for (csint j = 0; j < N; j++) {
        C.p_[j] = nz;  // column j of C starts here

        // Compute x = A @ B[:, j]
        for (csint p = B.p_[j]; p < B.p_[j+1]; p++) {
            // Compute x += A[:, B.i_[p]] * B.v_[p]
            nz = scatter(B.i_[p], B.v_[p], w, x, j+1, C, nz, fs);
            fs = false;
        }

        // Gather values into the correct locations in C
        for (csint p = C.p_[j]; p < nz; p++) {
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
    assert((N_ == 1) && (y.N_ == 1));  // both must be column vectors
    assert(M_ == y.M_);
    double z = 0.0;

    if (has_sorted_indices_ && y.has_sorted_indices_) {
        csint p = 0, q = 0;  // pointer to row index of each vector

        while ((p < nnz()) && (q < y.nnz())) {
            csint i = i_[p];    // row index of each vector
            csint j = y.i_[q];

            if (i == j) {
                z += v_[p++] * y.v_[q++];
            } else if (i < j) {
                p++;
            } else {  // (j < i)
                q++;
            }
        }

    } else {  // unsorted indices
        std::vector<double> w(M_);  // workspace

        // Expand this vector
        for (csint p = 0; p < nnz(); p++) {
            w[i_[p]] = v_[p];
        }

        // Multiply by non-zero entries in y and sum
        for (csint q = 0; q < y.nnz(); q++) {
            csint i = y.i_[q];
            if (w[i] != 0) {
                z += w[i] * y.v_[q];
            }
        }
    }

    return z;
}


std::vector<double> operator*(const CSCMatrix& A, const std::vector<double>& x) { return A.dot(x); }
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
    assert(A.shape() == B.shape());
    auto [M, N] = A.shape();

    bool values = !A.v_.empty() && !B.v_.empty();

    CSCMatrix C({M, N}, A.nnz() + B.nnz(), values);  // output

    // Allocate workspaces
    std::vector<csint> w(M);
    std::vector<double> x;
    if (values) {
        x.resize(M);
    }

    csint nz = 0;    // track total number of non-zeros in C
    bool fs = true;  // Exercise 2.19 -- first call to scatter

    for (csint j = 0; j < N; j++) {
        C.p_[j] = nz;  // column j of C starts here
        nz = A.scatter(j, alpha, w, x, j+1, C, nz, fs);  // alpha * A(:, j)
        fs = false;
        nz = B.scatter(j,  beta, w, x, j+1, C, nz, fs);  //  beta * B(:, j)

        // Gather results into the correct column of C
        if (values) {
            for (csint p = C.p_[j]; p < nz; p++) {
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
std::vector<csint> saxpy(
    const CSCMatrix& a,
    const CSCMatrix& b,
    std::vector<csint>& w,
    std::vector<double>& x
    )
{
    assert(a.shape() == b.shape());
    assert((a.N_ == 1) && (b.N_ == 1));  // both must be column vectors

    for (csint p = 0; p < a.nnz(); p++) {
        csint i = a.i_[p];
        w[i] = 1;        // mark as non-zero
        x[i] = a.v_[p];  // copy x into w
    }

    for (csint p = 0; p < b.nnz(); p++) {
        csint i = b.i_[p];
        if (w[i] == 0) {
            w[i] = 1;         // mark as non-zero
            x[i] = b.v_[p];   // copy b into w
        } else {
            x[i] += b.v_[p];  // add b to x
        }
    }

    return w;
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


csint CSCMatrix::scatter(
    csint j,
    double beta,
    std::vector<csint>& w,
    std::optional<std::reference_wrapper<std::vector<double>>> x_ref,
    csint mark,
    CSCMatrix& C,
    csint nz,
    bool fs  // Exercise 2.19
) const
{
    // Check if x is passed as a reference
    std::vector<double> empty_vec;
    std::vector<double>& x = x_ref ? x_ref->get() : empty_vec;
    bool values = !v_.empty() && !x.empty();

    // Exercise 2.19
    if (fs) {
        // If it's the first call, we can just copy the (scaled) column
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            csint i = i_[p];       // A(i, j) is non-zero
            w[i] = mark;             // i is new entry in column j
            C.i_[nz++] = i;          // add i to sparsity pattern of C(:, j)
            if (values) {
                x[i] = beta * v_[p];   // x = beta * A(i, j)
            }
        }
    } else {
        // Original scatter
        // Otherwise, we need to accumulate the values
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            csint i = i_[p];           // A(i, j) is non-zero
            if (w[i] < mark) {
                w[i] = mark;             // i is new entry in column j
                C.i_[nz++] = i;          // add i to pattern of C(:, j)
                if (values) {
                    x[i] = beta * v_[p];   // x = beta * A(i, j)
                }
            } else {
                if (values) {
                    x[i] += beta * v_[p];  // i exists in C(:, j) already
                }
            }
        }
    }

    return nz;
}


/*------------------------------------------------------------------------------
 *         Permutations 
 *----------------------------------------------------------------------------*/
CSCMatrix CSCMatrix::permute(
    const std::vector<csint>& p_inv,
    const std::vector<csint>& q,
    bool values
) const
{
    CSCMatrix C({M_, N_}, nnz(), values);
    csint nz = 0;

    for (csint k = 0; k < N_; k++) {
        C.p_[k] = nz;                   // column k of C is column q[k] of A
        csint j = q[k];

        for (csint t = p_[j]; t < p_[j+1]; t++) {
            if (values) {
                C.v_[nz] = v_[t];       // row i of A is row p_inv[i] of C
            }
            C.i_[nz++] = p_inv[i_[t]];
        }
    }

    C.p_[N_] = nz;

    return C;
}


CSCMatrix CSCMatrix::permute_rows(const std::vector<csint>& p_inv, bool values) const
{
    std::vector<csint> q(N_);
    std::iota(q.begin(), q.end(), 0);  // identity permutation
    return permute(p_inv, q, values);
}


CSCMatrix CSCMatrix::permute_cols(const std::vector<csint>& q, bool values) const
{
    std::vector<csint> p_inv(M_);
    std::iota(p_inv.begin(), p_inv.end(), 0);  // identity permutation
    return permute(p_inv, q, values);
}


CSCMatrix CSCMatrix::symperm(const std::vector<csint>& p_inv, bool values) const
{
    assert(M_ == N_);  // matrix must be square. Symmetry not checked.

    CSCMatrix C({N_, N_}, nnz(), values);
    std::vector<csint> w(N_);  // workspace for column counts

    // Count entries in each column of C
    for (csint j = 0; j < N_; j++) {
        csint j2 = p_inv[j];  // column j of A is column j2 of C

        for (csint p = p_[j]; p < p_[j+1]; p++) {
            csint i = i_[p];

            if (i > j)
                continue;   // skip lower triangular part of A

            csint i2 = p_inv[i];    // row i of A is row i2 of C
            w[std::max(i2, j2)]++;  // column count of C
        }
    }

    // Row pointers are the cumulative sum of the counts, starting with 0.
    C.p_ = cumsum(w);
    w = C.p_;  // copy back into workspace

    for (csint j = 0; j < N_; j++) {
        csint j2 = p_inv[j];  // column j of A is column j2 of C

        for (csint p = p_[j]; p < p_[j+1]; p++) {
            csint i = i_[p];

            if (i > j)
                continue;   // skip lower triangular part of A

            csint i2 = p_inv[i];  // row i of A is row i2 of C
            csint q = w[std::max(i2, j2)]++;
            C.i_[q] = std::min(i2, j2);
            if (values) {
                C.v_[q] = v_[p];
            }
        }
    }

    return C;
}


// Exercise 2.26
CSCMatrix CSCMatrix::permute_transpose(
    const std::vector<csint>& p_inv,
    const std::vector<csint>& q_inv,
    bool values
) const
{
    std::vector<csint> w(M_);            // workspace
    CSCMatrix C({N_, M_}, nnz(), values);  // output

    // Compute number of elements in each permuted row (aka column of C)
    for (csint p = 0; p < nnz(); p++)
        w[p_inv[i_[p]]]++;

    C.p_ = cumsum(w);
    w = C.p_;  // copy back into workspace

    // place A(i, j) as C(j, i) (permuted)
    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            csint t = w[p_inv[i_[p]]]++;
            C.i_[t] = q_inv[j];
            if (values) {
                C.v_[t] = v_[p];
            }
        }
    }

    return C;
}


double CSCMatrix::norm() const
{
    double the_norm = 0;

    for (csint j = 0; j < N_; j++) {
        double s = 0;

        for (csint p = p_[j]; p < p_[j+1]; p++) {
            s += std::fabs(v_[p]);
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
    if (p_.size() != (N_ + 1)) {
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
    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j+1]; p++) {
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
            for (csint p = p_[j]; p < p_[j+1]-1; p++) {
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
    assert(A.M_ == B.M_);

    // Copy the first matrix
    CSCMatrix C = A;
    C.N_ += B.N_;
    C.realloc(A.nnz() + B.nnz());

    // Copy the second matrix
    for (csint j = 0; j < B.N_; j++) {
        C.p_[A.N_ + j] = B.p_[j] + A.nnz();
    }

    std::copy(B.i_.begin(), B.i_.end(), C.i_.begin() + A.nnz());
    std::copy(B.v_.begin(), B.v_.end(), C.v_.begin() + A.nnz());

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
    assert(A.N_ == B.N_);

    CSCMatrix C({A.M_ + B.M_, A.N_}, A.nnz() + B.nnz());

    csint nz = 0;

    for (csint j = 0; j < C.N_; j++) {
        C.p_[j] = nz;  // column j of C starts here

        // Copy column j from the first matrix
        for (csint p = A.p_[j]; p < A.p_[j+1]; p++) {
            C.i_[nz] = A.i_[p];
            C.v_[nz] = A.v_[p];
            nz++;
        }

        // Copy column j from the second matrix
        for (csint p = B.p_[j]; p < B.p_[j+1]; p++) {
            C.i_[A.p_[j+1] + p] = A.M_ + B.i_[p];
            C.v_[A.p_[j+1] + p] = B.v_[p];
            nz++;
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
    assert((i_start >= 0) && (i_end <= M_) && (i_start <= i_end));
    assert((j_start >= 0) && (j_end <= N_) && (j_start <= j_end));

    CSCMatrix C({i_end - i_start, j_end - j_start}, nnz());

    csint nz = 0;

    for (csint j = j_start; j < j_end; j++) {
        C.p_[j - j_start] = nz;  // column j of C starts here

        for (csint p = p_[j]; p < p_[j+1]; p++) {
            csint i = i_[p];

            if ((i >= i_start) && (i < i_end)) {
                C.i_[nz] = i - i_start;
                C.v_[nz] = v_[p];
                nz++;
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
    const std::vector<csint>& rows,
    const std::vector<csint>& cols
    ) const
{
    csint M = rows.size();
    csint N = cols.size();
    CSCMatrix C({M, N}, nnz());

    csint nz = 0;

    for (csint j = 0; j < N; j++) {
        C.p_[j] = nz;  // column j of C starts here

        // Iterate over `rows` and find the corresponding indices in `i_`.
        for (csint k = 0; k < M; k++) {
            double val = (*this)(rows[k], cols[j]);
            if (val != 0) {
                C.i_[nz] = k;
                C.v_[nz] = val;
                nz++;
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

    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            out[i_[p]] += v_[p];
        }
    }

    return out;
}


std::vector<double> CSCMatrix::sum_cols() const
{
    std::vector<double> out(N_, 0.0);

    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            out[j] += v_[p];
        }
    }

    return out;
}


/*------------------------------------------------------------------------------
 *         Printing
 *----------------------------------------------------------------------------*/
void CSCMatrix::print_dense(int precision, bool suppress, std::ostream& os) const
{
    print_dense_vec(to_dense_vector('F'), M_, N_, 'F', precision, suppress, os);
}


std::string CSCMatrix::to_string(bool verbose, csint threshold) const
{
    csint nnz_ = nnz();
    std::stringstream ss;

    ss << std::format(
        "<{} matrix\n"
        "        with {} stored elements and shape ({}, {})>",
        format_desc_, nnz_, M_, N_);

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


void CSCMatrix::write_elems_(std::stringstream& ss, csint start, csint end) const
{
    // Determine whether to use scientific notation
    double abs_max = 0.0;
    for (const auto& val : v_) {
        if (std::isfinite(val)) {
            abs_max = std::max(abs_max, std::fabs(val));
        }
    }

    bool use_scientific = (abs_max < 1e-4 || abs_max > 1e4);
    // Leading space aligns for "-" signs
    const std::string fmt = use_scientific ? " .4e" : " .4g";

    // Compute index width from maximum index
    int row_width = std::to_string(M_ - 1).size();
    int col_width = std::to_string(N_ - 1).size();

    const std::string format_string = "({0:>{1}d}, {2:>{3}d}): {4:" + fmt + "}";

    csint n = 0;  // number of elements printed
    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j + 1]; p++) {
            if ((n >= start) && (n < end)) {
                ss << std::vformat(
                    format_string,
                    std::make_format_args(
                        i_[p], row_width,
                        j, col_width,
                        v_[p]
                    )
                );

                if (n < end - 1) {
                    ss << std::endl;
                }
            }
            n++;
        }
    }
}


void CSCMatrix::print(std::ostream& os, bool verbose, csint threshold) const
{
    os << to_string(verbose, threshold) << std::endl;
}


std::ostream& operator<<(std::ostream& os, const CSCMatrix& A)
{
    A.print(os, true);  // verbose printing assumed
    return os;
}


}  // namespace cs

/*==============================================================================
 *============================================================================*/
