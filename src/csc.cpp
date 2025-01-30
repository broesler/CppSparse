/*==============================================================================
 *     File: csc.cpp
 *  Created: 2024-10-09 20:58
 *   Author: Bernie Roesler
 *
 *  Description: Implements the compressed sparse column matrix class
 *
 *============================================================================*/

#include <algorithm>  // for std::lower_bound
#include <cmath>      // for std::fabs
#include <ranges>     // for std::views::reverse

#include "csparse.h"

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


CSCMatrix::CSCMatrix(csint M, csint N, csint nzmax, bool values)
    : i_(nzmax),
      p_(N + 1),
      M_(M),
      N_(N)
{
    if (values) {
        v_.resize(nzmax);
    } else {
        v_.resize(0);
        v_.shrink_to_fit();
    }
}


CSCMatrix::CSCMatrix(const COOMatrix& A) : CSCMatrix(A.compress())
{
    sum_duplicates();  // O(N) space, O(nnz) time
    dropzeros();       // O(nnz) time
    sort();            // O(M) space, O(M + N + nnz) time
    has_canonical_format_ = true;
}


CSCMatrix::CSCMatrix(const std::vector<double>& A, csint M, csint N)
    : M_(M),
      N_(N)
{
    assert(A.size() == (M * N));  // ensure input is valid

    // Allocate memory
    v_.reserve(A.size());
    i_.reserve(A.size());
    p_.reserve(N);

    csint nz = 0;  // count number of non-zeros

    for (csint j = 0; j < N; j++) {
        p_.push_back(nz);

        for (csint i = 0; i < M; i++) {
            double val = A[i + j * N];  // linear index for column-major order

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


CSCMatrix& CSCMatrix::realloc(csint nzmax)
{
    csint Z = (nzmax <= 0) ? p_[N_] : nzmax;

    p_.resize(N_ + 1);  // always contains N_ columns + nz
    i_.resize(Z);
    v_.resize(Z);

    p_.shrink_to_fit();  // deallocate memory
    i_.shrink_to_fit();
    v_.shrink_to_fit();

    return *this;
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


bool CSCMatrix::is_symmetric() const
{
    assert(has_canonical_format_);

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


CSCMatrix& CSCMatrix::assign(csint i, csint j, double v)
{
    (*this)(i, j) = v;
    return *this;
}


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
COOMatrix CSCMatrix::tocoo() const { return COOMatrix(*this); }


std::vector<double> CSCMatrix::toarray(const char order) const
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

            if (has_canonical_format_) {
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
    CSCMatrix C(N_, M_, nnz(), values);  // output

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
CSCMatrix CSCMatrix::T(bool values) const { return this->transpose(values); }


CSCMatrix CSCMatrix::tsort() const
{
    CSCMatrix C = this->transpose().transpose();
    C.has_sorted_indices_ = true;
    return C;
}


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


CSCMatrix& CSCMatrix::sort()
{
    // ----- first transpose
    std::vector<csint> w(M_);   // workspace
    CSCMatrix C(N_, M_, nnz());  // intermediate transpose

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
            C.v_[q] = v_[p];
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
            v_[q] = C.v_[p];
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


CSCMatrix& CSCMatrix::fkeep(
    bool (*fk) (csint, csint, double, void *),
    void *other
)
{
    csint nz = 0;  // count actual number of non-zeros

    for (csint j = 0; j < N_; j++) {
        csint p = p_[j];  // get current location of column j
        p_[j] = nz;       // record new location of column j
        for (; p < p_[j+1]; p++) {
            if (fk(i_[p], j, v_[p], other)) {
                v_[nz] = v_[p];  // keep A(i, j)
                i_[nz++] = i_[p];
            }
        }
    }

    p_[N_] = nz;    // finalize A
    realloc();

    return *this;
};


CSCMatrix CSCMatrix::fkeep(
    bool (*fk) (csint i, csint j, double Aij, void *tol),
    void *other
) const
{
    CSCMatrix C(*this);
    return C.fkeep(fk, other);
}


bool CSCMatrix::nonzero(csint i, csint j, double Aij, void *other)
{
    return (Aij != 0);
}


CSCMatrix& CSCMatrix::dropzeros()
{
    return fkeep(&nonzero, nullptr);
}


bool CSCMatrix::abs_gt_tol(csint i, csint j, double Aij, void *tol)
{
    return (std::fabs(Aij) > *((double *) tol));
}


CSCMatrix& CSCMatrix::droptol(double tol)
{
    return fkeep(&abs_gt_tol, &tol);
}


bool CSCMatrix::in_band(csint i, csint j, double Aij, void *limits)
{
    auto [kl, ku] = *((Shape *) limits);
    return ((i <= (j - kl)) && (i >= (j - ku)));
};


CSCMatrix& CSCMatrix::band(const csint kl, const csint ku)
{
    assert(kl <= ku);
    Shape limits {kl, ku};

    return fkeep(&in_band, &limits);
}


CSCMatrix CSCMatrix::band(const csint kl, const csint ku) const
{
    assert(kl <= ku);
    Shape limits {kl, ku};

    return fkeep(&in_band, &limits);
}


/*------------------------------------------------------------------------------
       Math Operations
----------------------------------------------------------------------------*/
std::vector<double> CSCMatrix::gaxpy(
    const std::vector<double>& x,
    const std::vector<double>& y
    ) const
{
    assert(M_ == y.size());  // addition
    assert(N_ == x.size());  // multiplication

    std::vector<double> out = y;  // copy the input vector

    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            out[i_[p]] += v_[p] * x[j];
        }
    }

    return out;
};


std::vector<double> CSCMatrix::gatxpy(
    const std::vector<double>& x,
    const std::vector<double>& y
    ) const
{
    assert(M_ == x.size());  // multiplication
    assert(N_ == y.size());  // addition

    std::vector<double> out = y;  // copy the input vector

    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            out[j] += v_[p] * x[i_[p]];
        }
    }

    return out;
};


std::vector<double> CSCMatrix::sym_gaxpy(
    const std::vector<double>& x,
    const std::vector<double>& y
    ) const
{
    assert(M_ == N_);  // matrix must be square to be symmetric
    assert(N_ == x.size());
    assert(x.size() == y.size());

    std::vector<double> out = y;  // copy the input vector

    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            csint i = i_[p];

            if (i > j)
                continue;  // skip lower triangular

            // Add the upper triangular elements
            out[i] += v_[p] * x[j];

            // If off-diagonal, also add the symmetric element
            if (i < j)
                out[j] += v_[p] * x[i];
        }
    }

    return out;
};


std::vector<double> CSCMatrix::gaxpy_col(
    const std::vector<double>& X,
    const std::vector<double>& Y
    ) const
{
    assert(X.size() % N_ == 0);  // check that X.size() is a multiple of N_
    assert(Y.size() == M_ * (X.size() / N_));

    std::vector<double> out = Y;  // copy the input matrix

    csint K = X.size() / N_;  // number of columns in X

    // For each column of X
    for (csint k = 0; k < K; k++) {
        // Compute one column of Y (see gaxpy)
        for (csint j = 0; j < N_; j++) {
            double x_val = X[j + k * N_];  // cache value

            // Only compute if x_val is non-zero
            if (x_val != 0.0) {
                for (csint p = p_[j]; p < p_[j+1]; p++) {
                    // Indexing in column-major order
                    out[i_[p] + k * M_] += v_[p] * x_val;
                }
            }
        }
    }

    return out;
}


std::vector<double> CSCMatrix::gaxpy_block(
    const std::vector<double>& X,
    const std::vector<double>& Y
    ) const
{
    assert(X.size() % N_ == 0);  // check that X.size() is a multiple of N_
    assert(Y.size() == M_ * (X.size() / N_));

    std::vector<double> out = Y;  // copy the input matrix

    csint K = X.size() / N_;  // number of columns in X

    const csint BLOCK_SIZE = 32;  // block size for column operations

    // For each column of X
    for (csint k = 0; k < K; k++) {
        // Take a block of columns
        for (csint j_start = 0; j_start < N_; j_start += BLOCK_SIZE) {
            csint j_end = std::min(j_start + BLOCK_SIZE, N_);
            // Compute one column of Y (see gaxpy)
            for (csint j = j_start; j < j_end; j++) {
                double x_val = X[j + k * N_];  // cache value

                // Only compute if x_val is non-zero
                if (x_val != 0.0) {
                    for (csint p = p_[j]; p < p_[j+1]; p++) {
                        // Indexing in column-major order
                        out[i_[p] + k * M_] += v_[p] * x_val;
                    }
                }
            }
        }
    }

    return out;
}


std::vector<double> CSCMatrix::gaxpy_row(
    const std::vector<double>& X,
    const std::vector<double>& Y
    ) const
{
    assert(X.size() % N_ == 0);  // check that X.size() is a multiple of N_
    assert(Y.size() == M_ * (X.size() / N_));

    std::vector<double> out = Y;  // copy the input matrix

    csint K = X.size() / N_;  // number of columns in X

    // For each column of X
    for (csint k = 0; k < K; k++) {
        // Compute one column of Y (see gaxpy)
        for (csint j = 0; j < N_; j++) {
            double x_val = X[k + j * K];  // cache value (row-major indexing)

            // Only compute if x_val is non-zero
            if (x_val != 0.0) {
                for (csint p = p_[j]; p < p_[j+1]; p++) {
                    // Indexing in row-major order
                    out[k + i_[p] * K] += v_[p] * x_val;
                }
            }
        }
    }

    return out;
}


std::vector<double> CSCMatrix::gatxpy_col(
    const std::vector<double>& X,
    const std::vector<double>& Y
    ) const
{
    assert(X.size() % M_ == 0);  // check that X.size() is a multiple of M_
    assert(Y.size() == N_ * (X.size() / M_));

    std::vector<double> out = Y;  // copy the input matrix

    csint K = X.size() / M_;  // number of columns in X

    // For each column of X
    for (csint k = 0; k < K; k++) {
        // Compute one column of Y (see gaxpy)
        for (csint j = 0; j < N_; j++) {
            for (csint p = p_[j]; p < p_[j+1]; p++) {
                // Indexing in column-major order
                out[j + k * N_] += v_[p] * X[i_[p] + k * M_];
            }
        }
    }

    return out;
}


std::vector<double> CSCMatrix::gatxpy_block(
    const std::vector<double>& X,
    const std::vector<double>& Y
    ) const
{
    assert(X.size() % M_ == 0);  // check that X.size() is a multiple of N_
    assert(Y.size() == N_ * (X.size() / M_));

    std::vector<double> out = Y;  // copy the input matrix

    csint K = X.size() / M_;  // number of columns in X

    const csint BLOCK_SIZE = 32;  // block size for column operations

    // For each column of X
    for (csint k = 0; k < K; k++) {
        // Take a block of columns
        for (csint j_start = 0; j_start < N_; j_start += BLOCK_SIZE) {
            csint j_end = std::min(j_start + BLOCK_SIZE, N_);
            // Compute one column of Y (see gaxpy)
            for (csint j = j_start; j < j_end; j++) {
                for (csint p = p_[j]; p < p_[j+1]; p++) {
                    // Indexing in column-major order
                    out[j + k * N_] += v_[p] * X[i_[p] + k * M_];
                }
            }
        }
    }

    return out;
}


std::vector<double> CSCMatrix::gatxpy_row(
    const std::vector<double>& X,
    const std::vector<double>& Y
    ) const
{
    assert(X.size() % M_ == 0);  // check that X.size() is a multiple of N_
    assert(Y.size() == N_ * (X.size() / M_));

    std::vector<double> out = Y;  // copy the input matrix

    csint K = X.size() / M_;  // number of columns in X

    // For each column of X
    for (csint k = 0; k < K; k++) {
        // Compute one column of Y (see gaxpy)
        for (csint j = 0; j < N_; j++) {
            for (csint p = p_[j]; p < p_[j+1]; p++) {
                // Indexing in row-major order
                out[k + j * K] += v_[p] * X[k + i_[p] * K];
            }
        }
    }

    return out;
}


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

    // NOTE See Problem 2.20 for how to compute nnz(A*B)
    CSCMatrix C(M, N, nnz() + B.nnz());  // output

    // Allocate workspaces
    std::vector<csint> w(M);
    std::vector<double> x(M);

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
    CSCMatrix C(M, N, nz_C);

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

    CSCMatrix C(M, N, A.nnz() + B.nnz());  // output

    // Allocate workspaces
    std::vector<csint> w(M);
    std::vector<double> x(M);

    csint nz = 0;    // track total number of non-zeros in C
    bool fs = true;  // Exercise 2.19 -- first call to scatter

    for (csint j = 0; j < N; j++) {
        C.p_[j] = nz;  // column j of C starts here
        nz = A.scatter(j, alpha, w, x, j+1, C, nz, fs);  // alpha * A(:, j)
        fs = false;
        nz = B.scatter(j,  beta, w, x, j+1, C, nz, fs);  //  beta * B(:, j)

        // Gather results into the correct column of C
        for (csint p = C.p_[j]; p < nz; p++) {
            C.v_[p] = x[C.i_[p]];
        }
    }

    // Finalize and deallocate unused memory
    C.p_[N] = nz;
    C.realloc();

    return C;
}


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


CSCMatrix operator+(const CSCMatrix& A, const CSCMatrix& B) { return A.add(B); }


csint CSCMatrix::scatter(
    csint j,
    double beta,
    std::vector<csint>& w,
    std::vector<double>& x,
    csint mark,
    CSCMatrix& C,
    csint nz,
    bool fs
    ) const
{
    if (fs) {
        // If it's the first call, we can just copy the (scaled) column
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            csint i = i_[p];       // A(i, j) is non-zero
            w[i] = mark;             // i is new entry in column j
            C.i_[nz++] = i;          // add i to sparsity pattern of C(:, j)
            x[i] = beta * v_[p];   // x = beta * A(i, j)
        }
    } else {
        // Otherwise, we need to accumulate the values
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            csint i = i_[p];           // A(i, j) is non-zero
            if (w[i] < mark) {
                w[i] = mark;             // i is new entry in column j
                C.i_[nz++] = i;          // add i to pattern of C(:, j)
                x[i] = beta * v_[p];   // x = beta * A(i, j)
            } else {
                x[i] += beta * v_[p];  // i exists in C(:, j) already
            }
        }
    }

    return nz;
}


/*------------------------------------------------------------------------------
 *         Permutations 
 *----------------------------------------------------------------------------*/
CSCMatrix CSCMatrix::permute(
    const std::vector<csint> p_inv,
    const std::vector<csint> q
    ) const
{
    CSCMatrix C(M_, N_, nnz());
    csint nz = 0;

    for (csint k = 0; k < N_; k++) {
        C.p_[k] = nz;                   // column k of C is column q[k] of A
        csint j = q[k];

        for (csint t = p_[j]; t < p_[j+1]; t++) {
            C.v_[nz] = v_[t];           // row i of A is row p_inv[i] of C
            C.i_[nz++] = p_inv[i_[t]];
        }
    }

    C.p_[N_] = nz;

    return C;
}


CSCMatrix CSCMatrix::permute_rows(const std::vector<csint> p_inv) const
{
    std::vector<csint> q(N_);
    std::iota(q.begin(), q.end(), 0);  // identity permutation
    return permute(p_inv, q);
}


CSCMatrix CSCMatrix::permute_cols(const std::vector<csint> q) const
{
    std::vector<csint> p_inv(M_);
    std::iota(p_inv.begin(), p_inv.end(), 0);  // identity permutation
    return permute(p_inv, q);
}


CSCMatrix CSCMatrix::symperm(const std::vector<csint> p_inv, bool values) const
{
    assert(M_ == N_);  // matrix must be square. Symmetry not checked.

    CSCMatrix C(N_, N_, nnz());
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


CSCMatrix CSCMatrix::permute_transpose(
    const std::vector<csint>& p_inv,
    const std::vector<csint>& q_inv
    ) const
{
    std::vector<csint> w(M_);    // workspace
    CSCMatrix C(N_, M_, nnz());  // output

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
            C.v_[t] = v_[p];
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


bool CSCMatrix::is_valid(const bool sorted, const bool values) const
{
    // Check number of columns
    if (p_.size() != (N_ + 1)) {
        // std::cout << "Columns inconsistent!" << std::endl;
        return false;
    }

    // TODO not sure how we're supposed to use O(M) space? Column counts can't
    // be independently checked.

    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            csint i = i_[p];

            if (i > M_) {
                // std::cout << "Invalid row index!" << std::endl;
                return false;  // invalid row index
            }

            if (sorted && (p < (p_[j+1] - 1)) && (i > i_[p+1])) {
                // std::cout << "Columns not sorted!" << std::endl;
                return false;  // not sorted in ascending order
            }

            if (values && v_[p] == 0.0) {
                // std::cout << "Explicit zeros!" << std::endl;
                return false;  // no zeros allowed
            }
        }
    }

    return true;
}


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


CSCMatrix vstack(const CSCMatrix& A, const CSCMatrix& B)
{
    assert(A.N_ == B.N_);

    CSCMatrix C(A.M_ + B.M_, A.N_, A.nnz() + B.nnz());

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


CSCMatrix CSCMatrix::slice(
    const csint i_start,
    const csint i_end,
    const csint j_start,
    const csint j_end
    ) const
{
    assert((i_start >= 0) && (i_end <= M_) && (i_start < i_end));
    assert((j_start >= 0) && (j_end <= N_) && (j_start < j_end));

    CSCMatrix C(i_end - i_start, j_end - j_start, nnz());

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


CSCMatrix CSCMatrix::index(
    const std::vector<csint>& rows,
    const std::vector<csint>& cols
    ) const
{
    CSCMatrix C(rows.size(), cols.size(), nnz());

    csint nz = 0;

    for (csint j = 0; j < cols.size(); j++) {
        C.p_[j] = nz;  // column j of C starts here

        // Iterate over `rows` and find the corresponding indices in `i_`.
        for (csint k = 0; k < rows.size(); k++) {
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


CSCMatrix CSCMatrix::add_empty_top(const csint k) const
{
    CSCMatrix C = *this;  // copy the matrix
    C.M_ += k;

    // Increate all row indices by k
    for (auto& i : C.i_) {
        i += k;
    }

    return C;
}


CSCMatrix CSCMatrix::add_empty_bottom(const csint k) const
{
    CSCMatrix C = *this;  // copy the matrix
    C.M_ += k;
    return C;
}


CSCMatrix CSCMatrix::add_empty_left(const csint k) const
{
    CSCMatrix C = *this;  // copy the matrix
    C.N_ += k;
    C.p_.insert(C.p_.begin(), k, 0);  // insert k zeros at the beginning
    return C;
}


CSCMatrix CSCMatrix::add_empty_right(const csint k) const
{
    CSCMatrix C = *this;  // copy the matrix
    C.N_ += k;
    C.p_.insert(C.p_.end(), k, nnz());  // insert k nnz() at the end
    return C;
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
 *      Triangular Matrix Solutions 
 *----------------------------------------------------------------------------*/
std::vector<double> CSCMatrix::lsolve(const std::vector<double>& b) const
{
    assert(M_ == N_);
    assert(M_ == b.size());

    std::vector<double> x = b;

    for (csint j = 0; j < N_; j++) {
        x[j] /= v_[p_[j]];
        for (csint p = p_[j] + 1; p < p_[j+1]; p++) {
            x[i_[p]] -= v_[p] * x[j];
        }
    }

    return x;
}


std::vector<double> CSCMatrix::ltsolve(const std::vector<double>& b) const
{
    assert(M_ == N_);
    assert(M_ == b.size());

    std::vector<double> x = b;

    for (csint j = N_ - 1; j >= 0; j--) {
        for (csint p = p_[j] + 1; p < p_[j+1]; p++) {
            x[j] -= v_[p] * x[i_[p]];
        }
        x[j] /= v_[p_[j]];
    }

    return x;
}


std::vector<double> CSCMatrix::usolve(const std::vector<double>& b) const
{
    assert(M_ == N_);
    assert(M_ == b.size());

    std::vector<double> x = b;

    for (csint j = N_ - 1; j >= 0; j--) {
        x[j] /= v_[p_[j+1] - 1];  // diagonal entry
        for (csint p = p_[j]; p < p_[j+1] - 1; p++) {
            x[i_[p]] -= v_[p] * x[j];
        }
    }

    return x;
}


std::vector<double> CSCMatrix::utsolve(const std::vector<double>& b) const
{
    assert(M_ == N_);
    assert(M_ == b.size());

    std::vector<double> x = b;

    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j+1] - 1; p++) {
            x[j] -= v_[p] * x[i_[p]];
        }
        x[j] /= v_[p_[j+1] - 1];  // diagonal entry
    }

    return x;
}


std::vector<double> CSCMatrix::lsolve_opt(const std::vector<double>& b) const
{
    assert(M_ == N_);
    assert(M_ == b.size());

    std::vector<double> x = b;

    for (csint j = 0; j < N_; j++) {
        double& x_val = x[j];  // cache reference to value
        // Exercise 3.8: improve performance by checking for zeros
        if (x_val != 0) {
            x_val /= v_[p_[j]];
            for (csint p = p_[j] + 1; p < p_[j+1]; p++) {
                x[i_[p]] -= v_[p] * x_val;
            }
        }
    }

    return x;
}


std::vector<double> CSCMatrix::usolve_opt(const std::vector<double>& b) const
{
    assert(M_ == N_);
    assert(M_ == b.size());

    std::vector<double> x = b;

    for (csint j = N_ - 1; j >= 0; j--) {
        double& x_val = x[j];  // cache reference to value
        if (x_val != 0) {
            x_val /= v_[p_[j+1] - 1];  // diagonal entry
            for (csint p = p_[j]; p < p_[j+1] - 1; p++) {
                x[i_[p]] -= v_[p] * x_val;
            }
        }
    }

    return x;
}


std::vector<csint> CSCMatrix::find_lower_diagonals() const
{
    assert(M_ == N_);

    std::vector<bool> marked(N_, false);  // workspace
    std::vector<csint> p_diags(N_);  // diagonal indicies (inverse permutation)

    for (csint j = N_ - 1; j >= 0; j--) {
        csint N_unmarked = 0;

        for (csint p = p_[j]; p < p_[j+1]; p++) {
            csint i = i_[p];
            // Mark the rows viewed so far
            if (!marked[i]) {
                marked[i] = true;
                p_diags[j] = p;
                N_unmarked++;
            }
        }

        // If 0 or > 1 "diagonal" entries found, the matrix is not permuted.
        if (N_unmarked != 1) {
            throw std::runtime_error("Matrix is not a permuted lower triangular matrix!");
        }
    }

    return p_diags;
}


std::vector<double> CSCMatrix::lsolve_rows(const std::vector<double>& b) const
{
    assert(M_ == N_);
    assert(M_ == b.size());

    // First (backward) pass to find diagonal entries
    // p_diags is a vector of pointers to the diagonal entries
    std::vector<csint> p_diags = find_lower_diagonals();

    // Compute the row permutation vector
    std::vector<csint> permuted_rows(N_);
    for (csint i = 0; i < N_; i++) {
        permuted_rows[i_[p_diags[i]]] = i;
    }

    // Second (forward) pass to solve the system
    std::vector<double> x = b;

    // Perform the permuted forward solve
    for (csint j = 0; j < N_; j++) {
        csint d = p_diags[j];  // pointer to the diagonal entry
        double& x_val = x[j];  // cache diagonal value

        if (x_val != 0) {
            x_val /= v_[d];    // solve for x[d]
            for (csint p = p_[j]; p < p_[j+1]; p++) {
                csint i = permuted_rows[i_[p]];
                if (p != d) {
                    x[i] -= v_[p] * x_val;  // update the off-diagonals
                }
            }
        }
    }

    return x;
}


std::vector<double> CSCMatrix::lsolve_cols(const std::vector<double>& b) const
{
    assert(M_ == N_);
    assert(M_ == b.size());

    // First O(N) pass to find the diagonal entries
    // Assume that the first entry in each column has the smallest row index
    std::vector<csint> p_diags(N_, -1);
    for (csint j = 0; j < N_; j++) {
        if (p_diags[j] == -1) {
            p_diags[j] = p_[j];  // pointer to the diagonal entry
        } else {
            // We have seen this column index before
            throw std::runtime_error("Matrix is not a permuted lower triangular matrix!");
        }
    }

    // Compute the column permutation vector
    std::vector<csint> permuted_cols(N_);
    for (csint i = 0; i < N_; i++) {
        permuted_cols[i_[p_diags[i]]] = i;
    }

    // Second (forward) pass to solve the system
    std::vector<double> x = b;

    // Perform the permuted forward solve
    for (const auto& j : permuted_cols) {
        csint d = p_diags[j];      // pointer to the diagonal entry
        double& x_val = x[i_[d]];  // cache diagonal value

        if (x_val != 0) {
            x_val /= v_[d];  // solve for x[i_[d]]
            for (csint p = p_[j]+1; p < p_[j+1]; p++) {
                x[i_[p]] -= v_[p] * x_val;  // update the off-diagonals
            }
        }
    }

    return x;
}


std::vector<csint> CSCMatrix::find_upper_diagonals() const
{
    assert(M_ == N_);

    std::vector<bool> marked(N_, false);  // workspace
    std::vector<csint> p_diags(N_);  // diagonal indicies (inverse permutation)

    for (csint j = 0; j < N_; j++) {
        csint N_unmarked = 0;

        for (csint p = p_[j]; p < p_[j+1]; p++) {
            csint i = i_[p];
            // Mark the rows viewed so far
            if (!marked[i]) {
                marked[i] = true;
                p_diags[j] = p;
                N_unmarked++;
            }
        }

        // If 0 or > 1 "diagonal" entries found, the matrix is not permuted.
        if (N_unmarked != 1) {
            throw std::runtime_error("Matrix is not a permuted upper triangular matrix!");
        }
    }

    return p_diags;
}


std::vector<double> CSCMatrix::usolve_rows(const std::vector<double>& b) const
{
    assert(M_ == N_);
    assert(M_ == b.size());

    // First (backward) pass to find diagonal entries
    // p_diags is a vector of pointers to the diagonal entries
    std::vector<csint> p_diags = find_upper_diagonals();

    // Compute the row permutation vector
    std::vector<csint> permuted_rows(N_);
    for (csint i = 0; i < N_; i++) {
        permuted_rows[i_[p_diags[i]]] = i;
    }

    // Second (forward) pass to solve the system
    std::vector<double> x = b;

    // Perform the permuted forward solve
    for (csint j = N_ - 1; j >= 0; j--) {
        csint d = p_diags[j];  // pointer to the diagonal entry
        double& x_val = x[j];  // cache diagonal value

        if (x_val != 0) {
            x_val /= v_[d];    // solve for x[d]
            for (csint p = p_[j]; p < p_[j+1]; p++) {
                csint i = permuted_rows[i_[p]];
                if (p != d) {
                    x[i] -= v_[p] * x_val;  // update the off-diagonals
                }
            }
        }
    }

    return x;
}


std::vector<double> CSCMatrix::usolve_cols(const std::vector<double>& b) const
{
    assert(M_ == N_);
    assert(M_ == b.size());

    // First O(N) pass to find the diagonal entries
    // Assume that the last entry in each column has the largest row index
    std::vector<csint> p_diags(N_, -1);
    for (csint j = 0; j < N_; j++) {
        if (p_diags[j] == -1) {
            p_diags[j] = p_[j+1] - 1;  // pointer to the diagonal entry
        } else {
            // We have seen this column index before
            throw std::runtime_error("Matrix is not a permuted lower triangular matrix!");
        }
    }

    // Compute the column permutation vector
    std::vector<csint> permuted_cols(N_);
    for (csint i = 0; i < N_; i++) {
        permuted_cols[i_[p_diags[i]]] = i;
    }

    // Second (forward) pass to solve the system
    std::vector<double> x = b;

    // Perform the permuted forward solve
    for (const auto& j : std::views::reverse(permuted_cols)) {
        csint d = p_diags[j];      // pointer to the diagonal entry
        double& x_val = x[i_[d]];  // cache diagonal value

        if (x_val != 0) {
            x_val /= v_[d];  // solve for x[i_[d]]
            for (csint p = p_[j]; p < p_[j+1] - 1; p++) {
                x[i_[p]] -= v_[p] * x_val;  // update the off-diagonals
            }
        }
    }

    return x;
}


std::tuple<std::vector<csint>, std::vector<csint>, std::vector<csint>>
CSCMatrix::find_tri_permutation() const
{
    assert(M_ == N_);

    // Create a vector of row counts and corresponding set vector
    std::vector<csint> r(N_, 0);
    std::vector<csint> z(N_, 0);  // z[i] is XORed with each column j in row i

    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            r[i_[p]]++;
            z[i_[p]] ^= j;
        }
    }

    // Create a list of singleton row indices
    std::vector<csint> singles;
    singles.reserve(N_);

    for (csint i = 0; i < N_; i++) {
        if (r[i] == 1) {
            singles.push_back(i);
        }
    }

    // Iterate through the columns to get the permutation vectors
    std::vector<csint> p_inv(N_, -1);
    std::vector<csint> q_inv(N_, -1);
    std::vector<csint> p_diags(N_, -1);

    for (csint k = 0; k < N_; k++) {
        // Take a singleton row
        if (singles.empty()) {
            throw std::runtime_error("Matrix is not a permuted triangular matrix!");
        }

        csint i = singles.back();
        singles.pop_back();
        csint j = z[i];  // column index

        // Update the permutations
        p_inv[k] = i;
        q_inv[k] = j;

        // Decrement each row count, and update the set vector
        for (csint p = p_[j]; p < p_[j+1]; p++) {
            csint t = i_[p];
            if (--r[t] == 1) {
                singles.push_back(t);
            }
            z[t] ^= j;  // removes j from the set
            if (t == i) {
                p_diags[k] = p;  // store the pointers to the diagonal entries
            }
        }
    }

    return std::make_tuple(p_inv, q_inv, p_diags);
}


std::vector<double> CSCMatrix::tri_solve_perm(
    const std::vector<double>& b,
    bool is_upper
) const
{
    assert(M_ == N_);
    assert(M_ == b.size());

    // Get the permutation vectors
    // NOTE If upper triangular, the permutation vectors are reversed
    auto [p_inv, q_inv, p_diags] = find_tri_permutation();

    // Get the non-inverse row-permutation vector O(N)
    std::vector<csint> p = inv_permute(p_inv);

    // Copy the RHS vector
    std::vector<double> x =
        (is_upper) ? std::vector<double>(b.rbegin(), b.rend()) : b;

    // Solve the system
    for (csint k = 0; k < N_; k++) {
        csint j = q_inv[k];    // permuted column
        csint d = p_diags[k];  // pointer to the diagonal entry

        // Update the solution
        double& x_val = x[k];  // diagonal of un-permuted row of x
        if (x_val != 0) {
            x_val /= v_[d];  // diagonal entry
            for (csint t = p_[j]; t < p_[j+1]; t++) {
                // off-diagonals from un-permuted row
                if (t != d) {
                    x[p[i_[t]]] -= v_[t] * x_val;
                }
            }
        }
    }

    if (is_upper) {
        std::reverse(x.begin(), x.end());
    }

    return x;
}


std::pair<std::vector<csint>, std::vector<double>> CSCMatrix::spsolve(
    const CSCMatrix& B,
    csint k,
    bool lo
    ) const
{
    // Populate xi with the non-zero indices of x
    std::vector<csint> xi = reach(B, k);
    std::vector<double> x(N_);  // dense output vector

    // Clear non-zeros of x
    for (auto& i : xi) {
        x[i] = 0.0;
    }

    // scatter B(:, k) into x
    for (csint p = B.p_[k]; p < B.p_[k+1]; p++) {
        x[B.i_[p]] = B.v_[p];
    }

    // Solve Lx = b_k or Ux = b_k
    for (auto& j : xi) {  // x(j) is nonzero
        csint J = j;  // j maps to col J of G (NOTE ignore for now)
        if (J < 0) {
            continue;                                // x(j) is not in the pattern of G
        }
        x[j] /= v_[lo ? p_[J] : p_[J+1] - 1];  // x(j) /= G(j, j)
        csint p = lo ? p_[J] + 1 : p_[J];        // lo: L(j,j) 1st entry
        csint q = lo ? p_[J+1]   : p_[J+1] - 1;  // up: U(j,j) last entry
        for (; p < q; p++) {
            x[i_[p]] -= v_[p] * x[j];            // x[i] -= G(i, j) * x[j]
        }
    }

    return std::make_pair(xi, x);
}


std::vector<csint> CSCMatrix::reach(const CSCMatrix& B, csint k) const
{
    std::vector<bool> marked(N_, false);
    std::vector<csint> xi;  // do not initialize for dfs call!
    xi.reserve(N_);

    for (csint p = B.p_[k]; p < B.p_[k+1]; p++) {
        csint j = B.i_[p];  // consider nonzero B(j, k)
        if (!marked[j]) {
            xi = dfs(j, marked, xi);
        }
    }

    // xi is returned from dfs in reverse order, since it is a stack
    return std::vector<csint>(xi.rbegin(), xi.rend());
}


std::vector<csint>& CSCMatrix::dfs(
    csint j,
    std::vector<bool>& marked,
    std::vector<csint>& xi
    ) const
{
    std::vector<csint> rstack, pstack;  // recursion and pause stacks
    rstack.reserve(N_);
    pstack.reserve(N_);

    rstack.push_back(j);       // initialize the recursion stack

    bool done = false;  // true if no unvisited neighbors

    while (!rstack.empty()) {
        j = rstack.back();  // get j from the top of the recursion stack
        csint jnew = j;  // j maps to col jnew of G (NOTE ignore p_inv for now)

        if (!marked[j]) {
            marked[j] = true;  // mark node j as visited
            pstack.push_back((jnew < 0) ? 0 : p_[jnew]);
        }

        done = true;  // node j done if no unvisited neighbors
        csint q = (jnew < 0) ? 0 : p_[jnew+1];

        // examine all neighbors of j
        for (csint p = pstack.back(); p < q; p++) {
            csint i = i_[p];        // consider neighbor node i
            if (!marked[i]) {
                pstack.back() = p;    // pause dfs of node j
                rstack.push_back(i);  // start dfs at node i
                done = false;         // node j has unvisited neighbors
                break;
            }
        }

        if (done) {
            pstack.pop_back();
            rstack.pop_back();  // node j is done; pop it from the stack
            xi.push_back(j);    // node j is the next on the output stack
        }
    }

    return xi;
}


/*------------------------------------------------------------------------------
 *         Printing
 *----------------------------------------------------------------------------*/
void CSCMatrix::print_dense(std::ostream& os) const
{
    print_dense_vec(toarray('F'), M_, N_, 'F', os);
}


void CSCMatrix::print_elems_(std::ostream& os, const csint start, const csint end) const
{
    csint n = 0;  // number of elements printed
    for (csint j = 0; j < N_; j++) {
        for (csint p = p_[j]; p < p_[j + 1]; p++) {
            if ((n >= start) && (n < end)) {
                os << "(" << i_[p] << ", " << j << "): " << v_[p] << std::endl;
            }
            n++;
        }
    }
}


void CSCMatrix::print(std::ostream& os, const bool verbose, const csint threshold) const
{
    csint nnz_ = nnz();
    os << "<" << format_desc_ << " matrix" << std::endl;
    os << "        with " << nnz_ << " stored elements "
        << "and shape (" << M_ << ", " << N_ << ")>" << std::endl;

    if (verbose) {
        if (nnz_ < threshold) {
            // Print all elements
            print_elems_(os, 0, nnz_);  // FIXME memory leak?
        } else {
            // Print just the first and last 3 non-zero elements
            print_elems_(os, 0, 3);
            os << "..." << std::endl;
            print_elems_(os, nnz_ - 3, nnz_);
        }
    }
}


std::ostream& operator<<(std::ostream& os, const CSCMatrix& A)
{
    A.print(os, true);  // verbose printing assumed
    return os;
}


}  // namespace cs

/*==============================================================================
 *============================================================================*/
