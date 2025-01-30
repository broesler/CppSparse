/*==============================================================================
 *     File: coo.cpp
 *  Created: 2024-10-01 21:07
 *   Author: Bernie Roesler
 *
 *  Description: Implements the sparse matrix classes
 *
 *============================================================================*/

#include <algorithm>  // for std::max_element
#include <random>
#include <sstream>  // for std::stringstream

#include "csparse.h"

namespace cs {

/*------------------------------------------------------------------------------
 *     Constructors
 *----------------------------------------------------------------------------*/
COOMatrix::COOMatrix() {};


COOMatrix::COOMatrix(
    const std::vector<double>& v,
    const std::vector<csint>& i,
    const std::vector<csint>& j,
    csint M,
    csint N
)
    : v_(v),
      i_(i),
      j_(j),
      M_(M ? M : *std::max_element(i_.begin(), i_.end()) + 1),
      N_(N ? N : *std::max_element(j_.begin(), j_.end()) + 1)
{
    // Check that all vectors are the same size
    assert(v_.size() == i_.size());
    assert(v_.size() == j_.size());
    assert(M_ > 0 && N_ > 0);
}


COOMatrix::COOMatrix(csint M, csint N, csint nzmax)
    : M_(M),
      N_(N) 
{
    v_.reserve(nzmax);
    i_.reserve(nzmax);
    j_.reserve(nzmax);
}


COOMatrix::COOMatrix(const CSCMatrix& A)
    : v_(A.nnz()),
      i_(A.nnz()),
      j_(A.nnz())
{
    // Get the shape
    std::tie(M_, N_) = A.shape();
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


COOMatrix::COOMatrix(std::istream& fp)
{
    csint i, j;
    double v;

    while (fp) {
        std::string line;
        std::getline(fp, line);
        if (!line.empty()) {
            std::stringstream ss(line);
            if (!(ss >> i >> j >> v))
                throw std::runtime_error("File is not in (i, j, v) format!");
            else
                assign(i, j, v);
        }
    }
}


COOMatrix COOMatrix::random(csint M, csint N, double density, unsigned int seed)
{
    csint nzmax = M * N * density;
    COOMatrix A(M, N, nzmax);

    if (seed == 0)
        seed = std::random_device{}();

    std::default_random_engine rng(seed);
    std::uniform_int_distribution<csint> row_dist(0, M - 1);
    std::uniform_int_distribution<csint> col_dist(0, N - 1);
    std::uniform_real_distribution<double> value_dist(0.0, 1.0);

    for (csint k = 0; k < nzmax; k++) {
        csint i = row_dist(rng);
        csint j = col_dist(rng);
        double v = value_dist(rng);
        A.assign(i, j, v);
    }

    return A;
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
const std::vector<csint>& COOMatrix::column() const { return j_; }
const std::vector<double>& COOMatrix::data() const { return v_; }


// cs_entry
COOMatrix& COOMatrix::assign(csint i, csint j, double v)
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
COOMatrix& COOMatrix::assign(
    std::vector<csint> rows,
    std::vector<csint> cols,
    std::vector<double> vals
    )
{
    assert(rows.size() == cols.size());
    csint N = rows.size();
    assert(vals.size() == (N * N));

    for (csint i = 0; i < N; i++) {
        for (csint j = 0; j < N; j++) {
            i_.push_back(rows[i]);
            j_.push_back(cols[j]);
            v_.push_back(vals[i + j*N]);  // column-major order
        }
    }

    M_ = std::max(M_, *std::max_element(rows.begin(), rows.end()) + 1);
    N_ = std::max(N_, *std::max_element(cols.begin(), cols.end()) + 1);

    return *this;
}


/*------------------------------------------------------------------------------
 *          Format Conversions 
 *----------------------------------------------------------------------------*/
CSCMatrix COOMatrix::compress() const 
{
    csint nnz_ = nnz();
    CSCMatrix C(M_, N_, nnz_);
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
        C.v_[p] = v_[k];
    }

    return C;
}


// Exercise 2.9
CSCMatrix COOMatrix::tocsc() const { return CSCMatrix(*this); }


std::vector<double> COOMatrix::toarray(const char order) const
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
    return COOMatrix(this->v_, this->j_, this->i_, this->N_, this->M_);
}


COOMatrix COOMatrix::T() const { return this->transpose(); }


std::vector<double> COOMatrix::dot(const std::vector<double>& x) const
{
    assert(N_ == x.size());
    std::vector<double> out(x.size());

    for (csint p = 0; p < nnz(); p++) {
        out[i_[p]] += v_[p] * x[j_[p]];
    }

    return out;
}


// Exercise 2.10
std::vector<double> operator*(const COOMatrix& A, const std::vector<double>& x)
{ return A.dot(x); }


/*------------------------------------------------------------------------------
 *         Printing
 *----------------------------------------------------------------------------*/
void COOMatrix::print_elems_(std::ostream& os, const csint start, const csint end) const
{
    for (csint k = start; k < end; k++) {
        os << "(" << i_[k] << ", " << j_[k] << "): " << v_[k] << std::endl;
    }
}


void COOMatrix::print(std::ostream& os, const bool verbose, const csint threshold) const
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


std::ostream& operator<<(std::ostream& os, const COOMatrix& A)
{
    A.print(os, true);  // verbose printing assumed
    return os;
}


}  // namespace cs

/*==============================================================================
 *============================================================================*/
