/*==============================================================================
 *     File: csparse.cpp
 *  Created: 2024-10-01 21:07
 *   Author: Bernie Roesler
 *
 *  Description: Implements the sparse matrix classes
 *
 *============================================================================*/

#include <cassert>
#include <sstream>

#include "csparse.h"


/*------------------------------------------------------------------------------
 *     Constructors
 *----------------------------------------------------------------------------*/
COOMatrix::COOMatrix() {};

/** Construct a COOMatrix from arrays of values and coordinates.
 *
 * The entries are *not* sorted in any order, and duplicates are allowed. Any
 * duplicates will be summed.
 *
 * The matrix shape `(M, N)` will be inferred from the maximum indices given.
 *
 * @param v the values of the entries in the matrix
 * @param i, j the non-negative integer row and column indices of the values
 * @return a new COOMatrix object
 */
COOMatrix::COOMatrix(
    const std::vector<double>& v,
    const std::vector<csint>& i,
    const std::vector<csint>& j
    )
    : v_(v),
      i_(i),
      j_(j),
      M_(*std::max_element(i_.begin(), i_.end()) + 1),  // zero-based indexing
      N_(*std::max_element(j_.begin(), j_.end()) + 1)
{}


/** Allocate a COOMatrix for a given shape and number of non-zeros.
 *
 * @param M, N  integer dimensions of the rows and columns
 * @param nzmax integer capacity of space to reserve for non-zeros
 */
COOMatrix::COOMatrix(csint M, csint N, csint nzmax)
    : M_(M),
      N_(N)
{
    v_.reserve(nzmax);
    i_.reserve(nzmax);
    j_.reserve(nzmax);
}


/** Read a COOMatrix matrix from a file.
 *
 * The file is expected to be in "triplet format" `(i, j, v)`, where `(i, j)`
 * are the index coordinates, and `v` is the value to be assigned.
 *
 * @param fp    a reference to the file stream.
 * @throws std::runtime_error if file format is not in triplet format
 */
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

/*------------------------------------------------------------------------------
 *         Accessors
 *----------------------------------------------------------------------------*/
csint COOMatrix::nnz() const { return v_.size(); }
csint COOMatrix::nzmax() const { return v_.capacity(); }

std::array<csint, 2> COOMatrix::shape() const
{
    return std::array<csint, 2> {M_, N_};
}

const std::vector<csint>& COOMatrix::row() const { return i_; }
const std::vector<csint>& COOMatrix::column() const { return j_; }
const std::vector<double>& COOMatrix::data() const { return v_; }


/** Assign a value to a pair of indices.
 *
 * Note that there is no argument checking other than for positive indices.
 * Assigning to an index that is outside of the dimensions of the matrix will
 * just increase the size of the matrix accordingly.
 *
 * Duplicate entries are also allowed to ease incremental construction of
 * matrices from files, or, e.g., finite element applications. Duplicates will be
 * summed upon compression to sparse column/row form.
 *
 * @param i, j  integer indices of the matrix
 * @param v     the value to be assigned
 *
 * @see cs_entry Davis p 12.
 */
void COOMatrix::assign(csint i, csint j, double v)
{
    assert ((i >= 0) && (j >= 0));

    i_.push_back(i);
    j_.push_back(j);
    v_.push_back(v);

    assert(v_.size() == i_.size());
    assert(v_.size() == j_.size());

    M_ = std::max(M_, i+1);
    N_ = std::max(N_, j+1);
}


/*------------------------------------------------------------------------------
       Math Operations 
----------------------------------------------------------------------------*/
// Transpose the matrix as a copy
COOMatrix COOMatrix::T() const
{
    return COOMatrix(this->v_, this->j_, this->i_);
}


/*------------------------------------------------------------------------------
 *         Other
 *----------------------------------------------------------------------------*/
/** Print the matrix
 *
 * @param os          the output stream, defaults to std::cout
 * @param verbose     if True, print all non-zeros and their coordinates
 * @param threshold   if `nnz > threshold`, print only the first and last
 *        3 entries in the matrix. Otherwise, print all entries.
 */
void COOMatrix::print(std::ostream& os, bool verbose, csint threshold) const
{
    csint nnz_ = nnz();
    os << "<COOrdinate sparse matrix" << std::endl;
    os << "        with " << nnz_ << " stored elements "
        << "and shape (" << M_ << ", " << N_ << ")>" << std::endl;

    if (verbose) {
        if (nnz_ < threshold) {
            // Print all elements
            print_elems_(os, 0, nnz_);
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


/*==============================================================================
 *============================================================================*/
