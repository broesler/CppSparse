//==============================================================================
//    File: csc.h
// Created: 2024-10-09 20:57
//  Author: Bernie Roesler
//
//  Description: Implements the compressed sparse column matrix class.
//
//==============================================================================

#ifndef _CSC_H_
#define _CSC_H_

namespace cs {

class CSCMatrix
{
    // Private members
    static constexpr std::string_view format_desc_ = "Compressed Sparse Column";
    std::vector<double> v_;  // numerical values, size nzmax
    std::vector<csint> i_;   // row indices, size nzmax
    std::vector<csint> p_;   // column pointers (CSC size N_);
    csint M_ = 0;            // number of rows
    csint N_ = 0;            // number of columns
    bool has_sorted_indices_ = false;
    bool has_canonical_format_ = false;

    void print_elems_(std::ostream& os, const csint start, const csint end) const;

    public:
        friend class COOMatrix;

        // ---------- Constructors
        CSCMatrix();

        // Provide data, coordinates, and shape as vectors
        CSCMatrix(
            const std::vector<double>& vals,
            const std::vector<csint>& indices,
            const std::vector<csint>& indptr,
            const Shape& shape
        );

        CSCMatrix(csint M, csint N, csint nzmax=0);  // allocate dims + nzmax
        CSCMatrix(const COOMatrix& A);               // Exercise 2.2
        CSCMatrix(const std::vector<double>& A, csint M, csint N);  // Exercise 2.16

        CSCMatrix& realloc(csint nzmax=0);           // re-allocate vectors

        // ---------- Accessors
        csint nnz() const;                   // number of non-zeros
        csint nzmax() const;                 // maximum number of non-zeros
        Shape shape() const;  // the dimensions of the matrix

        CSCMatrix& to_canonical();
        bool has_sorted_indices() const;
        bool has_canonical_format() const;
        bool is_symmetric() const;  // Exercise 2.13

        const std::vector<csint>& indices() const;     // indices and data
        const std::vector<csint>& indptr() const;
        const std::vector<double>& data() const;

        // Access an element by index, but do not change its value
        const double operator()(csint i, csint j) const;  // v = A(i, j)
        double& operator()(csint i, csint j);             // A(i, j) = v

        // Exercise 2.25 assign by index
        CSCMatrix& assign(csint i, csint j, double v);
        CSCMatrix& assign(
            const std::vector<csint>& i,
            const std::vector<csint>& j,
            const std::vector<double>& C  // dense column-major
        );
        CSCMatrix& assign(
            const std::vector<csint>& rows,
            const std::vector<csint>& cols,
            const CSCMatrix& C
        );

        // Helper for assign and operator()
        double& insert(csint i, csint j, double v, csint p);

        // ---------- Format Conversions
        COOMatrix tocoo() const;  // Exercise 2.2 Matlab's find.

        // inverse of Exercise 2.16
        std::vector<double> toarray(const char order='F') const;

        // ---------- Math Operations
        CSCMatrix transpose() const;  // transpose a copy
        CSCMatrix T() const;          // transpose a copy (alias)

        CSCMatrix tsort() const;      //  Exercise 2.7
        CSCMatrix& qsort();           //  Exercise 2.8 sort in-place
        CSCMatrix& sort();            //  Exercise 2.11 efficient sort in-place

        CSCMatrix& sum_duplicates();

        // Keep or remove entries
        CSCMatrix& fkeep(
            bool (*fk) (csint i, csint j, double Aij, void *tol),
            void *other
        );

        // Overload for copies
        CSCMatrix fkeep(
            bool (*fk) (csint i, csint j, double Aij, void *tol),
            void *other
        ) const;

        CSCMatrix& dropzeros();
        CSCMatrix& droptol(double tol);

        // Exercise 2.15
        static bool in_band(csint i, csint j, double Aij, void *limits);
        CSCMatrix& band(csint kl, csint ku);
        CSCMatrix band(csint kl, csint ku) const;  // overload for copies

        // TODO possibly make these private? Or define in SparseMatrix base.
        static bool nonzero(csint i, csint j, double Aij, void *tol);
        static bool abs_gt_tol(csint i, csint j, double Aij, void *tol);

        // Matrix-vector multiply and add via C-style function
        std::vector<double> gaxpy(
            const std::vector<double>& x,
            const std::vector<double>& y
        ) const;

        // Exercise 2.1
        std::vector<double> gatxpy(
            const std::vector<double>& x,
            const std::vector<double>& y
        ) const;

        // Exercise 2.3
        std::vector<double> sym_gaxpy(
            const std::vector<double>& x,
            const std::vector<double>& y
        ) const;

        // Exercise 2.27(a) cs_gaxpy with matrix x, y in column-major order
        std::vector<double> gaxpy_col(
            const std::vector<double>& X,
            const std::vector<double>& Y
        ) const;

        // Exercise 2.27(b) cs_gaxpy with matrix x, y in row-major order
        std::vector<double> gaxpy_row(
            const std::vector<double>& X,
            const std::vector<double>& Y
        ) const;

        // Exercise 2.27(c) cs_gaxpy with matrix x, y in column-major order, but
        // operating on blocks of columns
        std::vector<double> gaxpy_block(
            const std::vector<double>& X,
            const std::vector<double>& Y
        ) const;

        // Exercise 2.28(a) cs_gatxpy with matrix x, y in column-major order
        std::vector<double> gatxpy_col(
            const std::vector<double>& X,
            const std::vector<double>& Y
        ) const;

        // Exercise 2.28(b) cs_gatxpy with matrix x, y in row-major order
        std::vector<double> gatxpy_row(
            const std::vector<double>& X,
            const std::vector<double>& Y
        ) const;

        // Exercise 2.28(c) cs_gatxpy with matrix x, y in column-major order, but
        // operating on blocks of columns
        std::vector<double> gatxpy_block(
            const std::vector<double>& X,
            const std::vector<double>& Y
        ) const;

        // Exercise 2.4
        CSCMatrix scale(const std::vector<double>& r, const std::vector<double> c) const;

        // Multiply (see cs_multiply)
        std::vector<double> dot(const std::vector<double>& x) const;
        CSCMatrix dot(const double c) const;
        CSCMatrix dot(const CSCMatrix& B) const;

        double vecdot(const CSCMatrix& y) const;  // Exercise 2.18 cs_dot

        CSCMatrix dot_2x(const CSCMatrix& B) const;  // Exercise 2.20

        // Matrix-matrix add via C-style function
        CSCMatrix add(const CSCMatrix& B) const;

        friend CSCMatrix add_scaled(
            const CSCMatrix& A,
            const CSCMatrix& B,
            double alpha,
            double beta
        );

        // Exercise 2.21
        friend std::vector<csint> saxpy(
            const CSCMatrix& a,
            const CSCMatrix& b,
            std::vector<csint>& w,
            std::vector<double>& x
        );

        // Helper for add and multiply
        csint scatter(
            csint j,
            double beta,
            std::vector<csint>& w,
            std::vector<double>& x,
            csint mark,
            CSCMatrix& C,
            csint nz,
            bool fs
        ) const;

        // Permutations
        CSCMatrix permute(
            const std::vector<csint> p_inv,
            const std::vector<csint> q
        ) const;

        CSCMatrix symperm(const std::vector<csint> p_inv) const;

        // Exercise 2.26 permuted transpose
        CSCMatrix permute_transpose(
            const std::vector<csint>& p_inv,
            const std::vector<csint>& q_inv
        ) const;

        CSCMatrix permute_rows(const std::vector<csint> p_inv) const;
        CSCMatrix permute_cols(const std::vector<csint> q) const;

        double norm() const;

        // Exercise 2.12 "cs_ok"
        bool is_valid(const bool sorted=false, const bool values=false) const;

        // Exercise 2.22 concatenation (see cs_hcat, cs_vcat)
        friend CSCMatrix hstack(const CSCMatrix& A, const CSCMatrix& B);
        friend CSCMatrix vstack(const CSCMatrix& A, const CSCMatrix& B);

        // Exercise 2.23 slice with contiguous indices
        CSCMatrix slice(
            const csint i_start,
            const csint i_end,
            const csint j_start,
            const csint j_end
        ) const;

        // Exercise 2.24 index with arbitrary indices
        CSCMatrix index(
            const std::vector<csint>& rows,
            const std::vector<csint>& cols
        ) const;

        // Exercise 2.28 add empty rows or columns
        CSCMatrix add_empty_top(const csint k) const;
        CSCMatrix add_empty_bottom(const csint k) const;
        CSCMatrix add_empty_left(const csint k) const;
        CSCMatrix add_empty_right(const csint k) const;

        // Sum rows or columns
        std::vector<double> sum_rows() const;
        std::vector<double> sum_cols() const;
        // double sum() const;
        // std::vector<double> sum(const int axis) const;

        //----------------------------------------------------------------------
        //        Triangular Matrix Solutions
        //----------------------------------------------------------------------
        std::vector<double>  lsolve(const std::vector<double>& b) const;
        std::vector<double> ltsolve(const std::vector<double>& b) const;
        std::vector<double>  usolve(const std::vector<double>& b) const;
        std::vector<double> utsolve(const std::vector<double>& b) const;

        // Exercise 3.8 optimized versions of [lu]solve
        std::vector<double> lsolve_opt(const std::vector<double>& b) const;
        std::vector<double> usolve_opt(const std::vector<double>& b) const;

        // Exercise 3.3, 3.4 solve row-permuted triangular system
        std::vector<double> lsolve_rows(const std::vector<double>& b) const;
        std::vector<double> usolve_rows(const std::vector<double>& b) const;

        // Exercise 3.5, 3.6 solve column-permuted triangular system
        std::vector<double> lsolve_cols(const std::vector<double>& b) const;
        std::vector<double> usolve_cols(const std::vector<double>& b) const;

        std::vector<csint> find_lower_diagonals() const;
        std::vector<csint> find_upper_diagonals() const;

        // Exercise 3.7 Generalized permuted triangular solve
        std::vector<double> tri_solve_perm(
            const std::vector<double>& b,
            bool is_upper=false
        ) const;

        std::tuple<std::vector<csint>, std::vector<csint>, std::vector<csint>>
            find_tri_permutation() const;

        // Sparse matrix solve
        std::pair<std::vector<csint>, std::vector<double>> spsolve(
            const CSCMatrix& B,
            csint k,
            bool lo
        ) const;

        std::vector<csint> reach(const CSCMatrix& B, csint k) const;

        std::vector<csint>& dfs(
            csint j,
            std::vector<bool>& is_marked,
            std::vector<csint>& xi
        ) const;

        //----------------------------------------------------------------------
        //        Cholesky Decomposition
        //----------------------------------------------------------------------
        std::vector<csint> etree(bool ata=false) const;
        std::vector<csint> ereach(
            csint k,
            const std::vector<csint>& parent
        ) const;

        // ---------- Other
        void print_dense(std::ostream& os=std::cout) const;

        void print(
            std::ostream& os=std::cout,
            const bool verbose=false,
            const csint threshold=1000
        ) const;

        // Type conversions
        operator const double() const {
            assert((M_ == 1) && (N_ == 1));
            return v_[0];
        }
};


/*------------------------------------------------------------------------------
 *          Free Functions
 *----------------------------------------------------------------------------*/
CSCMatrix operator+(const CSCMatrix& A, const CSCMatrix& B);

std::vector<double> operator*(const CSCMatrix& A, const std::vector<double>& B);
CSCMatrix operator*(const CSCMatrix& A, const CSCMatrix& B);
CSCMatrix operator*(const CSCMatrix& A, const double c);
CSCMatrix operator*(const double c, const CSCMatrix& A);

std::ostream& operator<<(std::ostream& os, const CSCMatrix& A);


}  // namespace cs

#endif  // _CSC_H_

//==============================================================================
//==============================================================================
