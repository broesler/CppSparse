//==============================================================================
//    File: csc.h
// Created: 2024-10-09 20:57
//  Author: Bernie Roesler
//
//  Description: The header file for the CSparse++ package with definitions of
//    the matrix classes and associated functions.
//
//==============================================================================

#ifndef _CSC_H_
#define _CSC_H_


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

        // Provide data, coordinates, and shsape as vectors
        CSCMatrix(
            const std::vector<double>& vals,
            const std::vector<csint>& indices,
            const std::vector<csint>& indptr,
            const std::array<csint, 2>& shape
        );

        CSCMatrix(csint M, csint N, csint nzmax=0);  // allocate dims + nzmax
        CSCMatrix(const COOMatrix& A);               // Exercise 2.2
        CSCMatrix(const std::vector<double>& A, csint M, csint N);  // Exercise 2.16

        CSCMatrix& realloc(csint nzmax=0);           // re-allocate vectors

        // ---------- Accessors
        csint nnz() const;                   // number of non-zeros
        csint nzmax() const;                 // maximum number of non-zeros
        std::array<csint, 2> shape() const;  // the dimensions of the matrix

        CSCMatrix to_canonical() const;
        bool has_sorted_indices() const;
        bool has_canonical_format() const;
        bool is_symmetric() const;  // Exercise 2.13

        const std::vector<csint>& indices() const;     // indices and data
        const std::vector<csint>& indptr() const;
        const std::vector<double>& data() const;

        // Access an element by index, but do not change its value
        const double operator()(csint i, csint j) const;
        // double& operator()(csint, csint);  // assignment

        // ---------- Format Conversions
        COOMatrix tocoo() const;  // Exercise 2.2 Matlab's find.

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

        CSCMatrix& dropzeros();
        CSCMatrix& droptol(double tol);

        // Exercise 2.15
        static bool in_band(csint i, csint j, double Aij, void *limits);
        CSCMatrix band(const int kl, const int ku);

        // TODO possibly make these private? Or define in SparseMatrix base.
        static bool nonzero(csint i, csint j, double Aij, void *tol);
        static bool abs_gt_tol(csint i, csint j, double Aij, void *tol);

        // Matrix-vector multiply and add via C-style function
        friend std::vector<double> gaxpy(
            const CSCMatrix& A,
            const std::vector<double>& x,
            std::vector<double> y
        );

        // Exercise 2.1
        friend std::vector<double> gatxpy(
            const CSCMatrix& A,
            const std::vector<double>& x,
            std::vector<double> y
        );

        // Exercise 2.3
        friend std::vector<double> sym_gaxpy(
            const CSCMatrix& A,
            const std::vector<double>& x,
            std::vector<double> y
        );

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
        friend csint scatter(
            const CSCMatrix& A,
            csint j,
            double beta,
            std::vector<csint>& w,
            std::vector<double>& x,
            csint mark,
            CSCMatrix& C,
            csint nz,
            bool fs
        );

        // Permutations
        CSCMatrix permute(const std::vector<csint> p_inv, const std::vector<csint> q) const;
        CSCMatrix symperm(const std::vector<csint> p_inv) const;

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
        CSCMatrix index_lazy(
            const std::vector<csint>& rows,
            const std::vector<csint>& cols
        ) const;

        CSCMatrix index(
            const std::vector<csint>& rows,
            const std::vector<csint>& cols
        ) const;

        // ---------- Other
        void print(
            std::ostream& os=std::cout,
            const bool verbose=false,
            const csint threshold=1000
        ) const;

        // Type conversions
        operator double() const {
            assert((M_ == 1) && (N_ == 1));
            return v_[0];
        }
};


/*------------------------------------------------------------------------------
 *          Free Functions
 *----------------------------------------------------------------------------*/
CSCMatrix operator+(const CSCMatrix&, const CSCMatrix&);

std::vector<double> operator*(const CSCMatrix&, const std::vector<double>&);
CSCMatrix operator*(const CSCMatrix&, const CSCMatrix&);
CSCMatrix operator*(const CSCMatrix&, const double);
CSCMatrix operator*(const double, const CSCMatrix&);

std::ostream& operator<<(std::ostream&, const CSCMatrix&);

#endif

//==============================================================================
//==============================================================================
