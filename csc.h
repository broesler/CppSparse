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

    void print_elems_(std::ostream&, csint, csint) const;

    public:
        // ---------- Constructors
        CSCMatrix();

        // Provide data, coordinates, and shsape as vectors
        CSCMatrix(
            const std::vector<double>&,
            const std::vector<csint>&,
            const std::vector<csint>&,
            const std::array<csint, 2>&
        );

        CSCMatrix(csint, csint, csint nzmax=0);  // allocate dims + nzmax

        // ---------- Accessors
        csint nnz() const;                   // number of non-zeros
        csint nzmax() const;                 // maximum number of non-zeros
        std::array<csint, 2> shape() const;  // the dimensions of the matrix

        const std::vector<csint>& indices() const;     // indices and data
        const std::vector<csint>& indptr() const;
        const std::vector<double>& data() const;

        // Access an element by index, but do not change its value
        const double operator()(csint, csint) const;
        // double& operator()(csint, csint);

        // ---------- Math Operations
        CSCMatrix T() const;  // transpose a copy
        CSCMatrix& sum_duplicates();

        // Keep or remove entries
        CSCMatrix& fkeep(
            bool (*fk) (csint, csint, double, void *),
            void *other
        );

        CSCMatrix& dropzeros();
        CSCMatrix& droptol(double);

        // TODO possibly make these private? Or define in SparseMatrix base.
        static bool nonzero(csint, csint, double, void *);
        static bool abs_gt_tol(csint, csint, double, void *);

        // Matrix-vector multiply and add via C-style function
        friend std::vector<double> gaxpy(
            const CSCMatrix&,
            const std::vector<double>&,
            std::vector<double>
        );

        // Matrix-vector multiply operator overload
        friend std::vector<double> operator*(
            const CSCMatrix&,
            const std::vector<double>&
        );

        // TODO implement algorithm
        friend CSCMatrix operator*(const CSCMatrix&, const CSCMatrix&);

        friend csint scatter(const CSCMatrix&, csint, double,
            std::vector<csint>&, std::vector<double>&, csint, CSCMatrix&, csint);

        // ---------- Other
        void print(
            std::ostream& os=std::cout,
            bool verbose=false,
            csint threshold=1000
        ) const;
};


/*------------------------------------------------------------------------------
 *          Free Functions
 *----------------------------------------------------------------------------*/
std::ostream& operator<<(std::ostream&, const CSCMatrix&);

std::vector<double> operator+(
    const std::vector<double>&,
    const std::vector<double>&
);

#endif

//==============================================================================
//==============================================================================
