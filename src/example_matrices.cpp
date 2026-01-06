/*==============================================================================
 *     File: example_matrices.cpp
 *  Created: 2025-03-20 15:11
 *   Author: Bernie Roesler
 *
 *  Description: Definitions of example matrices for testing.
 *
 *============================================================================*/

#include <numeric>  // iota
#include <random>
#include <vector>

#include "types.h"
#include "example_matrices.h"
#include "csc.h"
#include "coo.h"

namespace cs {


// 4 x 4 non-symmetric example. Davis, pp 7-8, Eqn (2.1)
COOMatrix davis_example_small()
{
    std::vector<csint>  i = {2,    1,    3,    0,    1,    3,    3,    1,    0,    2};
    std::vector<csint>  j = {2,    0,    3,    2,    1,    0,    1,    3,    0,    1};
    std::vector<double> v = {3.0,  3.1,  1.0,  3.2,  2.9,  3.5,  0.4,  0.9,  4.5,  1.7};
    return COOMatrix {v, i, j};
}


// 11 x 11 symmetric, positive definite Cholesky example. Davis, Fig 4.2, p 39.
CSCMatrix davis_example_chol()
{
    csint N = 11;  // total number of rows and columns

    // Only off-diagonal elements
    std::vector<csint> rows = {5, 6, 2, 7, 9, 10, 5, 9, 7, 10, 8, 9, 10, 9, 10, 10};
    std::vector<csint> cols = {0, 0, 1, 1, 2,  2, 3, 3, 4,  4, 5, 5,  6, 7,  7,  9};
    std::vector<double> vals(rows.size(), 1.0);

    // Values for the lower triangle
    CSCMatrix L = COOMatrix(vals, rows, cols, {N, N}).tocsc();

    // Create the symmetric matrix A
    CSCMatrix A = L + L.T();

    // Set the diagonal to ensure positive definiteness
    for (csint i = 0; i < N; i++) {
        A.assign(i, i, i + 10);
    }

    return A;
}


// 8 x 8, non-symmetric QR example. Davis, Figure 5.1, p 74.
CSCMatrix davis_example_qr(double add_diag, bool random_vals)
{
    // Define the test matrix A (See Davis, Figure 5.1, p 74)
    std::vector<csint> rows = {0, 1, 2, 3, 4, 5, 6,
                               3, 6, 1, 6, 0, 2, 5, 7, 4, 7, 0, 1, 3, 7, 5, 6};
    std::vector<csint> cols = {0, 1, 2, 3, 4, 5, 6,
                               0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7};

    std::vector<double> vals(rows.size());
    if (random_vals) {
        // Randomize the non-zero values
        unsigned int seed = std::random_device{}();
        std::default_random_engine rng(seed);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        std::generate(
            vals.begin(),
            vals.end(),
            [&rng, &uniform]() { return uniform(rng); }
        );
    } else {
        // Label the diagonal elements 1..7, skipping the 8th
        std::iota(vals.begin(), vals.begin() + 7, 1.0);
        // All non-diagonal values set to 1.0
        std::fill(vals.begin() + 7, vals.end(), 1.0);
    }

    COOMatrix A(vals, rows, cols);

    // NOTE no need to include A[7, 7], we previously thought we needed to
    // include it even though it is numerically zero to keep structural full
    // rank, but this is not necessary.
    // A.assign(7, 7, 0.0);

    // Set the diagonal
    if (add_diag) {
        for (csint i = 0; i < 8; i++) {
            A.insert(i, i, add_diag);  // duplicates will be added
        }
    }

    // Non-canonical format for testing
    return A.compress().sum_duplicates();
}


// 10 x 10 symmetric, positive definite AMD example. Davis, Figure 7.1, p 101.
CSCMatrix davis_example_amd()
{
    csint N = 10;  // total number of rows (and columns)

    // Only off-diagonal elements
    std::vector<csint> rows = {0, 3, 5, 1, 4, 5, 8, 2, 4, 5, 6, 3, 6, 7, 
                               4, 6, 8, 5, 6, 7, 8, 9, 7, 8, 9, 8, 9, 9};
    std::vector<csint> cols = {0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3,
                               4, 4, 4, 5, 6, 6, 6, 6, 7, 7, 7, 8, 8, 9};
    std::vector<double> vals(rows.size(), 1.0);

    // Values for the lower triangle
    CSCMatrix L = COOMatrix(vals, rows, cols, {N, N}).tocsc();

    // Create the symmetric matrix A
    CSCMatrix A = L + L.T();

    // Set the diagonal to ensure positive definiteness
    for (csint i = 0; i < N; i++) {
        A.assign(i, i, i + 10);
    }

    return A.to_canonical();
}


// Build matrices with sorted columns for internal testing
CSCMatrix E_mat()
{
    return COOMatrix(
        std::vector<double> {1, -2, 1, 1},  // vals
        std::vector<csint>  {0,  1, 1, 2},  // rows
        std::vector<csint>  {0,  0, 1, 2}   // cols
    ).tocsc();
}


CSCMatrix A_mat()
{
    return COOMatrix(
        std::vector<double> {2, 4, -2, 1, -6, 7, 1, 2},  // vals
        std::vector<csint>  {0, 1,  2, 0,  1, 2, 0, 2},  // rows
        std::vector<csint>  {0, 0,  0, 1,  1, 1, 2, 2}   // cols
    ).tocsc();
}


}  // namespace cs


/*==============================================================================
 *============================================================================*/
