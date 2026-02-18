/*==============================================================================
 *     File: printmatrix_demo.cpp
 *  Created: 2025-04-09 10:23
 *   Author: Bernie Roesler
 *
 *  Description: Test printing functions.
 *
 *============================================================================*/

#include <cmath>    // nan
#include <print>
#include <vector>

#include "csparse.h"


using namespace cs;


int main()
{
    // Define a rectangular matrix with some values
    csint M = 5;
    csint N = 4;

    double pi = 4.0 * atan(1.0);

    // ---------- Default:
    //         1.0000       nan         0      -inf
    //            inf    2.0000 1096.6332         0
    //              0   20.0855    3.0000   -2.7183
    //              0    3.1416         0    4.0000
    //              0         0    0.7071         0

    // Define the diagonal
    std::vector<csint> i{0, 1, 2, 3, 1, 0, 0, 3, 2, 2, 1, 4};
    std::vector<csint> j{0, 1, 2, 3, 0, 3, 1, 1, 3, 1, 2, 2};
    std::vector<double> v{
        1, 2, 3, 4,
        INFINITY, -INFINITY, NAN,
        pi, -exp(1), exp(3), exp(7), sqrt(2) / 2
    };

    COOMatrix Ac{v, i, j, {M, N}};

    auto A = Ac.tocsc();
    
    std::println("---------- Default:");
    A.print_dense();

    std::println("---------- Scale a column extra large:");
    for (csint i = 0; i < M; ++i) {
        A(i, 2) *= 1e10;
    }
    A = A.to_canonical();
    A.print_dense();

    std::println("---------- Scale a column extra small:");
    for (csint i = 0; i < M; ++i) {
        A(i, 2) *= 1e-20;
    }
    A = A.to_canonical();
    A.print_dense();

    std::println("---------- suppress=false:");
    A.print_dense(4, false);

    std::println("---------- precision=16:");
    A.print_dense(16);

    std::println("---------- precision=16, suppress=false:");
    A.print_dense(16, false);

    return EXIT_SUCCESS;
}

/*==============================================================================
 *============================================================================*/
