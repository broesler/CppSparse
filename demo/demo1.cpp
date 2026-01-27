/*==============================================================================
 *     File: demo1.cpp
 *  Created: 2025-05-06 11:20
 *   Author: Bernie Roesler
 *
 *  Description: Demonstration of basic CSparse library functions.
 *
 *============================================================================*/

#include <iostream>

#include "csparse.h"


using namespace cs;


int main(int argc, char *argv[]) {
    COOMatrix T;

    // Load a matrix from either a filename or stdin
    if (argc > 2) {
        std::cerr << "Usage: " << argv[0] << " [filename]\n";
        return EXIT_FAILURE;
    } else if (argc == 2) {
        T = COOMatrix::from_file(argv[1]);
    } else {
        std::cerr << "Usage: " << argv[0] << " [filename]\n";
        std::cerr << "Reading from stdin...\n";
        T = COOMatrix::from_stream(std::cin);
    }

    std::cout << "T:\n" << T << "\n";

    // Convert to CSCMatrix
    CSCMatrix A(T);
    std::cout << "A:\n" << A << "\n";

    CSCMatrix AT = A.transpose();
    std::cout << "AT:\n" << AT << "\n";

    // Create an identity matrix
    auto [M, N] = A.shape();
    COOMatrix I({M, M}, M);  // only zeros on the diagonal
    for (csint i = 0; i < M; ++i) {
        I.insert(i, i, 1.0);
    }
    CSCMatrix Eye(I);

    // Do some math
    CSCMatrix C = A * AT;  // (M, N) * (N, M) = (M, M)
    CSCMatrix D = C + Eye * C.norm();  // D = C + Eye * norm(C, 1)
    std::cout << "D:\n" << D << "\n";

    return EXIT_SUCCESS;
}


/*==============================================================================
 *============================================================================*/
