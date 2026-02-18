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
        std::println(std::cerr, "Usage: {} [filename]", argv[0]);
        return EXIT_FAILURE;
    } else if (argc == 2) {
        T = COOMatrix::from_file(argv[1]);
    } else {
        std::println(std::cerr, "Usage: {} [filename]", argv[0]);
        std::println(std::cerr, "Reading from stdin...");
        T = COOMatrix::from_stream(std::cin);
    }

    std::println("T:\n{:v}", T);

    // Convert to CSCMatrix
    CSCMatrix A{T};
    std::println("A:\n{:v}", A);

    auto AT = A.transpose();
    std::println("AT:\n{:v}", AT);

    // Create an identity matrix
    const auto [M, N] = A.shape();
    COOMatrix I{{M, M}, M};  // only zeros on the diagonal
    for (auto i : A.row_range()) {
        I.insert(i, i, 1.0);
    }
    CSCMatrix Eye{I};

    // Do some math
    auto C = A * AT;  // (M, N) * (N, M) = (M, M)
    auto D = C + Eye * C.norm();  // D = C + Eye * norm(C, 1)
    std::println("D:\n{:v}", D);

    return EXIT_SUCCESS;
}


/*==============================================================================
 *============================================================================*/
