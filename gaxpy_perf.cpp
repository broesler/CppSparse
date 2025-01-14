/*==============================================================================
 *     File: gaxpy_perf.cpp
 *  Created: 2025-01-06 14:07
 *   Author: Bernie Roesler
 *
 *  Davis, Exercise 2.27: Test gaxpy performance for column-major, row-major,
 *  and column-major block operations.
 *
 *  To compare their performance, we will use a large, sparse matrix and two
 *  dense matrices. We will plot the time taken for each operation as a function
 *  of the number of columns in the dense matrices.
 *
 *============================================================================*/

#include <chrono>
#include <functional>
#include <fstream>
#include <iostream>
#include <map>
#include <ranges>  // for std::views:keys

#include "csparse.h"

using namespace cs;


/*------------------------------------------------------------------------------
 *         Main Loop 
 *----------------------------------------------------------------------------*/
int main()
{
    // Declare constants
    const bool VERBOSE = true;
    const unsigned int SEED = 565656;
#ifdef GATXPY
    const std::string filename = "./plots/gatxpy_perf.json";
#else
    const std::string filename = "./plots/gaxpy_perf.json";
#endif

    // Run the tests
    using gaxpy_prototype = std::vector<double>
        (CSCMatrix::*)(
            const std::vector<double>&,
            const std::vector<double>&
        ) const;

    const std::map<std::string, gaxpy_prototype> gaxpy_funcs = {
#ifdef GATXPY
        {"gatxpy_col", &CSCMatrix::gatxpy_col},
        {"gatxpy_row", &CSCMatrix::gatxpy_row},
        {"gatxpy_block", &CSCMatrix::gatxpy_block}
#else
        {"gaxpy_col", &CSCMatrix::gaxpy_col},
        {"gaxpy_row", &CSCMatrix::gaxpy_row},
        {"gaxpy_block", &CSCMatrix::gaxpy_block}
#endif
    };

    // Store the results
    std::map<std::string, TimeStats> times;

    // const std::vector<int> Ns = {10, 100, 1000};
    const std::vector<int> Ns = {10, 20, 50, 100, 200, 500, 1000, 2000, 5000};
    const float density = 0.25;  // density of the sparse matrix

    // Time sampling
    const int N_repeats = 1;
    const int N_samples = 3;  // should adjust to get total time ~0.2 s

    // Initialize the results struct
    for (const auto& name : std::views::keys(gaxpy_funcs)) {
        times[name] = TimeStats(Ns.size());
    }


    /*--------------------------------------------------------------------------
     *         Run the tests 
     *------------------------------------------------------------------------*/
    for (const int N : Ns) {
        if (VERBOSE)
            std::cout << "Running N = " << N << "..." << std::endl;

        int M = (int)(0.9 * N);  // number of rows in sparse matrix and added dense matrix
        int K = (int)(0.8 * N);  // number of columns in multiplied dense matrix
        // int M = N, K = N;  // square matrices

        // Create a large, sparse matrix
        CSCMatrix A = COOMatrix::random(M, N, density, SEED).tocsc();
#ifdef GATXPY
        A = A.T();
#endif

        // Create a compatible random, dense matrix
        const COOMatrix X = COOMatrix::random(N, K, density, SEED);
        const COOMatrix Y = COOMatrix::random(M, K, density, SEED);

        // Convert to dense column-major order
        const std::vector<double> X_col = X.toarray('F');
        const std::vector<double> Y_col = Y.toarray('F');

        // Use same matrices in row-major order
        const std::vector<double> X_row = X.toarray('C');
        const std::vector<double> Y_row = Y.toarray('C');

        for (const auto& [name, gaxpy_func] : gaxpy_funcs) {
            // Time the function runs
            Stats ts = timeit_member(
                gaxpy_func,
                A,
                N_repeats,
                N_samples,
                name == "gaxpy_row" ? X_row : X_col,
                name == "gaxpy_row" ? Y_row : Y_col
            );

            // Store results
            times[name].means.push_back(ts.mean);
            times[name].std_devs.push_back(ts.std_dev);

            if (VERBOSE) {
                std::cout << name
                    << ", Time: " << ts.mean
                    << " ± " << ts.std_dev << " s" 
                    << std::endl;
            }
        }
    }

    if (VERBOSE)
        std::cout << "done." << std::endl;

    //--------------------------------------------------------------------------
    //        Write the results to a file
    //--------------------------------------------------------------------------
    if (VERBOSE)
        std::cout << "Writing results to '" << filename << "'..." << std::endl;

    write_json_results(filename, density, Ns, times);

    if (VERBOSE)
        std::cout << "done." << std::endl;

    return EXIT_SUCCESS;
};

/*==============================================================================
 *============================================================================*/
