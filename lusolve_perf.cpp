/*==============================================================================
 *     File: lusolve_perf.cpp
 *  Created: 2025-01-11 09:38
 *   Author: Bernie Roesler
 *
 *  Davis, Exercise 3.8: Test [lu]solve performance for original and optimized
 *  functions.
 *
 *  To compare their performance, we will use a large, sparse matrix and a dense
 *  RHS vector that becomes increasingly sparse. We will plot the time taken for
 *  each operation as a function of the number of columns in the dense matrices.
 *
 *============================================================================*/

#include <chrono>
#include <functional>
#include <fstream>
#include <iostream>
#include <map>

#include "csparse.h"


// Define function prototypes here to make them visible to main()
// See: 
// <https://stackoverflow.com/questions/69558521/friend-function-name-undefined>
std::vector<double> lsolve(const CSCMatrix& L, const std::vector<double>& b);
std::vector<double> usolve(const CSCMatrix& U, const std::vector<double>& b);
std::vector<double> lsolve_opt(const CSCMatrix& L, const std::vector<double>& b);
std::vector<double> usolve_opt(const CSCMatrix& U, const std::vector<double>& b);


int main()
{
    // Declare constants
    const bool VERBOSE = true;
    const unsigned int SEED = 565656;
    const std::string filename = "./plots/lusolve_perf.json";

    // Run the tests
    using lusolve_prototype = std::function<
        std::vector<double>(const CSCMatrix&, const std::vector<double>&)
    >;

    const std::map<std::string, lusolve_prototype> lusolve_funcs = {
        {"lsolve", lsolve},
        {"usolve", usolve},
        {"lsolve_opt", lsolve_opt},
        {"usolve_opt", usolve_opt}
    };

    // Store the results
    std::map<std::string, TimeStats> times;

    // const std::vector<int> Ns = {10};
    // const std::vector<int> Ns = {10, 100, 1000};
    const std::vector<int> Ns = {10, 20, 50, 100, 200, 500, 1000, 2000, 5000};
    const float density = 0.4;  // density of the sparse matrix

    // Time sampling
    const int N_repeats = 1;
    const int N_samples = 3;  // should adjust to get total time ~0.2 s

    // Define temporary vector to store sample times
    std::vector<double> sample_times(N_samples);

    // TODO swap order of loops so that we only create the matrix once
    for (const auto& [name, lusolve_func] : lusolve_funcs) {
        // Initialize the results struct
        times[name] = TimeStats(Ns.size());

        if (VERBOSE)
            std::cout << "Running " << name << "..." << std::endl;

        for (const int N : Ns) {
            // Create a large, square sparse matrix
            CSCMatrix A = COOMatrix::random(N, N, density, SEED).tocsc();

            // Ensure all diagonal elements are non-zero
            for (int i = 0; i < N; i++) {
                A.assign(i, i, 1.0);
            }

            // Take the lower triangular
            CSCMatrix L = A.band(-N, 0);
            CSCMatrix U = L.T();

            // Create a dense column vector that is the sum of the rows of L
            const std::vector<double> bL = L.sum_rows();
            const std::vector<double> bU = U.sum_rows();

            const std::vector<double> expect = std::vector<double>(N, 1.0);

            // Run the function r times (sample size) for n loops (samples)
            TimeStats ts(N_repeats);

            for (int r = 0; r < N_repeats; r++) {
                for (int s = 0; s < N_samples; s++) {
                    // Compute and time the function
                    const auto tic = std::chrono::high_resolution_clock::now();

                    if (name[0] == 'l') {
                        const std::vector<double> x = lusolve_func(L, bL);
                    } else {
                        const std::vector<double> x = lusolve_func(U, bU);
                    }

                    const auto toc = std::chrono::high_resolution_clock::now();
                    const std::chrono::duration<double> elapsed = toc - tic;

                    sample_times[s] = elapsed.count();
                }

                // Compute the mean and std
                double mean = std::accumulate(sample_times.begin(), sample_times.end(), 0.0) / N_samples;

                double sq_sum = std::inner_product(sample_times.begin(), sample_times.end(), sample_times.begin(), 0.0);
                double std_dev = std::sqrt(sq_sum / N_samples - mean * mean);

                // Store the results
                ts.mean.push_back(mean);
                ts.std_dev.push_back(std_dev);
            }

            // Take the mean over the repeats
            double μ = std::accumulate(ts.mean.begin(), ts.mean.end(), 0.0) / N_repeats;
            double σ = std::accumulate(ts.std_dev.begin(), ts.std_dev.end(), 0.0) / N_repeats;

            // Store results
            times[name].mean.push_back(μ);
            times[name].std_dev.push_back(σ);

            if (VERBOSE) {
                std::cout << "N = " << N 
                    << ", Time: " << μ
                    << " ± " << σ << " s" 
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
