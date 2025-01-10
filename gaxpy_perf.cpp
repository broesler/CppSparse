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

#include "csparse.h"


// Define function prototypes here to make them visible to main()
// See: 
// <https://stackoverflow.com/questions/69558521/friend-function-name-undefined>
#ifdef GATXPY
std::vector<double> gatxpy_col(
#else
std::vector<double> gaxpy_col(
#endif
    const CSCMatrix& A,
    const std::vector<double>& X,
    const std::vector<double>& Y
);

#ifdef GATXPY
std::vector<double> gatxpy_row(
#else
std::vector<double> gaxpy_row(
#endif
    const CSCMatrix& A,
    const std::vector<double>& X,
    const std::vector<double>& Y
);

#ifdef GATXPY
std::vector<double> gatxpy_block(
#else
std::vector<double> gaxpy_block(
#endif
    const CSCMatrix& A,
    const std::vector<double>& X,
    const std::vector<double>& Y
);


// Define a structure to store the mean and standard deviation of the times
struct TimeStats {
    std::vector<double> mean;
    std::vector<double> std_dev;

    TimeStats() {};
    TimeStats(const int N) {
        mean.reserve(N);
        std_dev.reserve(N);
    };
};


// Write the results to a JSON file
void write_json_results(
    const std::string filename,
    const double density,
    const std::vector<int>& Ns,
    const std::map<std::string, TimeStats>& times
    )
{
    // Open the file and check for success
    std::ofstream fp(filename);
    if (!fp.is_open()) {
        std::cerr << "Error: could not open file " << filename << std::endl;
        return;
    }

    // Opening brace
    fp << "{\n";

    // Write the density
    fp << "  \"density\": " << density << ",\n";

    // Write the Ns vector
    fp << "  \"Ns\": ";
    print_vec(Ns, fp, ",\n");

    // Write the result times
    for (auto it = times.begin(); it != times.end(); it++) {
        const std::string name = it->first;
        const TimeStats ts = it->second;

        fp << "  \"" << name << "\": {\n";
        fp << "    \"mean\": ";
        print_vec(ts.mean, fp, ",\n");

        fp << "    \"std_dev\": ";
        print_vec(ts.std_dev, fp, "\n");

        if (std::next(it) == times.end())  // last entry in sorted map
            fp << "  }\n";
        else
            fp << "  },\n";
    }

    // Closing brace
    fp << "}\n";
    fp.close();
}


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
    using gaxpy_prototype = std::function<
        std::vector<double>(
            const CSCMatrix&,
            const std::vector<double>&,
            const std::vector<double>&
        )
    >;

    const std::map<std::string, gaxpy_prototype> gaxpy_funcs = {
#ifdef GATXPY
        {"gatxpy_col", gatxpy_col},
        {"gatxpy_row", gatxpy_row},
        {"gatxpy_block", gatxpy_block}
#else
        {"gaxpy_col", gaxpy_col},
        {"gaxpy_row", gaxpy_row},
        {"gaxpy_block", gaxpy_block}
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

    // Define temporary vector to store sample times
    std::vector<double> sample_times(N_samples);

    for (const auto& [name, gaxpy_func] : gaxpy_funcs) {
        // Initialize the results struct
        times[name] = TimeStats(Ns.size());

        if (VERBOSE)
            std::cout << "Running " << name << "..." << std::endl;

        for (const int N : Ns) {
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

            // Run the function r times (sample size) for n loops (samples)
            TimeStats ts(N_repeats);

            for (int r = 0; r < N_repeats; r++) {
                for (int s = 0; s < N_samples; s++) {
                    // Compute and time the function
                    const auto tic = std::chrono::high_resolution_clock::now();

                    if (name == "gaxpy_row") {
                        const std::vector<double> Y_out = gaxpy_func(A, X_row, Y_row);
                    } else {
                        const std::vector<double> Y_out = gaxpy_func(A, X_col, Y_col);
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
