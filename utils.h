//==============================================================================
//    File: utils.h
// Created: 2024-11-02 17:29
//  Author: Bernie Roesler
//
//  Description: Utility functions for CSparse++.
//
//==============================================================================

#ifndef _UTILS_H_
#define _UTILS_H_

#include <fstream>
#include <map>
#include <numeric>  // std::iota

std::vector<double> operator+(
    const std::vector<double>& a,
    const std::vector<double>& b
);

std::vector<double> operator-(
    const std::vector<double>& a,
    const std::vector<double>& b
);

std::vector<double> operator-(const std::vector<double>& a);

std::vector<double> operator*(const double c, const std::vector<double>& x);
std::vector<double> operator*(const std::vector<double>& x, const double c);
std::vector<double>& operator*=(std::vector<double>& x, const double c);

std::vector<double> pvec(const std::vector<csint>& p, const std::vector<double>& b);
std::vector<double> ipvec(const std::vector<csint>& p, const std::vector<double>& b);

std::vector<csint> inv_permute(const std::vector<csint>& p);

std::vector<csint> cumsum(const std::vector<csint>& w);


/*------------------------------------------------------------------------------
 *          Performance Testing
 *----------------------------------------------------------------------------*/
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


// Basic structure to store mean and standard deviation.
struct Stats {
    double mean;
    double std_dev;
};


void write_json_results(
    const std::string filename,
    const double density,
    const std::vector<int>& Ns,
    const std::map<std::string, TimeStats>& times
);


/** Compute the mean and standard deviation of a vector of samples. */
Stats compute_stats(const std::vector<double>& samples);


/*------------------------------------------------------------------------------
 *         Declare Template Functions
 *----------------------------------------------------------------------------*/
/** Print a std::vector. */
template <typename T>
void print_vec(
    const std::vector<T>& vec,
    std::ostream& os=std::cout,
    const std::string end="\n"
)
{
    os << "[";
    for (int i = 0; i < vec.size(); i++) {
        os << vec[i];
        if (i < vec.size() - 1) {
            os << ", ";
        }
    }
    os << "]" << end;
}


/** Print a std::vector to an output stream. */
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
{
    print_vec(vec, os, "");
    return os;
}


/** Sort the indices of a vector. */
template <typename T>
std::vector<csint> argsort(const std::vector<T>& vec)
{
    std::vector<csint> idx(vec.size());
    std::iota(idx.begin(), idx.end(), 0);

    // Sort the indices by referencing the vector
    std::sort(
        idx.begin(),
        idx.end(),
        [&vec](csint i, csint j) { return vec[i] < vec[j]; }
    );

    return idx;
}


/** Time a function call.
 *
 * @param func  the function to time
 * @param N_repeats  the number of times to repeat the function call
 * @param N_samples  the number of samples to take
 * @param args...  any arguments to pass to the function
 *
 * @return ts  a TimeStats object with the mean and standard deviation of the
 *         times taken.
 */
template <typename Func, typename... Args>
Stats timeit(
    Func func,
    const int N_repeats = 1,
    const int N_samples = 7,
    Args... args
)
{
    TimeStats ts(N_repeats);
    std::vector<double> sample_times(N_samples);

    for (int r = 0; r < N_repeats; r++) {
        for (int s = 0; s < N_samples; s++) {
            // Compute and time the function
            const auto tic = std::chrono::high_resolution_clock::now();

            // Run the function
            func(std::forward<Args>(args)...);

            const auto toc = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double> elapsed = toc - tic;

            sample_times[s] = elapsed.count();
        }

        // Compute the mean and std_dev of the sampled times
        auto stats = compute_stats(sample_times);

        // Store the results
        ts.mean.push_back(stats.mean);
        ts.std_dev.push_back(stats.std_dev);
    }

    // Take the mean over the repeats
    double μ = std::accumulate(ts.mean.begin(), ts.mean.end(), 0.0) / N_repeats;
    double σ = std::accumulate(ts.std_dev.begin(), ts.std_dev.end(), 0.0) / N_repeats;

    return {μ, σ};
}


#endif

//==============================================================================
//==============================================================================
