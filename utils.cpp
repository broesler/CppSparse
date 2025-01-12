/*==============================================================================
 *     File: utils.cpp
 *  Created: 2024-11-02 17:32
 *   Author: Bernie Roesler
 *
 *  Description: Utility functions.
 *
 *============================================================================*/

#include "csparse.h"


/*------------------------------------------------------------------------------
 *         Vector Operators 
 *----------------------------------------------------------------------------*/
/** Vector-vector addition */
std::vector<double> operator+(
    const std::vector<double>& a,
    const std::vector<double>& b
    )
{
    assert(a.size() == b.size());

    std::vector<double> out(a.size());

    for (csint i = 0; i < a.size(); i++) {
        out[i] = a[i] + b[i];
    }

    return out;
}

/** Unary minus operator for a vector */
std::vector<double> operator-(const std::vector<double>& a)
{
    std::vector<double> out(a.size());

    for (csint i = 0; i < a.size(); i++) {
        out[i] = -a[i];
    }

    return out;
}


/** Vector-vector subtraction */
std::vector<double> operator-(
    const std::vector<double>& a,
    const std::vector<double>& b
    )
{
    assert(a.size() == b.size());

    std::vector<double> out(a.size());

    for (csint i = 0; i < a.size(); i++) {
        out[i] = a[i] - b[i];
    }

    return out;
}


/** Scale a vector by a scalar */
std::vector<double> operator*(const double c, const std::vector<double>& vec)
{
    std::vector<double> out(vec);
    for (auto& x : out) {
        x *= c;
    }
    return out;
}


std::vector<double> operator*(const std::vector<double>& vec, const double c)
{
    return c * vec;
}


std::vector<double>& operator*=(std::vector<double>& vec, const double c)
{
    for (auto& x : vec) {
        x *= c;
    }
    return vec;
}


/*------------------------------------------------------------------------------
 *          Vector Permutations
 *----------------------------------------------------------------------------*/

/** Compute \f$ x = Pb \f$ where P is a permutation matrix, represented as
 * a vector.
 *
 * @param p  permutation vector, where `p[k] = i` means `p_{ki} = 1`.
 * @param b  vector of data to permute
 *
 * @return x  `x = Pb` the permuted vector, like `x = p(b)` in Matlab.
 */
std::vector<double> pvec(
    const std::vector<csint>& p,
    const std::vector<double>& b
    )
{
    std::vector<double> x(b.size());

    for (csint k = 0; k < b.size(); k++)
        x[k] = b[p[k]];

    return x;
}


/** Compute \f$ x = P^T b = P^{-1} b \f$ where P is a permutation matrix,
 * represented as a vector.
 *
 * @param p  permutation vector, where `p[k] = i` means `p_{ki} = 1`.
 * @param b  vector of data to permute
 *
 * @return x  `x = Pb` the permuted vector, like `x = p(b)` in Matlab.
 */
std::vector<double> ipvec(
    const std::vector<csint>& p,
    const std::vector<double>& b
    )
{
    std::vector<double> x(b.size());

    for (csint k = 0; k < b.size(); k++)
        x[p[k]] = b[k];

    return x;
}


/** Compute the inverse (or transpose) of a permutation vector.
 *
 * @note This function is named `cs_pinv` in CSparse, but we have changed the
 * name to avoid conflict with similarly named variables, and the well-known
 * Matlab funvtion to compute the pseudo-inverse of a matrix.
 *
 * @param p  permutation vector
 *
 * @return pinv  inverse permutation vector
 */
std::vector<csint> inv_permute(const std::vector<csint>& p)
{
    std::vector<csint> out(p.size());

    for (csint k = 0; k < p.size(); k++)
        out[p[k]] = k;

    return out;
}


/** Compute the cumulative sum of a vector, starting with 0.
 *
 * @param w  a reference to a vector of length N.
 *
 * @return p  the cumulative sum of `w`, of length N + 1.
 */
std::vector<csint> cumsum(const std::vector<csint>& w)
{
    std::vector<csint> out(w.size() + 1);

    // Row pointers are the cumulative sum of the counts, starting with 0
    std::partial_sum(w.begin(), w.end(), out.begin() + 1);

    return out;
}


/** Compute the mean and standard deviation of a vector of samples. */
Stats compute_stats(const std::vector<double>& samples)
{
    int N = samples.size();

    // Compute the mean and std of the samples
    double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / N;

    double sq_sum = std::inner_product(samples.begin(), samples.end(), samples.begin(), 0.0);
    double std_dev = std::sqrt(sq_sum / N - mean * mean);

    return {mean, std_dev};
}


/** Write the results of a performance test to a JSON file.
 *
 * @param filename  the name of the file to write
 * @param density   the density of the matrix
 * @param Ns        the vector of matrix sizes
 * @param times     a map of test names to TimeStats objects
 */
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
    fp << Ns << ",\n";

    // Write the result times
    for (auto it = times.begin(); it != times.end(); it++) {
        const std::string name = it->first;
        const TimeStats ts = it->second;

        fp << "  \"" << name << "\": {\n";
        fp << "    \"mean\": ";
        fp << ts.means << ",\n";

        fp << "    \"std_dev\": ";
        fp << ts.std_devs << "\n";

        if (std::next(it) == times.end())  // last entry in sorted map
            fp << "  }\n";
        else
            fp << "  },\n";
    }

    // Closing brace
    fp << "}\n";
    fp.close();
}

/*==============================================================================
 *============================================================================*/
