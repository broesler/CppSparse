//==============================================================================
//     File: pybind11_conversion.h
//  Created: 2025-05-20 16:26
//   Author: Bernie Roesler
//
//  Description: Header file for pybind11 wrapper.
//
//==============================================================================

#ifndef _CSPARSE_PYBIND11_H_
#define _CSPARSE_PYBIND11_H_

#include <array>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>
#include <utility>  // forward

#include "csparse.h"

namespace py = pybind11;


/** Convert a string to an AMDOrder enum.
 *
 * @param order  the string to convert
 *
 * @return the AMDOrder enum
 */
inline cs::AMDOrder string_to_amdorder(const std::string& order)
{
    if (order == "Natural") { return cs::AMDOrder::Natural; }
    if (order == "APlusAT") { return cs::AMDOrder::APlusAT; }
    if (order == "ATANoDenseRows") { return cs::AMDOrder::ATANoDenseRows; }
    if (order == "ATA") { return cs::AMDOrder::ATA; }
    throw std::runtime_error("Invalid AMDOrder specified.");
}


namespace pybind11::detail {
// -----------------------------------------------------------------------------
//         Custom Type Caster for std::vector<T> <=> py::array_t<T>
// -----------------------------------------------------------------------------
template <typename T>
struct type_caster<std::vector<T>> {
    // macro to define the `value` member and internals
    PYBIND11_TYPE_CASTER(std::vector<T>, _("Sequence[Union[int, float]]"));

    // C++ (std::vector<T>) to Python (py::array_t<T>)
    // Output conversion (when a C++ function returns std::vector)
    static handle cast(
        const std::vector<T>& src,
        return_value_policy policy,
        handle parent
    ) {
        py::array_t<T> arr(src.size(), src.data());
        return arr.release();
    }

    // Python (py::array_t<T> or list) to C++ (std::vector<T>)
    // Input conversion (when a C++ function takes std::vector)
    bool load(handle src, bool convert) {
        // --- Try to load as a NumPy array ---
        if (py::isinstance<py::array>(src)) {
            try {
                // Cast to a C-style array
                auto arr = src.cast<py::array_t<T, py::array::c_style | py::array::forcecast>>();
                py::buffer_info buf_info = arr.request();

                if (buf_info.ndim != 1) {
                    std::cerr << "  NumPy array is not 1D." << std::endl;
                    return false;
                }

                if (buf_info.itemsize != sizeof(T)) {
                    std::cerr << "  Internal itemsize mismatch "
                        << "(should be " << sizeof(T)
                        << " , but is " << buf_info.itemsize << ")."
                        << std::endl;
                    return false;
                }

                // Assign data directly into the buffer
                value.assign(static_cast<T*>(buf_info.ptr),
                                static_cast<T*>(buf_info.ptr) + buf_info.shape[0]);

                return true;

            } catch (const py::error_already_set& e) {
                std::cerr << "  Failed to request NumPy buffer for type "
                    << typeid(T).name() << ": " << e.what() << std::endl;
                PyErr_Print();  // print Python traceback
                return false;
            } catch (const std::exception& e) {
                std::cerr << "  Failed to cast NumPy array to std::vector<"
                    << typeid(T).name() << ">: " << e.what() << std::endl;
                return false;
            }
        }

        // --- if not a NumPy array, try loading as a Python list ---
        if (py::isinstance<py::list>(src)) {
            try {
                // Iterate through the list and cast each item individually
                py::list py_list = src.cast<py::list>();
                value.clear();
                value.reserve(py::len(py_list));
                for (auto item : py_list) {
                    value.push_back(item.cast<T>());
                }
                return true;
            } catch (const py::cast_error& e) {
                std::cerr << "  Failed to cast Python list elements to "
                    << typeid(T).name() << ": " << e.what() << std::endl;
                return false;
            }
        }

        return false;
    }  // load
};


}  // namespace pybind11::detail


/** Convert an array to a NumPy array.
 *
 * @param self  the array to convert
 *
 * @return a NumPy array with the same data as the array
 */
template <typename T, std::size_t N>
inline py::array_t<T> array_to_numpy(const std::array<T, N>& arr)
{
    return py::array_t<T>(arr.size(), arr.data());
};


/** Convert a dense matrix to a NumPy array.
 *
 * @param self  the dense matrix to convert
 * @param order the order of the NumPy array ('C' or 'F')
 *
 * @return a NumPy array with the same data as the matrix
 */
template <typename T>
auto sparse_to_ndarray(const T& self, const char order)
{
    // Get the matrix in dense column-major order
    std::vector<double> v = self.to_dense_vector('C');
    auto [M, N] = self.shape();

    // Create a NumPy array with specified dimensions
    py::array_t<double> result({M, N});

    // Get a pointer to the underlying data of the NumPy array.
    auto buffer_info = result.request();
    double* ptr = static_cast<double*>(buffer_info.ptr);

    // Calculate strides based on order
    std::vector<ssize_t> strides;
    if (order == 'C') { // C-style (row-major)
        strides = {
            static_cast<ssize_t>(N * sizeof(double)),
            sizeof(double)
        };
    } else if (order == 'F') { // Fortran-style (column-major)
        strides = {
            sizeof(double),
            static_cast<ssize_t>(M * sizeof(double))
        };
    } else {
        throw std::runtime_error("Invalid order specified. Use 'C' or 'F'.");
    }

    // Assign strides to the buffer info. This is crucial!
    buffer_info.strides = strides;

    // Copy the data from the vector to the NumPy array.  This is the most
    // straightforward way.
    std::copy(v.begin(), v.end(), ptr);

    return result;
};


/** Convert a COOMatrix to a scipy.sparse.coo_array
 *
 * @param A  the COOMatrix to convert
 *
 * @return a scipy.sparse.coo_array with the same data as the COOMatrix
 */
py::object scipy_from_coo(const cs::COOMatrix& A);


/** Convert a CSCMatrix to a scipy.sparse.csc_array
 *
 * @param A  the CSCMatrix to convert
 *
 * @return a scipy.sparse.csc_array with the same data as the CSCMatrix
 */
py::object scipy_from_csc(const cs::CSCMatrix& A);


/** Convert a scipy.sparse.sparray to a CSCMatrix.
 *
 * @param A  the scipy.sparse.sparray to convert
 *
 * @return a CSCMatrix with the same data as the scipy.sparse.sparray
 */
cs::CSCMatrix csc_from_scipy(const py::object& obj);


/** Wrap a function to convert a scipy.sparse.sparray on input. 
 *
 * This function takes a function that operates on a CSCMatrix as the first
 * argument, and a variable number of other arguments. It converts the first
 * argument from a python object to a CSCMatrix, and forwards the rest of the
 * arguments to the function.
 *
 * @param f  the function to wrap
 *
 * @return a lambda function that takes a `scipy.sparse.sparray` and 
 *         forwards the rest of the arguments to the wrapped function.
 */
template <typename... Args>
auto wrap_vector_func(
    std::vector<double>(*func)(const cs::CSCMatrix& A, Args...)
)
{
    return [func](const py::object& A_scipy, Args... args) {
        cs::CSCMatrix A = csc_from_scipy(A_scipy);
        std::vector<double> x = func(A, std::forward<Args>(args)...);
        return py::cast(x);
    };
}


/** Wrap a function to convert a scipy.sparse.sparray on input. 
 *
 * This function takes a function that operates on a CSCMatrix as the first
 * argument, and a variable number of other arguments. It converts the first
 * argument from a python object to a CSCMatrix, and forwards the rest of the
 * arguments to the function.
 *
 * @param f  the function to wrap
 *
 * @return a lambda function that takes a `scipy.sparse.sparray` and 
 *         forwards the rest of the arguments to the wrapped function.
 */
template <typename Func>
auto wrap_gaxpy_mat(Func&& f)
{
    return [f = std::forward<Func>(f)](
        const py::object& A_scipy,
        const std::vector<double>& X,
        const std::vector<double>& Y
    ) {
        cs::CSCMatrix A = csc_from_scipy(A_scipy);
        // TODO X and Y are dense matrices! Accept numpy matrices
        std::vector<double> Z = f(A, X, Y);
        return py::cast(Z);  // TODO Z is actually a dense matrix!
    };
}


/** Wrap a solve function that takes a matrix, vector, and order. */
template <typename... Args>
auto wrap_solve(
    std::vector<double> (*f)(
        const cs::CSCMatrix& A,
        const std::vector<double>& b,
        cs::AMDOrder order,
        Args...
    )
)
{
    return [f](
        const py::object& A_scipy,
        const std::vector<double>& b,
        const std::string& order,
        Args... args
    ) {
        cs::CSCMatrix A = csc_from_scipy(A_scipy);
        cs::AMDOrder order_enum = string_to_amdorder(order);
        std::vector<double> x = f(A, b, order_enum, std::forward<Args>(args)...);
        return py::cast(x);
    };
}


/** Dispatch the vector permutation functions for appropriate types.
 *
 * @param p  the permutation vector
 * @param b_obj  the vector to permute, can be a vector of doubles or integers
 * @param func_double  function to handle double vectors, e.g. &cs::pvec<double>
 * @param func_int  function to handle double vectors, e.g. &cs::pvec<csint>
 *
 * @return  a new vector with the elements of `b_obj` permuted according to `p`
 */
template <typename FuncD, typename FuncI>
py::object dispatch_pvec_ipvec(
    const std::vector<cs::csint>& p,
    const py::object& b_obj,
    FuncD func_double,
    FuncI func_int
) {
    try {
        std::vector<double> b = b_obj.cast<std::vector<double>>();
        return py::cast(func_double(p, b));
    } catch (const py::cast_error&) {
        try {
            std::vector<cs::csint> b = b_obj.cast<std::vector<cs::csint>>();
            return py::cast(func_int(p, b));
        } catch (const py::cast_error&) {
            throw py::type_error("Input must be a vector of doubles or integers.");
        }
    }
}


/** Solve a triangular system with the given function. */
template <typename DenseSolver>
auto make_trisolver(DenseSolver dense_solver)
{
    return [dense_solver](const py::object& A_scipy, const py::object& b_obj) {
        const cs::CSCMatrix A = csc_from_scipy(A_scipy);
        py::module_ sparse = py::module_::import("scipy.sparse");

        if (sparse.attr("issparse")(b_obj).cast<bool>()) {
            const cs::CSCMatrix B = csc_from_scipy(b_obj);

            if (B.shape()[1] != 1) {
                throw std::invalid_argument(
                    "b must be a column vector (shape (N, 1))."
                );
            }

            cs::SparseSolution sol = cs::spsolve(A, B, 0);

            // Solution is an (N, 1) CSCMatrix
            cs::csint N = B.shape()[0];
            cs::COOMatrix x({N, 1}, sol.xi.size());

            for (const auto& i : sol.xi) {
                x.insert(i, 0, sol.x[i]);
            }

            return scipy_from_csc(x.tocsc());
        } else {
            // Assume b is a dense vector, return a dense vector solution
            try {
                py::module_ np = py::module_::import("numpy");
                // Guarantee array is 1D, or fail
                std::vector<double> b = np.attr("atleast_1d")(
                    b_obj.attr("squeeze")()
                ).cast<std::vector<double>>();
                return py::cast(dense_solver(A, b));
            } catch (const py::cast_error&) {
                throw py::type_error("b must have shape (N,) or (N, 1).");
            }
        }
    };
}


#endif  // _CSPARSE_PYBIND11_H_

//==============================================================================
//==============================================================================
