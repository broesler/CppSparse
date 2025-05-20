//==============================================================================
//     File: pybind11.h
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

#include "csparse.h"

namespace py = pybind11;


namespace pybind11::detail {
    // -------------------------------------------------------------------------
    //         Custom Type Caster for std::vector<T> <=> py::array_t<T>
    // -------------------------------------------------------------------------
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
                    // Fall through to failure message if individual cast fails
                }
            }

            std::cerr << "  Failed to load from either NumPy array or list for "
                "std::vector<" << typeid(T).name() << ">." << std::endl;
            return false;
        }  // load
    };


    // -------------------------------------------------------------------------
    //         Custom Type Caster for scipy.sparse.sparray <=> cs::CSCMatrix
    // -------------------------------------------------------------------------
    template <>  // specialization for cs::CSCMatrix
    struct type_caster<cs::CSCMatrix> {
        // macro to define the `value` member and internals
        PYBIND11_TYPE_CASTER(cs::CSCMatrix, _("scipy.sparse.sparray"));

        // C++ to Python conversion (cast method)
        static handle cast(
            const cs::CSCMatrix& src,
            return_value_policy policy,
            handle parent
        ) {
            py::module_ sparse = py::module_::import("scipy.sparse");

            auto data_array = py::cast(src.data());
            auto indices_array = py::cast(src.indices());
            auto indptr_array = py::cast(src.indptr());

            // Create the SciPy CSC matrix
            py::object scipy_csc_array = sparse.attr("csc_array")(
                py::make_tuple(data_array, indices_array, indptr_array),
                py::arg("shape") = src.shape()
            );

            return scipy_csc_array.release();
        }

        // Python to C++ conversion (load method)
        bool load(handle src, bool convert) {
            py::object A;

            // Check if it's already a csc_array, or convertible to one
            if (py::hasattr(src, "tocsc")) {
                std::cerr << "  Convering to csc." << std::endl;
                A = src.attr("tocsc")(); // Convert to CSC if not already
            } else {
                std::cerr << "Error: Input object is not convertible to a "
                    "SciPy CSC matrix (missing .tocsc() method)." << std::endl;
                return false;
            }

            // Verify it has the expected attributes
            if (!py::hasattr(A, "data") ||
                !py::hasattr(A, "indices") ||
                !py::hasattr(A, "indptr") ||
                !py::hasattr(A, "shape"))
            {
                std::cerr << "Error: Converted object is not a valid SciPy CSC "
                    "matrix (missing data, indices, indptr, or shape)." 
                    << std::endl;
                return false;
            }

            try {
                // Cast SciPy attributes to py::array_t and then to std::vector
                // Our std::vector type caster will handle the numpy.ndarray -> std::vector conversion automatically.
                std::vector<double> data_vec = A.attr("data").cast<std::vector<double>>();
                std::vector<cs::csint> indices_vec = A.attr("indices").cast<std::vector<cs::csint>>();
                std::vector<cs::csint> indptr_vec = A.attr("indptr").cast<std::vector<cs::csint>>();

                // Get shape
                auto shape_tuple = A.attr("shape").cast<std::tuple<cs::csint, cs::csint>>();
                std::array<cs::csint, 2> c_shape_arr = {std::get<0>(shape_tuple), std::get<1>(shape_tuple)};

                // Construct the C++ CSCMatrix using the loaded data
                // The 'value' member is the target CSCMatrix object
                value = cs::CSCMatrix(data_vec, indices_vec, indptr_vec, c_shape_arr);
                return true;

            } catch (const py::cast_error& e) {
                std::cerr << "Error in SciPy CSC to C++ CSCMatrix cast: "
                    << e.what() << std::endl;
                return false;

            } catch (const py::error_already_set& e) {
                std::cerr << "Python error in SciPy CSC to C++ CSCMatrix cast: "
                    << e.what() << std::endl;
                PyErr_Print();  // print Python traceback
                return false;
            }
        }  // load
    };  // type_caster<cs::CSCMatrix>
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
auto matrix_to_ndarray(const T& self, const char order)
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



#endif  // _CSPARSE_PYBIND11_H_

//==============================================================================
//==============================================================================
