#!/usr/bin/env python3
# =============================================================================
#     File: davis_example.py
#  Created: 2025-02-05 11:45
#   Author: Bernie Roesler
#
"""
Description: Davis book example from Equation (2.1) on pages 7--8.
"""
# =============================================================================

import numpy as np

from scipy import sparse
from scipy import linalg as la
from scipy.sparse import linalg as spla

import csparse

# See Davis pp 7-8, Eqn (2.1)
# A_dense = np.array(
#     [[4.5, 0. , 3.2, 0. ],
#      [3.1, 2.9, 0. , 0.9],
#      [0. , 1.7, 3. , 0. ],
#      [3.5, 0.4, 0. , 1. ]]
# )

A = csparse.davis_example_small().tocsc()

# NOTE 
#   * numpy returns an array for lists or slices, and a np.float for scalars.
#       The array has the same dimensions as the inputs, e.g.,
#       - A[:, k] returns a 1D array of shape (4,)
#       - A[k, :] returns a 1D array of shape (4,)
#       - A[:, [k]] returns a 2D array of shape (4, 1).
#
#   * scipy.sparse.csc_matrix returns coo_matrix for slices, and a np.float for
#       scalars. The coo_matrix does *not* have the same dimensions as the
#       inputs, e.g.,
#       - both A[:, k] and A[:, [k]] return a 2D csc_matrix of size (4, 1).
#
#   * scipy.sparse.csc_array is similar to np.ndarray. It returns a np.float
#       for scalars.
#       - A[:, k] returns a 1D coo_array of shape(4,)
#       - A[k, :] returns a 1D coo_array of shape(4,)
#       - A[:, [k]] returns a 2D csc_array of shape (4, 1).
#       - scipy.sparse.csc_array must be exactly 2D. coo_array must be used for
#       1D or > 2D arrays.
#       
#   * MATLAB/octave returns a sparse matrix regardless of the type of index.

# Test indexing and slicing
atol = 1e-15
np.testing.assert_allclose(A[0, 0], 4.5, atol=atol)
np.testing.assert_allclose(A[:, 0].toarray(), np.r_[4.5, 3.1, 0, 3.5], atol=atol)
np.testing.assert_allclose(A[0, :].toarray(), np.r_[4.5, 0, 3.2, 0], atol=atol)
np.testing.assert_allclose(A[1:3, 0].toarray(), np.r_[3.1, 0], atol=atol)
np.testing.assert_allclose(A[0, 1:3].toarray(), np.r_[0, 3.2], atol=atol)
np.testing.assert_allclose(A[::2, 0].toarray(), np.r_[4.5, 0], atol=atol)
np.testing.assert_allclose(A[0, ::2].toarray(), np.r_[4.5, 3.2], atol=atol)
np.testing.assert_allclose(A[1:3, 1:3].toarray(), np.array([[2.9, 0], [1.7, 3.0]]), atol=atol)

# =============================================================================
# =============================================================================
