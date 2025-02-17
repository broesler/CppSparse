#!/usr/bin/env python3
# =============================================================================
#     File: __init__.py
#  Created: 2025-02-14 08:59
#   Author: Bernie Roesler
#
"""
csparse: A Python wrapper for the CSparse++ library.

This module provides bindings for sparse matrix operations using pybind11.

Example usage:
    import csparse
    rows = [0, 1, 2, 0, 1, 2, 0, 2]
    cols = [0, 0, 0, 1, 1, 1, 2, 2]
    vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    A = csparse.CSparse(rows, cols, vals, 3, 3)
    print(A)

Author: Bernie Roesler
Date: 2025-02-14
Version: 0.1
"""
# =============================================================================

# Import all bindings from the csparse module
from .csparse import *
from .qr_utils import *


# TODO move to own file
def davis_example():
    """Create the matrix from Davis, "Direct Methods for Sparse Linear Systems"
    p. 7-8, Eqn (2.1).

    .. math:: 
        A = \begin{bmatrix}
            4.5 & 0   & 3.2 & 0   \\
            3.1 & 2.9 & 0   & 0.9 \\
            0   & 1.7 & 3   & 0   \\
            3.5 & 0.4 & 0   & 1 
        \end{bmatrix}

    Returns
    -------
    A : (4, 4) csparse.CSCMatrix
        The example matrix from Davis.
    """
    N = 4
    rows = np.r_[2,    1,    3,    0,    1,    3,    3,    1,    0,    2]
    cols = np.r_[2,    0,    3,    2,    1,    0,    1,    3,    0,    1]
    vals = np.r_[3.0,  3.1,  1.0,  3.2,  2.9,  3.5,  0.4,  0.9,  4.5,  1.7]
    return csparse.COOMatrix(vals, rows, cols, N, N).tocsc()


__all__ = dir(csparse) + dir(qr_utils)

# =============================================================================
# =============================================================================
