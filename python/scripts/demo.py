#!/usr/bin/env python3
# =============================================================================
#     File: demo.py
#  Created: 2025-05-19 16:15
#   Author: Bernie Roesler
#
"""
Description: Helper functions for the C++Sparse demo programs.
"""
# =============================================================================

import numpy as np
import scipy.linalg as la

from scipy import sparse
from scipy.sparse import linalg as spla

import csparse


# TODO create pybind bindings for this function
def print_resid(A, x, b):
    """Print the norm of the residual of the linear system `Ax = b`.

    Parameters
    ----------
    A : (M, N) array_like
        The matrix `A`.
    x : (N,) array_like
        The solution vector `x`.
    b : (M,) array_like
        The right-hand side vector `b`.
    """
    r = A @ x - b
    resid = (
        la.norm(r, np.inf) /
        (spla.norm(A, 1) * la.norm(x, np.inf) + la.norm(b, np.inf))
    )
    print(f"resid: {resid:.2e}")

# =============================================================================
# =============================================================================
