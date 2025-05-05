#!/usr/bin/env python3
# =============================================================================
#     File: test_solve.py
#  Created: 2025-05-05 11:22
#   Author: Bernie Roesler
#
"""
Unit tests for csparse solve functions.
"""
# =============================================================================

import pytest 
import numpy as np
import scipy.linalg as la

from scipy import sparse
from scipy.sparse import linalg as spla

import csparse

ATOL = 1e-14

SOLVE_FUNCS = [
    csparse.chol_solve,
    csparse.lu_solve,
    csparse.qr_solve,
    csparse.dm_solve
]


# TODO generate classes of tests matrices:
#   * symmetric positive definite (Cholesky)
#   * square, symmetric indefinite (LU)
#   * square, asymmetric (LU, QR, DM)
#   * overdetermined (QR, DM)
#   * underdetermined (QR, DM)


@pytest.mark.parametrize('solve_func', SOLVE_FUNCS)
def test_solve_func(solve_func):
    """Test the solve function with a known right-hand side."""
    A = csparse.davis_example_chol()
    M, N = A.shape
    expect = np.arange(1, M + 1)  # use range to test permutations
    b = np.array(A @ expect)
    x = np.array(solve_func(A, b))
    assert x is not None
    assert x.shape == (N,)
    assert np.allclose(np.array(A @ x), b, atol=ATOL)


# =============================================================================
# =============================================================================
