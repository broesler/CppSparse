#!/usr/bin/env python3
# =============================================================================
#     File: test_lu.py
#  Created: 2025-03-20 12:15
#   Author: Bernie Roesler
#
"""
Unit tests for the csparse.lu*() functions.
"""
# =============================================================================

import pytest
import numpy as np

from scipy import linalg as la

import csparse

ATOL = 1e-15  # testing tolerance

PYTHON_LU_FUNCS = [
    csparse.lu_left,
    csparse.lu_right,
    csparse.lu_rightp,
]


@pytest.fixture
def A():
    """Define a dense, square matrix."""
    return csparse.davis_example_qr(format='ndarray')


@pytest.mark.parametrize(
    "permute_rows, lu_func",
    [pytest.param(False, func, id=f"permute_rows=False, func={func.__name__}")
     for func in PYTHON_LU_FUNCS]
    +
    [pytest.param(True, func, id=f"permute_rows=True, func={func.__name__}")
     for func in [csparse.lu_left, csparse.lu_rightp]]
)
def test_nonpivoting_LU(A, permute_rows, lu_func):
    """Test the non-pivoting LU decomposition."""
    if permute_rows:
        # Permute the rows of A so pivoting is required
        seed = 56
        print(f"Testing LU with permuted rows, {seed=}")
        p = np.arange(A.shape[0])
        rng = np.random.default_rng(seed)
        rng.shuffle(p)
        A = A[p]

    # Compare to scipy PLU = A
    P_, L_, U_ = la.lu(A)

    np.testing.assert_allclose(P_ @ L_ @ U_, A, atol=ATOL)

    # Computes LU = PA
    P, L, U = lu_func(A)

    np.testing.assert_allclose(P, P_.T, atol=ATOL)
    np.testing.assert_allclose(L, L_, atol=ATOL)
    np.testing.assert_allclose(U, U_, atol=ATOL)
    np.testing.assert_allclose(L @ U, P @ A, atol=ATOL)


# =============================================================================
# =============================================================================
