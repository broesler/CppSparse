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

LU_NOPIVOT_FUNCS = [
    csparse.lu_rightr,
    csparse.lu_right,
]

LU_PIVOT_FUNCS = [
    csparse.lu_left,
    csparse.lu_rightpr,
    csparse.lu_rightprv,
    csparse.lu_rightp,
]

LU_FUNCS = LU_NOPIVOT_FUNCS + LU_PIVOT_FUNCS


@pytest.fixture
def A():
    """Define a dense, square matrix."""
    return csparse.davis_example_qr(format='ndarray')


def lu_helper(A, lu_func):
    """Helper function to test the LU decomposition."""
    # Compare to scipy PLU = A
    P_, L_, U_ = la.lu(A)

    np.testing.assert_allclose(P_ @ L_ @ U_, A, atol=ATOL)

    # Computes LU = PA
    P, L, U = lu_func(A)

    if lu_func is csparse.lu_rightprv:
        # P is a vector, so create the matrix
        P = np.eye(A.shape[0])[:, P]

    np.testing.assert_allclose(P, P_.T, atol=ATOL)
    np.testing.assert_allclose(L, L_, atol=ATOL)
    np.testing.assert_allclose(U, U_, atol=ATOL)
    np.testing.assert_allclose(L @ U, P @ A, atol=ATOL)


@pytest.mark.parametrize("lu_func", LU_FUNCS)
def test_nonpivoting_LU(A, lu_func):
    """Test the LU decomposition without pivoting."""
    lu_helper(A, lu_func)


# Only parametrize with functions that include pivoting
@pytest.mark.parametrize("lu_func", LU_PIVOT_FUNCS)
def test_pivoting_LU(A, lu_func):
    """Test the LU decomposition with pivoting."""
    # Permute the rows of A so pivoting is required
    seed = 56
    rng = np.random.default_rng(seed)
    A_in = A.copy()  # preserve the original matrix
    for i in range(A.shape[0]):
        print(f"Testing LU with permuted rows, {seed=}, {i=}")
        p = np.arange(A.shape[0])
        rng.shuffle(p)
        A = A_in[p]

        lu_helper(A, lu_func)


# =============================================================================
# =============================================================================
