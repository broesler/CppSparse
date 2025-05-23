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

from scipy import linalg as la, sparse
from scipy.sparse import linalg as spla

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
    csparse.lu_crout,
]

LU_FUNCS = LU_NOPIVOT_FUNCS + LU_PIVOT_FUNCS


@pytest.fixture
def A():
    """Define a dense, square matrix."""
    return csparse.davis_example_qr().toarray()


@pytest.mark.parametrize("order", ['Natural', 'ATANoDenseRows', 'ATA'])
def test_lu_interface(order):
    """Test the LU decomposition python interface."""
    Ac = csparse.davis_example_qr()
    A = Ac.toarray()

    # Test the LU decomposition with the default order
    L, U, p, q = csparse.lu(Ac, order=order)

    np.testing.assert_allclose((L @ U).toarray(), A[p][:, q], atol=ATOL)


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


def test_1norm_estimate(A):
    """Test the 1-norm and condition number estimates."""
    # Notes are for csparse.davis_example_qr(10)
    As = sparse.csc_matrix(A)

    normd = la.norm(A, 1)
    norms = spla.norm(As, 1)
    norm_est = spla.onenormest(A)

    np.testing.assert_allclose(normd, norms, atol=ATOL)
    np.testing.assert_allclose(normd, norm_est, atol=ATOL)

    # Condition number == ||A||_1 * ||A^-1||_1
    condd = np.linalg.cond(A, 1)

    Ainv = la.inv(A)
    Asinv = spla.inv(As)

    normd_inv = la.norm(Ainv, 1)
    norms_inv = spla.norm(Asinv, 1)
    norm_est_inv = spla.onenormest(Ainv)

    # Test out condition number estimate
    # CSparse version:
    #
    # >> [L, U, P, Q] = lu(A);
    # >> norm1est(L, U, P, Q)  % CSparse 1-norm estimate
    # ans = 0.1153750055167834

    # >> cond1est(A)  % CSparse 1-norm condition number estimate
    # ans = 2.422875115852452

    # >> condest(A)  % built-in 1-norm condition number estimate
    # ans = 2.422875115852452

    # C++Sparse version:
    normc_inv = csparse.norm1est_inv(As)  # == 0.11537500551678347

    np.testing.assert_allclose(normd_inv, norms_inv, atol=ATOL)
    np.testing.assert_allclose(normd_inv, norm_est_inv, atol=ATOL)
    np.testing.assert_allclose(normd_inv, normc_inv, atol=ATOL)

    κc = csparse.cond1est(As)          # == 2.422875115852453

    np.testing.assert_allclose(condd, κc, atol=ATOL)

    print("---------- 1-norm estimate:")
    print("    normd:", normd)
    print("normd_inv:", normd_inv)
    print("    condd:", condd)


# =============================================================================
# =============================================================================
