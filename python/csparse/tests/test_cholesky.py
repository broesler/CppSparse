#!/usr/bin/env python3
# =============================================================================
#     File: test_cholesky.py
#  Created: 2025-02-20 14:55
#   Author: Bernie Roesler
#
"""
Unit tests for the python Cholesky algorithms.
"""
# =============================================================================

import pytest
import numpy as np

from scipy import (linalg as la,
                   sparse as sparse)

from csparse._cholesky import (chol_up, chol_left, chol_left_amp, chol_right,
                               chol_super,
                               chol_update, chol_downdate, chol_updown)

ATOL = 1e-15  # testing tolerance


# TODO break into multiple individual tests with A as a fixture
def test_cholesky():
    """Test the Cholesky decomposition algorithms."""
    # Create the example matrix A
    N = 11
    # Only off-diagonal elements
    rows = np.r_[5, 6, 2, 7, 9, 10, 5, 9, 7, 10, 8, 9, 10, 9, 10, 10]
    cols = np.r_[0, 0, 1, 1, 2,  2, 3, 3, 4,  4, 5, 5,  6, 7,  7,  9]
    vals = np.ones((rows.size,))

    # Values for the lower triangle
    L = sparse.csc_matrix((vals, (rows, cols)), shape=(N, N))

    # Create the symmetric matrix A
    A = L + L.T

    # Get the sum of the off-diagonal elements to ensure positive definiteness
    # diag_A = np.max(np.sum(A, axis=0))
    A.setdiag(np.arange(10, 21))

    A = A.toarray()

    # NOTE Scipy Cholesky is only implemented for dense matrices!
    R = la.cholesky(A, lower=True)
    R_up = chol_up(A, lower=True)
    R_left = chol_left(A, lower=True)
    R_left_amp = chol_left_amp(A, lower=True)
    R_right = chol_right(A, lower=True)

    # Define "supernodes" as ones, so we get the same result as left-looking
    s = np.ones(A.shape[0], dtype=int)
    R_super = chol_super(A, s, lower=True)

    # NOTE etree is not implemented in scipy!
    # Get the elimination tree
    # [parent, post] = etree(A)
    # Rp = la.cholesky(A(post, post), lower=True)
    # assert (R.nnz == Rp.nnz)  # post-ordering does not change nnz

    # Compute the row counts of the post-ordered Cholesky factor
    row_counts = np.sum(R != 0, 1)
    col_counts = np.sum(R != 0, 0)

    # -------------------------------------------------------------------------
    #         Test Algorithms
    # -------------------------------------------------------------------------
    # Check that algorithms work
    for L in [R, R_up, R_left, R_left_amp, R_super, R_right]:
        np.testing.assert_allclose(L @ L.T, A, atol=ATOL)

    # Test (up|down)date
    # Generate random update with same sparsity pattern as a column of L
    rng = np.random.default_rng(565656)
    k = rng.integers(0, N)
    idx = np.nonzero(R[:, k])[0]
    w = np.zeros((N,))
    w[idx] = rng.random(idx.size)

    wwT = np.outer(w, w)  # == w[:, np.newaxis] @ w[np.newaxis, :]

    A_up = A + wwT

    L_up, w_up = chol_update(R, w)
    np.testing.assert_allclose(L_up @ L_up.T, A_up, atol=ATOL)
    np.testing.assert_allclose(la.solve(R, w), w_up, atol=ATOL)

    L_upd, w_upd = chol_updown(R, w, update=True)
    np.testing.assert_allclose(L_up, L_upd, atol=ATOL)
    np.testing.assert_allclose(L_upd @ L_upd.T, A_up, atol=ATOL)
    np.testing.assert_allclose(la.solve(R, w), w_upd, atol=ATOL)

    # Just downdate back to the original matrix!
    A_down = A.copy()  # A_down == A_up - wwT == A
    L_down, w_down = chol_downdate(L_up, w)
    np.testing.assert_allclose(L_down @ L_down.T, A_down, atol=ATOL)
    np.testing.assert_allclose(la.solve(L_up, w), w_down, atol=ATOL)

    L_downd, w_downd = chol_updown(L_up, w, update=False)
    np.testing.assert_allclose(L_down, L_downd, atol=ATOL)
    np.testing.assert_allclose(L_downd @ L_downd.T, A_down, atol=ATOL)
    np.testing.assert_allclose(la.solve(L_up, w), w_downd, atol=ATOL)

    # Count the nonzeros of the Cholesky factor of A^T A
    ATA = A.T @ A
    L_ATA = la.cholesky(ATA, lower=True)
    nnz_cols = np.diff(sparse.csc_matrix(L_ATA).indptr)

# =============================================================================
# =============================================================================
