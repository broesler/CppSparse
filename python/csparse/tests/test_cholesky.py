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

@pytest.fixture
def A_matrix():
    """Define a symmetric, positive definite matrix."""
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

    # NOTE scipy Cholesky is only implemented for dense matrices!
    return A.toarray()


CHOL_FUNCS = [
    chol_up,
    chol_left,
    chol_left_amp,
    chol_right,
    chol_super
]


@pytest.mark.parametrize("chol_func", CHOL_FUNCS)
def test_cholesky(A_matrix, chol_func):
    """Test the Cholesky decomposition algorithms."""
    A = A_matrix

    # Compute the Cholesky factor
    L_ = la.cholesky(A, lower=True)

    if chol_func == chol_super:
        # Define "supernodes" as ones to get the same result as left-looking
        s = np.ones(A.shape[0], dtype=int)
        L = chol_super(A, s, lower=True)
    else:
        L = chol_func(A, lower=True)

    # Check that algorithms work
    np.testing.assert_allclose(L, L_, atol=ATOL)
    np.testing.assert_allclose(L @ L.T, A, atol=ATOL)


def test_cholesky_update(A_matrix):
    """Test the Cholesky update and downdate algorithms."""
    A = A_matrix
    N = A.shape[0]

    L = la.cholesky(A, lower=True)

    seed = 565656
    rng = np.random.default_rng(seed)

    # Test multiple random vectors
    for i in range(10):
        print(f"Testing update {i} with seed {seed}")  # keep for failures
        # Generate random update with same pattern as a random column of L
        k = rng.integers(0, N)
        idx = np.nonzero(L[:, k])[0]
        w = np.zeros((N,))
        w[idx] = rng.random(idx.size)

        wwT = np.outer(w, w)  # == w[:, np.newaxis] @ w[np.newaxis, :]

        A_up = A + wwT

        L_up, w_up = chol_update(L, w)
        np.testing.assert_allclose(L_up @ L_up.T, A_up, atol=ATOL)
        np.testing.assert_allclose(la.solve(L, w), w_up, atol=ATOL)

        L_upd, w_upd = chol_updown(L, w, update=True)
        np.testing.assert_allclose(L_up, L_upd, atol=ATOL)
        np.testing.assert_allclose(L_upd @ L_upd.T, A_up, atol=ATOL)
        np.testing.assert_allclose(la.solve(L, w), w_upd, atol=ATOL)

        # Just downdate back to the original matrix!
        A_down = A.copy()  # A_down == A_up - wwT == A
        L_down, w_down = chol_downdate(L_up, w)
        np.testing.assert_allclose(L_down @ L_down.T, A_down, atol=ATOL)
        np.testing.assert_allclose(la.solve(L_up, w), w_down, atol=ATOL)

        L_downd, w_downd = chol_updown(L_up, w, update=False)
        np.testing.assert_allclose(L_down, L_downd, atol=ATOL)
        np.testing.assert_allclose(L_downd @ L_downd.T, A_down, atol=ATOL)
        np.testing.assert_allclose(la.solve(L_up, w), w_downd, atol=ATOL)


# =============================================================================
# =============================================================================
