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

from scipy import linalg as la

import csparse

ATOL = 1e-15  # testing tolerance


@pytest.fixture
def A_matrix():
    """Define a symmetric, positive definite matrix."""
    return csparse.davis_example_chol(format='ndarray')


PYTHON_CHOL_FUNCS = [
    csparse.chol_up,
    csparse.chol_left,
    csparse.chol_left_amp,
    csparse.chol_right,
    csparse.chol_super
]


@pytest.mark.parametrize("use_postorder", [False, True])
@pytest.mark.parametrize("order", ['Natural', 'APlusAT'])
def test_cholesky_interface(order, use_postorder):
    """Test the Cholesky decomposition python interface."""
    A = csparse.davis_example_chol()

    L, _ = csparse.chol(A, order, use_postorder)
    Ls = csparse.symbolic_cholesky(A, order, use_postorder)
    Ll = csparse.leftchol(A, order, use_postorder)
    Lr = csparse.rechol(A, order, use_postorder)

    np.testing.assert_allclose(L.indptr, Ls.indptr)
    np.testing.assert_allclose(L.indices, Ls.indices)
    np.testing.assert_allclose(L.toarray(), Ll.toarray())
    np.testing.assert_allclose(Ll.toarray(), Lr.toarray())


@pytest.mark.parametrize("chol_func", PYTHON_CHOL_FUNCS)
def test_python_cholesky(A_matrix, chol_func):
    """Test the Cholesky decomposition algorithms."""
    A = A_matrix

    # Compute the Cholesky factor
    L_ = la.cholesky(A, lower=True)

    if chol_func == csparse.chol_super:
        # Define "supernodes" as ones to get the same result as left-looking
        s = np.ones(A.shape[0], dtype=int)
        L = csparse.chol_super(A, s, lower=True)
    else:
        L = chol_func(A, lower=True)

    # Check that algorithms work
    np.testing.assert_allclose(L, L_, atol=ATOL)
    np.testing.assert_allclose(L @ L.T, A, atol=ATOL)


def test_python_cholesky_update(A_matrix):
    """Test the Cholesky update and downdate algorithms.

    .. note:: These tests only cover the python implementations of the
    update/downdate functions, not the C++ cs::chol_update function.
    """
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

        L_up, w_up = csparse.chol_update(L, w)
        np.testing.assert_allclose(L_up @ L_up.T, A_up, atol=ATOL)
        np.testing.assert_allclose(la.solve(L, w), w_up, atol=ATOL)

        L_upd, w_upd = csparse.chol_updown(L, w, update=True)
        np.testing.assert_allclose(L_up, L_upd, atol=ATOL)
        np.testing.assert_allclose(L_upd @ L_upd.T, A_up, atol=ATOL)
        np.testing.assert_allclose(la.solve(L, w), w_upd, atol=ATOL)

        # Just downdate back to the original matrix!
        A_down = A.copy()  # A_down == A_up - wwT == A
        L_down, w_down = csparse.chol_downdate(L_up, w)
        np.testing.assert_allclose(L_down @ L_down.T, A_down, atol=ATOL)
        np.testing.assert_allclose(la.solve(L_up, w), w_down, atol=ATOL)

        L_downd, w_downd = csparse.chol_updown(L_up, w, update=False)
        np.testing.assert_allclose(L_down, L_downd, atol=ATOL)
        np.testing.assert_allclose(L_downd @ L_downd.T, A_down, atol=ATOL)
        np.testing.assert_allclose(la.solve(L_up, w), w_downd, atol=ATOL)


# =============================================================================
# =============================================================================
