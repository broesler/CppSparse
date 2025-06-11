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

from numpy.testing import assert_allclose
from scipy import linalg as la
from scipy import sparse
from scipy.sparse import linalg as sla

from .helpers import generate_suitesparse_matrices

import csparse

ATOL = 1e-15  # testing tolerance


@pytest.fixture
def A():
    """Define a dense, symmetric, positive definite matrix."""
    return csparse.davis_example_chol().toarray()


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

    L, p, _ = csparse.chol(A, order, use_postorder)
    Ls = csparse.symbolic_cholesky(A, order, use_postorder).L
    Ll = csparse.leftchol(A, order, use_postorder).L
    Lr = csparse.rechol(A, order, use_postorder).L

    assert_allclose(L.indptr, Ls.indptr)
    assert_allclose(L.indices, Ls.indices)
    assert_allclose(L.toarray(), Ll.toarray())
    assert_allclose(Ll.toarray(), Lr.toarray())

    assert_allclose((L @ L.T).toarray(), A[p][:, p].toarray(), atol=ATOL)


@pytest.mark.parametrize("chol_func", PYTHON_CHOL_FUNCS)
def test_python_cholesky(A, chol_func):
    """Test the Cholesky decomposition algorithms."""
    # Compute the Cholesky factor
    L_ = la.cholesky(A, lower=True)

    if chol_func == csparse.chol_super:
        # Define "supernodes" as ones to get the same result as left-looking
        s = np.ones(A.shape[0], dtype=int)
        L = csparse.chol_super(A, s, lower=True)
    else:
        L = chol_func(A, lower=True)

    # Check that algorithms work
    assert_allclose(L, L_, atol=ATOL)
    assert_allclose(L @ L.T, A, atol=ATOL)


def test_python_cholesky_update(A):
    """Test the Cholesky update and downdate algorithms.

    .. note:: These tests only cover the python implementations of the
    update/downdate functions, not the C++ cs::chol_update function.
    """
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
        assert_allclose(L_up @ L_up.T, A_up, atol=ATOL)
        assert_allclose(la.solve(L, w), w_up, atol=ATOL)

        L_upd, w_upd = csparse.chol_updown(L, w, update=True)
        assert_allclose(L_up, L_upd, atol=ATOL)
        assert_allclose(L_upd @ L_upd.T, A_up, atol=ATOL)
        assert_allclose(la.solve(L, w), w_upd, atol=ATOL)

        # Just downdate back to the original matrix!
        A_down = A.copy()  # A_down == A_up - wwT == A
        L_down, w_down = csparse.chol_downdate(L_up, w)
        assert_allclose(L_down @ L_down.T, A_down, atol=ATOL)
        assert_allclose(la.solve(L_up, w), w_down, atol=ATOL)

        L_downd, w_downd = csparse.chol_updown(L_up, w, update=False)
        assert_allclose(L_down, L_downd, atol=ATOL)
        assert_allclose(L_downd @ L_downd.T, A_down, atol=ATOL)
        assert_allclose(la.solve(L_up, w), w_downd, atol=ATOL)


# -----------------------------------------------------------------------------
#         Test 3
# -----------------------------------------------------------------------------
# TODO refactor as a class and separate out tests
@pytest.mark.parametrize(
    "problem",
    generate_suitesparse_matrices(square_only=True)
)
def test_trisolve_cholesky(problem):
    """Test triangular solvers with Cholesky factors."""
    A = problem.A
    order = 'APlusAT'
    M, N = A.shape

    # Make the matrix symmetric positive definite
    A = A @ A.T + 2 * N * sparse.eye_array(N)

    # Get the Cholesky factorization using scipy (dense matrices only)
    try:
        L0 = sparse.csc_array(la.cholesky(A.toarray(), lower=True))
    except Exception:
        print(f"Skipping {problem.name} due to Cholesky failure.")
        return

    # Get the permutation from AMD
    # TODO symamd? not in python
    p = csparse.amd(A, order=order)

    # RHS
    rng = np.random.default_rng(problem.id)
    b = rng.random(N)

    # Reorder the matrix
    C = A[p][:, p]
    κ = csparse.cond1est(A)
    print(f"cond1est: {κ:.4e} ({problem.name})")

    # Check lsolve
    x1 = sla.spsolve(L0, b)
    x2 = csparse.lsolve(L0, b)
    assert_allclose(x1, x2, atol=1e-12 * κ)

    # Check ltsolve
    x1 = sla.spsolve(L0.T, b)
    x2 = csparse.ltsolve(L0, b)
    err = la.norm(x1 - x2, ord=1)
    assert_allclose(x1, x2, atol=1e-10 * κ)

    # usolve
    U = L0.T

    x1 = sla.spsolve(U, b)
    x2 = csparse.usolve(U, b)
    assert_allclose(x1, x2, atol=1e-10 * κ)

    # utsolve (not in test3.m)
    x1 = sla.spsolve(U.T, b)
    x2 = csparse.utsolve(U, b)
    assert_allclose(x1, x2, atol=1e-10 * κ)

    # C++Sparse Cholesky
    L2 = csparse.chol(A).L
    assert_allclose(L0.toarray(), L2.toarray(), atol=1e-8 * κ)

    # Cholesky of re-ordered A
    L1 = sparse.csc_array(la.cholesky(C.toarray(), lower=True))
    L2 = csparse.chol(C).L
    assert_allclose(L1.toarray(), L2.toarray(), atol=1e-8 * κ)

    # Cholesky of A, then reorder
    res = csparse.chol(A, order=order)
    L3 = res.L
    p3 = res.p
    C3 = A[p3][:, p3]
    L4 = sparse.csc_array(la.cholesky(C3.toarray(), lower=True))
    assert_allclose(L3.toarray(), L4.toarray(), atol=1e-8 * κ)
    






# =============================================================================
# =============================================================================
