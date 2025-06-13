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

from numpy.testing import assert_allclose
from scipy import sparse
from scipy.sparse import linalg as spla

from .helpers import (
    generate_suitesparse_matrices,
    generate_random_matrices,
    BaseSuiteSparseTest
)

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
    assert_allclose(A @ x, b, atol=ATOL)


# -----------------------------------------------------------------------------
#         Test 8
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    'problem',
    list(generate_suitesparse_matrices(square_only=True)),
    indirect=True
)
class TestCholLUSolve(BaseSuiteSparseTest):
    """Test Cholesky and LU solve functions on SuiteSparse matrices."""
    @pytest.fixture(scope='class', autouse=True)
    def setup_problem(self, request, problem):
        """Setup the problem matrix."""
        cls = request.cls
        A = problem.A
        M, N = A.shape
        is_spd = False

        if M == N:
            is_symmetric = (A - A.T).nnz == 0
            if is_symmetric:
                p = csparse.amd(A, order='APlusAT')
                try:
                    la.cholesky(A[p][:, p].toarray(), lower=True)
                    is_spd = True
                except la.LinAlgError:
                    print("`A` is not symmetric positive definite.")

        if is_spd:
            cls.C = A
        else:
            cls.C = A @ A.T + N * sparse.eye_array(N)
            p = csparse.amd(cls.C, order='APlusAT')
            try:
                la.cholesky(cls.C[p][:, p].toarray(), lower=True)
            except la.LinAlgError:
                pytest.skip("`C` is not symmetric positive definite.")

        # Define the rhs
        rng = np.random.default_rng(565656)
        cls.b = rng.random((M,))

    def test_chol_solve(self):
        """Test Cholesky solve on the problem matrix."""
        x_ = spla.spsolve(self.C, self.b)
        x = csparse.chol_solve(self.C, self.b)
        np.testing.assert_allclose(x, x_, atol=ATOL)

    @pytest.mark.parametrize('order', ['APlusAT', 'ATANoDenseRows'])
    def test_lu_solve(self, order):
        """Test LU solve on the problem matrix."""
        tol = 0.001 if order == 'APlusAT' else 1.0
        x_ = spla.spsolve(self.C, self.b)
        x = csparse.lu_solve(self.C, self.b, order=order, tol=tol)
        np.testing.assert_allclose(x, x_, atol=ATOL)


# -----------------------------------------------------------------------------
#         Test 17
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('A', list(generate_random_matrices(square_only=True)))
def test_qr_solve(request, A):
    """Test QR solve on a random matrix."""
    test_id = request.node.name
    order = 'ATA'  # column ordering for QR

    M, N = A.shape

    if M < N:
        A = A.T

    M, N = A.shape

    rng = np.random.default_rng(565656)
    b = rng.random((M,))

    # Compute CSparse QR
    V, beta, R, p, q = csparse.qr(A, order=order)

    V = V.toarray()
    R = R.toarray()

    Q = csparse.apply_qright(V, beta, p)

    # Add extra rows to match Q
    A_long = A.copy()
    A_long.resize((Q.shape[0], N))

    # Solve using scipy QR
    Aq = A[:, q].toarray()
    Q_, R_ = la.qr(Aq)

    # Make sure the factorizations worked
    assert_allclose(Q @ R, A_long[:, q].toarray(), atol=ATOL)
    assert_allclose(Q_ @ R_, Aq, atol=ATOL)

    # Test the actual problem solution
    fail_ids = [6, 12, 18, 53, 71, 78, 91, 93]
    if test_id in [f"test_qr_solve[random_{i}]" for i in fail_ids]:
        # These tests fail because the last entry in U[-1, -1] is either
        # exactly 0.0 or extremely small (e.g. ~1e-21), so when we do "x /=
        # U[-1, -1]" it results is ~1e+20, and blows up the rest of the
        # solution.
        #
        # When we run the same matrix and RHS through cs_qrsol -> same result.
        #
        # Neither function supports the least squares solution.
        pytest.xfail("Matrix is singular, expected failure.")

    x_ = la.lstsq(R_, Q_.T @ b)[0]
    x_[q] = x_  # inverse permutation
    r_ = la.norm(A @ x_ - b)

    # Solve using csparse QR
    x = csparse.qr_solve(A, b, order=order)  # FIXME
    x[np.isnan(x)] = 0
    r = la.norm(A @ x - b)

    # Solve using Q matrix created from csparse QR
    QTPb = csparse.apply_qtleft(V, beta, p, b).reshape(-1)
    xq = la.lstsq(R, QTPb)[0]
    xq[q] = xq  # inverse permutation
    rq = la.norm(A @ xq - b)

    # Check the solution
    print(f"{test_id}: r_={r_:.2e}, r={r:.2e}, rq={rq:.2e}")
    assert_allclose(xq, x_, atol=1e-10)  # always passes
    assert_allclose(x, x_, atol=1e-10)   # fails

# =============================================================================
# =============================================================================
