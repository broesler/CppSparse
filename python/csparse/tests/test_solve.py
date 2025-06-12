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

from .helpers import generate_suitesparse_matrices

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
class TestCholLUSolve:
    """Test Cholesky and LU solve functions on SuiteSparse matrices."""
    @pytest.fixture(scope='class')
    def problem(self, request):
        """Fixture to provide a SuiteSparse matrix."""
        return request.param

    @pytest.fixture(scope='class', autouse=True)
    def setup_problem(self, request, problem):
        """Setup the problem matrix."""
        cls = request.cls
        cls.problem = problem
        print(f"---------- {cls.problem.name}")

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

# =============================================================================
# =============================================================================
