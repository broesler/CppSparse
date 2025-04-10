#!/usr/bin/env python3
# =============================================================================
#     File: test_qr.py
#  Created: 2025-02-20 11:49
#   Author: Bernie Roesler
#
"""
Unit tests for the csparse.qr() function.
"""
# =============================================================================

import pytest
import numpy as np

from scipy import sparse
from scipy import linalg as la

import csparse


ATOL = 1e-12  # random matrices need a bit larger tolerance than 1e-15

# Group the matrices for parameterization
N = 7  # arbitrary matrix size for testing
TEST_MATRICES = [
    ("Identity", sparse.eye_array(N).tocsc()),
    ("Diagonal", sparse.diags(np.arange(1, N)).tocsc()),
    ("Asymmetric Banded",
        sparse.diags([np.ones(N-1), np.arange(1, N+1)], [-1, 0]).tocsc()),
    ("Laplacian (Symmetric Banded)",
        sparse.diags([1, -2, 1], [-1, 0, 1]).tocsc()),
    ("Davis 8x8", csparse.davis_example_qr(format='csc')),
    ("Davis 4x4", csparse.davis_example_small(format='csc')),
    # See: Strang Linear Algebra p 203.
    ("Strang 3x3", sparse.csc_array(
        np.array([[1, 1, 2], [0, 0, 1], [1, 0, 0]], dtype=float)
    )),
    ("Davis 8x5 (M > N)", csparse.davis_example_qr(format='csc')[:, :5]),
    ("Davis 8x5 (M < N)", csparse.davis_example_qr(format='csc')[:5, :])
]


@pytest.mark.parametrize("case_name, A", TEST_MATRICES)
def test_qr_fixed(case_name, A):
    """Test QR decomposition with various matrices."""
    _test_qr_decomposition(A, case_name)


@pytest.mark.parametrize("N", [2, 7, 10])
def test_qr_random(N):
    """Test QR decomposition with random matrices."""
    N_runs = 10
    seed = 565656
    rng = np.random.default_rng(seed)
    for i in range(N_runs):
        A = sparse.random(N, N, density=0.5, format='csc', random_state=rng)
        A.setdiag(N * np.arange(1, N+1))  # ensure structural full rank
        _test_qr_decomposition(A, f"Random {N}x{N} ({seed=}, {i=})")


class TestQRColumnPivoting:
    @pytest.fixture
    def A(self):
        """Create a test matrix for column pivoting."""
        A = csparse.davis_example_qr(format='ndarray')
        # Add 10 to diagonal to enforce expected pivoting
        M, N = A.shape
        i = range(M)
        A[i, i] += 10
        return A

    @pytest.mark.parametrize("ks", [[], [3], [2, 3, 5]])
    def test_qr_pivoting(self, A, ks):
        """Test QR decomposition with column pivoting."""
        tol = 0.1  # tolerance for column pivoting

        # Create small columns so that they pivot
        for k in ks:
            A_kk = A[k, k]
            A[:, k] *= 0.95 * tol / A_kk

        As = sparse.csc_array(A)

        def qr_func(A):
            return csparse.qr_pivoting(A, tol)

        _test_qr_decomposition(As, qr_func=qr_func)


def _test_qr_decomposition(A, case_name='', qr_func=csparse.qr):
    """Test QR decomposition with various matrices using parametrization."""
    print(case_name)
    Ac = csparse.from_scipy_sparse(A, format='csc')
    A_dense = A.toarray()
    M, N = A.shape

    # ---------- Compute csparse QR
    QRres = qr_func(Ac)
    V, beta, R, p_inv, q = QRres.V, QRres.beta, QRres.R, QRres.p_inv, QRres.q

    # Convert to numpy arrays
    V = V.toarray()
    R = R.toarray()

    p = csparse.inv_permute(p_inv)
    Q = csparse.apply_qright(V, beta, p)
    Ql = csparse.apply_qtleft(V, beta, p).T

    # Test the apply functions both get the same Q
    np.testing.assert_allclose(Q, Ql, atol=ATOL)

    # ---------- scipy QR
    # Apply the row and column permutation to A_dense
    PAQ = A_dense[p][:, q]
    (Qraw, tau), Rraw = la.qr(PAQ, mode='raw')
    Q_, R_ = la.qr(PAQ)
    # Handle case when M < N
    V_ = np.tril(Qraw, -1)[:, :M] + np.eye(M, min(M, N))
    Qr_ = csparse.apply_qright(V_, tau, p)

    # Now we get the same Householder vectors and weights
    np.testing.assert_allclose(V, V_, atol=ATOL)
    np.testing.assert_allclose(beta, tau, atol=ATOL)
    np.testing.assert_allclose(R, R_, atol=ATOL)

    # Q is the same up to row permutation
    np.testing.assert_allclose(Q, Qr_, atol=ATOL)
    np.testing.assert_allclose(Q, Q_[p_inv], atol=ATOL)
    np.testing.assert_allclose(Q[p], Q_, atol=ATOL)

    # Reproduce A = QR
    np.testing.assert_allclose(Q_ @ R_, PAQ, atol=ATOL)
    np.testing.assert_allclose(Q_[p_inv] @ R_, A_dense, atol=ATOL)
    np.testing.assert_allclose(Q @ R, A_dense, atol=ATOL)


def test_apply_q():
    """Test application of the Householder reflectors."""
    A = csparse.davis_example_qr(format='ndarray')
    N = A.shape[0]

    (Qraw, tau), Rraw = la.qr(A, mode='raw')

    # Check that the raw LAPACK output is as expected
    np.testing.assert_allclose(np.triu(Qraw), Rraw, atol=ATOL)

    # Get the Householder reflectors from the raw LAPACK output
    V_ = np.tril(Qraw, -1) + np.eye(N)

    # Apply them to the identity to get back Q itself
    Q_r = csparse.apply_qright(V_, tau)
    Q_l = csparse.apply_qtleft(V_, tau).T
    np.testing.assert_allclose(Q_r, Q_l, atol=ATOL)

    # Compare to the scipy output
    Q_, R_ = la.qr(A)
    np.testing.assert_allclose(Q_r, Q_, atol=ATOL)

    # Ensure scipy is self-consistent
    np.testing.assert_allclose(R_, Rraw, atol=ATOL)


def test_qrightleft():
    """Test the python QR decomposition algorithms."""
    A = csparse.davis_example_small(format='ndarray')

    # Test our own python QR decomposition
    V_r, beta_r, R_r = csparse.qr_right(A)
    V_l, beta_l, R_l = csparse.qr_left(A)

    Q_r = csparse.apply_qright(V_r, beta_r)
    Q_l = csparse.apply_qtleft(V_l, beta_l).T

    # Compare to each other
    np.testing.assert_allclose(V_r, V_l, atol=ATOL)
    np.testing.assert_allclose(beta_r, beta_l, atol=ATOL)
    np.testing.assert_allclose(Q_r, Q_l, atol=ATOL)
    np.testing.assert_allclose(R_r, R_l, atol=ATOL)

    # Compare to scipy
    (Qraw, tau), Rraw = la.qr(A, mode='raw')
    V = np.tril(Qraw, -1) + np.eye(Qraw.shape[0])

    np.testing.assert_allclose(V_r, V, atol=ATOL)
    np.testing.assert_allclose(beta_r, tau, atol=ATOL)

    # Compare to scipy's QR
    Q, R = la.qr(A)
    np.testing.assert_allclose(Q, Q_r, atol=ATOL)
    np.testing.assert_allclose(R, R_r, atol=ATOL)

    # Reproduce A = QR
    np.testing.assert_allclose(Q_r @ R_r, A, atol=ATOL)


# =============================================================================
# =============================================================================
