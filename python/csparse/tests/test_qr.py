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
    ("Davis 5x8 (M < N)", csparse.davis_example_qr(format='csc')[:5, :])
]


def categorize_shape(M, N):
    if M < N:
        return "under"
    elif M == N:
        return "square"
    else:
        return "over"


def generate_test_matrices():
    """Generate all matrices for testing."""
    # Fixed test matrices
    for case_name, A in TEST_MATRICES:
        M, N = A.shape
        shape_cat = categorize_shape(M, N)
        test_id = f"{shape_cat}::{case_name}"
        yield pytest.param(shape_cat, case_name, A, id=test_id)

    # Random test matrices
    seed = 565656
    rng = np.random.default_rng(seed)
    Ns = np.r_[2, 7, 10]
    Ms = (Ns * np.r_[0.6, 1, 1.4]).astype(int)
    N_runs = 10

    for N in Ns:
        for i in range(N_runs):
            for M in Ms:
                A = sparse.random(M, N,
                                  density=0.5, format='csc', random_state=rng)
                # ensure structural full rank
                A.setdiag(N * np.arange(1, min(M, N) + 1))

                shape_cat = categorize_shape(M, N)
                case_name = f"Random {M}x{N} ({seed=}, {i=})"
                test_id = f"{shape_cat}::{case_name}"
                yield pytest.param(shape_cat, case_name, A,
                                   id=test_id,
                                   marks=pytest.mark.random)


@pytest.mark.parametrize("order", ['Natural', 'ATA'])
@pytest.mark.parametrize("shape_cat, case_name, A", generate_test_matrices())
def test_csparse_qr(shape_cat, case_name, A, order):
    """Test QR decomposition with various matrices using parametrization."""
    Ac = csparse.from_scipy_sparse(A, format='csc')
    A_dense = A.toarray()
    M, N = A.shape

    # FIXME
    if M < N:
        pytest.skip(f"Skipping {case_name} ({order}) because M < N")

    # ---------- Compute csparse QR
    QRres = csparse.qr(Ac, order)
    V, beta, R, p_inv, q = QRres.V, QRres.beta, QRres.R, QRres.p_inv, QRres.q

    if order == 'Natural':
        np.testing.assert_allclose(q, np.arange(N))

    # Convert to numpy arrays
    V = V.toarray()
    R = R.toarray()

    p = csparse.inv_permute(p_inv)
    Q = csparse.apply_qright(V, beta, p)
    Ql = csparse.apply_qtleft(V, beta, p).T

    # Test the apply functions both get the same Q
    np.testing.assert_allclose(Q, Ql, atol=ATOL)

    # ---------- scipy QR
    # Apply the row permutation to A_dense
    Apq = A_dense[p][:, q]
    (Qraw, tau), Rraw = la.qr(Apq, mode='raw')
    Q_, R_ = la.qr(Apq)
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
    np.testing.assert_allclose(Q_ @ R_, Apq, atol=ATOL)
    np.testing.assert_allclose(Q_[p_inv] @ R_, A_dense[:, q], atol=ATOL)
    np.testing.assert_allclose(Q @ R, A_dense[:, q], atol=ATOL)


@pytest.mark.parametrize("shape_cat, case_name, A", generate_test_matrices())
def test_apply_q(shape_cat, case_name, A):
    """Test application of the Householder reflectors."""
    A = A.toarray()  # only test with dense matrices
    M, N = A.shape

    if M > N:
        pytest.skip(f"Skipping {case_name} because M > N")

    (Qraw, tau), Rraw = la.qr(A, mode='raw')

    # Check that the raw LAPACK output is as expected
    np.testing.assert_allclose(np.triu(Qraw), Rraw, atol=ATOL)

    # Get the Householder reflectors from the raw LAPACK output
    V_ = np.tril(Qraw, -1)[:, :M] + np.eye(M, min(M, N))

    # Apply them to the identity to get back Q itself
    Q_r = csparse.apply_qright(V_, tau)
    Q_l = csparse.apply_qtleft(V_, tau).T
    np.testing.assert_allclose(Q_r, Q_l, atol=ATOL)

    # Compare to the scipy output
    Q_, R_ = la.qr(A)
    np.testing.assert_allclose(Q_r, Q_, atol=ATOL)

    # Ensure scipy is self-consistent
    np.testing.assert_allclose(R_, Rraw, atol=ATOL)


@pytest.mark.parametrize("shape_cat, case_name, A", generate_test_matrices())
@pytest.mark.parametrize("qr_func", [csparse.qr_right, csparse.qr_left])
def test_qrightleft(shape_cat, case_name, A, qr_func):
    """Test the python QR decomposition algorithms."""
    A = A.toarray()  # only test with dense matrices
    M, N = A.shape

    if qr_func == csparse.qr_right and M < N:
        pytest.skip(f"Skipping {case_name} ({qr_func.__name__}) because M < N")

    # Test our own python QR decomposition
    V, beta, R = qr_func(A)
    Q = csparse.apply_qright(V, beta)

    # Compare to scipy
    (Qraw, tau), Rraw = la.qr(A, mode='raw')
    V = np.tril(Qraw, -1)[:, :M] + np.eye(M, min(M, N))

    np.testing.assert_allclose(V, V, atol=ATOL)
    np.testing.assert_allclose(beta, tau, atol=ATOL)

    # Compare to scipy's QR
    Q_, R_ = la.qr(A)
    np.testing.assert_allclose(Q_, Q, atol=ATOL)
    np.testing.assert_allclose(R_, R, atol=ATOL)

    # Reproduce A = QR
    np.testing.assert_allclose(Q @ R, A, atol=ATOL)


# =============================================================================
# =============================================================================
