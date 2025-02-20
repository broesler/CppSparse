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

atol = 1e-14


# ---------- Matrix from Davis Figure 5.1, p 74.
N = 8
rows = np.r_[0, 1, 2, 3, 4, 5, 6, 7,
             3, 6, 1, 6, 0, 2, 5, 7, 4, 7, 0, 1, 3, 7, 5, 6]
cols = np.r_[0, 1, 2, 3, 4, 5, 6, 7,
             0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7]
vals = np.r_[np.arange(1, N), 0, np.ones(rows.size - N)]
A_davis8 = sparse.csc_array((vals, (rows, cols)), shape=(N, N))

# ---------- Davis 4x4 example
A_davis4 = csparse.to_scipy_sparse(csparse.davis_example())

# Group the matrices for parameterization
TEST_MATRICES = [
    ("Identity", sparse.eye_array(7).tocsc()),
    ("Diagonal", sparse.diags(np.arange(1, N)).tocsc()),
    ("Asymmetric Banded", sparse.diags([np.ones(N-1), np.arange(1, N+1)], [-1, 0]).tocsc()),
    ("Laplacian (Symmetric Banded)", sparse.diags([1, -2, 1], [-1, 0, 1]).tocsc()),
    ("Random", sparse.random(N, N, density=0.5, format='csc', random_state=56)),
    ("Davis 8x8", A_davis8),
    ("Davis 4x4", A_davis4),
]


@pytest.mark.parametrize("case_name, A", TEST_MATRICES)
def test_qr_decomposition(case_name, A):
    """Test QR decomposition with various matrices using parametrization."""
    Ac = csparse.from_scipy_sparse(A, format='csc')
    A_dense = A.toarray()
    N = A.shape[0]

    # scipy QR
    (Qraw, tau), Rraw = la.qr(A_dense, mode='raw')
    Q_, R_ = la.qr(A_dense)
    V_ = np.tril(Qraw, -1) + np.eye(N)

    # Compute csparse QR
    S = csparse.sqr(Ac)
    QRres = csparse.qr(Ac, S)
    V, beta, R = QRres.V, QRres.beta, QRres.R

    # Convert to numpy arrays
    V = V.toarray()
    beta = np.r_[beta]
    R = R.toarray()

    p = csparse.inv_permute(S.p_inv)
    Q = csparse.qright(V, beta, p)
    Ql = csparse.qleft(V, beta, p)

    np.testing.assert_allclose(Q, Ql.T, atol=atol)

    # Compare Householder reflectors (only if no row permutation)
    if np.all(p == np.arange(N)):
        np.testing.assert_allclose(V, V_, atol=atol)
        np.testing.assert_allclose(beta, tau, atol=atol)

    # Compare Q and R except for signs
    np.testing.assert_allclose(np.abs(Q), np.abs(Q_), atol=atol)
    np.testing.assert_allclose(np.abs(R), np.abs(R_), atol=atol)

    # Reproduce A = QR
    np.testing.assert_allclose(Q @ R, A_dense, atol=atol)


def test_qapply():
    """Test application of the Householder reflectors."""
    A_dense = A_davis8.toarray()

    (Qraw, tau), Rraw = la.qr(A_dense, mode='raw')
    Q_, R_ = la.qr(A_dense)
    V_ = np.tril(Qraw, -1) + np.eye(N)

    Qr_ = csparse.qright(V_, tau)
    Ql_ = csparse.qleft(V_, tau).T

    np.testing.assert_allclose(Qr_, Ql_, atol=atol)
    np.testing.assert_allclose(Qr_, Q_, atol=atol)
    np.testing.assert_allclose(np.triu(Qraw), Rraw, atol=atol)


# =============================================================================
# =============================================================================
