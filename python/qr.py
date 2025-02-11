#!/usr/bin/env python3
# =============================================================================
#     File: qr.py
#  Created: 2025-02-11 15:18
#   Author: Bernie Roesler
#
"""
Python implementation of various QR decomposition algorithms, as presented in
Davis, Chapter 5.
"""
# =============================================================================

import numpy as np

from scipy import linalg as la
# from scipy import sparse

tol = 1e-14

# -----------------------------------------------------------------------------
#         Define QR Algorithms
# -----------------------------------------------------------------------------
def qr_right(A):
    """Compute the QR decomposition of A using the right-looking algorithm.

    Parameters
    ----------
    A : (M, N) ndarray
        Matrix of M vectors in N dimensions

    Returns
    -------
    V : (M, N) ndarray
        A matrix with the Householder reflectors as columns.
    beta : (N,) ndarray
        A vector with the scaling factors for each reflector.
    R : (M, N) ndarray
        The upper triangular matrix.
    """
    M, N = A.shape
    V = np.zeros((M, N))
    R = np.copy(A)
    beta = np.zeros(N)

    for k in range(N):
        # Compute the Householder reflector
        (v, b), _ = la.qr(R[k:, [k]], mode='raw')
        b = b[0]  # singleton
        V[k:, [k]] = v  # column vector
        beta[k] = b
        R[k:, k:] -= v @ (b * (v.T @ R[k:, k:]))

    return V, beta, R


def extract_householder_reflectors(Q):
    """Extract the Householder reflectors from the compact representation given
    by `scipy.linalg.qr(..., mode='raw')`.

    Parameters
    ----------
    Q : (M, N) ndarray
        Matrix of M vectors in N dimensions

    Returns
    -------
    V : (M, N) ndarray
        A matrix with the Householder reflectors as columns.
    """
    M, N = Q.shape
    reflectors = []

    for j in range(min(M, N)):
        v = np.zeros(M)       # initialize the vector for the reflector
        v[j] = 1.0            # diagonal element is 1
        v[j+1:] = Q[j+1:, j]  # elements below the diagonal are taken from Q

        reflectors.append(v)

    return reflectors


def construct_householder_matrix(v, tau=None):
    """Constructs the Householder matrix from its defining vector."""
    v = np.asarray(v, dtype=float)
    M = v.size
    v = v.reshape(M, 1)  # make it a column vector
    if tau is None:
        tau = 2 / (v.T @ v)
    return np.eye(M) - tau * (v @ v.T)


if __name__ == '__main__':
    # See: Strang Linear Algebra p 203.
    A = np.array([[1, 1, 2],
                  [0, 0, 1],
                  [1, 0, 0]],
                 dtype=float)

    # Get the raw LAPACK output
    # Qc stores the Householder reflectors *below* the diagonal*
    (Qc, tau), R_ = la.qr(A, mode='raw')

    reflectors = extract_householder_reflectors(Qc)

    Hs = [construct_householder_matrix(v, t)
          for v, t in zip(reflectors, tau)]

    Qr = np.eye(A.shape[0])
    for v, t in zip(reflectors, tau):
        Qr @= construct_householder_matrix(v, t)

    np.testing.assert_allclose(Qr @ R_, A, atol=tol)

    V_r, beta_r, R_r = qr_right(A)
    print(V_r)
    print(beta_r)
    print(R_r)

    # Compare to scipy's QR
    Q, R = la.qr(A)

    # np.testing.assert_allclose(R_r, R)

# =============================================================================
# =============================================================================
