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

from csparse import davis_example, to_ndarray, qright, qleft

tol = 1e-14


# -----------------------------------------------------------------------------
#         Define QR Algorithms
# -----------------------------------------------------------------------------
def qr_right(A):
    """Compute the QR decomposition of A using the right-looking algorithm.

    From the LAPACK documentation for `dgeqrf.f`:

    The matrix Q is represented as a product of elementary reflectors

       Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

       H(i) = I - tau * v * v**T

    where tau is a real scalar, and v is a real vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
    and tau in TAU(i).

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
        (Qraw, b), _ = la.qr(R[k:, [k]], mode='raw')
        v = np.vstack([1.0, Qraw[1:]])  # extract the reflector
        b = float(b.squeeze())          # get the scalar
        V[k:, [k]] = v
        beta[k] = b
        R[k:, k:] -= v @ (b * (v.T @ R[k:, k:]))

    return V, beta, R


def qr_left(A):
    """Compute the QR decomposition of A using the left-looking algorithm.

    From the LAPACK documentation for `dgeqrf.f`:

    The matrix Q is represented as a product of elementary reflectors

       Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

       H(i) = I - tau * v * v**T

    where tau is a real scalar, and v is a real vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
    and tau in TAU(i).

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
    R = np.zeros((M, N))
    beta = np.zeros(N)

    for k in range(N):
        x = A[:, [k]]

        for i in range(k):
            v = V[i:, [i]]
            b = beta[i]
            x[i:] -= v @ (b * (v.T @ x[i:]))

        (Qraw, b), _ = la.qr(x[k:], mode='raw')
        V[k:, [k]] = np.vstack([1.0, Qraw[1:]])  # extract the reflector
        beta[k] = float(b[0])                    # get the scalar
        R[:k, [k]] = x[:k]
        R[k, k] = Qraw[0, 0]  # == Rraw[0, 0] == s = -sign(x[0]) * norm(x)

    return V, beta, R



def build_H(v, tau=None):
    """Constructs the Householder matrix from its defining vector.

    Parameters
    ----------
    v : (M,) array_like
        A vector defining the Householder reflector.
    tau : float, optional
        The scaling factor for the reflector. If not provided, it is computed
        as `2 / (v.T @ v)`.

    Returns
    -------
    H : (M, N) ndarray
        The Householder matrix.
    """
    v = np.asarray(v, dtype=float)
    M = v.size
    v = v.reshape(M, 1)  # make it a column vector
    if tau is None:
        tau = 2 / (v.T @ v)
    return np.eye(M) - tau * (v @ v.T)


# -----------------------------------------------------------------------------
#         Main Script
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # See: Strang Linear Algebra p 203.
    # A = np.array([[1, 1, 2],
    #               [0, 0, 1],
    #               [1, 0, 0]],
    #              dtype=float)

    A = davis_example().toarray()

    # Get the raw LAPACK output for the entire matrix
    # Qraw stores the Householder reflectors *below* the diagonal
    (Qraw, tau), Rraw = la.qr(A, mode='raw')
    V = np.tril(Qraw, -1) + np.eye(Qraw.shape[0])
    Hs = [build_H(v, t) for v, t in zip(V.T, tau)]
    Qr = qright(V, tau)
    np.testing.assert_allclose(Qr @ Rraw, A, atol=tol)

    # Test our own python QR decomposition
    V_r, beta_r, R_r = qr_right(A)
    V_l, beta_l, R_l = qr_left(A)

    # NOTE that V is scaled to have 1 on the diagonal, whereas the MATLAB
    # output from `qr_right.m` is not.
    # The MATLAB output is just scaled by v(1):
    #   v := v / v(1)
    #   beta_M := 2 / (v' * v) == tau / (v(1)**2)

    print("V = ")
    print(V_r)
    print("beta = ")
    print(beta_r)
    print("R = ")
    print(R_r)

    Q_r = qright(V_r, beta_r)
    Q_l = qleft(V_l, beta_l).T

    # Reproduce A = QR
    np.testing.assert_allclose(Q_r @ R_r, A, atol=tol)
    np.testing.assert_allclose(Q_l @ R_l, A, atol=tol)

    # Compare to scipy's QR
    Q, R = la.qr(A)
    np.testing.assert_allclose(Q, Q_r, atol=tol)
    np.testing.assert_allclose(R, R_r, atol=tol)
    np.testing.assert_allclose(Q, Q_l, atol=tol)
    np.testing.assert_allclose(R, R_l, atol=tol)

    # Get the raw LAPACK output for a test vector
    # x = np.c_[[3, 4]]
    # (Qraw, tau), _ = la.qr(x, mode='raw')
    # v = np.vstack([1.0, Qraw[1:]])
    # H = np.eye(x.size) - tau * (v @ v.T)
    # Hx = H @ x
    # print("v = ")
    # print(v)
    # print("beta = ")
    # print(tau)
    # print("Hx = ")
    # print(Hx)
    # # NOTE sign of Hx[0] is negative when x[0] is positive
    # np.testing.assert_allclose(Hx[0], -la.norm(x))

# =============================================================================
# =============================================================================
