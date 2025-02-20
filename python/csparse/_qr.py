#!/usr/bin/env python3
# =============================================================================
#     File: _qr.py
#  Created: 2025-02-11 15:18
#   Author: Bernie Roesler
#
"""
Python implementation of various QR decomposition algorithms, as presented in
Davis, Chapter 5.
"""
# =============================================================================

import numpy as np

from scipy import sparse
from scipy import linalg as la

from .csparse import CSCMatrix
from .utils import to_scipy_sparse


def apply_qright(V, beta, p=None, Y=None):
    r"""Apply Householder vectors on the right.

    Computes :math:`X = Y P^T H_1 \dots H_N = Y Q`, where :math:`Q` is
    represented by the Householder vectors stored in `V`, coefficients `beta`,
    and permutation `p`. To obtain :math:`Q` itself, pass `Y = sparse.eye(M)`.

    Parameters
    ----------
    Y : (M, N) ndarray or sparse array
        The matrix to which the Householder transformations are applied.
    V : (M, N) CSCMatrix
        The matrix of Householder vectors.
    beta : (N,) ndarray
        The Householder coefficients.
    p : (N,) ndarray, optional
        The column permutation vector.
    Y : (M, N) ndarray or sparse array, optional
        The matrix to which the Householder transformations are applied. If not
        given, the identity matrix is used, resulting in the full `Q` matrix.

    Returns
    -------
    result : (M, N) ndarray
        The result of applying the Householder transformations to `Y`.
    """
    if isinstance(V, CSCMatrix):
        V = to_scipy_sparse(V)

    if Y is None:
        Y = sparse.eye_array(V.shape[1]).tocsc()

    M, N = V.shape
    X = Y.copy()
    if p is not None:
        X = X[:, p]
    for j in range(N):
        X -= X @ (beta[j] * V[:, [j]]) @ V[:, [j]].T
    return X


def apply_qleft(V, beta, p=None, Y=None):
    r"""Apply Householder vectors on the left.

    Computes :math:`X = H_N \dots H_1 P Y = Q^T Y`, where :math:`Q` is
    represented by the Householder vectors stored in `V`, coefficients `beta`,
    and permutation `p`. To obtain :math:`Q^T` itself, pass `Y = sparse.eye(M)`.

    Parameters
    ----------
    Y : (M2, N) ndarray or sparse array
        The matrix to which the Householder transformations are applied.
    V : (M, NY) CSCMatrix
        The matrix of Householder vectors.
    beta : (N,) ndarray
        The Householder coefficients.
    p : (N,) ndarray, optional
        The row permutation vector.
    Y : (M, N) ndarray or sparse array, optional
        The matrix to which the Householder transformations are applied. If not
        given, the identity matrix is used, resulting in the full `Q` matrix.

    Returns
    -------
    result : (M, N) ndarray
        The result of applying the Householder transformations to `Y`.
    """
    if isinstance(V, CSCMatrix):
        V = to_scipy_sparse(V)

    if Y is None:
        Y = sparse.eye_array(V.shape[0]).tocsc()

    M2, N = V.shape
    M, NY = Y.shape
    X = Y.copy()

    if (M2 > M):
        # Add empty rows to the bottom of X
        if sparse.issparse(X):
            X = sparse.vstack([X, sparse.csc_array((M2 - M, NY))])
        else:
            X = np.vstack([X, np.zeros((M2 - M, NY))])

    if p is not None:
        X = X[p, :]

    for j in range(N):
        X -= V[:, [j]] @ (beta[j] * V[:, [j]].T @ X)

    return X


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

        # Compute the Householder reflector
        x_k = x[k:]
        (Qraw, b), _ = la.qr(x_k, mode='raw')
        V[k:, [k]] = np.vstack([1.0, Qraw[1:]])  # extract the reflector
        beta[k] = float(b[0])                    # get the scalar
        R[:k, [k]] = x[:k]
        # NOTE If beta == 0, H is the identity matrix, so Hx == x:
        # if beta[k] == 0:
        #     R[k, k] = x_k[0]
        # else:
        #     R[k, k] = -np.sign(x_k[0]) * la.norm(x_k)
        #
        # Qraw computes Hx internally to give the correct result.
        R[k, k] = Qraw[0, 0]

    return V, beta, R


# =============================================================================
# =============================================================================
