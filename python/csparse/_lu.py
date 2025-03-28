#!/usr/bin/env python3
# =============================================================================
#     File: _lu.py
#  Created: 2025-02-28 14:39
#   Author: Bernie Roesler
#
"""
Python implmentation of various LU decomposition algorithms, as presented in
Davis, Chapter 6.
"""
# =============================================================================

import numpy as np
import scipy.linalg as la


def lu_left(A):
    """Compute the LU decomposition of a matrix using a left-looking algorithm
    with partial pivoting.

    .. math::
        PA = LU

    Parameters
    ----------
    A : (M, M) array_like
        Square matrix to decompose.

    Returns
    -------
    P : (M, M) ndarray
        Row permutation matrix.
    L : (M, M) ndarray
        Lower triangular matrix. The diagonal elements are all 1.
    U : (M, M) ndarray
        Upper triangular matrix.
    """
    M, N = A.shape

    if M != N:
        raise ValueError("Input matrix must be square.")

    P = np.eye(N)
    L = np.zeros((N, N))
    U = np.zeros((N, N))

    for k in range(N):
        if k == 0:
            x = P @ A[:, k]  # no need to solve for the first column
        else:
            # Solve Eqn (6.2)
            z = np.vstack([np.zeros((k, N-k)), np.eye(N-k)])  # (N, N-k)
            Y = np.c_[L[:, :k], z]
            x = la.solve(Y, P @ A[:, k])
        U[:k, k] = x[:k]                  # the column of U
        i = np.argmax(np.abs(x[k:])) + k  # get the pivot index
        L[[i, k]] = L[[k, i]]             # swap rows
        P[[i, k]] = P[[k, i]]
        x[[i, k]] = x[[k, i]]
        U[k, k] = x[k]
        L[k, k] = 1
        L[k+1:, k] = x[k+1:] / x[k]       # divide the column by the pivot

    return P, L, U


def lu_right(A):
    """Compute the LU decomposition of a matrix using a right-looking algorithm
    without pivoting.

    .. math::
        PA = LU

    Parameters
    ----------
    A : (M, M) array_like
        Square matrix to decompose.

    Returns
    -------
    P : (M, M) ndarray
        Row permutation matrix (identity matrix).
    L : (M, M) ndarray
        Lower triangular matrix. The diagonal elements are all 1.
    U : (M, M) ndarray
        Upper triangular matrix.
    """
    A = np.copy(A)
    M, N = A.shape

    if M != N:
        raise ValueError("Input matrix must be square.")

    L = np.eye(N)
    U = np.zeros((N, N))

    for k in range(N):
        U[k, k:] = A[k, k:]
        L[k+1:, k] = A[k+1:, k] / U[k, k]
        A[k+1:, k+1:] -= L[k+1:, [k]] @ U[[k], k+1:]

    return np.eye(N), L, U


def lu_rightp(A):
    """Compute the LU decomposition of a matrix using a right-looking algorithm
    with partial pivoting.

    .. math::
        PA = LU

    Parameters
    ----------
    A : (M, M) array_like
        Square matrix to decompose.

    Returns
    -------
    P : (M, M) ndarray
        Row permutation matrix (identity matrix).
    L : (M, M) ndarray
        Lower triangular matrix. The diagonal elements are all 1.
    U : (M, M) ndarray
        Upper triangular matrix.
    """
    A = np.copy(A)
    M, N = A.shape

    if M != N:
        raise ValueError("Input matrix must be square.")

    P = np.eye(N)

    for k in range(N):
        i = np.argmax(np.abs(A[k:, k])) + k           # partial pivoting
        P[[k, i]] = P[[i, k]]                         # (6.10) swap rows
        A[[k, i]] = A[[i, k]]                         # (6.11)
        A[k+1:, k] = A[k+1:, k] / A[k, k]             # (6.12)
        A[k+1:, k+1:] -= A[k+1:, [k]] @ A[[k], k+1:]  # (6.9)

    L = np.tril(A, -1) + np.eye(N)  # L is unit diagonal
    U = np.triu(A)

    return P, L, U


# =============================================================================
# =============================================================================
