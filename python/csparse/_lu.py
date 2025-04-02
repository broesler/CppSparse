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


def lu_rightr(A):
    """Compute the LU decomposition of a matrix using a recursive,
    right-looking algorithm without pivoting.

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
    N = A.shape[0]
    if N == 1:
        L = np.array([[1]])  # 2D array
        U = A.copy()
    else:
        u11 = A[0, 0]                                   # (6.4)
        u12 = A[[0], 1:]                                # (6.5)
        l21 = A[1:, [0]] / u11                          # (6.6)
        _, L22, U22 = lu_rightr(A[1:, 1:] - l21 @ u12)  # (6.7)
        L = np.block([[1, np.zeros((1, N-1))],
                      [l21, L22]])
        U = np.block([[u11, u12],
                      [np.zeros((N-1, 1)), U22]])

    return np.eye(N), L, U



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


def lu_rightpr(A):
    """Compute the LU decomposition of a matrix using a recursive,
    right-looking algorithm with pivoting.

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
    A = np.copy(A)
    N = A.shape[0]
    if N == 1:
        P = np.eye(1)
        L = np.array([[1]])  # 2D array
        U = A.copy()
    else:
        i = np.argmax(np.abs(A[:, 0]))  # partial pivoting
        P1 = np.eye(N)
        P1[[0, i]] = P1[[i, 0]]  # swap rows
        A = P1 @ A
        u11 = A[0, 0]                                     # (6.10)
        u12 = A[[0], 1:]                                  # (6.11)
        l21 = A[1:, [0]] / u11                            # (6.12)
        P2, L22, U22 = lu_rightpr(A[1:, 1:] - l21 @ u12)  # (6.9) or (6.13)
        o = np.zeros((1, N-1))
        P = np.block([[1, o], [o.T, P2]]) @ P1
        L = np.block([[1, o], [P2 @ l21, L22]])           # (6.13)
        U = np.block([[u11, u12], [o.T, U22]])

    return P, L, U



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
