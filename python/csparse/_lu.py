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
from scipy.linalg import norm, solve
from scipy.sparse import linalg as spla

from csparse import inv_permute


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
            x = solve(Y, P @ A[:, k])
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


def lu_rightprv(A):
    """Compute the LU decomposition of a matrix using a recursive,
    right-looking algorithm with pivoting.

    .. math::
        PA = LU

    This function is identical to `lu_rightpr`, but uses a permutation vector
    instead of a permutation matrix.

    See: Davis, Exercise 6.12.

    Parameters
    ----------
    A : (M, M) array_like
        Square matrix to decompose.

    Returns
    -------
    p : (M,) ndarray
        Row permutation vector.
    L : (M, M) ndarray
        Lower triangular matrix. The diagonal elements are all 1.
    U : (M, M) ndarray
        Upper triangular matrix.
    """
    A = np.copy(A)
    N = A.shape[0]
    if N == 1:
        p = np.array([0])
        L = np.array([[1]])  # 2D array
        U = A.copy()
    else:
        i = np.argmax(np.abs(A[:, 0]))  # partial pivoting
        p1 = np.arange(N)
        p1[0], p1[i] = p1[i], p1[0]  # swap indices
        A = A[p1]
        u11 = A[0, 0]                                      # (6.10)
        u12 = A[[0], 1:]                                   # (6.11)
        l21 = A[1:, [0]] / u11                             # (6.12)
        p2, L22, U22 = lu_rightprv(A[1:, 1:] - l21 @ u12)  # (6.9) or (6.13)
        o = np.zeros((1, N-1))
        p = np.r_[0, p2 + 1][p1]
        L = np.block([[1, o], [l21[inv_permute(p2)], L22]])  # (6.13)
        U = np.block([[u11, u12], [o.T, U22]])

    return p, L, U


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


def cond1est(A):
    """Estimate the 1-norm condition number of a real, square matrix.

    See: Davis, Exercise 6.15.

    Parameters
    ----------
    A : (N, N) array_like
        Matrix of N vectors in N dimensions.

    Returns
    -------
    result : float
        An estimate of the 1-norm condition number of the matrix.
    """
    M, N = A.shape

    if M != N:
        raise ValueError("Input matrix must be square.")

    if M == 0:
        raise ValueError("Condition number of an empty matrix is undefined.")

    # Compute the LU decomposition of A
    try:
        lu = spla.splu(A)
    except RuntimeError as e:
        if "Factor is exactly singular" in str(e):
            return np.inf
        else:
            raise

    if np.any(lu.U.diagonal() == 0):
        return np.inf
    else:
        return spla.norm(A, 1) * norm1est_inv(lu)


def _norm1est_inv_lu(lu):
    """Estimate the 1-norm of the inverse of a sparse matrix, given its LU
    decomposition.

    See: Davis, Exercise 6.15.

    Parameters
    ----------
    lu : SuperLU object
        The result of the LU decomposition of a sparse matrix, as computed by
        `scipy.sparse.linalg.splu`.

    Returns
    -------
    result : float
        An estimate of the 1-norm of the matrix.
    """
    N = lu.L.shape[0]

    for k in range(5):
        if k == 0:
            est = 0
            x = np.ones(N) / N
            jold = -1
        else:
            # j = np.min(np.argwhere(np.abs(x) == norm(x, np.inf)))
            j = np.min(np.argmax(np.abs(x)))

            if j == jold:
                break

            x = np.zeros(N)
            x[j] = 1
            jold = j

        x = lu.solve(x)
        est_old = est
        est = norm(x, 1)

        if k > 0 and est <= est_old:
            break

        s = np.ones(N)
        s[x < 0] = -1
        x = lu.solve(s, trans='T')

    return est


def norm1est_inv(A):
    """Estimate the 1-norm of the inverse of a sparse matrix.

    See: Davis, Exercise 6.15.

    Parameters
    ----------
    A : (N, N) array_like or SuperLU
        Matrix of N vectors in N dimensions, or the result of the LU
        decomposition as computed by `scipy.sparse.linalg.splu`.

    Returns
    -------
    result : float
        An estimate of the 1-norm of the inverse of the matrix.
    """
    if isinstance(A, spla.SuperLU):
        lu = A
    else:
        lu = spla.splu(A)
    return _norm1est_inv_lu(lu)


def scipy_cond1est(A):
    """Estimate the 1-norm condition number of a real, square matrix.

    This function uses only scipy functions to compute the condition number, as
    opposed to `cond1est` that uses our own algorithm.

    See: Davis, Exercise 6.15.

    Parameters
    ----------
    A : (N, N) array_like
        Matrix of N vectors in N dimensions.

    Returns
    -------
    result : float
        An estimate of the 1-norm condition number of the matrix.
    """
    M, N = A.shape

    if M != N:
        raise ValueError("Input matrix must be square.")

    if A.size == 0:
        return 0.0

    # Compute ||A||_1 exactly
    norm_A = spla.norm(A, 1)

    # Compute the LU decomposition of A
    try:
        lu = spla.splu(A)
    except RuntimeError as e:
        if "Factor is exactly singular" in str(e):
            return np.inf
        else:
            raise

    # Estimate ||A^-1||_1 using the LU decomposition
    A_inv_op = spla.LinearOperator(
        shape=A.shape,
        matvec=lu.solve,                           # x = A \ b
        rmatvec=lambda b: lu.solve(b, trans='T'),  # x = A.T \ b
        dtype=A.dtype
    )

    # Estimate the 1-norm of the inverse
    norm_A_inv = spla.onenormest(A_inv_op)

    return norm_A * norm_A_inv


def lu_crout(A):
    """Compute the LU decomposition of a matrix using Crout's algorithm
    with partial pivoting.

    This method computes the `k`th column of `L` and `k`th row of `U` at each
    step of the factorization.

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
    A = A.copy()
    M, N = A.shape

    if M != N:
        raise ValueError("Input matrix must be square.")

    P = np.eye(N)  # no pivoting (yet)

    for k in range(N):
        # Find the pivot element
        i = np.argmax(np.abs(A[k:, k])) + k  # partial pivoting

        # Swap rows in P and A
        if i != k:
            P[[k, i]] = P[[i, k]]
            A[[k, i]] = A[[i, k]]

        # update row of U
        A[k, k:] = A[k, k:] - A[k, :k] @ A[:k, k:]

        # update column of L (unit diagonal not stored)
        A[k+1:, k] = (A[k+1:, k] - A[k+1:, :k] @ A[:k, k]) / A[k, k]

    # Get the individual factors
    L = np.tril(A, -1) + np.eye(N)  # L is unit diagonal
    U = np.triu(A)

    return P, L, U


# =============================================================================
# =============================================================================
