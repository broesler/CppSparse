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
            z = np.vstack([np.zeros((k, N-k)), np.eye(N-k)])  # (N, N-k)
            Y = np.c_[L[:, :k], z]
            x = la.solve(Y, P @ A[:, k])
        U[:k, k] = x[:k]  # the column of U
        i = np.argmax(np.abs(x[k:])) + k  # get the pivot index
        L[[i, k]] = L[[k, i]]  # swap rows
        P[[i, k]] = P[[k, i]]
        x[[i, k]] = x[[k, i]]
        U[k, k] = x[k]
        L[k, k] = 1
        L[k+1:, k] = x[k+1:] / x[k]  # divide the column by the pivot

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



if __name__ == "__main__":
    import utils
    A = utils.davis_example_qr(format='ndarray')

    # Compute using scipy (computes A = PLU -> LU = P^T A)
    P_, L_, U_ = la.lu(A)

    atol = 1e-15

    for lu_func in [lu_left, lu_right]:
        # Computes LU = PA
        P, L, U = lu_func(A)

        # Check the results
        np.testing.assert_allclose(P, P_.T, atol=atol)
        np.testing.assert_allclose(L, L_, atol=atol)
        np.testing.assert_allclose(U, U_, atol=atol)
        np.testing.assert_allclose(L @ U, P @ A, atol=atol)


# =============================================================================
# =============================================================================
