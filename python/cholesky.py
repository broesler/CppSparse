#!/usr/bin/env python3
# =============================================================================
#     File: cholesky.py
#  Created: 2025-01-28 11:10
#   Author: Bernie Roesler
#
"""
Python implementation of various Cholesky decomposition algorithms, as
presented in Davis, Chapter 4.
"""
# =============================================================================

import numpy as np

from scipy import (linalg as la,
                   sparse as sp)
# import scipy.sparse.linalg as spla


# -----------------------------------------------------------------------------
#         Define Cholesky Algorithms
# -----------------------------------------------------------------------------
def chol_up(A, lower=False):
    """Up-looking Cholesky decomposition.

    .. note:: See Davis, p 38.

    Parameters
    ----------
    A : ndarray
        Symmetric positive definite matrix to be decomposed.
    lower : bool, optional
        Whether to compute the lower triangular Cholesky factor.
        Default is False, which computes the upper triangular Cholesky
        factor.

    Returns
    -------
    R : (N, N) ndarray
        Triangular Cholesky factor of A.
    """
    N = A.shape[0]
    L = np.zeros((N, N), dtype=A.dtype)

    for k in range(N):
        L[k, :k] = la.solve(L[:k, :k], A[:k, k]).T  # solve Lx = b
        L[k, k] = np.sqrt(A[k, k] - L[k, :k] @ L[k, :k])

    return L if lower else L.T


def chol_left(A, lower=False):
    """Left-looking Cholesky decomposition.

    .. note:: See Davis, p 60.

    Parameters
    ----------
    A : (N, N) ndarray
        Symmetric positive definite matrix to be decomposed.
    lower : bool, optional
        Whether to compute the lower triangular Cholesky factor.
        Default is False, which computes the upper triangular Cholesky
        factor.

    Returns
    -------
    R : (N, N) ndarray
        Triangular Cholesky factor of A.
    """
    N = A.shape[0]
    L = np.zeros((N, N), dtype=A.dtype)

    for k in range(N):
        L[k, k] = np.sqrt(A[k, k] - L[k, :k] @ L[k, :k])
        L[k + 1:, k] = (A[k + 1:, k] - L[k + 1:, :k] @ L[k, :k].T) / L[k, k]

    return L if lower else L.T


def chol_left_amp(A, lower=False):
    """Left-looking Cholesky decomposition, "amplified" for sparse matrices.

    .. note:: See Davis, p 61.

    Parameters
    ----------
    A : (N, N) ndarray
        Symmetric positive definite matrix to be decomposed.
    lower : bool, optional
        Whether to compute the lower triangular Cholesky factor.
        Default is False, which computes the upper triangular Cholesky
        factor.

    Returns
    -------
    R : (N, N) ndarray
        Triangular Cholesky factor of A.
    """
    N = A.shape[0]
    L = np.zeros((N, N), dtype=A.dtype)
    a = np.zeros((N,), dtype=A.dtype)

    for k in range(N):
        a[k:] = A[k:, k]
        for j in np.argwhere(L[k]).flat:
            a[k:] -= L[k:, j] * L[k, j]
        L[k, k] = np.sqrt(a[k])
        L[k + 1:, k] = a[k + 1:] / L[k, k]

    return L if lower else L.T


def chol_super(A, s, lower=False):
    r"""Supernodal Cholesky decomposition.

    .. note:: See Davis, p 61.

    Parameters
    ----------
    A : (N, N) ndarray
        Symmetric positive definite matrix to be decomposed.
    s : (N,) ndarray of int
        Supernode structure. The `j`th supernode consists of `s[j]` columns of
        `L` which can be stored as a dense matrix of dimension
        :math:`|\mathcal{L}_f| \\times s_j`, where :math:`f` is the column of
        `L` represented as the leftmost column in the `j`th supernode. The
        values of `s` should satisfy `all(s > 0)` and `sum(s) == N`.
    lower : bool, optional
        Whether to compute the lower triangular Cholesky factor.
        Default is False, which computes the upper triangular Cholesky
        factor.

    Returns
    -------
    R : (N, N) ndarray
        Triangular Cholesky factor of A.
    """
    N = A.shape[0]

    assert np.sum(s) == N
    assert np.all(s > 0)

    L = np.zeros((N, N), dtype=A.dtype)
    ss = np.cumsum(np.r_[1, s], dtype=int)

    for j in range(len(s)):
        k1 = ss[j]
        k2 = ss[j + 1]
        k = slice(k1, k2)
        L[k, k] = la.cholesky(A[k, k] - L[k, :k1] @ L[k, :k1].T).T
        L[k2:, k] = (A[k2:, k] - L[k2:, :k1] @ L[k, :k1].T) / L[k, k].T

    return L if lower else L.T


if __name__ == "__main__":
    # Create the example matrix A
    N = 11
    rows = np.r_[np.arange(N),
                 [5, 6, 2, 7, 9, 10, 5, 9, 7, 10, 8, 9, 10, 9, 10, 10]]
    cols = np.r_[np.arange(N),
                 [0, 0, 1, 1, 2,  2, 3, 3, 4,  4, 5, 5,  6, 7,  7,  9]]
    lnz = rows.size
    vals = np.ones((lnz,))

    # Values for the lower triangle
    L = sp.csc_matrix((vals, (rows, cols)), shape=(N, N))

    # Get the sum of the off-diagonal elements to ensure positive definiteness
    diag_A = np.max(np.sum(L + L.T - 2 * sp.diags(L.diagonal()), 0))

    # Create the symmetric matrix A
    A = L + sp.triu(L.T, 1) + diag_A * sp.eye(N)

    A = A.toarray()

    # NOTE Scipy Cholesky is only implemented for dense matrices!
    R = la.cholesky(A, lower=True)
    R_up = chol_up(A, lower=True)
    R_left = chol_left(A, lower=True)
    R_left_amp = chol_left_amp(A, lower=True)
    R_super = chol_super(A, np.ones(A.shape[0], dtype=int), lower=True)

    # NOTE etree is not implemented in scipy!
    # Get the elimination tree
    # [parent, post] = etree(A)
    # Rp = la.cholesky(A(post, post), lower=True)
    # assert (R.nnz == Rp.nnz)  # post-ordering does not change nnz

    # Compute the row counts of the post-ordered Cholesky factor
    row_counts = np.sum(R != 0, 1)
    col_counts = np.sum(R != 0, 0)

    # print("A = \n", A)
    # print("L = \n", R)

    # Check that algorithms work
    for L in [R, R_up, R_left, R_left_amp, R_super]:
        np.testing.assert_allclose(L @ L.T, A, atol=1e-15)

# =============================================================================
# =============================================================================
