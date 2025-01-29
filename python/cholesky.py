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
                   sparse as sparse)
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
        L[k, k] = np.sqrt(A[k, k] - L[k, :k] @ L[k, :k].T)
        L[k+1:, k] = (A[k+1:, k] - L[k+1:, :k] @ L[k, :k].T) / L[k, k]

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
        L[k+1:, k] = a[k+1:] / L[k, k]

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
    ss = np.cumsum(np.r_[0, s], dtype=int)

    for j in range(len(s)):
        k1 = ss[j]
        k2 = ss[j + 1]
        k = slice(k1, k2)
        L[k, k] = la.cholesky(A[k, k] - L[k, :k1] @ L[k, :k1].T).T
        L[k2:, k] = (A[k2:, k] - L[k2:, :k1] @ L[k, :k1].T) / L[k, k].T

    return L if lower else L.T


def chol_right(A, lower=False):
    """Right-looking Cholesky decomposition.

    .. note:: See Davis, p 62.

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
    A = np.ascontiguousarray(A).copy()
    N = A.shape[0]
    L = np.zeros((N, N), dtype=A.dtype)

    for k in range(N):
        L[k, k] = np.sqrt(A[k, k])
        L[k+1:, k] = A[k+1:, k] / L[k, k]
        A[k+1:, k+1:] -= L[k+1:, [k]] @ L[k+1:, [k]].T

    return L if lower else L.T


def chol_update(L, w):
    """Update the Cholesky factor L of a matrix `A = L @ L.T + w @ w.T`.

    Parameters
    ----------
    L : (N, N) ndarray
        The Cholesky factor of a matrix A, assumed to be lower triangular.
    w : (N,) ndarray
        The update vector.

    Returns
    -------
    L : (N, N) ndarray
        The updated Cholesky factor of A + w @ w.T.
    w : (N,) ndarray
        The updated vector, which is the solution to `Lx = w`.
    """
    L = np.ascontiguousarray(L).copy()
    w = np.ascontiguousarray(w).copy()
    β = 1  # scaling factor
    N = L.shape[0]

    for j in range(N):
        α = w[j] / L[j, j]
        β2 = np.sqrt(β**2 + α**2)
        γ = α / (β2 * β)
        δ = β / β2
        L[j, j] = δ * L[j, j] + γ * w[j]
        w[j] = α
        β = β2
        if (j == N-1):
            break
        w1 = w[j+1:]  # store vector before updating w
        w[j+1:] -= α * L[j+1:, j]
        L[j+1:, j] = δ * L[j+1:, j] + γ * w1

    return L, w


def chol_downdate(L, w):
    """Downdate the Cholesky factor L of a matrix `A = L @ L.T - w @ w.T`.

    Parameters
    ----------
    L : (N, N) ndarray
        The Cholesky factor of a matrix A, assumed to be lower triangular.
    w : (N,) ndarray
        The update vector.

    Returns
    -------
    L : (N, N) ndarray
        The updated Cholesky factor of A + w @ w.T.
    w : (N,) ndarray
        The updated vector, which is the solution to `Lx = w`.
    """
    L = np.ascontiguousarray(L).copy()
    w = np.ascontiguousarray(w).copy()
    β = 1  # scaling factor
    N = L.shape[0]

    for j in range(N):
        α = w[j] / L[j, j]
        if α**2 >= β**2:
            raise ValueError("L is not positive definite.")
        β2 = np.sqrt(β**2 - α**2)
        γ = α / (β2 * β)
        δ = β / β2
        L[j, j] = δ * L[j, j]
        w[j] = α
        β = β2
        if (j == N-1):
            break
        w[j+1:] -= α * L[j+1:, j]
        L[j+1:, j] = δ * L[j+1:, j] - γ * w[j+1:]

    return L, w


def chol_updown(L, w, update=True):
    """Up/downdate the Cholesky factor L of a matrix `A = L @ L.T ± w @ w.T`.

    Parameters
    ----------
    L : (N, N) ndarray
        The Cholesky factor of a matrix A, assumed to be lower triangular.
    w : (N,) ndarray
        The update vector.
    update : bool, optional
        Whether to update (True) or downdate (False) the Cholesky factor.

    Returns
    -------
    L : (N, N) ndarray
        The updated Cholesky factor of A + w @ w.T.
    w : (N,) ndarray
        The updated vector, which is the solution to `Lx = w`.
    """
    L = np.ascontiguousarray(L).copy()
    w = np.ascontiguousarray(w).copy()

    beta = 1
    N = L.shape[0]
    sigma = 1 if update else -1

    if N == 1:
        L = np.sqrt(L*L.T + sigma*w*w.T)
        return L, w

    for k in range(N):
        alpha = w[k] / L[k, k]
        beta2 = np.sqrt(beta**2 + sigma*alpha**2)
        gamma = sigma * alpha / (beta2 * beta)

        if update:
            delta = beta / beta2
            L[k, k] = delta * L[k, k] + gamma * w[k]
            w1 = w[k+1:]
            w[k+1:] -= alpha * L[k+1:, k]
            L[k+1:, k] = delta * L[k+1:, k] + gamma * w1
        else:  # downdate
            delta = beta2 / beta
            L[k, k] = delta * L[k, k]
            w[k+1:] -= alpha * L[k+1:, k]
            L[k+1:, k] = delta * L[k+1:, k] + gamma * w[k+1:]

        w[k] = alpha
        beta = beta2

    return L, w


# -----------------------------------------------------------------------------
#        Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Create the example matrix A
    N = 11
    rows = np.r_[np.arange(N),
                 [5, 6, 2, 7, 9, 10, 5, 9, 7, 10, 8, 9, 10, 9, 10, 10]]
    cols = np.r_[np.arange(N),
                 [0, 0, 1, 1, 2,  2, 3, 3, 4,  4, 5, 5,  6, 7,  7,  9]]
    vals = np.ones((rows.size,))

    # Values for the lower triangle
    L = sparse.csc_matrix((vals, (rows, cols)), shape=(N, N))

    # Get the sum of the off-diagonal elements to ensure positive definiteness
    diag_A = np.max(np.sum(L + L.T - 2 * sparse.diags(L.diagonal()), 0))

    # Create the symmetric matrix A
    A = L + sparse.triu(L.T, 1) + diag_A * sparse.eye(N)

    A = A.toarray()

    # NOTE Scipy Cholesky is only implemented for dense matrices!
    R = la.cholesky(A, lower=True)
    R_up = chol_up(A, lower=True)
    R_left = chol_left(A, lower=True)
    R_left_amp = chol_left_amp(A, lower=True)
    R_right = chol_right(A, lower=True)

    # Define "supernodes" as ones, so we get the same result as left-looking
    s = np.ones(A.shape[0], dtype=int)
    R_super = chol_super(A, s, lower=True)

    # NOTE etree is not implemented in scipy!
    # Get the elimination tree
    # [parent, post] = etree(A)
    # Rp = la.cholesky(A(post, post), lower=True)
    # assert (R.nnz == Rp.nnz)  # post-ordering does not change nnz

    # Compute the row counts of the post-ordered Cholesky factor
    row_counts = np.sum(R != 0, 1)
    col_counts = np.sum(R != 0, 0)

    # Check that algorithms work
    for L in [R, R_up, R_left, R_left_amp, R_super, R_right]:
        np.testing.assert_allclose(L @ L.T, A, atol=1e-15)

    # Test (up|down)date
    # Generate random update with same sparsity pattern as a column of L
    rng = np.random.default_rng(565656)
    k = 3
    idx = np.nonzero(R[:, k])[0]
    w = np.zeros((N,))
    w[idx] = rng.random(idx.size)

    wwT = np.outer(w, w)  # == w[:, np.newaxis] @ w[np.newaxis, :]

    A_up = A + wwT

    L_up, w_up = chol_update(R, w)
    print("A_up =")
    print(A_up)
    print("L_up @ L_up.T =")
    print(L_up @ L_up.T)
    # np.testing.assert_allclose(L_up @ L_up.T, A_up, atol=1e-15)
    np.testing.assert_allclose(la.solve(R, w), w_up, atol=1e-15)

    # This function *does* produce the same result as chol_update
    L_up, w_up = chol_updown(R, w, update=True)
    print("A_up =")
    print(A_up)
    print("L_up @ L_up.T =")
    print(L_up @ L_up.T)
    # np.testing.assert_allclose(L_up @ L_up.T, A_up, atol=1e-15)
    np.testing.assert_allclose(la.solve(R, w), w_up, atol=1e-15)

    # Just downdate back to the original matrix!
    A_down = A.copy()  # A_down == A_up - wwT == A
    L_down, w_down = chol_downdate(L_up, w)
    # np.testing.assert_allclose(L_down @ L_down.T, A_down, atol=1e-15)
    np.testing.assert_allclose(la.solve(L_up, w), w_down, atol=1e-15)

# =============================================================================
# =============================================================================
