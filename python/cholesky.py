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

import matplotlib.pyplot as plt
import numpy as np

from scipy import (linalg as la,
                   sparse as sp)
import scipy.sparse.linalg as spla


# -----------------------------------------------------------------------------
#         Define Cholesky Algorithms
# -----------------------------------------------------------------------------
def chol_up(A, lower=False):
    """Up-looking Cholesky decomposition.

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
    R : ndarray
        Triangular Cholesky factor of A.
    """
    N = A.shape[0]
    L = np.zeros((N, N), dtype=A.dtype)
    for k in range(N):
        L[k, :k] = la.solve(L[:k, :k], A[:k, k]).T  # solve Lx = b
        L[k, k] = np.sqrt(A[k, k] - L[k, :k] @ L[k, :k])
    return L if lower else L.T



if __name__ == "__main__":
    # Create the example matrix A
    N = 11
    rows = np.r_[np.arange(N), [5, 6, 2, 7, 9, 10, 5, 9, 7, 10, 8, 9, 10, 9, 10, 10]]
    cols = np.r_[np.arange(N), [0, 0, 1, 1, 2,  2, 3, 3, 4,  4, 5, 5,  6, 7,  7,  9]]
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
    Rup = chol_up(A, lower=True)

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
    np.testing.assert_allclose(R @ R.T, A, atol=1e-15)
    np.testing.assert_allclose(Rup @ Rup.T, A, atol=1e-15)

# =============================================================================
# =============================================================================
