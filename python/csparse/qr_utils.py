#!/usr/bin/env python3
# =============================================================================
#     File: qr_utils.py
#  Created: 2025-02-14 09:26
#   Author: Bernie Roesler
#
"""
Additional functions to support QR decomposition.
"""
# =============================================================================

import numpy as np
from scipy import sparse

from csparse import CSCMatrix
from .utils import to_scipy_sparse


def qright(V, beta, p=None, Y=None):
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
        Y = sparse.eye_array(V.shape[1])

    M, N = V.shape
    X = Y.copy()
    if p is not None:
        X = X[:, p]
    for j in range(N):
        X -= X @ (beta[j] * V[:, [j]]) @ V[:, [j]].T
    return X


def qleft(V, beta, p=None, Y=None):
    r"""Apply Householder vectors on the left.

    Computes :math:`X = H_N \dots H_1 P Y = Q^T Y`, where :math:`Q` is
    represented by the Householder vectors stored in `V`, coefficients `beta`,
    and permutation `p`. To obtain :math:`Q` itself, pass `Y = sparse.eye(M)`.

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
        Y = sparse.eye_array(V.shape[0])

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

# =============================================================================
# =============================================================================
