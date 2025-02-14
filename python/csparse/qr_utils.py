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

import csparse


# TODO In order to make these functions work with my csparse matrices, we need
# to either: have a way to convert to sparse.csc_array, or implement the
# slicing operations in the csparse module.

def qright(Y, V, beta, p=None):
    r"""Apply Householder vectors on the right.

    Computes :math:`X = Y P^T H_1 \dots H_N = Y Q`, where :math:`Q` is
    represented by the Householder vectors stored in `V`, coefficients `beta`,
    and permutation `p`. To obtain :math:`Q` itself, pass `Y = sparse.eye(M)`.

    Parameters
    ----------
    Y : (M, N) ndarray or sparse matrix
        The matrix to which the Householder transformations are applied.
    V : (M, N) CSCMatrix
        The matrix of Householder vectors.
    beta : (N,) ndarray
        The Householder coefficients.
    p : (N,) ndarray, optional
        The column permutation vector.

    Returns
    -------
    result : (M, N) ndarray
        The result of applying the Householder transformations to `Y`.
    """
    M, N = V.shape
    X = Y.copy()
    if p is not None:
        X = X[:, p]
    for j in range(N):
        X -= X @ (beta[j] * V[:, [j]]) @ V[:, [j]].T()
    return X


def qleft(Y, V, beta, p=None):
    r"""Apply Householder vectors on the left.

    Computes :math:`X = H_N \dots H_1 P Y = Q^T Y`, where :math:`Q` is
    represented by the Householder vectors stored in `V`, coefficients `beta`,
    and permutation `p`. To obtain :math:`Q` itself, pass `Y = sparse.eye(M)`.

    Parameters
    ----------
    Y : (M2, N) ndarray or sparse matrix
        The matrix to which the Householder transformations are applied.
    V : (M, NY) CSCMatrix
        The matrix of Householder vectors.
    beta : (N,) ndarray
        The Householder coefficients.
    p : (N,) ndarray, optional
        The row permutation vector.

    Returns
    -------
    result : (M, N) ndarray
        The result of applying the Householder transformations to `Y`.
    """
    M2, N = V.shape
    M, NY = Y.shape
    X = Y.copy()

    if (M2 > M):
        # Add empty rows to the bottom of X
        if (isinstance(X, csparse.CSCMatrix)
                or isinstance(X, csparse.COOMatrix)):
            X = X.add_empty_bottom(M2 - M)
        elif sparse.issparse(X):
            X = sparse.vstack([X, sparse.csc_matrix((M2 - M, NY))])
        else:
            X = np.vstack([X, np.zeros((M2 - M, NY))])

    if p is not None:
        X = X[p, :]

    for j in range(N):
        X -= V[:, [j]] @ (beta[j] * V[:, [j]].T() @ X)

    return X

# =============================================================================
# =============================================================================
