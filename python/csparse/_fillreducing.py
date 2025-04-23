#!/usr/bin/env python3
# =============================================================================
#     File: _fillreducing.py
#  Created: 2025-04-22 19:03
#   Author: Bernie Roesler
#
"""
Python implementation of fill-reducing algorithms, as presented in Davis,
Chapter 7.
"""
# =============================================================================

import numpy as np

from scipy import sparse
from scipy.sparse import linalg as spla

from csparse import dmperm


def fiedler(A):
    """Compute the Fiedler vector of a connected graph.

    The Fiedler vector is the eigenvector corresponding to the second smallest
    eigenvalue of the Laplacian of `A + A.T`.

    Parameters
    ----------
    A : (M, N) sparse array
        Matrix of M vectors in N dimensions, corresponding to a connected
        graph.

    Returns
    -------
    p : (M,) ndarray
        The permutation vector obtained when `v` is sorted.
    v : (M,) ndarray
        The Fiedler vector of the graph.
    d : float
        The second smallest eigenvalue of the Laplacian of `A + A.T`.
    """
    N = A.shape[1]

    if N < 2:
        return np.ones(N), np.ones(N), 0.0

    # Compute the structure of the Laplacian matrix
    S = (A + A.T + sparse.eye_array(N)).astype(bool).astype(float)

    # Create a diagonal matrix with the sum of each column
    D = sparse.diags(S.sum(axis=0))

    # Compute the Laplacian matrix itself
    L = D - S

    # Get the eigenvalues and eigenvectors of the Laplacian matrix
    λ, x = spla.eigsh(L, k=2, which='SA', tol=np.sqrt(np.finfo(float).eps))

    # Take the second smallest eigenvalue and its corresponding eigenvector
    d = λ[1]
    v = x[:, 1]
    p = np.argsort(v)

    return p, v, d


# =============================================================================
# =============================================================================
