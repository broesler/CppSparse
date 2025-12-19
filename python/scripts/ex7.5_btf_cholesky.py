#!/usr/bin/env python3
# =============================================================================
#     File: ex7.5_btf_cholesky.py
#  Created: 2025-12-19 14:19
#   Author: Bernie Roesler
# =============================================================================

"""Solution to Exercise 7.5: Cholesky Factorization from BTF."""

import numpy as np
from scipy import sparse

import csparse


def create_matrix_from_tree(parent):
    """Create a symmetric positive definite matrix from an elimination tree.

    Parameters
    ----------
    parent : array_like
        The parent array representing the elimination tree.

    Returns
    -------
    F : ndarray
        The generated symmetric positive definite matrix.
    """
    N = len(parent)
    L = sparse.eye_array(N, format="dok")

    for i, p in enumerate(parent):
        if p != -1:
            L[p, i] = 1  # arbitrary non-zero entry

    A = L @ L.T
    A.setdiag(np.arange(10, N + 10))  # easy reading

    return A.tocsc(), L.tocsc()


# --- Example Usage ---
if __name__ == "__main__":
    # Arbitrary elimination tree parent array
    parent = np.r_[5, 2, 7, 5, 7, 8, 8, 10, -1, 10, -1]

    A = csparse.davis_example_chol()

    # Generate the new matrix A_new
    F, L = create_matrix_from_tree(parent)

    # Ensure etree of F is parent
    F_parent = csparse.etree(F)
    np.testing.assert_array_equal(F_parent, parent)

# =============================================================================
# =============================================================================
