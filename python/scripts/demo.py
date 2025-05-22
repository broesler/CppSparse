#!/usr/bin/env python3
# =============================================================================
#     File: demo.py
#  Created: 2025-05-19 16:15
#   Author: Bernie Roesler
#
"""
Description: Helper functions for the C++Sparse demo programs.
"""
# =============================================================================

import numpy as np
import scipy.linalg as la

from scipy import sparse
from scipy.sparse import linalg as spla

import csparse


def is_triangular(A):
    """Check if a sparse matrix is triangular.

    Parameters
    ----------
    A : (M, N) array_like
        The sparse matrix to check.

    Returns
    -------
    result : int
        - -1 if `A` is square and lower triangular
        - 1 if `A` is square and upper triangular
        - 0 otherwise.
    """
    M, N = A.shape
    is_upper = False
    is_lower = False

    if M == N:
        is_upper = sparse.tril(A, -1).nnz == 0
        is_lower = sparse.triu(A, 1).nnz == 0

    return 1 if is_upper else -1 if is_lower else 0


# TODO create pybind11 interface for csparse/demo/get_problem
def get_problem(filename, droptol=0.0):
    """Read a matrix from a file and drop entries with values less than `droptol`.

    Parameters
    ----------
    filename : str or Path
        The name of the file containing the matrix in triplet format.
    droptol : float, optional
        The absolute tolerance for dropping small entries. Default is 0.0 (no
        entries dropped).

    Returns
    -------
    C : (M, N) CSCMatrix
        If the matrix `A` defined in the file is symmetric and stored as
        a triangular matrix, `C = A + A.T`, otherwise, `C = A`.
    is_sym : int
        - -1 if `C` is square and lower triangular
        - 1 if `C` is square and upper triangular
        - 0 otherwise.
    """
    # Read the matrix from the file
    A = csparse.COOMatrix.from_file(str(filename)).tocsc().toscipy()

    M, N = A.shape
    nnz = A.nnz

    if droptol > 0:
        A.data[np.abs(A.data) < droptol] = 0
        A.eliminate_zeros()

    # Check if the matrix is symmetric
    is_sym = is_triangular(A)

    if is_sym:
        C = A + (A.T - A.diagonal() * sparse.eye_array(M))
    else:
        C = A

    print(f"--- Matrix: {M}-by-{N}, nnz: {A.nnz} "
          f"(sym: {is_sym}, nnz: {abs(is_sym) * C.nnz}), "
          f"norm: {spla.norm(C, 1):.4g}")

    if nnz != A.nnz:
        print(f"tiny entries dropped: {nnz - A.nnz}")

    return C, is_sym


def print_resid(A, x, b):
    """Print the norm of the residual of the linear system `Ax = b`.

    Parameters
    ----------
    A : (M, N) array_like
        The matrix `A`.
    x : (N,) array_like
        The solution vector `x`.
    b : (M,) array_like
        The right-hand side vector `b`.
    """
    r = A @ x - b
    resid = (
        la.norm(r, np.inf) /
        (spla.norm(A, 1) * la.norm(x, np.inf) + la.norm(b, np.inf))
    )
    print(f"resid: {resid:.2e}")

# =============================================================================
# =============================================================================
