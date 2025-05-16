#!/usr/bin/env python3
# =============================================================================
#     File: demo2.py
#  Created: 2025-05-15 21:02
#   Author: Bernie Roesler
#
"""
A Python version of the C++Sparse/demo/demo2.cpp program.

Solve a linear system using Cholesky, LU, and QR decompositions with various
column orderings.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

from pathlib import Path
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

    if M == N:
        is_upper = sparse.tril(A, -1).nnz == 0
        is_lower = sparse.triu(A, 1).nnz == 0

    return 1 if is_upper else -1 if is_lower else 0


# TODO create pybind11 interface for csparse/demo/get_problem
def get_problem(filename, tol=0.0):
    """Read a matrix from a file and drop entries with values less than `tol`.

    Parameters
    ----------
    filename : str or Path
        The name of the file containing the matrix in triplet format.
    tol : float, optional
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
    data = np.genfromtxt(filename, delimiter=' ', dtype=[
        ('rows', np.int32),
        ('cols', np.int32),
        ('vals', np.float64)
    ])

    # Build the matrix
    # A = csparse.COOMatrix(data['vals'], data['rows'], data['cols']).tocsc()
    A = sparse.coo_array((data['vals'], (data['rows'], data['cols']))).tocsc()

    M, N = A.shape
    nnz = A.nnz

    if tol > 0:
        A.data[np.abs(A.data) < tol] = 0
        A.eliminate_zeros()
        # A = A.droptol(tol)  # drop small entries

    # Check if the matrix is symmetric
    is_sym = is_triangular(A)
    # is_sym = A.is_triangular()

    if is_sym:
        C = A + (A.T - A.diagonal() * sparse.eye_array(M))
    else:
        C = A

    print(f"--- Matrix: {M}-by-{N}, nnz: {A.nnz} "
          f"(sym: {is_sym}, nnz: {abs(is_sym) * C.nnz}), "
          f"norm: {spla.norm(C, 1)}")

    if nnz != A.nnz:
        print(f"tiny entries dropped: {nnz - A.nnz}")

    return C, is_sym


def demo2(A, is_sym, name):
    """Solve a linear system using Cholesky, LU, and QR decompositions with
    various column orderings.

    Parameters
    ----------
    A : (M, N) CSCMatrix
        The matrix to solve.
    is_sym : int
        - -1 if `A` is square and lower triangular
        - 1 if `A` is square and upper triangular
        - 0 otherwise.
    name : str
        The name of the matrix.
    """
    pass  # TODO implement demo2


matrix_path = Path('../../data/')

matrix_names = [
    't1',
    # 'fs_183_1',
    # 'west0067',
    # 'lp_afiro',
    # 'ash219',
    # 'mbeacxc',
    # 'bcsstk01',
    # 'bcsstk16'
]


for m in matrix_names:
    C, is_sym = get_problem(matrix_path / m, tol=0.0)
    demo2(C, is_sym, m)

# =============================================================================
# =============================================================================
