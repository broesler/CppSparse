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
import time

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
    is_upper = False
    is_lower = False

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
    data = np.genfromtxt(
        filename, 
        dtype=[
            ('rows', np.int32),
            ('cols', np.int32),
            ('vals', np.float64)
        ]
    )

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
    resid = (la.norm(r, np.inf) / (spla.norm(A, 1) * la.norm(x, np.inf))
             + la.norm(b, np.inf))
    print(f"resid: {resid:.2e}")


def demo2(C, is_sym, name='', axs=None):
    """Solve a linear system using Cholesky, LU, and QR decompositions with
    various column orderings.

    Parameters
    ----------
    C : (M, N) scipy.sparse.sparray
        The matrix to solve.
    is_sym : int
        - -1 if `C` is square and lower triangular
        - 1 if `C` is square and upper triangular
        - 0 otherwise.
    name : str, optional
        The name of the matrix.
    axs : (4,) array_like of matplotlib.axes.Axes, optional
        An array of 4 axes to plot the matrix on.
    """
    if axs is None:
        fig, axs = plt.subplots(nrows=2, ncols=2, clear=True)
        fig.set_size_inches((8, 8), forward=True)
    else:
        axs = np.asarray(axs).flatten()

    if axs.size < 4:
        raise ValueError("`axs` must have at least 4 axes")

    fig = axs[0].figure
    fig.suptitle(name)

    # Plot the matrix
    csparse.cspy(C, ax=axs[0])
    axs[0].set_title('cspy')

    # Compute the Dulmage-Mendelsohn (DM) ordering
    M, N = C.shape
    res = csparse.dmperm(csparse.from_scipy_sparse(C))
    r, s, rr = res.r, res.s, res.rr

    # TODO Plot the DM ordering
    # csparse.ccspy(C, ax=axs[1])  # connected components
    # csparse.dmspy(C, ax=axs[2])  # DM ordering highlighted

    sprank = rr[3]
    Nb = r.size - 1
    Ns = np.sum((r[1:Nb+1] == r[:Nb] + 1) & (s[1:Nb+1] == s[:Nb] + 1))
    print(f"blocks: {Nb}, singletons: {Ns}, structural rank: {sprank}")

    # FIXME all matrices show singular?
    # if sprank != sparse.csgraph.structural_rank(C):
    #     print("Structural rank mismatch")

    if sprank < min(M, N):
        print(f"Matrix is structurally singular ({sprank=})! Exiting.")
        return axs

    # Code that is unique to the python demo (not in C++ demo)
    # Compute and plot the appropriate decomposition of the matrix

    # TODO update pybind11 interface to accept scipy.sparse.csc_array
    Cc = csparse.from_scipy_sparse(C)

    ax = axs[3]

    if M == N:
        if is_sym:
            try:
                # Cholesky
                # TODO return permutation vector
                # L, p = csparse.cholesky(C)
                L = csparse.to_scipy_sparse(csparse.cholesky(Cc))
                csparse.cspy(L + sparse.triu(L.T, 1), ax=ax)
                ax.set_title('L + L.T')
            except Exception:
                # LU
                lu_res = csparse.lu(Cc)
                L, U = lu_res.L, lu_res.U
                csparse.cspy(L + U - sparse.eye_array(M), ax=ax)
                ax.set_title('L + U')
        else:
            # LU
            lu_res = csparse.lu(Cc)
            L, U = lu_res.L, lu_res.U
            csparse.cspy(L + U - sparse.eye_array(M), ax=ax)
            ax.set_title('L + U')
    else:
        # QR
        if M < N:
            res = csparse.qr(Cc.T)
        else:
            res = csparse.qr(Cc)

        V, R = res.V, res.R
        csparse.cspy(V + R, ax=ax)
        ax.set_title('V + R')

    # Continue with C++ demo
    b = np.ones(M) + np.arange(M) / M

    # QR Decomposition
    for order in ['Natural', 'ATA']:
        if order == 'Natural' and M > 1000:
            continue

        print(f"QR    ({order})", end='')
        tic = time.perf_counter()
        x = csparse.qr_solve(Cc, b, order=order)
        t = time.perf_counter() - tic
        print(f"time {t:8.2f} s ", end='')
        print_resid(C, x, b)

    if M != N:
        return

    # LU Decomposition
    for order in ['Natural', 'APlusAT', 'ATANoDenseRows', 'ATA']:
        print(f"LU    ({order})", end='')
        tic = time.perf_counter()
        x = csparse.lu_solve(Cc, b, order=order)
        t = time.perf_counter() - tic
        print(f"time {t:8.2f} s ", end='')
        print_resid(C, x, b)

    if not is_sym:
        return

    # Cholesky Decomposition
    for order in ['Natural', 'APlusAT']:
        print(f"Cholesky ({order})", end='')
        tic = time.perf_counter()
        x = csparse.chol_solve(Cc, b, order=order)
        t = time.perf_counter() - tic
        print(f"time {t:8.2f} s ", end='')
        print_resid(C, x, b)

    return axs


# -----------------------------------------------------------------------------
#         Main Script
# -----------------------------------------------------------------------------
matrix_path = Path('../../data/')

matrix_names = [
    't1',
    'fs_183_1',
    'west0067',
    'lp_afiro',
    'ash219',
    'mbeacxc',
    'bcsstk01',
    'bcsstk16'
]


for i, m in enumerate(matrix_names):
    C, is_sym = get_problem(matrix_path / m, tol=0.0)
    fig, axs = plt.subplots(num=i, nrows=2, ncols=2, clear=True)
    fig.set_size_inches((8, 8), forward=True)
    demo2(C, is_sym, m, axs=axs)
    plt.show()


# =============================================================================
# =============================================================================
