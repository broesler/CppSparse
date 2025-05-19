#!/usr/bin/env python3
# =============================================================================
#     File: demo3.py
#  Created: 2025-05-19 16:11
#   Author: Bernie Roesler
#
"""
A Python version of the C++Sparse/demo/demo3.cpp program.

Perform Cholesky up/downdate on a sparse matrix.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import time

from pathlib import Path
from scipy import sparse

import csparse

from demo import get_problem, print_resid


def demo3(C, is_sym, name='', axs=None):
    """Perform Cholesky up/downdate on a sparse matrix.

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
    else:
        axs = np.asarray(axs).flatten()

    if axs.size < 4:
        raise ValueError("`axs` must have at least 4 axes")

    fig = axs[0].figure
    fig.suptitle(name)

    # Plot the matrix
    axs[0].set_title('C')
    _, cb = csparse.cspy(C, ax=axs[0], norm='log')

    # Move the colorbar to the right, outside the subplots
    cb.remove()
    fig.colorbar(cb.mappable, ax=axs, location='right', shrink=0.8)

    # -------------------------------------------------------------------------
    #         Compute the Cholesky decomposition
    # -------------------------------------------------------------------------
    M, N = C.shape

    if M != N or not is_sym:
        raise ValueError("Matrix must be square and symmetric")

    order = 'Natural'
    print('chol then update/downdate ({order})')

    Cc = csparse.from_scipy_sparse(C)
    b = np.ones(M) + np.arange(M) / M

    tic = time.perf_counter()
    Lc = csparse.chol(Cc, order)
    t = time.perf_counter() - tic
    print(f"chol  time: {t:.2e} s")

    L = csparse.to_scipy_sparse(Lc)
    axs[1].set_title('L')
    csparse.cspy(L, ax=axs[1], colorbar=False, norm='log')

    # Solve the linear system
    tic = time.perf_counter()
    x = b.copy()  # TODO permutation vector b[p]
    x = csparse.lsolve(Lc, x)
    x = csparse.ltsolve(Lc, x)
    # TODO x[p] = x
    t = time.perf_counter() - tic
    print(f"solve time: {t:.2e} s")

    print('original: ', end='')
    print_resid(C, x, b)

    # Build the random update vector from a column of L
    k = N // 2
    rng = np.random.default_rng(565656)

    Lk = L[:, [k]].tocoo()
    rows = Lk.row
    cols = Lk.col
    vals = rng.random(Lk.data.size)

    w = L[k, k] * sparse.coo_array((vals, (rows, cols)), shape=(M, 1))
    wc = csparse.from_scipy_sparse(w)

    parent = csparse.etree(Cc)  # TODO permutation

    tic = time.perf_counter()
    Lupc = csparse.chol_update_(Lc, True, wc, parent)
    t1 = time.perf_counter() - tic
    print(f"update:   time: {t1:.2e} s")

    Lup = csparse.to_scipy_sparse(Lupc)

    # Plot the updated matrix and the difference from the original
    csparse.cspy(Lup, ax=axs[2], colorbar=False, norm='log')
    csparse.cspy(L - Lup, ax=axs[3], colorbar=False, norm='log')

    axs[2].set_title('Updated L')
    axs[3].set_title('L - updated L')

    # Solve the linear system
    tic = time.perf_counter()
    x = b.copy()  # TODO permutation vector b[p]
    x = csparse.lsolve(Lupc, x)
    x = csparse.ltsolve(Lupc, x)
    # TODO x[p] = x
    t = time.perf_counter() - tic

    # Compute the residuals
    # wp = sparse.csc_array(shape=(M, 1))
    # wp[p] = w  # TODO permutation vector
    wp = w.copy()
    E = C + wp * wp.T
    Ec = csparse.from_scipy_sparse(E)

    print(f"update:   time: {t1 + t:.2e} s (incl solve) ", end='')
    print_resid(E, x, b)

    tic = time.perf_counter()
    Lc = csparse.chol(Ec, order)
    x = b.copy()  # TODO permutation vector b[p]
    x = csparse.lsolve(Lc, x)
    x = csparse.ltsolve(Lc, x)
    # TODO x[p] = x
    t = time.perf_counter() - tic

    print(f"rechol:   time: {t:.2e} s (incl solve) ", end='')
    print_resid(E, x, b)

    # Downdate
    tic = time.perf_counter()
    Ldownc = csparse.chol_update_(Lupc, False, wc, parent)
    t1 = time.perf_counter() - tic

    print(f"downdate: time: {t1:.2e} s")

    tic = time.perf_counter()
    x = b.copy()  # TODO permutation vector b[p]
    x = csparse.lsolve(Ldownc, x)
    x = csparse.ltsolve(Ldownc, x)
    # TODO x[p] = x
    t = time.perf_counter() - tic

    print(f"downdate: time: {t1 + t:.2e} s (incl solve) ", end='')
    print_resid(C, x, b)

    return axs


# -----------------------------------------------------------------------------
#         Main Script
# -----------------------------------------------------------------------------
matrix_path = Path('../../data/')

matrix_names = [
    'bcsstk01',
    'bcsstk16'
]


for i, m in enumerate(matrix_names):
    C, is_sym = get_problem(matrix_path / m, tol=0.0)
    fig, axs = plt.subplots(num=i, nrows=2, ncols=2, clear=True)
    demo3(C, is_sym, m, axs=axs)
    print()  # extra newline on output for clarity
    plt.show()

# =============================================================================
# =============================================================================
