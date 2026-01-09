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

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

import csparse


def demo3(prob, name='', axs=None):
    """Perform Cholesky up/downdate on a sparse matrix.

    Parameters
    ----------
    prob : csparse.Problem
        The problem to solve, containing the matrix C and its properties.
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

    C, is_sym, b = prob.C, prob.is_sym, prob.b

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

    # cs_chol_mex.c uses A+A' ordering when nargout == 2
    order = 'APlusAT'
    print(f"chol then update/downdate ({order})")

    tic = time.perf_counter()
    L, p = csparse.chol(C, order)
    parent = csparse.etree(C[p[:, np.newaxis], p])
    t = time.perf_counter() - tic
    print(f"chol  time: {t:.2e} s")

    axs[1].set_title('L')
    csparse.cspy(L, ax=axs[1], colorbar=False, norm='log')

    # Solve the linear system
    tic = time.perf_counter()
    x = b[p]
    x = csparse.lsolve(L, x)
    x = csparse.ltsolve(L, x)
    x[p] = x
    t = time.perf_counter() - tic
    print(f"solve time: {t:.2e} s")

    print('original: ', end='')
    resid = csparse.residual_norm(C, x, b)
    print(f"residual: {resid:.2e}")

    # Build the random update vector from a column of L
    k = N // 2
    rng = np.random.default_rng(565656)

    Lk = L[:, [k]].tocoo()
    rows = Lk.row
    cols = Lk.col
    vals = rng.random(Lk.data.size)

    w = L[k, k] * sparse.coo_array((vals, (rows, cols)), shape=(M, 1))

    tic = time.perf_counter()
    Lup = csparse.chol_update_(L, True, w, parent)
    t1 = time.perf_counter() - tic
    print(f"update:   time: {t1:.2e} s")

    # Plot the updated matrix and the difference from the original
    csparse.cspy(Lup, ax=axs[2], colorbar=False, norm='log')
    csparse.cspy(L - Lup, ax=axs[3], colorbar=False, norm='log')

    axs[2].set_title('Updated L')
    axs[3].set_title('L - updated L')

    # Solve the linear system
    tic = time.perf_counter()
    x = b[p]
    x = csparse.lsolve(Lup, x)
    x = csparse.ltsolve(Lup, x)
    x[p] = x
    t = time.perf_counter() - tic

    # Compute the residuals
    wp = sparse.dok_array((M, 1))
    wp[p] = w
    E = C + wp * wp.T

    print(f"update:   time: {t1 + t:.2e} s (incl solve) ", end='')
    resid = csparse.residual_norm(E, x, b)
    print(f"residual: {resid:.2e}")

    tic = time.perf_counter()
    L2, p2 = csparse.chol(E, order)
    x = b[p2]
    x = csparse.lsolve(L2, x)
    x = csparse.ltsolve(L2, x)
    x[p2] = x
    t = time.perf_counter() - tic

    print(f"rechol:   time: {t:.2e} s (incl solve) ", end='')
    resid = csparse.residual_norm(E, x, b)
    print(f"residual: {resid:.2e}")

    # Downdate
    tic = time.perf_counter()
    Ldown = csparse.chol_update_(Lup, False, w, parent)
    t1 = time.perf_counter() - tic

    print(f"downdate: time: {t1:.2e} s")

    tic = time.perf_counter()
    x = b[p]
    x = csparse.lsolve(Ldown, x)
    x = csparse.ltsolve(Ldown, x)
    x[p] = x
    t = time.perf_counter() - tic

    print(f"downdate: time: {t1 + t:.2e} s (incl solve) ", end='')
    resid = csparse.residual_norm(C, x, b)
    print(f"residual: {resid:.2e}")

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
    T = csparse.COOMatrix.from_file(str(matrix_path / m))
    prob = csparse.Problem.from_matrix(T, droptol=1e-14)
    fig, axs = plt.subplots(num=i, nrows=2, ncols=2, clear=True)
    demo3(prob, m, axs=axs)
    print()  # extra newline on output for clarity
    plt.show()

# =============================================================================
# =============================================================================
