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
import time

from pathlib import Path
from scipy import sparse

import csparse


def demo2(problem, name='', axs=None):
    """Solve a linear system using Cholesky, LU, and QR decompositions with
    various column orderings.

    Parameters
    ----------
    problem : csparse.Problem
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

    C = problem.C
    is_sym = problem.is_sym
    b = problem.b

    fig = axs[0].figure
    fig.suptitle(name)

    # Plot the matrix
    _, cb = csparse.cspy(C, ax=axs[0], norm='log')
    axs[0].set_title('C')

    # Compute the Dulmage-Mendelsohn (DM) ordering
    M, N = C.shape
    res = csparse.dmperm(C)
    r, s, rr = res.r, res.s, res.rr

    csparse.ccspy(C, ax=axs[1], colorbar=False, norm='log')
    csparse.dmspy(C, ax=axs[2], colorbar=False, norm='log')

    # Move the colorbar to the right, outside the subplots
    cb.remove()
    fig.colorbar(cb.mappable, ax=axs, location='right', shrink=0.8)

    sprank = rr[3]
    Nb = r.size - 1
    Ns = np.sum((r[1:Nb+1] == r[:Nb] + 1) & (s[1:Nb+1] == s[:Nb] + 1))
    print(f"blocks: {Nb}, singletons: {Ns}, structural rank: {sprank}")

    if sprank != sparse.csgraph.structural_rank(C):
        print("Structural rank mismatch")

    if sprank < min(M, N):
        print(f"Matrix is structurally singular ({sprank=:d})! Exiting.")
        axs[3].set_visible(False)
        return axs

    # -------------------------------------------------------------------------
    #         Compute and plot a decomposition of the matrix
    # -------------------------------------------------------------------------
    # Code is unique to the python demo (not in C++ demo)

    ax = axs[3]

    pad = dict(
        Natural=' ' * 8,
        APlusAT=' ' * 8,
        ATANoDenseRows=' ',
        ATA=' ' * 12
    )

    if M == N:
        if is_sym:
            try:
                # Cholesky
                L, p, _ = csparse.chol(C, order='APlusAT')
                csparse.cspy(L + sparse.triu(L.T, 1),
                             colorbar=False, norm='log', ax=ax)
                ax.set_title('L + L.T')
            except Exception:
                # LU
                res = csparse.lu(C, tol=0.001, order='APlusAT')
                L, U = res.L, res.U
                csparse.cspy(L + U - sparse.eye_array(M),
                             colorbar=False, norm='log', ax=ax)
                ax.set_title('L + U')
        else:
            # LU
            res = csparse.lu(C, order='ATANoDenseRows')
            L, U = res.L, res.U
            csparse.cspy(L + U - sparse.eye_array(M),
                         colorbar=False, norm='log', ax=ax)
            ax.set_title('L + U')
    else:
        # QR
        if M < N:
            res = csparse.qr(C.T, order='ATA')
        else:
            res = csparse.qr(C, order='ATA')

        V, R = res.V, res.R
        csparse.cspy(V + R, colorbar=False, norm='log', ax=ax)
        ax.set_title('V + R')

    # -------------------------------------------------------------------------
    #        Solve the linear system
    # -------------------------------------------------------------------------
    # Continue with C++ demo
    # QR Decomposition
    for order in ['Natural', 'ATA']:
        if order == 'Natural' and M > 1000:
            continue

        print(f"QR    ({order})", end=pad[order])
        tic = time.perf_counter()
        x = csparse.qr_solve(C, b, order=order)
        t = time.perf_counter() - tic
        print(f"time {t:.2e} s ", end='')
        resid = csparse.residual_norm(C, x, b)
        print(f"residual: {resid:.2e} ")

    if M != N:
        return

    # LU Decomposition
    for order in ['Natural', 'APlusAT', 'ATANoDenseRows', 'ATA']:
        print(f"LU    ({order})", end=pad[order])
        tic = time.perf_counter()
        x = csparse.lu_solve(C, b, order=order)
        t = time.perf_counter() - tic
        print(f"time {t:.2e} s ", end='')
        resid = csparse.residual_norm(C, x, b)
        print(f"residual: {resid:.2e} ")

    if not is_sym:
        return

    # Cholesky Decomposition
    for order in ['Natural', 'APlusAT']:
        print(f"Chol  ({order})", end=pad[order])
        tic = time.perf_counter()
        x = csparse.chol_solve(C, b, order=order)
        t = time.perf_counter() - tic
        print(f"time {t:.2e} s ", end='')
        resid = csparse.residual_norm(C, x, b)
        print(f"residual: {resid:.2e} ")

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
    # 'bcsstk16'
]


for i, m in enumerate(matrix_names):
    T = csparse.COOMatrix.from_file(str(matrix_path / m))
    prob = csparse.Problem.from_matrix(T, droptol=1e-14)
    fig, axs = plt.subplots(num=i, nrows=2, ncols=2, clear=True)
    demo2(prob, m, axs=axs)
    print()  # extra newline on output for clarity
    plt.show()


# =============================================================================
# =============================================================================
