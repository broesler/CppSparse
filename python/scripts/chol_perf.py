#!/usr/bin/env python3
# =============================================================================
#     File: chol_perf.py
#  Created: 2025-02-10 21:34
#   Author: Bernie Roesler
#
"""
Compare the performance of the Cholesky decomposition using different methods.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import re
import timeit

from collections import defaultdict
from functools import partial
from pathlib import Path

from scipy.sparse.linalg import LaplacianNd

import csparse


LAPLACE = True
SAVE_FIG = False

SEED = 565656

if LAPLACE:
    filestem = 'chol_perf_laplace'
else:
    filestem = 'chol_perf_random'

# -----------------------------------------------------------------------------
#         Create the data
# -----------------------------------------------------------------------------
chol_funcs = [csparse.chol, csparse.leftchol, csparse.rechol]

if LAPLACE:
    # for Lapalacian, define sqrtN and N = sqrtN^2
    Ns = [3, 4, 7, 10, 14, 22, 31, 44, 70]
    N_cols = [N**2 for N in Ns]
else:
    Ns = [10, 20, 50, 100, 200, 500, 1000, 2000]
    N_cols = Ns
    density = 0.2

N_repeats = 3  # number of "runs" in %timeit (7 is default)
N_samples = 1  # number of samples in each run (100,000 is default)

times = defaultdict(list)
fill_in = []

rng = np.random.default_rng(SEED)

for N in Ns:
    if LAPLACE:
        # Use a non-random matrix like a 2D Laplacian
        A = LaplacianNd((N, N)).tosparse().tocsc().astype(float)
    else:
        # Create a random matrix
        A = csparse.COOMatrix.random(N, N, density, SEED).tocsc().toscipy()

    # Ensure all diagonals are non-zero so that L is non-singular
    for i in range(A.shape[0]):
        A[i, i] = N * N

    # Make sure the matrix is symmetric, positive definite
    A = A + A.T

    print(f"---------- N = {A.shape[1]:6,d} ----------")

    for func in chol_funcs:
        func_name = func.__name__
        partial_func = partial(func, A)

        # Time the function
        ts = timeit.repeat(partial_func, repeat=N_repeats, number=N_samples)

        ts = np.array(ts) / N_samples  # time per loop
        ts_min = np.min(ts)

        times[func_name].append(ts_min)

        print(f"{func_name}: {ts_min:.4g} s per loop, "
              f"({N_repeats} runs, {N_samples} loops each)")

    # Compute fill-in
    L = csparse.chol(A).L
    fill_in.append((L.nnz - A.nnz) / A.nnz)


print(np.c_[Ns, fill_in])

# -----------------------------------------------------------------------------
#         Plot the data
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(num=1, clear=True)
fig.set_size_inches(6.4, 4.8, forward=True)
for i, (key, val) in enumerate(times.items()):
    ax.plot(N_cols, val, '.-', label=key)

ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(which='both')
ax.legend()

ax.set_xlabel('Number of Columns')
ax.set_ylabel('Time (s)')

title = f"{filestem.split('_')[0]}"

if not LAPLACE:
    title += f", density {density}"

ax.set_title(title)

if SAVE_FIG:
    fig_fullpath = f"../plots/{filestem}.png"

    if not LAPLACE:
        fig_fullpath = re.sub(r'\.png', f"_d{int(100*density):02d}.png", fig_fullpath)

    try:
        fig.savefig(Path(fig_fullpath))
        print(f"Saved figure to {fig_fullpath}.")
    except Exception as e:
        print(f"Could not save figure to {fig_fullpath}: {e}")
        raise e


# Plot the spy of each matrix to see fill-in
fig, axs = plt.subplots(num=2, ncols=2, clear=True)
axs[0].spy(A.toarray())
axs[1].spy(L.toarray())

plt.show()

# =============================================================================
# =============================================================================
