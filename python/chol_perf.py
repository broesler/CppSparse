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
import timeit

from collections import defaultdict
from functools import partial
from pathlib import Path

import csparse as cs


SAVE_FIG = True

SEED = 565656

filestem = 'chol_perf_py'

# -----------------------------------------------------------------------------
#         Create the data
# -----------------------------------------------------------------------------
chol_funcs = [cs.chol, cs.leftchol, cs.rechol]

# Ns = [10, 100, 1000]
Ns = [10, 20, 50, 100, 200, 500, 1000]

density = 0.2

N_repeats = 3  # number of "runs" in %timeit (7 is default)
N_samples = 1  # number of samples in each run (100,000 is default)

times = defaultdict(list)

rng = np.random.default_rng(SEED)

for N in Ns:
    print(f"---------- N = {N:6,d} ----------")
    # for density in densities:
    # print(f"---------- Density = {density:6.2g} ----------")

    # TODO try a non-random matrix like a 2D Laplacian

    # Create a random matrix
    A = cs.COOMatrix.random(N, N, density, SEED).tocsc()

    # Ensure all diagonals are non-zero so that L is non-singular
    for i in range(N):
        A[i, i] = N

    # Make sure the matrix is symmetric, positive definite
    A = A + A.T()

    # Get the symbolic factorization (same for all methods)
    S = cs.schol(A)

    for func in chol_funcs:
        func_name = func.__name__

        args = [A, S]

        if func_name == 'leftchol' or func_name == 'rechol':
            L = cs.symbolic_cholesky(A, S)
            args.append(L)

        partial_func = partial(func, *args)

        # Time the function
        ts = timeit.repeat(partial_func, repeat=N_repeats, number=N_samples)

        ts = np.array(ts) / N_samples  # time per loop
        ts_min = np.min(ts)

        times[func_name].append(ts_min)

        print(f"{func_name}: {ts_min:.4g} s per loop, "
              f"({N_repeats} runs, {N_samples} loops each)")


# -----------------------------------------------------------------------------
#         Plot the data
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(num=1, clear=True)
fig.set_size_inches(6.4, 4.8, forward=True)
for i, (key, val) in enumerate(times.items()):
    ax.plot(Ns, val, '.-', label=key)

ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(which='both')
ax.legend()

ax.set_xlabel('Number of Columns')
ax.set_ylabel('Time (s)')
ax.set_title(f"{filestem.split('_')[0]}, density {density}")

plt.show()

if SAVE_FIG:
    fig_fullpath = Path(f"../plots/{filestem}_d{int(100*density):02d}.png")
    try:
        fig.savefig(fig_fullpath)
        print(f"Saved figure to {fig_fullpath}.")
    except Exception as e:
        print(f"Could not save figure to {fig_fullpath}: {e}")
        raise e

# =============================================================================
# =============================================================================

# =============================================================================
# =============================================================================
