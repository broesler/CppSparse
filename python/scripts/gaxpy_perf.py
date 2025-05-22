#!/usr/bin/env python3
# =============================================================================
#     File: gaxpy_perf.py
#  Created: 2025-01-07 12:49
#   Author: Bernie Roesler
#
"""
Create and plot the gaxpy performance data.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import timeit

from collections import defaultdict
from functools import partial
from pathlib import Path

import csparse


SAVE_FIG = False

SEED = 565656

filestem = 'gaxpy_perf_py'

# -----------------------------------------------------------------------------
#         Create the data
# -----------------------------------------------------------------------------
# Ns = np.r_[10, 100, 500]
Ns = np.r_[10, 20, 50, 100, 200, 500, 1000, 2000, 5000]

density = 0.1  # density of the matrices

N_repeats = 7  # number of "runs" in %timeit (7 is default)
N_samples = 1  # number of samples in each run (100,000 is default)

gaxpy_funcs = dict(
    regular=[
        csparse.gaxpy_col,
        csparse.gaxpy_row,
        csparse.gaxpy_block
    ],
    transpose=[
        csparse.gatxpy_col,
        csparse.gatxpy_row,
        csparse.gatxpy_block
    ]
)

# Store the results
times = dict(regular=defaultdict(list),
             transpose=defaultdict(list))

for kind in ['regular', 'transpose']:
    for N in Ns:
        print(f"---------- N = {N:6,d} ----------")

        # Create a large, random, sparse matrix with different dimensions
        M = int(0.9 * N)
        K = int(0.8 * N)
        A = csparse.COOMatrix.random(M, N, density, SEED).tocsc().toscipy()

        if kind == 'transpose':
            A = A.T

        # Create compatible random, dense matrix for multiplying and adding
        X = csparse.COOMatrix.random(N, K, density, SEED)
        Y = csparse.COOMatrix.random(M, K, density, SEED)

        # Convert to row and column-major format
        X_col = X.to_dense_vector('F')
        Y_col = Y.to_dense_vector('F')

        X_row = X.to_dense_vector('C')
        Y_row = Y.to_dense_vector('C')

        for func in gaxpy_funcs[kind]:
            func_name = func.__name__
            args = ((X_row, Y_row) if func_name.endswith('row')
                    else (X_col, Y_col))

            # Create a partial function with the arguments for timing
            partial_method = partial(func, A, *args)

            # Run the function
            ts = timeit.repeat(partial_method,
                               repeat=N_repeats,
                               number=N_samples)

            # NOTE timeit returns the total time for all "number" loops for
            # each repeat, so we divide by N_samples to get the time per loop.
            #   len(ts) == N_repeats
            # We only care about the minimum time per loop, since additional
            # time is likely due to other processes running on the system.
            ts = np.array(ts) / N_samples  # time per loop
            ts_min = np.min(ts)

            times[kind][func_name].append(ts_min)

            print(f"{func_name}: {ts_min:.4g} s per loop, "
                  f"({N_repeats} runs, {N_samples} loops each)")


# -----------------------------------------------------------------------------
#         Plot the data
# -----------------------------------------------------------------------------
fig, axs = plt.subplots(num=1, ncols=2, sharey=True, clear=True)
fig.suptitle(f"{filestem.split('_')[0].upper()}, density {density}")
fig.set_size_inches(10, 4.8, forward=True)

for ax, kind in zip(axs, ['regular', 'transpose']):
    for i, (key, val) in enumerate(times[kind].items()):
        ax.plot(Ns, val, '.-', label=key)

    ax.set(
        title=f"{kind.capitalize()}",
        xscale='log',
        yscale='log',
        xlabel='Number of Columns',
    )

    if ax.get_subplotspec().is_first_col():
        ax.set(ylabel='Runtime [s]')

    ax.grid(True, which='both')
    ax.legend()


plt.show()


if SAVE_FIG:
    fig_fullpath = Path(f"../../plots/{filestem}_d{int(100*density):02d}.png")
    try:
        fig.savefig(fig_fullpath)
        print(f"Saved figure to {fig_fullpath}.")
    except Exception as e:
        print(f"Could not save figure to {fig_fullpath}: {e}")
        raise e

# =============================================================================
# =============================================================================
