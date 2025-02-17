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

import csparse as cs


SAVE_FIG = False

SEED = 565656

filestem = 'gaxpy_perf_py'
# filestem = 'gatxpy_perf_py'

# -----------------------------------------------------------------------------
#         Create the data
# -----------------------------------------------------------------------------
# Ns = np.r_[10, 100, 500]
Ns = np.r_[10, 20, 50, 100, 200, 500, 1000, 2000, 5000]

density = 0.25  # density of the matrices

N_repeats = 3  # number of "runs" in %timeit (7 is default)
N_samples = 1  # number of samples in each run (100,000 is default)

# TODO include the transpose versions and plot as subfigs
if filestem.startswith('gaxpy'):
    gaxpy_methods = ['gaxpy_col', 'gaxpy_row', 'gaxpy_block']
elif filestem.startswith('gatxpy'):
    gaxpy_methods = ['gatxpy_col', 'gatxpy_row', 'gatxpy_block']
else:
    raise ValueError(f"Unknown filestem: {filestem}")

# Store the results
times = defaultdict(list)

for N in Ns:
    print(f"---------- N = {N:6,d} ----------")

    # Create a large, random, sparse matrix
    M = int(0.9 * N)
    K = int(0.8 * N)
    A = cs.COOMatrix.random(M, N, density, SEED).tocsc()

    if filestem.startswith('gatxpy'):
        A = A.T()

    # Create compatible random, dense matrix for multiplying and adding
    X = cs.COOMatrix.random(N, K, density, SEED)
    Y = cs.COOMatrix.random(M, K, density, SEED)

    # Convert to row and column-major format
    X_col = X.to_dense_vector('F')
    Y_col = Y.to_dense_vector('F')

    X_row = X.to_dense_vector('C')
    Y_row = Y.to_dense_vector('C')

    for method_name in gaxpy_methods:
        args = (X_row, Y_row) if method_name.endswith('row') else (X_col, Y_col)
        method = getattr(A, method_name)

        # Create a partial function with the arguments for timing
        partial_method = partial(method, *args)

        # Run the function
        ts = timeit.repeat(partial_method, repeat=N_repeats, number=N_samples)

        # NOTE timeit returns the total time for all "number" loops for each
        # repeat, so we divide by N_samples to get the time per loop.
        #   len(ts) == N_repeats
        # We only care about the minimum time per loop, since additional time
        # is likely due to other processes running on the system.
        ts = np.array(ts) / N_samples  # time per loop
        ts_min = np.min(ts)

        times[method_name].append(ts_min)

        print(f"{method_name}: {ts_min:.4g} s per loop, "
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
