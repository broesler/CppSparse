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


SAVE_FIG = True

SEED = 565656

filestem = 'gaxpy_perf_py'
# filestem = 'gatxpy_perf_py'

# -----------------------------------------------------------------------------
#         Create the data
# -----------------------------------------------------------------------------
# Ns = np.r_[10, 100, 1000]
Ns = np.r_[10, 20, 50, 100, 200, 500, 1000, 2000, 5000]

density = 0.25  # density of the matrix

N_repeats = 1   # number of "runs" in %timeit (7 is default)
N_samples = 1  # number of samples in each run (100,000 is default)

# TODO include the transpose versions and plot as subfigs
gaxpy_methods = ['gaxpy_col', 'gaxpy_row', 'gaxpy_block']

# Store the results
times = defaultdict(lambda: {'mean': [], 'std_dev': []})

for N in Ns:
    print(f"---------- N = {N:6,d} ----------")

    # Create a large, random, sparse matrix
    M = int(0.9 * N)
    K = int(0.8 * N)
    A = cs.COOMatrix.random(M, N, density, SEED).tocsc()

    # Create compatible random, dense matrix for multiplying and adding
    X = cs.COOMatrix.random(N, K, density, SEED)
    Y = cs.COOMatrix.random(M, K, density, SEED)

    # Convert to row and column-major format
    X_col = X.toarray('F')
    Y_col = Y.toarray('F')

    X_row = X.toarray('C')
    Y_row = Y.toarray('C')

    for method_name in gaxpy_methods:
        args = (X_row, Y_row) if method_name.endswith('row') else (X_col, Y_col)
        method = getattr(A, method_name)

        # Create a partial function with the arguments for timing
        partial_method = partial(method, *args)

        # Run the function (len(ts) == N_repeats)
        ts = timeit.repeat(partial_method, repeat=N_repeats, number=N_samples)

        ts_mean = np.mean(ts)
        ts_std = np.std(ts)

        times[method_name]['mean'].append(ts_mean)
        times[method_name]['std_dev'].append(ts_std)

        print(f"{method_name}: {ts_mean:.4g} ± {ts_std:.4g} s per loop, "
              f"({N_repeats} runs, {N_samples} loops each)")


# -----------------------------------------------------------------------------
#         Plot the data
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(num=1, clear=True)
fig.set_size_inches(6.4, 4.8, forward=True)
for i, (key, val) in enumerate(times.items()):
    ax.errorbar(Ns, val['mean'],
                yerr=val['std_dev'], ecolor=f"C{i}", fmt='.-', label=key)

ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(which='both')
ax.legend()

ax.set_xlabel('Number of Columns')
ax.set_ylabel('Time (s)')
ax.set_title(f"{filestem.split('_')[0]}, density {density}")

plt.show()


if SAVE_FIG:
    fig_fullpath = Path(f"../plots/{filestem}.png")
    try:
        fig.savefig(fig_fullpath)
        print(f"Saved figure to {fig_fullpath}.")
    except Exception as e:
        print(f"Could not save figure to {fig_fullpath}: {e}")
        raise e

# =============================================================================
# =============================================================================
