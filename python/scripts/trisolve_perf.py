#!/usr/bin/env python3
# =============================================================================
#     File: trisolve_perf.py
#  Created: 2025-01-11 10:57
#   Author: Bernie Roesler
#
"""
Plot the triangular solve performance data.
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

filestem = 'trisolve_perf_py'

# -----------------------------------------------------------------------------
#         Create the data
# -----------------------------------------------------------------------------
trisolve_funcs = [cs.lsolve, cs.usolve, cs.lsolve_opt, cs.usolve_opt]

N = 1000  # size of the square matrix

densities = np.r_[
    0.001, 0.01, 0.1, 0.2, 0.5, 1.0
]

N_repeats = 7    # number of "runs" in %timeit (7 is default)
N_samples = 100  # number of samples in each run (100,000 is default)

times = defaultdict(list)

rng = np.random.default_rng(SEED)

for density in densities:
    print(f"---------- Density = {density:6.2g} ----------")

    # Create a random matrix
    A = cs.COOMatrix.random(N, N, density, SEED).tocsc()

    # Ensure all diagonals are non-zero so that L is non-singular
    for i in range(N):
        A[i, i] = 1.0

    # Take the lower and upper triangular
    L = A.band(-N, 0)
    U = L.T()

    # Create a dense column vector that is the sum of the rows of L
    bL = np.r_[L.sum_rows()]
    bU = np.r_[U.sum_rows()]

    # Create sparse RHS vectors by removing random elements
    idx_zero = rng.choice(N, int((1 - density) * N), replace=False)
    bL[idx_zero] = 0.0
    bU[idx_zero] = 0.0

    for func in trisolve_funcs:
        func_name = func.__name__
        args = (L, bL) if func_name.startswith('l') else (U, bU)
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
fig, axs = plt.subplots(num=1, nrows=2, sharex=True, clear=True)
fig.set_size_inches(6.4, 8, forward=True)
fig.suptitle(f"{filestem.split('_')[0]}, N = {N}")

colors = {
    'lsolve': 'C0',
    'usolve': 'C1',
    'lsolve_opt': 'C0',
    'usolve_opt': 'C1',
}

linestyles = {
    'lsolve': '-',
    'usolve': '-',
    'lsolve_opt': '--',
    'usolve_opt': '--',
}

ax = axs[0]
for i, (key, val) in enumerate(times.items()):
    ax.plot(densities, val,
            marker='.', color=colors[key], ls=linestyles[key], label=key)

# ax.set_xscale('log')
# ax.set_yscale('log')
ax.grid(which='both')
ax.legend()

ax.set_ylabel('Time (s)')

# Plot the difference between the two methods
ax = axs[1]
for i, k in enumerate(['l', 'u']):
    key = f"{k}solve"
    opt_key = key + '_opt'

    mean = np.r_[times[key]]
    opt_mean = np.r_[times[opt_key]]
    rel_diff = (mean - opt_mean) / mean

    ax.plot(densities, rel_diff,
            marker='.', color=colors[key], ls=linestyles[key], label=key)

ax.grid(which='both')
ax.legend()

ax.set_xlabel('Density of Matrix and RHS vector')
ax.set_ylabel('Relative Difference between Original and Optimized')

plt.show()

if SAVE_FIG:
    fig_fullpath = Path(f"../plots/{filestem}_N{N}.png")
    try:
        fig.savefig(fig_fullpath)
        print(f"Saved figure to {fig_fullpath}.")
    except Exception as e:
        print(f"Could not save figure to {fig_fullpath}: {e}")
        raise e

# =============================================================================
# =============================================================================
