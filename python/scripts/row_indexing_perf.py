# ==============================================================================
#     File: row_indexing.py
#  Created: 2025-04-26 20:27
#   Author: Bernie Roesler
#
"""Solution to Davis Exercise 2.30.

Experment with row indexing of sparse matrices in python. Does it use binary
search or linear search? Does it take advantage of special cases like A[1, :]
or A[M, :]?
"""
#
# ==============================================================================

import matplotlib.pyplot as plt
import numpy as np
import timeit

from tqdm import tqdm
from scipy import sparse

SAVE_FIGS = False

# Ms = np.r_[10, 20, 50, 100]
Ms = np.r_[10, 20, 50, 100, 200, 500, 1000]
# Ms = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]  # 5000 is SLOW
density = 0.2

times = np.zeros(len(Ms))
log_times = np.zeros(len(Ms))

fig1, ax = plt.subplots(num=1, clear=True)

# Create a random sparse matrix and time row indexing
with tqdm(total=len(Ms), desc="M", leave=False) as pbar0:
    for k, M in enumerate(Ms):
        A = sparse.random_array((M, M), density=density, format='csc')

        # Time row indexing for each row and average
        row_times = np.zeros(M)

        with tqdm(total=M, desc="i", leave=False) as pbar1:
            for i in range(M):
                ts = timeit.repeat(lambda A=A, i=i: A[i, :], repeat=5, number=7)
                row_times[i] = np.mean(ts)
                pbar1.update(1)

        times[k] = np.mean(row_times)

        if M == 1000:
            # Plot the distribution of times for each row
            # hist(mean(row_times, 1))

            # Plot the of times for each row
            ax.scatter(0, row_times[0], marker='x', c='C3')
            ax.scatter(np.arange(1, M-1), row_times[1:-1], marker='.', c='C0')
            ax.scatter(M-1, row_times[-1], marker='x', c='C3')
            ax.set(xlabel='Row index',
                ylabel='Time to index row (s)',
                title=f"Density = {density:0.2f}")

        # Compute the sum of the log of each column size
        # col_sizes = np.sum(A ~= 0, axis=1)
        # log_times[k] = np.mean(np.log(col_sizes))

        pbar0.set_postfix(M=M)
        pbar0.update(1)

# ------------------------------------------------------------------------------
#        Plot the results
# ------------------------------------------------------------------------------
fig2, ax = plt.subplots(num=2, clear=True)

# loglog(Ms, log_times, 'x-')
ax.plot(Ms, times, 'o-', label='Time to index row')
ax.plot(Ms, Ms * times[0] / Ms[0], '.-', label='Linear scaling')

ax.legend()
ax.set(
    xscale='log',
    yscale='log',
    xlabel='Matrix size M',
    ylabel='Time to index row (s)',
)
ax.grid(True, which='both')

if SAVE_FIGS:
    fig1.savefig('../../plots/row_indexing_distribution_py.png')
    fig2.savefig('../../plots/row_indexing_scaling_py.png')


# ==============================================================================
# ==============================================================================
