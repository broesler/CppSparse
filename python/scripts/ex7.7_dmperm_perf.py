#!/usr/bin/env python3
# =============================================================================
#     File: ex7.7_dmperm_perf.py
#  Created: 2025-07-16 14:57
#   Author: Bernie Roesler
#
"""
Solution to Davis, Exercise 7.7: Compare the run time of `cs_dmperm` with
different values of seed (0, -1, and 1) on a wide range of matrices from real
applications.

Symmetric indefinite matrices arising in optimization problems are of
particular interest (many of the matrices from Gould, Hu, and Scott in the
`GHS_indef` group from the SuiteSparse Matrix Collection, for example).

Find examples where the randomized order, reverse order, and natural order
methods each outperform the other methods (the `boyd2` matrix is an extreme
example).
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import suitesparseget as ssg
import timeit

from functools import partial
# from scipy import sparse
from pathlib import Path
from tqdm import tqdm

import csparse

pkl_file = Path('dmperm_perf.pkl')

# Get the index
ss_index = ssg.get_index()  # main database of matrices
ss_stats = ssg.get_stats()  # additional matrix info

# Filter for symmetric indefinite matrices
conditions = (ss_index['numerical_symmetry'] == 1.0) & (~ss_index['posdef'])

ghs_indef = ss_index.loc[ss_index['group'] == 'GHS_indef']  # 60 matrices
sym_indef = (
    # ss_index.loc[conditions]
    ghs_indef.loc[conditions]
    .merge(ss_stats, on='id', suffixes=(None, '_stats'))
)

# Drop the duplicate '_stats' columns
cols_to_drop = sym_indef.filter(like='_stats').columns
sym_indef = sym_indef.drop(columns=cols_to_drop)

# For each matrix, compute the time taken by cs_dmperm with different seeds
SEEDS = {'reverse': -1, 'natural': 0, 'random': 1}

if pkl_file.exists():
    df = pd.read_pickle(pkl_file)
else:
    df = pd.DataFrame(
        index=sym_indef['id'],
        columns=SEEDS.keys(),
        dtype=float
    )

    N_repeats = 7  # default for %timeit == 7

    for _idx, row in tqdm(sym_indef.iterrows(), total=len(sym_indef)):
        try:
            A = ssg.get_problem(index=ss_index, row=row).A
        except Exception as e:
            print(f"Error loading matrix {row['id']}: {e}")
            continue

        # Compute the dmperm orderings
        for col, seed in tqdm(
            SEEDS.items(),
            desc=f"Processing {row['group']}/{row['name']}",
            leave=False
        ):
            dm_func = partial(csparse.dmperm, A, seed=seed)

            # Determine how many times to repeat the function
            timer = timeit.Timer(dm_func)
            N_samples, _ = timer.autorange()
            ts = timer.repeat(repeat=N_repeats, number=N_samples)
            ts = np.array(ts) / N_samples

            df.loc[row['id'], col] = np.min(ts)

    # Compute the ratios of the times for each seed
    df['natural/reverse'] = df['natural'] / df['reverse']
    df['natural/random'] = df['natural'] / df['random']
    df['random/reverse'] = df['random'] / df['reverse']

    df.to_pickle('dmperm_perf.pkl')


# -----------------------------------------------------------------------------
#         Present Data
# -----------------------------------------------------------------------------
# Merge the results with the original index
sym_indef = sym_indef.merge(df, on='id')

# List top K and bottom K matrices for each seed
K = 5
timing_cols = ['natural', 'reverse', 'random']
ratio_cols = ['natural/reverse', 'natural/random', 'random/reverse']

for col in ratio_cols:
    print(f"----- {col} ratios:")
    print(f"Top {K} matrices:")
    disp_cols = ['id', 'name', col]
    print(sym_indef.nlargest(K, columns=[col])[disp_cols])

    print(f"\nBottom {K} matrices:")
    print(sym_indef.nsmallest(K, columns=[col])[disp_cols])

# Plot histograms of each ratio
# Melt the timing columns for easier plotting
tf = (
    df.reset_index()
    .melt(
        id_vars='id',
        value_vars=ratio_cols,
        var_name='seed',
        value_name='time ratio',
    )
)

# Bin data on a log scale since we're expecting a log normal distribution
# bins = np.logspace(np.log10(tf['time'].min()), np.log10(tf['time'].max()), 20)

fig, ax = plt.subplots(num=1, clear=True)

# sns.histplot(
#     tf,
# 	x='time',
# 	hue='seed',
# 	bins=bins,
# 	stat='probability',
# 	multiple='stack',
#     ax=ax
# )
# ax.set(xscale='log')

sns.kdeplot(
    tf,
	x='time ratio',
	hue='seed',
    log_scale=True,
    common_grid=True,
    ax=ax
)


plt.show()

# =============================================================================
# =============================================================================
