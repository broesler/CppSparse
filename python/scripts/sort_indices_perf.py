#!/usr/bin/env python3
# =============================================================================
#     File: sort_indices_perf.py
#  Created: 2025-10-13 13:12
#   Author: Bernie Roesler
#
"""Test and plot the performance of sorting the indices of a sparse matrix."""
# =============================================================================

import timeit
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import csparse


SAVE_FIGS = True

SEED = 565656

filestem = "sort_indices_perf"
fig_path = Path(__file__).absolute().parent.parent.parent / "plots"

# TODO test memory usage. what is expected?

# -----------------------------------------------------------------------------
#         Create the data
# -----------------------------------------------------------------------------
sort_methods = ["sort", "tsort", "qsort"]

N_repeats = 5  # length of output vector from %timeit (5 is default)

# Define the test cases
# Stats from SuiteSparse Matrix Collection:
# >>> import suitesparseget as ssg
# >>> df = ssg.get_index()
# >>> df['ar'] = df['nrows'] / df['ncols']  # tall: ar > 1, wide: ar < 1
# >>> np.count_nonzero(df['ar'] == 1.0) / len(df)
# === np.float64(0.7665289256198347)
# >>> tf = df.loc[df['ar'] != 1.0]
# >>> np.log10(tf['ar']).describe()
# ===
# count    678.000000
# mean      -0.234338
# std        0.701142
# min       -3.228024
# 25%       -0.489519
# 50%       -0.272248
# 75%        0.007986
# max        2.475286
# Name: ar, dtype: float64
# >>> np.log10(tf['ar']).quantile(0.95)
# === np.float64(0.9892992417881823)
# >>> np.log10(tf['ar']).quantile(0.05)
# === np.float64(-1.3740623687218865)
#
# i.e. 90% of non-square matrices have aspect ratios < ~20x

# --- Case 1: Varying density, fixed size
square_N = 10_000
min_log_d = -np.log10(square_N)  # diagonal matrix
ds = np.logspace(min_log_d, -1, 20)

cases = [(square_N, square_N, d) for d in ds]

# --- Case 2: Tall matrices, fixed density
fixed_N = 1_000
Ms = fixed_N * np.r_[1, 2, 5, 10, 20]
fixed_d = 10 / fixed_N  # ~10 nonzeros per column

cases.extend([(M, fixed_N, fixed_d) for M in Ms])

# --- Case 3: Wide matrices, fixed density
M = fixed_N
Ns = Ms

cases.extend([(M, N, fixed_d) for N in Ns])

# Build the DataFrame to hold results
flat_tuples = [(M, N, d, method) for M, N, d in cases for method in sort_methods]
index = pd.MultiIndex.from_tuples(flat_tuples, names=["M", "N", "density", "method"])

df = pd.DataFrame(index=index, columns=["time"]).sort_index()
df = df[~df.index.duplicated(keep="first")]  # remove duplicates


for M, N, density in tqdm(df.index.droplevel("method")):
    # Create a random sparse matrix
    A = csparse.COOMatrix.random(M, N, density=density, seed=SEED).tocsc()

    # Time the sorting of the indices
    for sort_method in sort_methods:
        sort_func = partial(getattr(A, sort_method))

        timer = timeit.Timer(sort_func)
        N_samples, _ = timer.autorange()
        # N_samples = 1  # fast for testing

        ts = timer.repeat(repeat=N_repeats, number=N_samples)
        ts = np.array(ts) / N_samples

        idx = (M, N, density, sort_method)
        df.loc[idx] = np.min(ts)  # time per loop


df = df.reset_index()
df["nnz"] = (df["density"] * df["M"] * df["N"]).astype(int)
df["nnz_per_col"] = df["nnz"] / df["N"]


# -----------------------------------------------------------------------------
#         Plots
# -----------------------------------------------------------------------------
# Plot the expected O(n log (n/N)) and O(n + M + N) scaling
Me = 1000
Nes = np.logspace(1, 5, 5, dtype=int)
des = np.logspace(-3, np.log10(0.3), 100)

nnz = np.outer(des, Me * Nes)
nnz_per_col = nnz / Nes

qsort_time = nnz * np.log2(np.maximum(nnz / Nes, 1))
tsort_time = nnz + Me + Nes

qsort_time[qsort_time == 0] = np.nan

alphas = np.linspace(0.3, 1, len(Nes))

# Plot vs density for varying N
fig, ax = plt.subplots(num=1, clear=True)

for i, N in enumerate(Nes):
    ax.plot(des, qsort_time[:, i], "C0-", alpha=alphas[i], lw=2, label=f"{N=:,d}")
    ax.plot(des, tsort_time[:, i], "C1-", alpha=alphas[i], lw=2)

# Plot dummy lines for legend
ax.plot([], [], "C0-", lw=2, label="qsort: O(n log(n/N))")
ax.plot([], [], "C1-", lw=2, label="tsort: O(n + M + N)")

ax.set(
    title="Expected scaling of sorting methods",
    xlabel=r"density = $\frac{|A|}{MN}$",
    ylabel="time",
    xscale="log",
    yscale="log",
)
ax.legend()
ax.grid(True, which="both")


# -----------------------------------------------------------------------------
#         Plot actual data
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(num=2, clear=True)

sns.lineplot(
    data=df.loc[(df["M"] == df["N"]) & (df["M"] == square_N)],
    x="density",
    y="time",
    hue="method",
    style="method",
    markers=True,
    ax=ax,
)


# Approx. square matrices in SuiteSparse Matrix Collection between 8k and 20k
# >>> import suitesparseget as ssg
# >>> df = ssg.get_index()
# >>> df['density'] = df['nnz'] / (df['nrows'] * df['ncols'])
# >>> tf = df.loc[
#   (df.nrows >= 8_000) & (df.ncols >= 8_000)
#   & (df.nrows <= 20_000) & (df.ncols <= 20_000)
# ]
# >>> tf['density'].quantile(0.5)
# === np.float64(0.0009235722900674114)
# >>> tf['density'].quantile(0.95)
# === np.float64(0.010556930751160046)

ax.axvline(
    x=0.00092,
    color="C3",
    ls="--",
    lw=1,
    label="0.50 Quantile of SuiteSparse Matrix Collection",
)
ax.axvline(
    x=0.01055,
    color="k",
    ls="--",
    lw=1,
    label="0.95 Quantile of SuiteSparse Matrix Collection",
)

ax.grid(True, which="both")
ax.set(
    title=f"Row Index Sorting Performance (M = N = {square_N:,d})",
    xlabel=r"density = $\frac{|A|}{MN}$",
    ylabel="time [s]",
    xscale="log",
    yscale="log",
)

if SAVE_FIGS:
    full_fig_path = fig_path / f"{filestem}_density.pdf"
    try:
        fig.savefig(full_fig_path)
    except Exception as e:
        print(f"Failed to save figure to {full_fig_path}: {e}")

# -----------------------------------------------------------------------------
#         Plot varying M and N
# -----------------------------------------------------------------------------
fig, axs = plt.subplots(num=3, nrows=1, ncols=2, clear=True)
fig.set_size_inches((10, 4.8), forward=True)

ax = axs[0]

sns.lineplot(
    data=df.loc[(df["M"] == fixed_N) & (df["density"] == fixed_d)],
    x="N",
    y="time",
    hue="method",
    style="method",
    markers=True,
    ax=ax,
)

ax.set(
    title=f"Wide Matrices (M={fixed_N:,d}, density={fixed_d})",
    ylabel="time [s]",
    xscale="log",
    yscale="log",
)
ax.grid(True, which="both")


ax = axs[1]

sns.lineplot(
    data=df.loc[(df["N"] == fixed_N) & (df["density"] == fixed_d)],
    x="M",
    y="time",
    hue="method",
    style="method",
    markers=True,
    ax=ax,
)

ax.set(
    title=f"Tall Matrices (N={fixed_N:,d}, density={fixed_d})",
    ylabel="time [s]",
    xscale="log",
    yscale="log",
)
ax.grid(True, which="both")

if SAVE_FIGS:
    full_fig_path = fig_path / f"{filestem}_MN.pdf"
    try:
        fig.savefig(full_fig_path)
    except Exception as e:
        print(f"Failed to save figure to {full_fig_path}: {e}")

plt.show()

# =============================================================================
# =============================================================================
