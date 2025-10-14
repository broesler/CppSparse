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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import csparse

SAVE_FIGS = False

SEED = 565656

filestem = "sort_indices_perf_py"

# TODO test memory usage. what is expected?

# -----------------------------------------------------------------------------
#         Create the data
# -----------------------------------------------------------------------------
sort_methods = ["sort", "tsort", "qsort"]

N_repeats = 7  # number of "runs" in %timeit (7 is default)

# --- Case 1: Varying density, fixed size
square_N = 10_000
min_log_d = -np.log10(square_N)  # diagonal matrix
ds = np.logspace(min_log_d, np.log10(0.1), 20)

cases = [(square_N, square_N, d) for d in ds]

# --- Case 2: Tall matrices, fixed density
fixed_N = 10_000
# Ms = fixed_N * np.r_[1, 2, 5, 10, 20, 50, 100]
Ms = fixed_N * np.r_[1, 2, 5, 10]
fixed_d = 0.001

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
        N_samples, _ = timer.autorange()  # TODO
        # N_samples = 1  # fast testing

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
    data=df.loc[df["M"] == df["N"]],
    x="density",
    y="time",
    hue="method",
    style="method",
    markers=True,
    ax=ax,
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
    fig.savefig(f"{filestem}_density.png")

# -----------------------------------------------------------------------------
#         Plot varying M and N
# -----------------------------------------------------------------------------
fig, axs = plt.subplots(num=3, nrows=1, ncols=2, clear=True)
fig.set_size_inches((10, 4.8), forward=True)

ax = axs[0]

sns.lineplot(
    data=df.loc[df["M"] == fixed_N],
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
    data=df.loc[df["N"] == fixed_N],
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
    fig.savefig(f"{filestem}_M_N.png")

plt.show()

# =============================================================================
# =============================================================================
