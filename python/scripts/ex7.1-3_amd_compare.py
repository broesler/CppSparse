#!/usr/bin/env python3
# =============================================================================
#     File: ex7.1-3_amd_compare.py
#  Created: 2025-12-16 16:38
#   Author: Bernie Roesler
#
"""
Exercise 7.1-3: Compare the performance of AMD, COLAMD, and AMA for ATA across
our local C++Sparse library and scikit-sparse (SuiteSparse wrapper).
"""
# =============================================================================

from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from scipy.sparse import linalg as spla
from sksparse.amd import amd as sk_amd
from sksparse.cholmod import symbfact
from sksparse.colamd import colamd as sk_colamd
from tqdm import tqdm

from csparse import amd as cs_amd
from utils import measure_perf

SEED = 565656

FORCE_UPDATE = False
SAVE_FIGS = False

DATA_PATH = Path(__file__).absolute().parent.parent.parent / "plots"
DATA_PATH.mkdir(parents=True, exist_ok=True)

# Ns = [100]
Ns = np.unique(np.logspace(1, 4, num=10, dtype=int))


def colamd_quality(A, q):
    """Return the total number of non-zeros in the LU factorization with column
    permutation q.
    """
    lu = spla.splu(A[:, q])
    return lu.L.nnz + lu.U.nnz


df_file = DATA_PATH / "ex7.1-3_amd_compare.pkl"

# -----------------------------------------------------------------------------
#         Run Tests
# -----------------------------------------------------------------------------
if not FORCE_UPDATE and df_file.exists():
    print("Loading existing results...")
    df = pd.read_pickle(df_file)
else:
    print(f"Running benchmarks for {df_file}...")

    # Compare the number of non-zeros in the factorization
    results = []

    # Create a random sparse matrix and time row indexing
    for N in tqdm(Ns):
        density = 0.1 if N <= 100 else (0.01 if N <= 1000 else 0.001)

        # Create a random (possibly symmetric) matrix with specified bandwidth
        A = sparse.random_array((N, N), density=density, rng=SEED)

        # Ensure the diagonal is non-zero
        A.setdiag(A.diagonal() + 1)

        # Keep only entries within the bandwidth
        bw = round(N / 10)
        i, j = A.nonzero()
        keep = np.abs(i - j) <= bw
        A = sparse.csc_array((A.data[keep], (i[keep], j[keep])), shape=(N, N))

        Asym = ((A + A.T) / 2).tocsc()

        amd_funcs = {
            "sksparse": {
                "amd": partial(sk_amd, Asym),
                "colamd": partial(sk_colamd, A),
                "amd_ATA": partial(lambda A: sk_amd((A.T @ A).tocsc()), A),
            },
            "csparse": {
                "amd": partial(cs_amd, Asym),
                "colamd": partial(cs_amd, A, order="ATANoDenseRows"),
                "amd_ATA": partial(cs_amd, A, order="ATA"),
            },
        }

        # Compute the nnz in the LU factorization as a quality metric
        quality_funcs = {
            "amd": lambda A, p: symbfact(A[p[:, np.newaxis], p])[0].sum(),
            "colamd": colamd_quality,
            "amd_ATA": lambda A, q: symbfact(A[:, q], kind="col")[0].sum(),
        }

        for package, funcs in tqdm(amd_funcs.items(), leave=False):
            for name, func in tqdm(funcs.items(), leave=False):
                time, peak_mem = measure_perf(func)
                M = Asym if name == "amd" else A
                quality = quality_funcs[name](M, func())
                results.append(
                    {
                        "package": package,
                        "function": name,
                        "N": N,
                        "time": time,
                        "memory": peak_mem,
                        "quality": quality,
                    }
                )

        # Display a matrix
        if N == max(Ns):
            fig = plt.figure(num=2, clear=True, figsize=(10, 6))
            gs = fig.add_gridspec(
                nrows=len(amd_funcs), ncols=1 + len(amd_funcs["sksparse"])
            )

            ax = fig.add_subplot(gs[:, 0])
            ax.set_title("A Matrix")
            ax.spy(A, markersize=1)
            ax.set_aspect("equal")

            for i, (package, funcs) in enumerate(amd_funcs.items()):
                for j, (name, func) in enumerate(funcs.items()):
                    ax = fig.add_subplot(gs[i, j + 1])
                    p = func()
                    ax.set_title(f"{package} {name.upper()}")
                    ax.spy(A[p[:, np.newaxis], p], markersize=1)
                    ax.set_aspect("equal")

            if SAVE_FIGS:
                fig_file = DATA_PATH / "ex7.1-3_amd_permuted_matrices.pdf"
                fig.savefig(fig_file)
                print(f"Saved figure 2 to {fig_file}")

    # Save results to DataFrame
    df = pd.DataFrame(results).set_index(["package", "function", "N"]).sort_index()
    df.columns.name = "metric"

    df.to_pickle(df_file)


# -----------------------------------------------------------------------------
#        Plot the results
# -----------------------------------------------------------------------------
# NOTE Same plot as below, but using seaborn's relplot (with FacetGrid)
# It actually requires more lines of code and is trickier to customize,
# so just stick with the manual subplots approach.

# df_long = df.reset_index().melt(
#     id_vars=["package", "function", "N"],
#     value_vars=["time", "memory", "quality"],
#     var_name="metric",
#     value_name="value",
# )

# g = sns.relplot(
#     data=df_long,
#     row="metric",
#     col="function",
#     kind="line",
#     x="N",
#     y="value",
#     hue="package",
#     style="package",
#     markers=True,
#     dashes=False,
#     height=3,
#     aspect=1,
#     facet_kws={
#         "sharey": "row",
#         "sharex": True,
#         "margin_titles": True,
#     },
# )

# g.set_titles(row_template="", col_template="{col_name}")
# g.set(xscale="log", yscale="log")
# g.set_axis_labels(x_var="Number of Rows/Columns (N)")

# for i, ax_row in enumerate(g.axes):
#     ax_row[0].set_ylabel(g.row_names[i])
#     for ax in ax_row:
#         ax.grid(True, which="both")

fig, axs = plt.subplots(
    num=1, nrows=3, ncols=3, sharex=True, sharey="row", clear=True,
)
fig.set_size_inches((9.5, 9), forward=True)

for i, col in enumerate(["time", "memory", "quality"]):
    for j, funcname in enumerate(["amd", "colamd", "amd_ATA"]):
        ax = axs[i, j]
        sns.lineplot(
            ax=ax,
            data=df.xs(funcname, level="function"),
            x="N",
            y=col,
            hue="package",
            style="package",
            markers=True,
            dashes=False,
            legend=(i == 1 and j == 2),
        )
        ax.grid(True, which="both")
        ax.set(yscale="log")

        if ax.get_subplotspec().is_first_row():
            ax.set_title(funcname.upper())

        if ax.get_subplotspec().is_last_row():
            ax.set(
                xscale="log",
                xlabel="Number of Rows/Columns (N)",
            )

axs[1, -1].legend(title="package", loc="center left", bbox_to_anchor=(1.01, 0.5))

if SAVE_FIGS:
    fig_file = DATA_PATH / "ex7.1-3_amd_performance.pdf"
    fig.savefig(fig_file)
    print(f"Saved figure 1 to {fig_file}")

# =============================================================================
# =============================================================================
