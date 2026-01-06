#!/usr/bin/env python3
# =============================================================================
#     File: ex8.6_lu_qr_bound.py
#  Created: 2026-01-06 11:14
#   Author: Bernie Roesler
# =============================================================================

"""Exercise 8.6: Experiment with the bounds on |L| and |U| computed in slu."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import suitesparseget as ssg
from tqdm import tqdm

import csparse

FORCE_UPDATE = False
SAVE_FIGS = False

PLOT_PATH = Path(__file__).absolute().parent.parent.parent / "plots"
PLOT_PATH.mkdir(parents=True, exist_ok=True)

DATA_PATH = Path(__file__).absolute().parent / "data"
pkl_file = DATA_PATH / "lu_qr_bound.pkl"

Nprobs = 500  # number of matrices to test

orders = ["Natural", "APlusAT", "ATANoDenseRows"]
tols = [1.0, 1e-3]

# Get the index
ss_index = ssg.get_index()  # main database of matrices

tf = ss_index.loc[
    (ss_index["nrows"] == ss_index["ncols"]) & ss_index["is_real"]
].sort_values("nrows")[:Nprobs]

if not FORCE_UPDATE and pkl_file.exists():
    df = pd.read_pickle(pkl_file)
else:
    results = []

    for _idx, row in tqdm(tf.iterrows(), total=len(tf)):
        try:
            A = ssg.get_problem(index=tf, row=row).A
        except Exception as e:
            print(f"Error loading matrix {row['id']}: {e}")
            continue

        N = A.shape[0]

        # Compute the optimistic estimates of L and U
        lnz_opt = 4 * A.nnz + N

        for order in orders:
            # Compute the QR upper bound of L and U
            lnz_bound, unz_bound, _ = csparse.slu(A, order, qr_bound=True)

            for tol in tols:
                # Compute the actual LU factorization
                lu = csparse.lu(A, order=order, tol=tol)
                lnz, unz = lu.L.nnz, lu.U.nnz

                results.append(
                    {
                        "id": row["id"],
                        "group": row["group"],
                        "name": row["name"],
                        "N": N,
                        "nnz": A.nnz,
                        "order": order,
                        "tol": tol,
                        "lnz_opt": lnz_opt,
                        "lnz_bound": lnz_bound,
                        "unz_bound": unz_bound,
                        "lnz": lnz,
                        "unz": unz,
                    }
                )

    df = pd.DataFrame(results).set_index(["id", "group", "name"]).sort_index()
    df["tol"] = df["tol"].astype("category")
    df.to_pickle(pkl_file)


# -----------------------------------------------------------------------------
#         Plot Data
# -----------------------------------------------------------------------------
# Plot ratios to show in single dimension?
df["lnz_bound_diff"] = (df["lnz_bound"] - df["lnz"]) / df["lnz_bound"]
df["lnz_opt_diff"] = (df["lnz_opt"] - df["lnz"]) / df["lnz_opt"]

fig, ax = plt.subplots(num=1, clear=True)

mf = df.melt(
    id_vars=["N", "nnz", "order", "tol"],
    # value_vars=["lnz_bound", "lnz", "unz_bound", "unz"],
    value_vars=["lnz", "lnz_bound"],
    var_name="type",
).loc[
    lambda x: (x["tol"] == 1e-3)  # & (x["order"] == "ATANoDenseRows")
]

sns.scatterplot(
    data=mf,
    x="nnz",
    y="value",
    hue="order",
    style="type",
    ax=ax,
)

sns.lineplot(
    data=df,
    x="nnz",
    y="lnz_opt",
    color="black",
    marker="o",
    markersize=3,
    markeredgecolor="none",
    alpha=0.5,
    label=r"4|A| + N",
    ax=ax,
)

ax.set(
    title="LU Factorization Nonzeros vs. QR Bound",
    xlabel=r"$|A|$",
    ylabel=r"$|L|$",
    xscale="log",
    yscale="log",
)
ax.grid(True, which="both")

if SAVE_FIGS:
    fig_file = PLOT_PATH / "ex8.6_lu_qr_bounds.pdf"
    fig.savefig(fig_file)
    print(f"Saved figure to {fig_file}")


# Plot relative difference between bounds and actual
fig, ax = plt.subplots(num=2, clear=True)

sf = df.reset_index().melt(
    id_vars=["id", "order", "tol"],
    value_vars=["lnz_bound_diff"],
    var_name="type",
)

sns.stripplot(
    data=sf,
    x="order",
    y="value",
    hue="tol",
    alpha=0.7,
    dodge=True,
    palette="deep",
    ax=ax,
)

ax.legend(loc="lower left", title="tol")
ax.set(
    title="LU Factorization Nonzeros vs. QR Bound",
    ylabel=r"$\frac{|L|_{\text{bound}} - |L|}{|L|}$",
    yscale="log",
)
ax.grid(True)


if SAVE_FIGS:
    fig_file = PLOT_PATH / "ex8.6_lu_rel_bounds.pdf"
    fig.savefig(fig_file)
    print(f"Saved figure to {fig_file}")


# =============================================================================
# =============================================================================
