#!/usr/bin/env python3
# =============================================================================
#     File: ex7.6_large_diags.py
#  Created: 2025-12-19 10:34
#   Author: Bernie Roesler
# =============================================================================

"""
Solution to Davis, Exercise 7.6: Heuristics for placing large entries on the
diagonal of a matrix.

This script runs an experiment on random matrices to determine how many
off-diagonal pivot elements are found after using a heuristic for pivoting
large entries to the diagonal.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from tqdm import tqdm

import csparse

FORCE_UPDATE = True
SAVE_FIGS = True

DATA_PATH = Path(__file__).absolute().parent.parent.parent / "plots"
DATA_PATH.mkdir(parents=True, exist_ok=True)

df_file = DATA_PATH / "ex7.6_pivoting.pkl"

# -----------------------------------------------------------------------------
#         Run Tests
# -----------------------------------------------------------------------------
if not FORCE_UPDATE and df_file.exists():
    print("Loading existing results...")
    df = pd.read_pickle(df_file)
else:
    print(f"Running tests for {df_file}...")

    # Compare the number of non-zeros in the factorization
    results = []

    N_trials = 100  # arbitrary number of matrices to test
    N = 100  # arbitrary matrix size (TODO test over multiple?)
    density = 0.1  # arbitrary density

    for i in tqdm(range(N_trials)):
        A = sparse.random_array((N, N), density=density, format="csc")
        C = csparse.permute_large_diag(A)

        for name, M in [("A", A), ("C", C)]:
            for order in ["Natural", "APlusAT"]:
                for tol in [1.0, 1e-3]:
                    lu = csparse.lu(M, order=order, tol=tol)
                    pivots = np.count_nonzero(lu.p != lu.q)
                    results.append(
                        {
                            "trial": i,
                            "matrix": name,
                            "order": order,
                            "tol": tol,
                            "pivots": pivots,
                        }
                    )

    df = (
        pd.DataFrame(results)
        .assign(
            tol=lambda x: x["tol"].astype("category"),
            pivot_ratio=lambda x: x["pivots"] / N,
        )
        .set_index(["trial", "matrix", "order", "tol"])
        .sort_index()
    )
    df.to_pickle(df_file)


# -----------------------------------------------------------------------------
#         Make Plot
# -----------------------------------------------------------------------------
g = sns.catplot(
    data=df,
    x="matrix",
    y="pivot_ratio",
    hue="tol",
    col="order",
    palette="deep",
    kind="strip",
    alpha=0.5,
    aspect=0.5,
)

g.axes[0, 0].set_ylabel("pivots / N")
g.figure.suptitle("Compare C = AQ (Large Entries on Diagonal)")
g.figure.tight_layout()

if SAVE_FIGS:
    fig_file = DATA_PATH / "ex7.6_pivoting.pdf"
    g.figure.savefig(fig_file)
    print(f"Saved figure to {fig_file}")


# =============================================================================
# =============================================================================
