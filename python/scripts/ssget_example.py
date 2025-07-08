#!/usr/bin/env python3
# =============================================================================
#     File: ssget_example.py
#  Created: 2025-06-10 12:28
#   Author: Bernie Roesler
#
"""Example script to demonstrat the csparse.suitesparseget submodule.

This script gets the index file of the SuiteSparse Matrix Collection,
and then loads in all symmetric non-binary matrices, in increasing order of
number of rows in the matrix.

See Also
--------
CSparse/MATLAB/ssget/ssget_example.m
"""
# =============================================================================

import matplotlib.pyplot as plt

from csparse.tests.suitesparseget import get_ss_index, get_ss_problem, ssweb

import csparse


# Load the index
df = get_ss_index()

# Get the symmetric non-binary matrices
tf = df.loc[df['numerical_symmetry'] == 1 & ~df['is_binary']]

# Sort by number of rows
tf = tf.sort_values('nrows')

fig = plt.figure(num=1, clear=True)
plt.ion()
plt.show()

for _i, row in tf.iterrows():
    print(f"Loading {row['name']} ({row['nrows']} x {row['ncols']})...")
    fig.clear()
    ax = fig.add_subplot()

    # Load the matrix
    problem = get_ss_problem(mat_id=row['id'], index=tf)
    print(problem)

    # Plot the matrix
    csparse.cspy(problem.A, ax=ax)
    ax.set_title(f"{problem.name}\n{problem.title}")

    # Open the web page for the problem
    ssweb(mat_id=problem.id)

    plt.draw()
    input("Press Enter to continue (or Ctrl-C to stop)... ")


# =============================================================================
# =============================================================================
