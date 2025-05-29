#!/usr/bin/env python3
# =============================================================================
#     File: test_basics.py
#  Created: 2025-05-28 15:46
#   Author: Bernie Roesler
#
"""
Test basic COOMatrix and CSCMatrix interface.
"""
# =============================================================================

import numpy as np
import tarfile

from scipy.io import loadmat, hb_read, mmread

from pathlib import Path

# from .helpers import generate_random_matrices, generate_suitesparse_matrices
from helpers import *
# import csparse


if __name__ == "__main__":
    df = get_ss_index()

    # -------------------------------------------------------------------------
    #         Test download process
    # -------------------------------------------------------------------------
    # TODO refactor into a function to loop over all matrices
    fmt = 'mat'  # 'RB' or 'mat' (or 'MAT')
    # k = 6  # ash219
    k = 2137  # JGD_Kocay/Trec4

    # Load the actual matrix
    problem = get_ss_problem(index=df, mat_id=k, fmt=fmt)

    # -------------------------------------------------------------------------
    #         Run the Test
    # -------------------------------------------------------------------------
    # Get the list of the 100 smallest SuiteSparse matrices
    # df['max_dim'] = df[['nrows', 'ncols']].max(axis=1)
    # tf = df.sort_values(by='max_dim').head(100)

    max_dim = df[['nrows', 'ncols']].max(axis=1)
    tf = df.loc[max_dim.sort_values().head(100).index]

    fmt = 'mat'     # 'RB' or 'mat' (or 'MAT')

    # for mat_id in tf['id']:
    #     problem = get_ss_problem(index=df, mat_id=mat_id, fmt=fmt)
    #     print(problem)

    #     A = problem.A

    #     # Check if A is real
    #     if not is_real(A):
    #         print(f"Matrix {mat_id} ({problem.name}) is not real, skipping.")
    #         continue

# =============================================================================
# =============================================================================
