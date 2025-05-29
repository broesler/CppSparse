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

# from .helpers import generate_random_matrices, generate_suitesparse_matrices
from helpers import get_ss_index, get_ss_problem, get_ss_problem_from_row
# import csparse


if __name__ == "__main__":
    df = get_ss_index()

    # -------------------------------------------------------------------------
    #         Test download process
    # -------------------------------------------------------------------------
    # TODO refactor into a function to loop over all matrices
    # k = 6  # ash219
    k = 2137  # JGD_Kocay/Trec4

    # This matrix has a 'b' array parameter for an RHS
    # Test how MM and RB handle this value
    # PosixPath('/Users/bernardroesler/.ssgetpy/mat/Grund/b1_ss.mat')

    # Load the actual matrix
    problem = get_ss_problem(index=df, mat_id=k, fmt='mat')

    # -------------------------------------------------------------------------
    #         Run the Test
    # -------------------------------------------------------------------------
    # Get the list of the 100 smallest SuiteSparse matrices
    # df['max_dim'] = df[['nrows', 'ncols']].max(axis=1)
    # tf = df.sort_values(by='max_dim').head(100)

    N = 10
    max_dim = df[['nrows', 'ncols']].max(axis=1)
    tf = df.loc[max_dim.sort_values().head(N).index]

    for index, row in tf.iterrows():
        print('-------------------')
        try:
            problem = get_ss_problem_from_row(row, fmt='mat')
            print(problem)
        except NotImplementedError as e:
            print(f"Skipping matrix {index} due to: {e}")
            continue

    #     A = problem.A
    #     # Check if A is real
    #     if not is_real(A):
    #         print(f"Matrix {mat_id} ({problem.name}) is not real, skipping.")
    #         continue

# =============================================================================
# =============================================================================
