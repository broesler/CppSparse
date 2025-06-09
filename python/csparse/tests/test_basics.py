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
import h5py

from pymatreader import read_mat
from scipy import sparse

# from .helpers import generate_random_matrices, generate_suitesparse_matrices
from helpers import (get_ss_index, get_ss_problem, get_path_from_row, #is_real,
    get_ss_problem_from_file, get_ss_problem_from_row)
# import csparse


if __name__ == "__main__":
    df = get_ss_index()

    # -------------------------------------------------------------------------
    #         Test download process
    # -------------------------------------------------------------------------
    # mat_id = 6     # arc130 TODO has 'Zeros'
    # mat_id = 7     # ash219
    # mat_id = 2137  # JGD_Kocay/Trec4 has 'notes'
    # mat_id = 449   # Grund/b1_ss  has 'b'
    # mat_id = 1759  # Meszaros/refine has 'b' as list?
    # mat_id = 2396  # Newman/dolphins has 'aux' (single item)
    # mat_id = 1487  # Pajek/GD95_c has 'aux' (list of ['nodename', 'coords'])
    # NOTE all Pajek matrices seem to be graphs with 'aux' data.

    # Load the actual matrix
    # problem = get_ss_problem(
    #     index=df,
    #     mat_id=mat_id,
    #     fmt='mat'
    # )

    # This .mat file is v7.3, which we need h5py to process:
    # PosixPath('/Users/bernardroesler/.ssgetpy/mat/Mycielski/mycielskian3.mat')

    # # Load the actual matrix
    # problem = get_ss_problem(
    #     index=df,
    #     group='Mycielski',
    #     name='mycielskian3',
    #     fmt='mat'
    # )

    row = df.set_index(['Group', 'Name']).loc['Mycielski', 'mycielskian3']
    row['Group'] = 'Mycielski'
    row['Name'] = 'mycielskian3'

    matrix_file = get_path_from_row(row, fmt='mat')

    # problem = get_ss_problem_from_file(matrix_file)

    # NOTE the h5py file has two keys {'#refs#', 'Problem'}
    # '#refs#' is a special key used by MATLAB to store references, so we don't
    # touch that one.
    # 'Problem' is a group that contains the actual matrix data and metadata,
    # keys: ['notes', 'author', 'date', 'ed', 'id', 'kind', 'name', 'title']

    # -------------------------------------------------------------------------
    #         Load the matrix using read_mat
    # -------------------------------------------------------------------------
    # NOTE read_mat does the best job at decoding the 'notes' field, but it
    # does not handle the sparse matrix 'A' correctly. It is the cleanest
    # option in terms of minimum lines of code, but there is a dependency on
    # pymatreader, which is not a standard library, 
    mat = read_mat(matrix_file)
    problem = mat['Problem']

    # Join the notes into a single string
    problem['notes'] = '\n'.join([x.rstrip() for x in problem['notes']])

    print('---------- read_mat:')
    print(problem)


    # -------------------------------------------------------------------------
    #         Run the Test
    # -------------------------------------------------------------------------
    # Get the list of the 100 smallest SuiteSparse matrices
    N = 100
    max_dim = df[['nrows', 'ncols']].max(axis=1)
    tf = df.loc[max_dim.sort_values().head(N).index]

    for idx, row in tf.iterrows():
        print('-------------------')
        try:
            problem = get_ss_problem_from_row(row, fmt='mat')
            print(problem)
        except NotImplementedError as e:
            print(f"Skipping matrix {idx} due to: {e}")
            continue

        A = problem.A

        # # Check if A is real
        # if not is_real(A):
        #     print(f"Matrix {mat_id} ({problem.name}) is not real, skipping.")
        #     continue

# =============================================================================
# =============================================================================
