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

# from .helpers import generate_random_matrices, generate_suitesparse_matrices
from helpers import (get_ss_index, get_ss_problem, get_path_from_row, is_real,
    get_ss_problem_from_file)
# import csparse


def decode_string_data(data_array):
    """Attempt to decode a NumPy array (presumably uint16) into a Python string.
    Handles 1D and 2D cases, trying common MATLAB string encodings/structures.
    """
    if data_array.dtype != np.uint16:
        print(f"  Data is not uint16 (it's {data_array.dtype}), attempting direct conversion or raw display.")
        if isinstance(data_array, bytes):  # should not happen if it's a dataset's value directly
            return data_array.decode('utf-8', errors='ignore').rstrip('\x00')
        if data_array.ndim == 0: # Scalar
            return str(data_array[()])  # get the scalar value
        return str(data_array)

    decoded_str = None
    try:
        if data_array.ndim == 1: # e.g. shape (N,)
            decoded_str = data_array.tobytes().decode('utf-16').rstrip('\x00')
        elif data_array.ndim == 2:
            # Common MATLAB: string stored as column vector of char codes (Length, 1)
            if data_array.shape[1] == 1:
                decoded_str = data_array[:, 0].tobytes().decode('utf-16').rstrip('\x00')
            # Or as a row vector (1, Length)
            elif data_array.shape[0] == 1:
                decoded_str = data_array[0, :].tobytes().decode('utf-16').rstrip('\x00')
            # Or a 2D char array (e.g. from char() in MATLAB)
            # The jumbled output often comes from this if not handled correctly.
            # Try to treat each row as a line, then join. Or try flattening.
            else:
                print(f"  Detected 2D uint16 array (shape {data_array.shape}). Trying multiple strategies:")
                # # Strategy 1: Each row is a string, strip nulls and join
                # lines_row_wise = []
                # for i in range(data_array.shape[0]):
                #     s_row = data_array[i, :].tobytes().decode('utf-16').replace('\x00', '').strip()
                #     lines_row_wise.append(s_row)
                # decoded_str_option1 = "\n".join(lines_row_wise)
                # print(f"    Option 1 (row-wise, joined by newline):\n{decoded_str_option1}")

                # Strategy 1b: Each row is a string, transpose, strip nulls and join
                rows = [
                    (data_array[:, j]
                     .tobytes()
                     .decode('utf-16')
                     .replace('\x00', '')
                     .rstrip())
                    for j in range(data_array.shape[1])
                ]
                decoded_str_option1b = '\n'.join(rows)
                # print(f"    Option 1b (row-wise, joined by newline):\n{decoded_str_option1b}")

                # Strategy 2: Transpose, flatten (often how single long strings stored as 2D char arrays appear)
                # This might be what causes the "e   l   a" if not for UTF-16 handling of nulls
                # flat_transposed_data = data_array.T.flatten()
                # decoded_str_option2 = flat_transposed_data.tobytes().decode('utf-16').replace('\x00', '').strip()
                # print(f"    Option 2 (transpose, flatten, decode):\n{decoded_str_option2}")

                # Strategy 3: Flatten directly
                # flat_data = data_array.flatten()
                # decoded_str_option3 = flat_data.tobytes().decode('utf-16').replace('\x00', '').strip()
                # print(f"    Option 3 (flatten, decode):\n{decoded_str_option3}")

                # Strategy 4: Build from chr(val) for non-nulls (good for weirdly spaced chars)
                # char_list = [chr(val) for val in data_array.flatten() if val != 0]
                # decoded_str_option4 = "".join(char_list)
                # print(f"    Option 4 (chr(val) for non-nulls):\n{decoded_str_option4}")

                # Heuristic: Often the longest non-empty result from options 2,3,4 is good for single strings
                # or option 1 if it has newlines.
                # You might need to pick the best one based on inspection.
                # For now, let's tentatively prefer option 4 if others look like typical jumbling.
                # The jumbled output "e  l  a..." sounds most like strategy 4 would fix it,
                # or strategy 2 if the spaces are actual null bytes in UTF-16.
                # if len(decoded_str_option4) > 0: # Often good if there were odd spacings
                #     decoded_str = decoded_str_option4
                # elif len(decoded_str_option2) > len(decoded_str_option3):
                #     decoded_str = decoded_str_option2
                # else:
                #     decoded_str = decoded_str_option3

                # Make the choice
                # decoded_str = decoded_str_option2
                decoded_str = decoded_str_option1b

                # # If option 1 has multiple lines and looks reasonable, it might be preferred for char arrays
                # if '\n' in decoded_str_option1 and len(decoded_str_option1) > (decoded_str if decoded_str else ""):
                #     print("    (Option 1 seems like multiple lines, might be preferable)")
                #     # decoded_str = decoded_str_option1 # Uncomment if this is the desired structure

    except UnicodeDecodeError as ude:
        print(f"  UnicodeDecodeError: {ude}. Trying 'utf-16-le'...")
        try:
            if data_array.ndim == 1:
                decoded_str = data_array.tobytes().decode('utf-16-le').rstrip('\x00')
            elif data_array.ndim == 2: # Simplified for LE attempt
                decoded_str = data_array.flatten().tobytes().decode('utf-16-le').rstrip('\x00')
        except Exception as e_le:
            print(f"  Error decoding as utf-16-le: {e_le}")
    except Exception as e:
        print(f"  Could not decode as UTF-16 string: {e}")

    return decoded_str if decoded_str is not None else str(data_array)



if __name__ == "__main__":
    df = get_ss_index()

    # -------------------------------------------------------------------------
    #         Test download process
    # -------------------------------------------------------------------------
    # mat_id = 6  # arc130 -> TODO has ['Problem']['Zeros'] attribute in .mat file
    # mat_id = 7  # ash219
    # mat_id = 2137  # JGD_Kocay/Trec4 has 'notes'

    # mat_id = 449
    # group = 'Grund'  # has RHS 'b' field
    # name = 'b1_ss'

    # Load the actual matrix
    # problem = get_ss_problem(
    #     index=df,
    #     mat_id=mat_id,
    #     fmt='mat'
    # )

    # This .mat file is v7.3, which we need h5py to process:
    # PosixPath('/Users/bernardroesler/.ssgetpy/mat/Mycielski/mycielskian2.mat')

    # # Load the actual matrix
    # problem = get_ss_problem(
    #     index=df,
    #     group='Mycielski',
    #     name='mycielskian2',
    #     fmt='mat'
    # )

    row = df.set_index(['Group', 'Name']).loc['Mycielski', 'mycielskian2']
    row['Group'] = 'Mycielski'
    row['Name'] = 'mycielskian2'

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

    # Build the sparse matrix
    problem['A'] = sparse.csc_array(
        (problem['A']['data'],
         problem['A']['ir'],
         problem['A']['jc'])
    )

    # Join the notes into a single string
    problem['notes'] = '\n'.join([x.rstrip() for x in problem['notes']])

    print('---------- read_mat:')
    print(problem)

    # -------------------------------------------------------------------------
    #         Write our own parser with hd5py
    # -------------------------------------------------------------------------
    # TODO
    # Since we know which fields are present and their datatypes, we can just
    # parse them directly from the h5py file without needing to use mat73 or
    # pymatreader. It is more code, but it avoids the dependencies and
    # allows us to control the parsing process more precisely.


    # # -------------------------------------------------------------------------
    # #         Run the Test
    # # -------------------------------------------------------------------------
    # # Get the list of the 100 smallest SuiteSparse matrices
    # # df['max_dim'] = df[['nrows', 'ncols']].max(axis=1)
    # # tf = df.sort_values(by='max_dim').head(100)

    # N = 10
    # max_dim = df[['nrows', 'ncols']].max(axis=1)
    # tf = df.loc[max_dim.sort_values().head(N).index]

    # for idx, row in tf.iterrows():
    #     print('-------------------')
    #     try:
    #         problem = get_ss_problem_from_row(row, fmt='mat')
    #         print(problem)
    #     except NotImplementedError as e:
    #         print(f"Skipping matrix {idx} due to: {e}")
    #         continue

    #     A = problem.A

    #     # Check if A is real
    #     if not is_real(A):
    #         print(f"Matrix {mat_id} ({problem.name}) is not real, skipping.")
    #         continue

# =============================================================================
# =============================================================================
