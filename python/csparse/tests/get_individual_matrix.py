#!/usr/bin/env python3
# =============================================================================
#     File: get_individual_matrix.py
#  Created: 2025-06-09 15:43
#   Author: Bernie Roesler
#
"""Test script to download and print a specific SuiteSparse matrix problem."""
# =============================================================================

from suitesparseget import get_ss_index, get_ss_problem

# import csparse


if __name__ == "__main__":
    df = get_ss_index()

    # -------------------------------------------------------------------------
    #         Test download process
    # -------------------------------------------------------------------------
    # mat_id = 6     # arc130 has 'Zeros'
    # mat_id = 7     # ash219  no special features
    # mat_id = 382   # DRIVCAV/cavity03 has 'x'
    # mat_id = 2137  # JGD_Kocay/Trec4 has 'notes'
    # mat_id = 449   # Grund/b1_ss  has 'b'
    # mat_id = 2396  # Newman/dolphins has 'aux' (single item)
    # mat_id = 1487  # Pajek/GD95_c has 'aux' (list of ['nodename', 'coords'])
    # mat_id = 1457  # Pajek/Cities has 'aux' data
    #                  (list of ['cluster', 'colname', 'rowname'])
    # NOTE all Pajek matrices seem to be graphs with 'aux' data.
    # mat_id = 1440  # Oberwolfach/LFATS 'notes' is one line
    mat_id = 2734  # VDOL/spaceStation_1 'aux' 'rowname' is long list of strings

    problem = get_ss_problem(
        index=df,
        mat_id=mat_id,
        fmt='mat'
    )

    print(problem)

# =============================================================================
# =============================================================================
