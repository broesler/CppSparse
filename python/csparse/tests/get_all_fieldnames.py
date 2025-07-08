#!/usr/bin/env python3
# =============================================================================
#     File: get_all_fieldnames.py
#  Created: 2025-06-09 14:48
#   Author: Bernie Roesler
#
"""Get a list of all field names in all SuiteSparse matrices' MAT files."""
# =============================================================================

from dataclasses import fields

from suitesparseget import get_ss_index, get_ss_problem_from_row


if __name__ == "__main__":
    N = 1000
    df = get_ss_index()
    max_dim = df[['nrows', 'ncols']].max(axis=1)
    tf = df.loc[max_dim.sort_values().head(N).index]

    fieldnames = set()

    for _, row in tf.iterrows():
        # Load the actual matrix
        problem = get_ss_problem_from_row(row, fmt='mat')

        # Get all field names
        new_names = {x.name for x in fields(problem)} - fieldnames

        if len(new_names) > 0:
            print("********** Found new field names in "
                  f"{row['group']}/{row['name']}: {new_names}")

        fieldnames.update([x.name for x in fields(problem)])

    print(sorted(fieldnames))


# =============================================================================
# =============================================================================
