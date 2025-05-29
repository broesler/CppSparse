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

import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import scipy.linalg as la
import tarfile

from scipy import sparse
from scipy.io import loadmat, hb_read, mmread
from scipy.sparse import linalg as spla

from pathlib import Path

# from .helpers import generate_random_matrices, generate_suitesparse_matrices
import csparse

# TODO refactor into a package
# Skip ssgetpy and implement my own using pandas.

# TODO move this to a config.py file
SS_DIR = Path.home() / ".ssgetpy"
SS_ROOT_URL = "https://sparse.tamu.edu"
SS_INDEX_URL = f"{SS_ROOT_URL}/files/ss_index.mat"
SSSTATS_CSV_URL = f"{SS_ROOT_URL}/files/ssstats.csv"


def download_file(url, path):
    """Download a file from a URL and save it to the specified path."""
    try:
        # Make any subdirectories
        path.parent.mkdir(parents=True, exist_ok=True)
        # Make the request to download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()  # raise an error for bad responses
        with open(path, 'wb') as fp:
            for chunk in response.iter_content(chunk_size=8192):
                fp.write(chunk)
        print(f"Downloaded {url} to {path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        raise e


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    #         Download the MAT and CSV Files
    # -------------------------------------------------------------------------
    index_mat = SS_DIR / "ss_index.mat"
    stats_csv = SS_DIR / "ssstats.csv"

    if not index_mat.exists():
        SS_DIR.mkdir(parents=True, exist_ok=True)
        download_file(SS_INDEX_URL, index_mat)

    if not stats_csv.exists():
        SS_DIR.mkdir(parents=True, exist_ok=True)
        download_file(SSSTATS_CSV_URL, stats_csv)

    # -------------------------------------------------------------------------
    #         Load the mat file
    # -------------------------------------------------------------------------
    mat = loadmat(index_mat)
    ss_index = mat['ss_index'][0][0]  # structured numpy array

    # NOTE ss_index is a `np.void` structured array object.
    # {'LastRevisionDate', 'DownloadTimeStamp'} are singletons, but every other
    # element is either (2904,), (1, 2904) or (2904, 1) shaped.
    col_names = ss_index.dtype.names
    data = {}

    for col in col_names:
        col_data = ss_index[col]

        if col_data.size > 1:
            col_data = col_data.flatten()

            if col_data.dtype == np.object_:
                # String arrays are nested, so unpack them
                col_data = [item.item() for item in col_data]

            data[col] = col_data

    df_index = pd.DataFrame(data)

    # Create id column at the front
    df_index['id'] = df_index.index + 1
    df_index = df_index.loc[:, np.roll(df_index.columns, 1)]

    # -------------------------------------------------------------------------
    #         Load the CSV file into a DataFrame
    # -------------------------------------------------------------------------
    with open(stats_csv, 'r') as f:
        # First row is the total number of matrices
        line = f.readline().strip()
        N_matrices = int(line.split(',')[0])
        # print(f"Total number of matrices: {first_line}")

        # Second row is the last modified date like "31-Oct-2023 18:12:37"
        line = f.readline().strip()
        last_modified = datetime.datetime.strptime(line, "%d-%b-%Y %H:%M:%S")

        # Read the rest of the CSV into a DataFrame (see 'ssgetpy/csvindex.py`)
        columns = [
            'group',
		    'name',
		    'nrows',
		    'ncols',
		    'nnz',
		    'is_real',
		    'is_logical',
            'is_2d3d',
		    'is_spd',
		    'pattern_symmmetry',
		    'numerical_symmetry',
		    'kind',
		    'pattern_entries'
        ]

        df = pd.read_csv(f, header=None, names=columns)

    # Add id column up front
    df['id'] = df.index + 1
    df = df.loc[:, np.roll(df.columns, 1)]

    # Both give the same results, but df_index has more columns of information
    assert df['name'].equals(df_index['Name']), "Names do not match"

    # Create the matrix url
    fmt = 'mat'  # 'RB' or 'mat' (or 'MAT')

    if fmt in ['MM', 'RB']:
        ext = '.tar.gz'
    elif fmt.lower() == 'mat':
        ext = '.mat'
    else:
        raise ValueError(f"Unknown format: {fmt}")

    directory = fmt.lower() if fmt == 'MAT' else fmt

    df['path'] = df.apply(
        lambda x: (Path(directory) / x['group'] / x['name']).with_suffix(ext),
        axis=1
    )

    df['local_tar_path'] = df.apply(
        lambda x: SS_DIR / x['path'],
        axis=1
    )

    df['url'] = df.apply(
        lambda x: f"{SS_ROOT_URL}/{x['path'].as_posix()}",
        axis=1
    )

    mat_ext = '.mtx' if fmt == 'MM' else ('.rb' if fmt == 'RB' else '.mat')
    df['local_filename'] = df.apply(
        lambda x: (
            (
                x['local_tar_path'].parent /
                x['name'] /  # add extra directory for MM and RB
                x['name']
            ).with_suffix(mat_ext)
            if fmt in ['MM', 'RB']
            else x['local_tar_path'].with_suffix(mat_ext)
        ),
        axis=1
    )

    # -------------------------------------------------------------------------
    #         Test download process
    # -------------------------------------------------------------------------
    # TODO refactor into a function to loop over all matrices
    k = 6  # ash219
    url = df.iloc[k]['url']
    local_tar_path = df.iloc[k]['local_tar_path']

    if not local_tar_path.exists():
        download_file(url, local_tar_path)

    extract = True

    if extract and fmt in ['MM', 'RB']:
        with tarfile.open(local_tar_path, 'r:gz') as tar:
            tar.extractall(path=local_tar_path.parent)
            print(f"Extracted {local_tar_path} to {local_tar_path.parent}")

    # Load the actual matrix
    matrix_file = df.iloc[k]['local_filename']

    # TODO load the "problem" metadata
    # It comes pre-loaded for the .mat file, but we'll have to parse the header
    # of the MatrixMarket files, and the associated *.txt file for the
    # Rutherford-Boeing files. Both fortunately have the same format.

    if fmt == 'MM':
        A = mmread(matrix_file)
    elif fmt == 'RB':
        A = hb_read(matrix_file)
    elif fmt.lower() == 'mat':
        mat = loadmat(matrix_file)
        problem = mat['Problem'][0][0]
        A = problem['A']


    # -------------------------------------------------------------------------
    #         Get the list of the 100 smallest SuiteSparse matrices
    # -------------------------------------------------------------------------
    df['max_dim'] = df[['nrows', 'ncols']].max(axis=1)
    df_index['max_dim'] = df_index[['nrows', 'ncols']].max(axis=1)

    tf = df.sort_values(by='max_dim').head(100)
    tf_index = df_index.sort_values(by='max_dim').head(100)

# =============================================================================
# =============================================================================
