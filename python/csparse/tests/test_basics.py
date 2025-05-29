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
import numpy as np
import pandas as pd
import re
import requests
import tarfile

from dataclasses import dataclass
from scipy import sparse
from scipy.io import loadmat, hb_read, mmread

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


@dataclass(frozen=True)
class MatrixProblem:
    name:    str = None
    title:   str = None
    A:       sparse.sparray = None
    id:      int = None
    date:    int = None
    author:  str = None
    ed:      str = None
    kind:    str = None
    notes:   str = None


def parse_header(path):
    r"""Parse the header of a SuiteSparse matrix file.

    The top of a MatrixMarket file will look like this:

    .. code::
        %%MatrixMarket matrix coordinate pattern general
        %-------------------------------------------------------------------------------
        % UF Sparse Matrix Collection, Tim Davis
        % http://www.cise.ufl.edu/research/sparse/matrices/HB/ash219
        % name: HB/ash219
        % [UNSYMMETRIC OVERDETERMINED PATTERN OF HOLLAND SURVEY. ASHKENAZI,1974]
        % id: 7
        % date: 1974
        % author: V. Askenazi
        % ed: A. Curtis, I. Duff, J. Reid
        % fields: title A name id date author ed kind
        % kind: least squares problem
        %-------------------------------------------------------------------------------
        % notes:
        % Brute force disjoint product matrices in tree algebra on n nodes, Nicolas Thiery
        % From Jean-Guillaume Dumas' Sparse Integer Matrix Collection,
        % ...
        %-------------------------------------------------------------------------------
        219 85 438
        1 1
        2 1
        3 1
        ...

    The header of a Rutherford-Boeing metadata file will be the same, but
    without the first line.

    This function only parses the leading comment lines for the pattern of
    "key: value", with the exception of the "title" that is in square brackets.

    Parameters
    ----------
    path : str or Path
        Path to the matrix file. It can be a MatrixMarket (.mtx) or
        Rutherford-Boeing metadata (.txt) file.

    Returns
    -------
    dict
        A dictionary containing the parsed metadata. The fields are:

        name : str
            The name of the matrix.
        title : str
            A descriptive title of the matrix.
        id : int
            The unique identifier of the matrix.
        date : int
            The year the matrix was created or last modified.
        author : str
            The author of the matrix or the data.
        ed : str
            Information about the editors or sources.
        kind : str
            The kind of problem from which the matrix arises ('least squares
            problem', 'structural mechanics', etc.)
        notes : str, optional
            Explanatory notes about the matrix.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"File not found: {path}")

    metadata = {}

    # Get the header
    header_lines = []

    with open(path, 'r') as fp:
        # Read the header lines until we find a non-comment line
        for line in fp:
            if not line.startswith('%'):
                break
            header_lines.append(line.strip())

    has_notes = False
    notes_line = None

    for i, line in enumerate(header_lines):
        # Parse the header line
        # Title is the odd one out in square brackets
        title_match = re.search(r'\[([^\]]+)\]', line)
        if title_match:
            metadata['title'] = title_match.group(1).strip()
            continue

        # Match the other key: value pairs
        g = re.match(r'^%\s*([^:]+):(.*)', line)
        if g:
            key = g.group(1).strip().lower()
            value = g.group(2).strip()

            if key == 'http' or key == 'fields':
                continue
            elif key == 'id' or key == 'date':
                # Convert id to int and date (year) to int
                try:
                    value = int(value)
                except ValueError:
                    pass
            elif key == 'notes':
                has_notes = True
                notes_line = i + 1  # Store the line number for notes
                break

            # Add the data to the output struct
            metadata[key] = value

    if has_notes:
        # Read all of the notes into one string
        notes = '\n'.join([line.split('%', 1)[1].lstrip()
                           for line in header_lines[notes_line:]
                           if not line.startswith('%---')])
        metadata['notes'] = notes

    return metadata


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
    fmt = 'MM'  # 'RB' or 'mat' (or 'MAT')

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
            # add extra directory for MM and RB tar files
            (x['local_tar_path'].parent / x['name'] / x['name'])
            if fmt in ['MM', 'RB']
            else x['local_tar_path']
        ).with_suffix(mat_ext),
        axis=1
    )

    # -------------------------------------------------------------------------
    #         Test download process
    # -------------------------------------------------------------------------
    # TODO refactor into a function to loop over all matrices
    k = 6  # ash219
    # k = 2137  # JGD_Kocay/Trec4
    url = df.iloc[k]['url']
    local_tar_path = df.iloc[k]['local_tar_path']

    if not local_tar_path.exists():
        download_file(url, local_tar_path)

    extract = True

    # Load the actual matrix
    matrix_file = df.iloc[k]['local_filename']

    if fmt in ['MM', 'RB'] and extract and not matrix_file.exists():
        with tarfile.open(local_tar_path, 'r:gz') as tar:
            tar.extractall(path=local_tar_path.parent)
            print(f"Extracted {local_tar_path} to {local_tar_path.parent}")

    # TODO load the "problem" metadata into a single object
    # It comes pre-loaded for the .mat file, but we'll have to parse the header
    # of the MatrixMarket files, and the associated *.txt file for the
    # Rutherford-Boeing files. Both fortunately have the same format.

    if fmt == 'MM':
        A = mmread(matrix_file)
        metadata = parse_header(matrix_file)
        problem = MatrixProblem(**dict(A=A, **metadata))
    elif fmt == 'RB':
        A = hb_read(matrix_file)
        metadata = parse_header(matrix_file.with_suffix('.txt'))
        problem = MatrixProblem(**dict(A=A, **metadata))
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
