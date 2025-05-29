#!/usr/bin/env python3
# =============================================================================
#     File: helpers.py
#  Created: 2025-05-28 15:44
#   Author: Bernie Roesler
#
"""
Helper functions for the C++Sparse python tests.
"""
# =============================================================================

# import datetime
import pandas as pd
import numpy as np
import re
import requests
import warnings
import tarfile

from dataclasses import dataclass
from pathlib import Path
from scipy import sparse
from scipy.io import loadmat, hb_read, mmread

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


def get_ss_index():
    """Download the SuiteSparse index file load it into a DataFrame.

    Returns
    -------
    index : DataFrame
        Loaded DataFrame containing the SuiteSparse index.
    """
    index_mat = SS_DIR / "ss_index.mat"

    if not index_mat.exists():
        SS_DIR.mkdir(parents=True, exist_ok=True)
        download_file(SS_INDEX_URL, index_mat)

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

    df = pd.DataFrame(data)

    # Create id column at the front
    df['id'] = df.index + 1
    df = df.loc[:, np.roll(df.columns, 1)]

    return df


def get_ss_stats():
    """Download the SuiteSparse statistics file and load it into a DataFrame.

    .. note:: The statistics file is not used in the CSparse testing.
              It is only used by the ``ssget`` Java application.

    Returns
    -------
    index : DataFrame
        Loaded DataFrame containing the SuiteSparse statistics from the
        ``ssstats.csv`` file.
    """
    # Load the secondary index from the CSV file
    stats_csv = SS_DIR / "ssstats.csv"

    if not stats_csv.exists():
        SS_DIR.mkdir(parents=True, exist_ok=True)
        download_file(SSSTATS_CSV_URL, stats_csv)

    # -------------------------------------------------------------------------
    #         Load the CSV file into a DataFrame
    # -------------------------------------------------------------------------
    with open(stats_csv, 'r') as f:
        # First row is the total number of matrices
        f.readline().strip()
        # N_matrices = int(line.split(',')[0])

        # Second row is the last modified date like "31-Oct-2023 18:12:37"
        f.readline().strip()
        # last_modified = datetime.datetime.strptime(line, "%d-%b-%Y %H:%M:%S")

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

    return df


def check_index_vs_csv():
    """Check if the index DataFrame is valid vs. the 'ssstats.csv' file.

    Returns
    -------
    bool
        True if the DataFrame is valid, False otherwise.
    """
    # Load the index from the mat file
    df = get_ss_index()
    df_csv = get_ss_stats()

    # Both give the same results, but df_index has more columns of information
    assert df_csv['name'].equals(df['Name']), "Names do not match"


@dataclass(frozen=True)
class MatrixProblem:
    """A class representing a matrix problem from the SuiteSparse collection.

    Attributes
    ----------
    name : str
        The name of the matrix.
    title : str
        A descriptive title of the matrix.
    A : sparse.sparray
        The sparse matrix in any subclass of `scipy.sparray`.
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
    name:    str = None
    title:   str = None
    A:       sparse.sparray = None
    id:      int = None
    date:    int = None
    author:  str = None
    ed:      str = None
    kind:    str = None
    notes:   str = None

    def __str__(self):
        items = [f"  {key}: {value}" if key != 'A'
                 else f"  {key}: {value.__repr__()}"
                 for key, value in self.__dict__.items()]
        return 'Matrix Problem:\n' + ('\n'.join(items))


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


def load_problem(matrix_path):
    """Load a SuiteSparse matrix problem from a file.

    Parameters
    ----------
    matrix_path : str or Path
        Path to the matrix file. It can be a MatrixMarket (.mtx),
        Rutherford-Boeing (.rb), or MATLAB (.mat) file.

    Returns
    -------
    MatrixProblem
        An instance of `MatrixProblem` containing the matrix and its metadata.
    """
    fmt = matrix_path.suffix

    if fmt == '.mtx':
        A = mmread(matrix_path)
        metadata = parse_header(matrix_path)
    elif fmt == '.rb':
        A = hb_read(matrix_path)
        metadata = parse_header(matrix_path.with_suffix('.txt'))
    elif fmt == '.mat':
        mat = loadmat(matrix_path)
        problem_mat = mat['Problem'][0][0]
        A = problem_mat['A']
        # Metadata is a structured numpy array
        metadata = {k: problem_mat[k].flatten().item()
                    for k in problem_mat.dtype.names
                    if k not in ['A', 'notes']}
        if 'notes' in problem_mat.dtype.names:
            metadata['notes'] = '\n'.join(problem_mat['notes'].tolist())
    else:
        raise ValueError(f"Unknown format: {fmt}")

    return MatrixProblem(A=A, **metadata)


def get_ss_problem(index=None, mat_id=None, group=None, name=None, fmt='mat'):
    """Get a SuiteSparse matrix problem by ID, group, or name.

    Parameters
    ----------
    index : DataFrame
        The DataFrame containing the SuiteSparse index.
    mat_id : int
        The unique identifier of the matrix.
    group : str
        The group name of the matrix.
    name : str
        The name or a pattern matching the name of the matrix.
    fmt : str in {'MM', 'RB', 'mat'}, optional
        The format of the matrix file to return. Defaults to 'mat'.

    Returns
    -------
    MatrixProblem
        The matrix problem instance containing the matrix and its metadata.
    """
    if index is None:
        index = get_ss_index()

    if mat_id is None and (group is None or name is None):
        raise ValueError("One of `mat_id` or the pair "
                         "(`group`, `name`) must be specified.")

    if fmt not in ['MM', 'RB', 'mat']:
        raise ValueError("Format must be one of 'MM', 'RB', 'mat'.")

    if mat_id is not None:
        if group is not None or name is not None:
            warnings.warn("If `mat_id` is specified, "
                          "`group` and `name` are ignored.")

        row = index.set_index('id').loc[mat_id]
    elif group is not None and name is not None:
        row = index.set_index(['Group', 'Name']).loc[group, name]

    return get_ss_problem_from_row(row, fmt=fmt)


def get_ss_problem_from_row(row, fmt='mat'):
    """Get a SuiteSparse matrix problem from a DataFrame row.

    This function is useful for iterating over rows in the SuiteSparse index,
    typically after filtering to a desired subset.

    .. code::
        for index, row in df.iterrows():
            problem = get_ss_problem_from_row(row, fmt='mat')
            A = problem.A
            # ... operate on the matrix ...

    It skips the checks and re-indexing used by `get_ss_problem`, so it is
    faster when iterating.

    Parameters
    ----------
    row : Series
        A row from the SuiteSparse index DataFrame containing the matrix.
    fmt : str in {'MM', 'RB', 'mat'}, optional
        The format of the matrix file to return. Defaults to 'mat'.

    Returns
    -------
    MatrixProblem
        The matrix problem instance containing the matrix and its metadata.
    """
    if fmt not in ['MM', 'RB', 'mat']:
        raise ValueError("Format must be one of 'MM', 'RB', 'mat'.")

    # Get the download path and URL
    has_tar = fmt in ['MM', 'RB']
    tar_ext = '.tar.gz' if has_tar else '.mat'
    path_tail = (Path(fmt) / row['Group'] / row['Name']).with_suffix(tar_ext)
    url = f"{SS_ROOT_URL}/{path_tail.as_posix()}"

    mat_extd = dict(
        MM='.mtx',
        RB='.rb',
        mat='.mat',
    )
    mat_ext = mat_extd[fmt]

    local_tar_path = SS_DIR / path_tail
    local_matrix_file = (
        # add extra directory for MM and RB tar files
        local_tar_path.parent / row['Name'] / row['Name']
        if has_tar
        else local_tar_path
    ).with_suffix(mat_ext)

    if not local_tar_path.exists():
        download_file(url, local_tar_path)

    if has_tar and not local_matrix_file.exists():
        with tarfile.open(local_tar_path, 'r:gz') as tar:
            tar.extractall(path=local_tar_path.parent)
            print(f"Extracted {local_tar_path} to {local_tar_path.parent}")

        # Remove the tar file after extraction
        local_tar_path.unlink()

    return load_problem(local_matrix_file)


# =============================================================================
# =============================================================================
