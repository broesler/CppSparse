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

import pytest

# import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re
import requests
import warnings
import webbrowser
import tarfile

from dataclasses import dataclass
from pathlib import Path
from pymatreader import read_mat
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

    df = df.rename(columns={
        'Group': 'group',
        'Name': 'name',
        'isBinary': 'is_binary',
        'isReal': 'is_real',
        'RBtype': 'rb_type',
        'isND': 'is_nd',
        'isGraph': 'is_graph'
    })

    for col in df.columns:
        if df[col].dtype == np.uint8:
            df[col] = df[col].astype(bool)

        if col.startswith('amd_'):
            df[col] = df[col].astype(int)

    df['group'] = df['group'].astype('category')
    df['rb_type'] = df['rb_type'].astype('category')

    df['nnz'] = df['nnz'].astype(int)
    df['nentries'] = df['nentries'].astype(int)

    # >>> df.info()
    # <class 'pandas.core.frame.DataFrame'>
    # RangeIndex: 2904 entries, 0 to 2903
    # Data columns (total 31 columns):
    #  #   Column              Non-Null Count  Dtype
    # ---  ------              --------------  -----
    #  0   id                  2904 non-null   int64
    #  1   group               2904 non-null   category
    #  2   name                2904 non-null   object
    #  3   nrows               2904 non-null   int32
    #  4   ncols               2904 non-null   int32
    #  5   nnz                 2904 non-null   int64
    #  6   nzero               2904 non-null   int32
    #  7   pattern_symmetry    2904 non-null   float64
    #  8   numerical_symmetry  2904 non-null   float64
    #  9   is_binary           2904 non-null   bool
    #  10  is_real             2904 non-null   bool
    #  11  nnzdiag             2904 non-null   int32
    #  12  posdef              2904 non-null   bool
    #  13  amd_lnz             2904 non-null   int64
    #  14  amd_flops           2904 non-null   int64
    #  15  amd_vnz             2904 non-null   int64
    #  16  amd_rnz             2904 non-null   int64
    #  17  nblocks             2904 non-null   int32
    #  18  sprank              2904 non-null   int32
    #  19  rb_type             2904 non-null   category
    #  20  cholcand            2904 non-null   bool
    #  21  ncc                 2904 non-null   int32
    #  22  is_nd               2904 non-null   bool
    #  23  is_graph            2904 non-null   bool
    #  24  lowerbandwidth      2904 non-null   int32
    #  25  upperbandwidth      2904 non-null   int32
    #  26  rcm_lowerbandwidth  2904 non-null   int32
    #  27  rcm_upperbandwidth  2904 non-null   int32
    #  28  xmin                2904 non-null   complex128
    #  29  xmax                2904 non-null   complex128
    #  30  nentries            2904 non-null   int64
    # dtypes: bool(6), category(2), complex128(2), float64(2), int32(11), int64(7), object(1)
    # memory usage: 478.4+ KB

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
    with open(stats_csv, 'r') as fp:
        # First row is the total number of matrices
        fp.readline().strip()
        # N_matrices = int(line.split(',')[0])

        # Second row is the last modified date like "31-Oct-2023 18:12:37"
        fp.readline().strip()
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

        df = pd.read_csv(fp, header=None, names=columns)

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
    assert df_csv['name'].equals(df['name']), "Names do not match"


@dataclass(frozen=True)
class MatrixProblem:
    """A class representing a matrix problem from the SuiteSparse collection.

    Attributes
    ----------
    id : int
        The unique identifier of the matrix.
    name : str
        The name of the matrix.
    title : str
        A descriptive title of the matrix.
    date : int
        The year the matrix was created or last modified.
    author : str
        The author of the matrix or the data.
    ed : str
        Information about the editors or sources.
    kind : str
        The kind of problem from which the matrix arises ('least squares
        problem', 'structural mechanics', etc.)
    A : sparse.sparray
        The sparse matrix in any subclass of `scipy.sparray`.
    Zeros : sparse.sparray
        A sparse matrix representing the locations of explicit zeros in `A`.
        The values in `Zeros` are all 1.0 so that sparse operations work.
    x : np.ndarray
        The solution vector or matrix, if available.
    b : sparse.sparray
        A right-hand side vector or matrix, if available.
    aux : dict, optional
        Auxiliary data that may include additional metadata or information
    notes : str, optional
        Explanatory notes about the matrix.
    """
    id:      int = None
    name:    str = None
    title:   str = None
    date:    int = None
    author:  str = None
    ed:      str = None
    kind:    str = None
    A:       sparse.sparray = None
    Zeros:   sparse.sparray = None
    x:       np.ndarray = None
    b:       np.ndarray = None
    aux:     dict = None
    notes:   str = None

    def __str__(self):
        def format_value(value):
            """Format the value for display."""
            if isinstance(value, sparse.sparray):
                return repr(value)
            elif isinstance(value, np.ndarray):
                return f"{value.shape} ndarray of dtype '{value.dtype}'"
            elif isinstance(value, list):
                return (f"({len(value)},) list of "
                        f"[{', '.join({type(v).__name__ for v in value})}]")
            elif isinstance(value, dict):
                # recursively format the aux dict
                return ('{\n' + ', \n'.join(
                    f"    {k}: {format_value(v)}" for k, v in value.items()
                ) + '\n}')
            else:
                return str(value)

        items = [
            f"{key}: {format_value(value)}"
            for key, value in self.__dict__.items()
        ]

        return '\n'.join(items)

    def __repr__(self):
        return f"<{self.__class__.__name__}:\n{self.__str__()}>"


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

            if key.startswith('http') or key == 'fields':
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


def load_matfile_ltv73(matrix_path):
    """Load a MAT-file with version < 7.3 using the scipy.io.loadmat.

    Parameters
    ----------
    matrix_path : str or Path
        Path to the MATLAB .mat file.

    Returns
    -------
    data : dict
        A dictionary containing the parsed data from the MAT-file.
    """
    mat = loadmat(
        matrix_path,
        squeeze_me=True,
        spmatrix=False    # return coo_array instead of coo_matrix
    )

    # `mat` will be a dictionary-like structure with MATLAB variables
    problem_mat = mat['Problem']

    # problem_mat is a structured numpy array of arrays, so get the
    # individual items as a dictionary
    data = {
        k: problem_mat[k].item()
        for k in problem_mat.dtype.names
        if k not in ['aux', 'notes']
    }

    # aux is another structured array, so convert it to a dictionary
    if 'aux' in problem_mat.dtype.names:
        aux = problem_mat['aux'].item()
        data['aux'] = {k: aux[k].item() for k in aux.dtype.names}

    # notes is a multi-line string (aka 2D character array)
    if 'notes' in problem_mat.dtype.names:
        notes = problem_mat['notes'].item()
        if isinstance(notes, str):
            data['notes'] = notes.rstrip()
        elif isinstance(notes, np.ndarray):
            # notes is an array of strings, join them
            data['notes'] = '\n'.join([x.rstrip() for x in notes.tolist()])
        else:
            raise ValueError(f"Unexpected type for notes: {type(notes)}")

    return data


def load_matfile_gev73(matrix_path):
    """Load a MAT-file with version >= 7.3 using the scipy.io.loadmat.

    Parameters
    ----------
    matrix_path : str or Path
        Path to the MATLAB .mat file.

    Returns
    -------
    data : dict
        A dictionary containing the parsed data from the MAT-file.
    """
    # Use the HDF5 interface
    mat = read_mat(matrix_path)

    data = mat['Problem']

    data['id'] = int(data['id'])

    # notes is a multi-line string (aka 2D character array)
    if 'notes' in data:
        data['notes'] = '\n'.join([x.rstrip() for x in data['notes']])

    return data


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
        rhs_path = matrix_path.with_stem(matrix_path.stem + '_b')
        b = mmread(rhs_path) if rhs_path.exists() else None
        metadata = parse_header(matrix_path)

        return MatrixProblem(A=A, b=b, **metadata)

    elif fmt == '.rb':
        try:
            A = hb_read(matrix_path)
        except ValueError as e:
            print(f"RB error: {e}")
            raise NotImplementedError(e)

        # RHS is in MatrixMarket format
        rhs_path = (
            matrix_path
            .with_stem(matrix_path.stem + '_b')
            .with_suffix('.mtx')
        )
        b = mmread(rhs_path) if rhs_path.exists() else None

        metadata = parse_header(matrix_path.with_suffix('.txt'))

        return MatrixProblem(A=A, b=b, **metadata)

    elif fmt == '.mat':
        # NOTE scipy.io.loadmat does not support v7.3+ MAT-files
        try:
            data = load_matfile_ltv73(matrix_path)
        except NotImplementedError:
            data = load_matfile_gev73(matrix_path)

        return MatrixProblem(**data)

    else:
        raise ValueError(f"Unknown format: {fmt}")


def get_ss_row(index=None, mat_id=None, group=None, name=None):
    """Get a SuiteSparse matrix row by ID, or group and name.

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
    Series
        The row from the SuiteSparse index DataFrame containing the matrix.
    """
    if index is None:
        index = get_ss_index()

    if mat_id is None and (group is None or name is None):
        raise ValueError("One of `mat_id` or the pair "
                         "(`group`, `name`) must be specified.")

    if mat_id is not None:
        if group is not None or name is not None:
            warnings.warn("If `mat_id` is specified, "
                          "`group` and `name` are ignored.")

        row = index.set_index('id').loc[mat_id]
        row['id'] = mat_id
    elif group is not None and name is not None:
        row = index.set_index(['group', 'name']).loc[group, name]
        row['group'] = group
        row['name'] = name

    return row


# FIXME either re-name this function or refactor it, since "get_path" does not
# imply actually downloading a file.
def get_path_from_row(row, fmt='mat'):
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
    path_tail = (Path(fmt) / row['group'] / row['name']).with_suffix(tar_ext)
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
        local_tar_path.parent / row['name'] / row['name']
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

    return local_matrix_file


def get_ss_problem(index=None, mat_id=None, group=None, name=None, fmt='mat'):
    """Get a SuiteSparse matrix problem by ID, or group and name.

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
    if fmt not in ['MM', 'RB', 'mat']:
        raise ValueError("Format must be one of 'MM', 'RB', 'mat'.")
    row = get_ss_row(index=index, mat_id=mat_id, group=group, name=name)
    return get_ss_problem_from_row(row, fmt=fmt)


def get_ss_problem_from_row(row, fmt='mat'):
    """Get a SuiteSparse matrix problem from a DataFrame row.

    Parameters
    ----------
    row : Series
        A row from the SuiteSparse index DataFrame containing the matrix.
    fmt : str in {'MM', 'RB', 'mat'}, optional
        The format of the matrix file to download. Defaults to 'mat'.

    Returns
    -------
    MatrixProblem
        The matrix problem instance containing the matrix and its metadata.
    """
    matrix_file = get_path_from_row(row, fmt=fmt)
    return load_problem(matrix_file)


def get_ss_problem_from_file(matrix_file):
    """Get a SuiteSparse matrix problem from a file path.

    Parameters
    ----------
    matrix_file : str or Path
        The path to the matrix file. It can be a MatrixMarket (.mtx),
        Rutherford-Boeing (.rb), or MATLAB (.mat) file.

    Returns
    -------
    MatrixProblem
        The matrix problem instance containing the matrix and its metadata.
    """
    return load_problem(matrix_file)


def ssweb(index=None, mat_id=None, group=None, name=None):
    """Open the SuiteSparse web page for a matrix in the browser.

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
    """
    row = get_ss_row(index=index, mat_id=mat_id, group=group, name=name)
    web_url = f"{SS_ROOT_URL}/{row['group']}/{row['name']}"
    try:
        webbrowser.open(web_url, new=0, autoraise=True)
    except webbrowser.Error as e:
        print(f"Error opening web page: {e}")
        raise e


# -----------------------------------------------------------------------------
#         Matrix Generators
# -----------------------------------------------------------------------------
def generate_suitesparse_matrices(N=100, real_only=True, square_only=False):
    """Generate a list of SuiteSparse matrices."""
    df = get_ss_index()

    # Get the list of the N smallest SuiteSparse matrices
    max_dim = df[['nrows', 'ncols']].max(axis=1)
    tf = df.loc[max_dim.sort_values().index]

    filters = (
        (tf['is_real'] if real_only else True) &
        (tf['nrows'] == tf['ncols'] if square_only else True)
    )

    tf = tf[filters]

    for idx, row in tf.head(N).iterrows():
        try:
            problem = get_ss_problem_from_row(row, fmt='mat')
        except NotImplementedError as e:
            print(f"Skipping matrix {idx} due to: {e}")
            continue

        yield pytest.param(
            problem,
            id=f"{problem.id}::{problem.name}",
            marks=pytest.mark.suitesparse
        )


def generate_random_matrices(
    seed=565656,
    N_trials=100,
    N_max=10,
    square_only=True,
    d_scale=1
):
    """Generate a list of random sparse matrices of maximum size N x N."""
    rng = np.random.default_rng(seed)
    for trial in range(N_trials):
        # Generate a random sparse matrix
        if square_only:
            M = N = rng.integers(1, N_max, endpoint=True)
        else:
            M, N = rng.integers(1, N_max, size=2, endpoint=True)

        d = d_scale * rng.random()  # density

        A = sparse.random_array(
            (M, N),
            density=d,
            format='csc',
            random_state=rng
        )

        print(f"Random matrix {trial} ({seed=}): "
              f"{A.shape} with {A.nnz} non-zeros")

        yield pytest.param(A, id=f"random_{trial}", marks=pytest.mark.random)


def generate_random_compatible_matrices(
    seed=565656, N_trials=100, N_max=10, kind='multiply'
):
    """Generate a list of random sparse matrices with compatible shapes."""
    rng = np.random.default_rng(seed)

    for trial in range(N_trials):
        # Generate a random sparse matrix
        M, N, K = rng.integers(1, N_max, size=3, endpoint=True)
        d = rng.random()  # density ∈ [0, 1]

        if kind == 'multiply':
            A_shape = (M, N)
            B_shape = (N, K)
        elif kind == 'add':
            A_shape = B_shape = (M, N)

        A = sparse.random_array(
            A_shape,
            density=d,
            format='csc',
            random_state=rng,
            data_sampler=rng.normal
        )

        B = sparse.random_array(
            B_shape,
            density=d,
            format='csc',
            random_state=rng,
            data_sampler=rng.normal
        )

        yield pytest.param(
            A, B,
            id=f"random_{trial}",
            marks=pytest.mark.random
        )


def generate_random_cholesky_matrices(seed=565656, N_trials=100, N_max=100):
    """Generate a list of random, square, lower-triangular matrices."""
    rng = np.random.default_rng(seed)

    for trial in range(N_trials):
        # Generate a random sparse matrix
        N = rng.integers(1, N_max, endpoint=True)
        d = 0.1 * rng.random()  # density ∈ [0, 0.1]

        A = sparse.random_array(
            (N, N),
            density=d,
            format='csc',
            random_state=rng,
            data_sampler=rng.normal
        )

        # Make it lower triangular
        D = sparse.diags(rng.random(N), 0, shape=(N, N))
        L = sparse.tril(A, -1) + D

        # RHS column vector
        b = sparse.random_array(
            (N, 1),
            density=d,
            format='csc',
            random_state=rng,
            data_sampler=rng.normal
        )

        yield pytest.param(
            L, b,
            id=f"random_{trial}",
            marks=pytest.mark.random
        )


def generate_pvec_params(seed=565656, N_trials=100, N_max=10):
    """Generate random permutation vectors and values."""
    rng = np.random.default_rng(seed)
    for i in range(N_trials):
        print(f"Trial {i+1} with seed {seed}")
        M = rng.integers(1, N_max, endpoint=True)
        p = rng.permutation(M)
        x = rng.random(M)
        yield pytest.param(
            p, x,
            id=f"trial_{i+1}",
            marks=pytest.mark.random
        )


# -----------------------------------------------------------------------------
#         Matrix Type Checking
# -----------------------------------------------------------------------------
def is_real(A):
    """Check if a sparse matrix is real-valued.

    Parameters
    ----------
    A : sparse.sparray
        The sparse matrix to check.

    Returns
    -------
    bool
        True if the matrix is real-valued, False otherwise.
    """
    return np.issubdtype(A.dtype, np.floating)


def is_complex(A):
    """Check if a sparse matrix is complex-valued.

    Parameters
    ----------
    A : sparse.sparray
        The sparse matrix to check.

    Returns
    -------
    bool
        True if the matrix is complex-valued, False otherwise.
    """
    return np.issubdtype(A.dtype, np.complexfloating)


def is_valid_permutation(p):
    """Check if a vector is a valid permutation."""
    return np.array_equal(np.sort(p), np.arange(len(p)))


# -----------------------------------------------------------------------------
#         Test Classes
# -----------------------------------------------------------------------------
class BaseSuiteSparseTest:
    """An abstract base class for tests."""
    @pytest.fixture(scope='class')
    def problem(self, request):
        """Fixture to provide the problem matrix."""
        return request.param

    @pytest.fixture(scope='class', autouse=True)
    def base_setup_problem(self, request, problem):
        """Setup method to initialize the problem matrix."""
        cls = request.cls

        if isinstance(problem, MatrixProblem):
            cls.problem = problem
        elif isinstance(problem, sparse.sparray):
            A = problem
            cls.problem = MatrixProblem(
                id=f"random_{A.shape[0]}_{A.shape[1]}_{A.nnz}",
                name=f"Random {A.shape}, {A.nnz} nnz",
                A=A
            )
        else:
            raise TypeError(f"Expected MatrixProblem or sparse.sparray, "
                            f"got {type(problem)}")

        print(f"Testing matrix {cls.problem.id} ({cls.problem.name})")
        # Subclasses should override this method to set up the problem


class BaseSuiteSparsePlot(BaseSuiteSparseTest):
    """An abstract base class for tests that require a plot."""
    # Default values for parameters
    _nrows = 1
    _ncols = 1
    _fig_dir = Path('test_suitesparse')
    _fig_title_prefix = ''

    @pytest.fixture(scope='class', autouse=True)
    def setup_plot(self, request, base_setup_problem):
        """Set up the problem and figure for plotting across tests."""
        cls = request.cls

        cls.make_figures = request.config.getoption('--make-figures')

        if not cls.make_figures:
            yield  # skip the setup if not making figures
            return

        cls.fig, cls.axs = plt.subplots(
            num=1,
            nrows=cls._nrows,
            ncols=cls._ncols,
            clear=True
        )
        cls.fig.suptitle(f"{cls._fig_title_prefix}{cls.problem.name}")
        cls.fig.set_size_inches((3 * cls._ncols, 4 * cls._nrows))

        def finalize_plot():
            """Finalize the plot after all tests."""
            if cls.make_figures:
                # Save the figure
                cls.fig_dir = Path('test_figures') / cls._fig_dir
                os.makedirs(cls.fig_dir, exist_ok=True)

                cls.figure_path = (
                    cls.fig_dir /
                    f"{cls.problem.name.replace('/', '_')}.pdf"
                )
                print(f"Saving figure to {cls.figure_path}")
                cls.fig.savefig(cls.figure_path)

            # Clean up
            if hasattr(cls, 'fig') and cls.fig is not None:
                plt.close(cls.fig)
                del cls.fig
                del cls.axs

        # Make sure to finalize the plot after all tests
        request.addfinalizer(finalize_plot)

        # Run the tests
        yield

# =============================================================================
# =============================================================================
