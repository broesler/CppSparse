#!/usr/bin/env python3
# =============================================================================
#     File: utils.py
#  Created: 2025-02-17 14:13
#   Author: Bernie Roesler
#
"""
Utility functions for the csparse module.
"""
# =============================================================================

import numpy as np

from scipy import sparse
from csparse import COOMatrix, CSCMatrix


def davis_example():
    r"""Create the matrix from Davis,
    "Direct Methods for Sparse Linear Systems" p. 7-8, Eqn (2.1).

    .. math::
        A = \begin{bmatrix}
            4.5 & 0   & 3.2 & 0   \\
            3.1 & 2.9 & 0   & 0.9 \\
            0   & 1.7 & 3   & 0   \\
            3.5 & 0.4 & 0   & 1
        \end{bmatrix}

    Returns
    -------
    A : (4, 4) csparse.CSCMatrix
        The example matrix from Davis.
    """
    N = 4
    rows = [2,    1,    3,    0,    1,    3,    3,    1,    0,    2]
    cols = [2,    0,    3,    2,    1,    0,    1,    3,    0,    1]
    vals = [3.0,  3.1,  1.0,  3.2,  2.9,  3.5,  0.4,  0.9,  4.5,  1.7]
    return COOMatrix(vals, rows, cols, (N, N)).tocsc()


def to_ndarray(A, order='C'):
    r"""Convert a csparse matrix to a numpy ndarray.

    Parameters
    ----------
    A : (M, N) csparse.CSCMatrix
        The matrix to convert.
    order : str, optional in {'C', 'F'}
        The order of the output array.

    Returns
    -------
    result : (M, N) ndarray
        The matrix as a numpy array.
    """
    # NOTE that the .toarray and .reshape orders must be the same, but the
    # np.array argument can be different to specify the in-memory layout.
    #
    # We use the csparse default 'F' order for the .toarray method, because it
    # is column-major like the CSC format, and thus more efficient to convert.
    return (np.array(A.to_dense_vector(order='F'), order=order)
              .reshape(A.shape, order='F'))


def to_scipy_sparse(A, format='csc'):
    r"""Convert a csparse matrix to a scipy.sparse matrix.

    Parameters
    ----------
    A : (M, N) csparse.CSCMatrix
        The matrix to convert.
    format : str, optional in {'bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil'}
        The format of the output matrix.

    Returns
    -------
    result : (M, N) sparse array
        The matrix in the specified format.
    """
    A_sparse = sparse.csc_array((A.data, A.indices, A.indptr), shape=A.shape)
    format_method_name = f"to{format}"
    try:
        format_method = getattr(A_sparse, format_method_name)
    except AttributeError:
        raise ValueError(f"Invalid format '{format}'")
    return format_method()


def from_scipy_sparse(A, format='csc'):
    r"""Convert a scipy.sparse matrix to a csparse matrix.

    Parameters
    ----------
    A : (M, N) sparse array
        The matrix to convert.
    format : str, optional in {'coo', 'csc'}
        The format of the output matrix.

    Returns
    -------
    result : (M, N) csparse.COOMatrix or csparse.CSCMatrix
        The matrix in the csparse format.
    """
    if format == 'coo':
        A = A.tocoo()
        return COOMatrix(A.data, A.row, A.col, A.shape)
    elif format == 'csc':
        A = A.tocsc()
        return CSCMatrix(A.data, A.indices, A.indptr, A.shape)
    else:
        raise ValueError(f"Invalid format '{format}'")


# =============================================================================
# =============================================================================
