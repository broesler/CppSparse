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
from .csparse import (COOMatrix, CSCMatrix,
                    _davis_example_small, _davis_example_chol,
                    _davis_example_qr)


def davis_example_small(format='csparse_csc'):
    r"""Create a 4x4 example matrix from Davis [0].

    .. code-block:: python
        array([[4.5,   0, 3.2,   0],
               [3.1, 2.9,   0, 0.9],
               [  0, 1.7,   3,   0],
               [3.5, 0.4,   0,   1]])

    Returns
    -------
    A : (4, 4) csparse.CSCMatrix
        The example matrix from Davis.

    References
    ----------
    .. [0] Davis, Timothy A. "Direct Methods for Sparse Linear Systems",
        Eqn (2.1), p. 7-8.
    """
    return _format_matrix(_davis_example_small().tocsc(), format)


def davis_example_chol(format='csparse_csc'):
    """Create an 11x11 example matrix from Davis, Figure 4.2 [0].

    .. code-block:: python
        array([[10.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.],
               [ 0., 11.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
               [ 0.,  1., 12.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
               [ 0.,  0.,  0., 13.,  0.,  1.,  0.,  0.,  0.,  1.,  0.],
               [ 0.,  0.,  0.,  0., 14.,  0.,  0.,  1.,  0.,  0.,  1.],
               [ 1.,  0.,  0.,  1.,  0., 15.,  0.,  0.,  1.,  1.,  0.],
               [ 1.,  0.,  0.,  0.,  0.,  0., 16.,  0.,  0.,  0.,  1.],
               [ 0.,  1.,  0.,  0.,  1.,  0.,  0., 17.,  0.,  1.,  1.],
               [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0., 18.,  0.,  0.],
               [ 0.,  0.,  1.,  1.,  0.,  1.,  0.,  1.,  0., 19.,  1.],
               [ 0.,  0.,  1.,  0.,  1.,  0.,  1.,  1.,  0.,  1., 20.]])

    Returns
    -------
    A : (11, 11) matrix in the specified format
        The example matrix from Davis.

    References
    ----------
    .. [0] Davis, Timothy A. "Direct Methods for Sparse Linear Systems",
        Figure 4.2, p 39.
    """
    return _format_matrix(_davis_example_chol(), format)


def davis_example_qr(format='csparse_csc'):
    r"""Create an 8x8 example matrix from Davis Figure 5.1 [0].

    .. code-block:: python
        array([[1., 0., 0., 1., 0., 0., 1., 0.,]
               [0., 2., 1., 0., 0., 0., 1., 0.,]
               [0., 0., 3., 1., 0., 0., 0., 0.,]
               [1., 0., 0., 4., 0., 0., 1., 0.,]
               [0., 0., 0., 0., 5., 1., 0., 0.,]
               [0., 0., 0., 0., 1., 6., 0., 1.,]
               [0., 1., 1., 0., 0., 0., 7., 1.,]
               [0., 0., 0., 0., 1., 1., 1., 0.,]])

    Returns
    -------
    A : (8, 8) csparse.CSCMatrix
        The example matrix from Davis.

    References
    ----------
    .. [0] Davis, Timothy A. "Direct Methods for Sparse Linear Systems",
        Figure 5.1, p. 74.
    """
    return _format_matrix(_davis_example_qr(), format)


def _format_matrix(A, format):
    """Convert a matrix to the specified format."""
    assert isinstance(A, CSCMatrix), "A must be a CSCMatrix"
    match format:
        case 'csparse_csc':
            return A
        case 'csparse_coo':
            return A.tocoo()
        case 'bsr' | 'coo' | 'csc' | 'csr' | 'dia' | 'dok' | 'lil':
            return to_scipy_sparse(A, format=format)
        case 'ndarray':
            return A.toarray()
        case _:
            raise ValueError(f"Invalid format '{format}'")


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


def from_ndarray(A, format='csc'):
    """Convert a numpy ndarray to a csparse matrix.

    Parameters
    ----------
    x : (M, N) ndarray
        The matrix to convert.
    format : str, optional in {'csc', 'coo'}

    Returns
    -------
    result : (M, N) csparse.CSCMatrix or csparse.COOMatrix
        The matrix in the specified format.
    """
    A = sparse.csc_array(A)
    return from_scipy_sparse(A, format=format)
    

def to_scipy_sparse(A, format='csc'):
    r"""Convert a csparse matrix to a scipy.sparse matrix.

    Parameters
    ----------
    A : (M, N) csparse.CSCMatrix or csparse.COOMatrix
        The matrix to convert.
    format : str, optional in {'bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil'}
        The format of the output matrix.

    Returns
    -------
    result : (M, N) sparse array
        The matrix in the specified format.
    """
    if isinstance(A, COOMatrix):
        A = A.tocsc()
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
