#!/usr/bin/env python3
# =============================================================================
#     File: _fillreducing.py
#  Created: 2025-04-22 19:03
#   Author: Bernie Roesler
#
"""
Python implementation of fill-reducing algorithms, as presented in Davis,
Chapter 7.
"""
# =============================================================================

import numpy as np

from scipy import sparse
from scipy.sparse import linalg as spla

from .csparse import CSCMatrix, amd, dmperm, lu_solve, qr_solve
from .utils import to_scipy_sparse


def fiedler(A):
    """Compute the Fiedler vector of a connected graph.

    The Fiedler vector is the eigenvector corresponding to the second smallest
    eigenvalue of the Laplacian of `A + A.T`.

    Parameters
    ----------
    A : (M, N) sparse array
        Matrix of M vectors in N dimensions, corresponding to a connected
        graph.

    Returns
    -------
    p : (M,) ndarray
        The permutation vector obtained when `v` is sorted.
    v : (M,) ndarray
        The Fiedler vector of the graph.
    d : float
        The second smallest eigenvalue of the Laplacian of `A + A.T`.
    """
    N = A.shape[1]

    if N < 2:
        return np.ones(N), np.ones(N), 0.0

    # Compute the structure of the Laplacian matrix
    Ab = A.astype(bool)
    S = Ab + Ab.T + sparse.eye_array(N)

    # Create a diagonal matrix with the sum of each column
    D = sparse.diags(S.sum(axis=0))

    # Compute the Laplacian matrix itself
    L = D - S

    # see also:
    # L = sparse.csgraph.laplacian(A)

    # Get the eigenvalues and eigenvectors of the Laplacian matrix
    λ, x = spla.eigsh(L, k=2, which='SA', tol=np.sqrt(np.finfo(float).eps))

    # Take the second smallest eigenvalue and its corresponding eigenvector
    d = λ[1]
    v = x[:, 1]
    p = np.argsort(v)

    return p, v, d


def edge_separator(A):
    """Compute an edge-separator of a symmetric matrix.

    See: Davis, §7.6, `cs_esep`.

    The edge-separator `s` splits the graph of `A` into two parts `a` and `b`
    of roughly equal size.

    The edge separator is the set of entries in `A[a][:, b]`.

    Parameters
    ----------
    A : (M, N) sparse array
        Matrix of M vectors in N dimensions.

    Returns
    -------
    a, b : ndarray of int
        The indices of the two parts of the edge-separator.
    """
    p = sparse.csgraph.reverse_cuthill_mckee(A, symmetric_mode=True)
    k = A.shape[1] // 2
    return p[:k], p[k:]


def node_from_edge_sep(A, a, b):
    """Convert an edge separator into a node separator.

    See: Davis, §7.6, `cs_sep`.

    The inputs `a` and `b` are a partition of `[0, N)`, thus the edges in
    `A[a][:, b]` are an edge separator of `A`. `sep` returns `s`, the node
    separator, consisting of a *node cover* of the edges of `A[a][:, b]`; `a_s`
    and `b_s`, the sets `a` and `b` with `s` removed.

    Parameters
    ----------
    A : (M, N) sparse array
        Matrix of M vectors in N dimensions.
    a, b : ndarray of int
        The indices of the two parts of the edge-separator.

    Returns
    -------
    s : ndarray of int
        The indices of the node separator.
    a_s, b_s : ndarray of int
        The sets `a` and `b` with `s` removed.
    """
    Ap = A[a][:, b]  # permute the matrix
    # dmperm requires a CSCMatrix
    res = dmperm(CSCMatrix(Ap.data, Ap.indices, Ap.indptr, Ap.shape))
    p, q, s, cc, rr = res.p, res.q, res.s, res.cc, res.rr
    s = np.r_[a[p[:rr[1]]], b[q[cc[2]:cc[4]]]]
    w = np.ones(A.shape[1]).astype(bool)
    w[s] = False
    a_s = a[w[a]]
    b_s = b[w[b]]
    return s, a_s, b_s


def node_separator(A):
    """Find a node separator of a symmetric matrix.

    See: Davis, §7.6, `cs_nsep`.

    The node-separator `s` splits the graph of `A` into two parts `a` and `b`
    of roughly equal size. If `A` is unsymmetric, use
    `node_separator(A + A.T)`. The permutation `p = np.r_[a, b, s]` is
    a one-level dissection of `A`.

    Parameters
    ----------
    A : (N, N) sparse matrix
        A square symmetric matrix.

    Returns
    -------
    s : ndarray of int
        The indices of the node separator.
    a, b : ndarray of int
        The indices of the two parts of the node-separator.
    """
    a, b = edge_separator(A)
    return node_from_edge_sep(A, a, b)


def nested_dissection(A):
    """Generalized nested dissection ordering of a matrix.

    See: Davis, §7.6, `cs_nd`.

    The nested dissection ordering is a recursive algorithm that finds a
    permutation of the rows and columns of `A` that minimizes the fill-in
    during LU factorization. Small submatrices (order 500 or less) are ordered
    via `amd`. `A` must be sparse and symmetric (use `A + A.T` if `A` is
    unsymmetric).

    Parameters
    ----------
    A : (N, N) sparse matrix
        A square symmetric matrix.

    Returns
    -------
    p : (N,) ndarray of int
        The permutation vector.
    """
    N = A.shape[0]

    if N == 1:
        return np.array([0])
    elif N < 500:
        # amd requires a CSCMatrix
        Ac = CSCMatrix(A.data, A.indices, A.indptr, A.shape)
        return amd(Ac, order='APlusAT')
    else:
        # Compute the node separator
        s, a, b = node_separator(A)

        # Recursively compute the nested dissection ordering
        a = a[nested_dissection(A[a][:, a])]
        b = b[nested_dissection(A[b][:, b])]

        # Concatenate the permutations
        p = np.r_[a, b, s]

        # Return the permutation vector
        return p


def dm_solve(A, b):
    """Solve `Ax = b` using the Dulmage-Mendelsohn decomposition.

    `A` may be rectangular and/or structurally rank deficient. `b` is a dense
    vector.

    Parameters
    ----------
    A : (M, N) array_like
        Matrix of M vectors in N dimensions
    b : (M,) array_like
        Right-hand side vector

    Returns
    -------
    x : (N,) ndarray
        Solution vector. If `A` is overdetermined, `x` is the least-squares
        solution.
    """
    M, N = A.shape

    # dmperm requires a CSCMatrix
    if isinstance(A, CSCMatrix):
        A = to_scipy_sparse(A)
    elif not isinstance(A, sparse.csc_array):
        A = sparse.csc_array(A)

    res = dmperm(CSCMatrix(A.data, A.indices, A.indptr, A.shape))
    p, q, cc, rr = res.p, res.q, res.cc, res.rr

    # Permute the matrix and the right-hand side
    C = A[p][:, q]
    b = b[p]
    x = np.zeros(N)

    # Solve the system
    if rr[2] <= M and rr[3] <= N:
        Cp = C[rr[2]:, rr[3]:]
        Cpc = CSCMatrix(Cp.data, Cp.indices, Cp.indptr, Cp.shape)
        x[cc[3]:] = qr_solve(Cpc, b[rr[2]:], order='ATA')
        b[:rr[2]] -= C[:rr[2], cc[3]:] @ x[cc[3]:]

    if rr[1] < rr[2] and cc[2] < cc[3]:
        Cp = C[rr[1]:rr[2], cc[2]:cc[3]]
        Cpc = CSCMatrix(Cp.data, Cp.indices, Cp.indptr, Cp.shape)
        x[cc[2]:cc[3]] = lu_solve(Cpc, b[rr[1]:rr[2]], order='ATANoDenseRows')
        b[:rr[1]] -= C[:rr[1], cc[2]:cc[3]] @ x[cc[2]:cc[3]]

    if rr[1] > 0 and cc[2] > 0:
        Cp = C[:rr[1], :cc[2]]
        Cpc = CSCMatrix(Cp.data, Cp.indices, Cp.indptr, Cp.shape)
        x[:cc[2]] = qr_solve(Cpc, b[:rr[1]], order='ATA')

    x[q] = x  # inverse permute the solution

    return x


# =============================================================================
# =============================================================================
