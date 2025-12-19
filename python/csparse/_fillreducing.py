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

import warnings

import numpy as np
from scipy import linalg as la
from scipy import sparse
from scipy.sparse import linalg as spla

from .csparse import amd, dmperm, lu_solve, qr_solve, scc


def profile(A):
    r"""Compute the profile of a sparse, symmetric matrix.

    The matrix *profile*, also called the *envelope*, is a measure of how close
    the entries of `A` are to the diagonal. It is defined as:

    .. math::
        \text{profile}(A) = \sum_{j=0}^{N-1} (j - \min \mathcal{A}_{*j})

    where `N` is the number of columns in `A`.

    This function assumes that `A` is symmetric, and that the diagonal is
    non-zero.

    Parameters
    ----------
    A : (N, N) sparse array
        A square symmetric matrix, with non-zero diagonal.

    Returns
    -------
    result : int
        The matrix profile of `A`.

    See Also
    --------
    bandwidth : Compute the bandwidth of a sparse matrix.
    scipy.linalg.bandwidth : Compute the upper and lower bandwidths of a dense
        matrix.
    """
    return np.sum(diag_dist(A))


def bandwidth(A):
    r"""Compute the bandwidth of a sparse, symmetric matrix.

    The matrix *bandwidth* is a measure of how close the entries of `A` are to
    the diagonal. It is defined as:

    .. math::
        \text{bandwidth}(A) = \max_j (j - \min \mathcal{A}_{*j})

    where `N` is the number of columns in `A`.

    This function assumes that `A` is symmetric, and that the diagonal is
    non-zero.

    Parameters
    ----------
    A : (N, N) sparse array
        A square symmetric matrix, with non-zero diagonal.

    Returns
    -------
    result : int
        The bandwidth of `A`.

    See Also
    --------
    matrix_profile : Compute the matrix profile of a sparse matrix.
    scipy.linalg.bandwidth : Compute the upper and lower bandwidths of a dense
        matrix.
    """
    return np.max(diag_dist(A))


def diag_dist(A):
    """Compute the distance to the diagonal of the first non-zero in each column. 

    Parameters
    ----------
    A : (N, N) sparse array
        A square symmetric matrix.

    Returns
    -------
    result : (N,) ndarray
        The distance to the diagonal of the first non-zero in each column.
    """
    if not sparse.issparse(A):
        raise ValueError("Matrix must be sparse.")

    M, N = A.shape

    if M != N:
        raise ValueError("Matrix must be square.")

    if (A != A.T).nnz > 0:
        warnings.warn(
            "Matrix is not symmetric; results may be incorrect.",
            UserWarning,
            stacklevel=2
        )

    # Ensure the diagonal is non-zero so that every column has at least one
    # entry and A.indptr is well-defined
    A = sparse.csc_array(A)
    A.setdiag(1.0)
    A.has_sorted_indices = False
    A.sort_indices()

    # Get the minimum row index for each column
    min_row = A.indices[A.indptr[:-1]]

    return np.arange(N) - min_row


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

    if L.nnz == 0:
        raise ValueError(
            "The Laplacian matrix is empty; the graph may not be connected."
        )

    # Get the eigenvalues and eigenvectors of the Laplacian matrix
    try:
        λ, x = spla.eigsh(L, k=2, which='SA', tol=np.sqrt(np.finfo(float).eps))
    except TypeError:
        # k must be < N for sparse.linalg.eigsh
        λ, x = la.eigh(L.toarray())
        λ = λ[:2]     # take the two smallest eigenvalues
        x = x[:, :2]  # and their corresponding eigenvectors

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
    p, q, r, s, cc, rr = dmperm(A[a][:, b])
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
        return amd(A, order='APlusAT')
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

    p, q, r, s, cc, rr = dmperm(A)

    # Permute the matrix and the right-hand side
    C = A[p[:, np.newaxis], q]
    b = b[p]
    x = np.zeros(N)

    # Backsolve the upper block triangular system
    # [ C00 C01 C02 ] [ x0 ]   [ b0 ]
    # [  0  C11 C12 ] [ x1 ] = [ b1 ]
    # [  0   0  C22 ] [ x2 ]   [ b2 ]
    #
    # where
    #   C00 = [A11, A12]
    #   C22 = [A34; A44]

    # Solve the overdetermined system [A34; A44]
    if rr[2] < M and cc[3] < N:
        C22 = C[rr[2]:, cc[3]:]
        b2 = b[rr[2]:]
        x2 = x[cc[3]:]
        x2[:] = qr_solve(C22, b2, order='ATA')
        # Update the right-hand side for the next system(s)
        C02_12 = C[:rr[2], cc[3]:]
        b0_1 = b[:rr[2]]
        b0_1 -= C02_12 @ x2

    # Solve the square system
    if rr[1] < rr[2] and cc[2] < cc[3]:
        x1 = x[cc[2]:cc[3]]
        x1[:] = _lu_solve_btf(C, b, r, s, cc, rr, order='ATANoDenseRows')
        # Update the right-hand side for the next system
        C01 = C[:rr[1], cc[2]:cc[3]]
        b0 = b[:rr[1]]
        b0 -= C01 @ x1

    # Solve the underdetermined system [A11 A12]
    if rr[1] > 0 and cc[2] > 0:
        C00 = C[:rr[1], :cc[2]]
        b0 = b[:rr[1]]
        x0 = x[:cc[2]]
        x0[:] = qr_solve(C00, b0, order='ATA')

    x[q] = x  # inverse permute the solution

    return x


# Exercise 7.4
def _lu_solve_btf(C, b, r, s, cc, rr, **kwargs):  # noqa:PLR0913
    """Solve `Cx = b` using LU factorization with BTF ordering.

    Parameters
    ----------
    C : (N, N) csc_array
        Square matrix in block triangular form.
    b : (N,) array_like
        Right-hand side vector.
    r : (Nb,) ndarray of int
        The row indices of the diagonal blocks.
    s : (Nb,) ndarray of int
        The column indices of the diagonal blocks.
    cc : (5,) ndarray of int
        The column indices of the Dulmage-Mendelsohn coarse decomposition.
    rr : (5,) ndarray of int
        The row indices of the Dulmage-Mendelsohn coarse decomposition.
    **kwargs
        Additional keyword arguments to pass to ``lu_solve``.

    Returns
    -------
    x : (N,) ndarray
        Solution vector. If ``A`` is overdetermined, ``x`` is the least-squares
        solution.
    """
    M, N = C.shape

    if M != N:
        raise ValueError(f"Matrix C must be square, got {C.shape}.")

    assert len(r) == len(s)

    Nb = len(r) - 1
    x = np.zeros(N)
    b = b.copy()

    # Solve all blocks by default
    hi = Nb - 1
    lo = -1

    # If [A34; A44] block exists, skip it
    if rr[2] < M and cc[3] < N:
        hi = Nb - 2

    # If [A11, A12] block exists, skip it
    if rr[1] > 0 and cc[2] > 0:
        lo = 0

    # Backsolve the middle, square blocks using LU
    for k in range(hi, lo, -1):
        # Solve the kth diagonal block
        rows = slice(r[k], r[k+1])
        cols = slice(s[k], s[k+1])
        Ckk = C[rows, cols]
        bk = b[rows]
        xk = x[cols]
        xk[:] = lu_solve(Ckk, bk, **kwargs)

        # Update the right-hand side for the next block
        if k > 0:
            Cik = C[:r[k], cols]
            b[:r[k]] -= Cik @ xk

    return x[s[lo+1]:s[hi+1]]


def scc_perm(A):
    """Compute the strongly connected components of a directed graph.

    The function finds a permutation such that `A[p][:, q]` is block upper
    triangular (if `A` is square). In this case, `r=s`, `p=q`, and the `k`th
    diagonal block is given by `A(t, t)`, where `t = r[k]:r[k]+1`. The diagonal
    of `A` is ignored. Each block is one strognly connected component of `A`.

    Parameters
    ----------
    A : (M, N) sparse matrix
        The adjacency matrix of the directed graph.

    Returns
    -------
    p : (M,) ndarray of int
        The row permutation vector.
    q : (N,) ndarray of int
        The column permutation vector.
    r : ndarray of int
        The row indices of the diagonal blocks.
    s : ndarray of int
        The column indices of the diagonal blocks.
    """
    M, N = A.shape

    if M == N:
        p, r, Nb = scc(A)
        q = p
        s = r
    else:
        # Find the connected components of [I A; A.T 0]
        S = spaugment(A)
        p_sym, r_sym, Nb = scc(S)
        p = p_sym[p_sym < M]
        q = p_sym[p_sym >= M] - M
        r = np.zeros(Nb + 1, dtype=int)
        s = np.zeros(Nb + 1, dtype=int)
        k_row = 0
        k_col = 0

        for k in range(Nb):
            # Find the rows and columns in the kth component
            r[k] = k_row
            s[k] = k_col
            k_sym = p_sym[r_sym[k]:r_sym[k+1]]
            k_row += len(k_sym < M)
            k_col += len(k_sym >= M)

        r[Nb] = M + 1
        s[Nb] = N + 1

    return p, q, r, s


def spaugment(A):
    r"""Build the sparse augmented matrix of a directed graph.

    The augmented matrix is defined as:

    .. math::
        A_{aug} =
        \begin{bmatrix}
              I & A \\
            A^T & 0
        \end{bmatrix}

    Parameters
    ----------
    A : (M, N) array_like
        Matrix of N vectors in M dimensions.

    Returns
    -------
    result : (M+N, M+N) ndarray
        The augmented matrix.
    """
    M, N = A.shape

    I_M = sparse.eye_array(M)

    return sparse.block_array(
        [[I_M, A],
         [A.T, None]],
        format=A.format
    )

# =============================================================================
# =============================================================================
