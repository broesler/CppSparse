#!/usr/bin/env python3
# =============================================================================
#     File: _qr.py
#  Created: 2025-02-11 15:18
#   Author: Bernie Roesler
#
"""
Python implementation of various QR decomposition algorithms, as presented in
Davis, Chapter 5.
"""
# =============================================================================

import numpy as np

from scipy import sparse
from scipy import linalg as la

from .csparse import CSCMatrix, etree
from .utils import from_scipy_sparse, to_scipy_sparse


# -----------------------------------------------------------------------------
#         Householder Reflections
# -----------------------------------------------------------------------------
# TODO implement apply_qleft for P^T Q b = P^T H1 H2 ... HN b. 
# See Trefethen Algorithm 10.3, p 74.

def apply_qright(V, beta, p=None, Y=None):
    r"""Apply Householder vectors on the right.

    Computes :math:`X = Y P^T H_1 \dots H_N = Y Q`, where :math:`Q` is
    represented by the Householder vectors stored in `V`, coefficients `beta`,
    and permutation `p`. To obtain :math:`Q` itself, pass `Y = sparse.eye(M)`.

    Parameters
    ----------
    V : (M, min(M, N)) CSCMatrix
        The matrix of Householder vectors.
    beta : (min(M, N),) ndarray
        The Householder coefficients.
    p : (M,) ndarray, optional
        The row permutation vector to apply to `Y`.
    Y : (M, N) ndarray or sparse array, optional
        The matrix to which the Householder transformations are applied. If not
        given, the identity matrix is used, resulting in the full `Q` matrix.

    Returns
    -------
    result : (M, N) ndarray
        The result of applying the Householder transformations to `Y`.

    See also
    --------
    cs_qright : The CSparse implementation of this function.
    apply_qtleft : Apply Householder vectors on the left as :math:`Q^T Y`.
    """
    if isinstance(V, CSCMatrix):
        V = to_scipy_sparse(V)

    if Y is None:
        Y = sparse.eye_array(V.shape[0]).tocsc()

    M, N = V.shape
    X = Y.copy()
    if p is not None:
        X = X[:, p]
    for j in range(N):
        X -= X @ (beta[j] * V[:, [j]]) @ V[:, [j]].T
    return X


def apply_qtleft(V, beta, p=None, Y=None):
    r"""Apply Householder vectors on the left.

    Computes :math:`X = H_N \dots H_1 P Y = Q^T Y`, where :math:`Q` is
    represented by the Householder vectors stored in `V`, coefficients `beta`,
    and permutation `p`. To obtain :math:`Q^T` itself, pass `Y = sparse.eye(M)`.

    Parameters
    ----------
    V : (M, min(M, N)) CSCMatrix
        The matrix of Householder vectors.
    beta : (min(M, N),) ndarray
        The Householder coefficients.
    p : (M,) ndarray, optional
        The row permutation vector to apply to `Y`.
    Y : (M, N) ndarray or sparse array, optional
        The matrix to which the Householder transformations are applied. If not
        given, the identity matrix is used, resulting in the full `Q` matrix.

    Returns
    -------
    result : (M, N) ndarray
        The result of applying the Householder transformations to `Y`.

    See also
    --------
    cs_qleft : The CSparse implementation of this function.
    apply_qright : Apply Householder vectors on the right as :math:`Y Q`.
    """
    if isinstance(V, CSCMatrix):
        V = to_scipy_sparse(V)

    if Y is None:
        Y = sparse.eye_array(V.shape[0]).tocsc()

    M2, N = V.shape
    M, NY = Y.shape
    X = Y.copy()

    if (M2 > M):
        # Add empty rows to the bottom of X
        if sparse.issparse(X):
            X = sparse.vstack([X, sparse.csc_array((M2 - M, NY))])
        else:
            X = np.vstack([X, np.zeros((M2 - M, NY))])

    if p is not None:
        X = X[p, :]

    for j in range(N):
        X -= V[:, [j]] @ (beta[j] * V[:, [j]].T @ X)

    return X


def qr_right(A):
    """Compute the QR decomposition of A using the right-looking algorithm.

    From the LAPACK documentation for `dgeqrf.f`:

    The matrix Q is represented as a product of elementary reflectors

       Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

       H(i) = I - tau * v * v**T

    where tau is a real scalar, and v is a real vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
    and tau in TAU(i).

    Parameters
    ----------
    A : (M, N) ndarray
        Matrix of M vectors in N dimensions

    Returns
    -------
    V : (M, N) ndarray
        A matrix with the Householder reflectors as columns.
    beta : (N,) ndarray
        A vector with the scaling factors for each reflector.
    R : (M, N) ndarray
        The upper triangular matrix.
    """
    M, N = A.shape
    V = np.zeros((M, N))
    R = np.copy(A)
    beta = np.zeros(N)

    for k in range(N):
        # Compute the Householder reflector
        (Qraw, b), _ = la.qr(R[k:, [k]], mode='raw')
        v = np.vstack([1.0, Qraw[1:]])  # extract the reflector
        b = float(b.squeeze())          # get the scalar
        V[k:, [k]] = v
        beta[k] = b
        R[k:, k:] -= v @ (b * (v.T @ R[k:, k:]))

    return V, beta, R


def qr_left(A):
    """Compute the QR decomposition of A using the left-looking algorithm.

    From the LAPACK documentation for `dgeqrf.f`:

    The matrix Q is represented as a product of elementary reflectors

       Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

       H(i) = I - tau * v * v**T

    where tau is a real scalar, and v is a real vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
    and tau in TAU(i).

    Parameters
    ----------
    A : (M, N) ndarray
        Matrix of M vectors in N dimensions

    Returns
    -------
    V : (M, N) ndarray
        A matrix with the Householder reflectors as columns.
    beta : (N,) ndarray
        A vector with the scaling factors for each reflector.
    R : (M, N) ndarray
        The upper triangular matrix.
    """
    M, N = A.shape
    Nv = min(M, N)  # number of Householder reflectors
    V = np.eye(M, Nv)
    beta = np.zeros(Nv)
    R = np.zeros((M, N))

    for k in range(Nv):
        x = A[:, [k]]

        # Apply the Householder reflectors to the current column
        for i in range(k):
            v = V[i:, [i]]
            b = beta[i]
            x[i:] -= v @ (b * (v.T @ x[i:]))

        # Compute the Householder reflector
        x_k = x[k:]
        (Qraw, b), _ = la.qr(x_k, mode='raw')
        V[k+1:, [k]] = Qraw[1:]  # extract the reflector
        beta[k] = b[0]    # get the scalar
        R[:k, [k]] = x[:k]
        # NOTE If beta == 0, H is the identity matrix, so Hx == x:
        # if beta[k] == 0:
        #     R[k, k] = x_k[0]
        # else:
        #     R[k, k] = -np.sign(x_k[0]) * la.norm(x_k)
        #
        # Qraw computes Hx internally to give the correct result.
        R[k, k] = Qraw[0, 0]

    if M < N:
        # If M < N, A = [A1 | A2], where A1 is (M, M) and A2 is (M, N-M).
        # Let Q1 R1 = A1 be a QR decomposition of A1.
        # Then, A = Q1 [ R1 | Q1.T @ A2 ] is a QR decomposition of A.
        #
        # We have found Q1 R1 = A1 in V, beta, R. So R2 = Q1.T @ A2
        # See:
        # <https://math.stackexchange.com/questions/678843/householder-qr-factorization-for-m-by-n-matrix-both-m-n-and-mn>
        # and Golub & Van Loan, 3rd ed., §5.7.2 *Underdetermined Systems*.
        R[:, M:] = apply_qtleft(V, beta, p=None, Y=A[:, M:])

    return V, beta, R


# -----------------------------------------------------------------------------
#         Givens Rotations
# -----------------------------------------------------------------------------
def givens(x):
    """Compute the 2x2 Givens rotation matrix.

    Parameters
    ----------
    x : (2,) ndarray
        The vector to rotate.

    Returns
    -------
    G : (2, 2) ndarray
        The Givens rotation matrix.
    """
    assert x.shape == (2,), "Input must be of shape (2,)"
    a, b = x

    if b == 0:
        c = 1
        s = 0
    elif abs(b) > abs(a):
        τ = -a / b
        s = 1 / np.sqrt(1 + τ**2)
        c = s * τ
    else:
        τ = -b / a
        c = 1 / np.sqrt(1 + τ**2)
        s = c * τ

    return np.array([[c, -s], [s, c]], dtype=float)


def qr_givens_full(A):
    """Compute the QR decomposition of A using Givens rotations for a full
    matrix.

    Parameters
    ----------
    A : (M, N) array_like
        Matrix of M vectors in N dimensions.

    Returns
    -------
    R : (M, N) ndarray
        The upper triangular matrix.
    """
    M, N = A.shape
    R = np.copy(A)

    for i in range(1, M):
        for k in range(min(i-1, N)):
            idx = np.r_[k, i]
            R[idx, k:] = givens(R[idx, k]) @ R[idx, k:]
            R[i, k] = 0

    return R


def qr_givens(A):
    """Compute the QR decomposition of A using Givens rotations for a sparse
    matrix.

    .. note:: This function assumes that `A` has a zero-free diagonal.

    Parameters
    ----------
    A : (M, N) array_like
        Matrix of M vectors in N dimensions.

    Returns
    -------
    R : (M, N) ndarray
        The upper triangular matrix.
    """
    M, N = A.shape
    R = np.copy(A)
    # Get the elimination tree of A^T A
    parent = etree(from_scipy_sparse(sparse.csc_array(R)), True)

    for i in range(1, M):
        nnz_idx = np.where(R[i, :])[0]
        if len(nnz_idx) == 0:
            continue
        k = np.min(nnz_idx)  # find the first non-zero element
        while (k > 0 and k <= min(i-1, N)):
            idx = np.r_[k, i]
            R[idx, k:] = givens(R[idx, k]) @ R[idx, k:]
            R[i, k] = 0
            k = parent[k]

    return R


# =============================================================================
# =============================================================================
