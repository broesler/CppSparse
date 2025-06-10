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

import pytest

import numpy as np

from scipy import linalg as la
from scipy import sparse
from scipy.sparse import linalg as sla

from .helpers import generate_suitesparse_matrices, generate_random_matrices

import csparse


# -----------------------------------------------------------------------------
#         Test 1
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("problem, A, Ac", generate_suitesparse_matrices())
def test_transpose(problem, A, Ac):
    """Test the transpose operation on SuiteSparse matrices."""
    print(f"Testing matrix {problem.id} ({problem.name})")
    B = A.T
    C = Ac.transpose()
    np.testing.assert_allclose(B.toarray(), C.toarray(), atol=1e-15)


@pytest.mark.parametrize("problem, A, Ac", generate_suitesparse_matrices())
def test_gaxpy(problem, A, Ac):
    """Test the GAXPY operation on SuiteSparse matrices."""
    print(f"Testing matrix {problem.id} ({problem.name})")
    M, N = A.shape
    rng = np.random.default_rng(problem.id)
    x = rng.random(N)
    y = rng.random(M)
    z = A @ x + y
    q = csparse.gaxpy(A, x, y)
    err = la.norm(z - q, ord=1) / la.norm(z, ord=1)
    assert err < 1e-14


@pytest.mark.parametrize("problem, A, Ac", generate_suitesparse_matrices())
def test_coo_matrix(problem, A, Ac):
    """Test the COO matrix interface."""
    A = A.tocoo()
    rows, cols, values = A.row, A.col, A.data
    rng = np.random.default_rng(problem.id)
    p = rng.permutation(len(values))

    # Create a new COO matrix with permuted values, rows, and cols
    A = sparse.coo_array((values[p], (rows[p], cols[p])), shape=A.shape)
    Ac = csparse.COOMatrix(values[p], rows[p], cols[p], shape=A.shape)
    np.testing.assert_allclose(A.toarray(), Ac.toarray(), atol=1e-15)


# -----------------------------------------------------------------------------
#         Test 2
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("A, Ac", generate_random_matrices())
def test_coo_construction(A, Ac):
    """Test the construction of a COO matrix."""
    err = sla.norm(A - csparse.scipy_from_csc(Ac), ord=1) / sla.norm(A, ord=1)
    assert err < 1e-15
    assert A.nnz == Ac.nnz
    assert max(1, Ac.nnz) == max(1, Ac.nzmax)


@pytest.mark.parametrize("A, Ac", generate_random_matrices())
def test_matrix_permutation(A, Ac):
    """Test matrix permutation."""
    M, N = A.shape
    rng = np.random.default_rng(56)
    p = rng.permutation(M)
    q = rng.permutation(N)
    C1 = A[p][:, q]
    p_inv = csparse.inv_permute(p)
    C2 = Ac.permute(p_inv, q)
    err = sla.norm(C1 - csparse.scipy_from_csc(C2), ord=1)
    assert err < 1e-13


# FIXME don't need this parametrization here
@pytest.mark.parametrize("A, Ac", generate_random_matrices())
def test_vector_permutation(A, Ac):
    """Test vector permutation."""
    M, N = A.shape
    rng = np.random.default_rng(56)
    p = rng.permutation(M)
    q = rng.permutation(N)
    x = rng.random(M)

    x1 = x[p]
    x2 = csparse.pvec(p, x)

    np.testing.assert_allclose(x1, x2, atol=1e-15)

    x1 = np.zeros(M)
    x1[p] = x
    p_inv = csparse.inv_permute(p)  # FIXME WHY??
    x2 = csparse.ipvec(p_inv, x)

    np.testing.assert_allclose(x1, x2, atol=1e-15)


@pytest.mark.parametrize("A, Ac", generate_random_matrices())
def test_symmetric_matrix_permutation(A, Ac):
    """Test symmetric matrix permutation."""
    M, N = A.shape
    rng = np.random.default_rng(56)

    # Make a symmetric matrix
    N = min(M, N)
    B = A[:N, :N]
    p = rng.permutation(N)
    B = B + B.T

    C1 = sparse.triu(B[p][:, p])
    p_inv = csparse.inv_permute(p)
    C2 = csparse.csc_from_scipy(B).symperm(p_inv)
    print(C1.toarray())
    print(C2.toarray())
    err = sla.norm(C1 - csparse.scipy_from_csc(C2), ord=1)
    assert err < 1e-14







# =============================================================================
# =============================================================================
