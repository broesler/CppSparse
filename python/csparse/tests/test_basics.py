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

from numpy.testing import assert_allclose
from scipy import linalg as la
from scipy import sparse
from scipy.sparse import linalg as sla

from .helpers import (
    generate_suitesparse_matrices,
    generate_random_matrices,
    generate_random_compatible_matrices,
    generate_pvec_params
)

import csparse


# -----------------------------------------------------------------------------
#         Test 1
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("problem", generate_suitesparse_matrices())
def test_transpose(problem):
    """Test the transpose operation on SuiteSparse matrices."""
    print(f"Testing matrix {problem.id} ({problem.name})")
    A = problem.A
    Ac = csparse.csc_from_scipy(A)
    B = A.T
    C = Ac.transpose()
    assert_allclose(B.toarray(), C.toarray(), atol=1e-15)


@pytest.mark.parametrize("problem", generate_suitesparse_matrices())
def test_gaxpy(problem):
    """Test the GAXPY operation on SuiteSparse matrices."""
    print(f"Testing matrix {problem.id} ({problem.name})")
    A = problem.A
    M, N = A.shape
    rng = np.random.default_rng(problem.id)
    x = rng.random(N)
    y = rng.random(M)
    z = A @ x + y
    q = csparse.gaxpy(A, x, y)
    err = la.norm(z - q, ord=1) / la.norm(z, ord=1)
    assert err < 1e-14


@pytest.mark.parametrize("problem", generate_suitesparse_matrices())
def test_coo_matrix(problem):
    """Test the COO matrix interface."""
    A = problem.A.tocoo()
    rows, cols, values = A.row, A.col, A.data
    rng = np.random.default_rng(problem.id)
    p = rng.permutation(len(values))

    # Create a new COO matrix with permuted values, rows, and cols
    A = sparse.coo_array((values[p], (rows[p], cols[p])), shape=A.shape)
    Ac = csparse.COOMatrix(values[p], rows[p], cols[p], shape=A.shape)
    assert_allclose(A.toarray(), Ac.toarray(), atol=1e-15)


# -----------------------------------------------------------------------------
#         Test 2
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("A, Ac", generate_random_matrices())
def test_coo_construction(A, Ac):
    """Test the construction of a COO matrix."""
    err = sla.norm(A - Ac.toscipy(), ord=1) / sla.norm(A, ord=1)
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
    C2 = Ac.permute(p, q)
    err = sla.norm(C1 - C2.toscipy(), ord=1)
    assert err < 1e-13


@pytest.mark.parametrize("p, x", generate_pvec_params())
def test_vector_permutation(p, x):
    """Test vector permutation."""
    M = len(p)
    x1 = x[p]
    x2 = csparse.pvec(p, x)

    assert_allclose(x1, x2, atol=1e-15)

    x1 = np.zeros(M)
    x1[p] = x
    x2 = csparse.ipvec(p, x)

    assert_allclose(x1, x2, atol=1e-15)


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
    C2 = csparse.csc_from_scipy(B).symperm(p)
    print(C1.toarray())
    print(C2.toarray())
    err = sla.norm(C1 - C2.toscipy(), ord=1)
    assert err < 1e-14


# -----------------------------------------------------------------------------
#         Test 4
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "A, B",
    generate_random_compatible_matrices(N_max=100)
)
def test_multiply(A, B):
    """Test the multiplication of a matrix with a vector."""
    C = A @ B
    D = csparse.csc_from_scipy(A) @ csparse.csc_from_scipy(B)
    assert_allclose(C.toarray(), D.toarray(), atol=1e-12)

# =============================================================================
# =============================================================================
