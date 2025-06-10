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

from .helpers import generate_suitesparse_matrices

import csparse


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


# =============================================================================
# =============================================================================
