#!/usr/bin/env python3
# =============================================================================
#     File: test_basics.py
#  Created: 2025-05-28 15:46
#   Author: Bernie Roesler
#
"""Test basic COOMatrix and CSCMatrix interface."""
# =============================================================================

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import linalg as la
from scipy import sparse
from scipy.sparse import linalg as sla

import csparse

from .helpers import (
    generate_pvec_params,
    generate_random_compatible_matrices,
    generate_random_matrices,
    generate_suitesparse_matrices,
)


# -----------------------------------------------------------------------------
#         Test 1
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("problem", generate_suitesparse_matrices())
def test_transpose(problem):
    """Test the transpose operation on SuiteSparse matrices."""
    A = problem.A
    Ac = csparse.csc_from_scipy(A)
    B = A.T
    C = Ac.transpose()
    assert_allclose(B.toarray(), C.toarray(), atol=1e-15)


@pytest.mark.parametrize("problem", generate_suitesparse_matrices())
def test_gaxpy(problem):
    """Test the GAXPY operation on SuiteSparse matrices."""
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
@pytest.mark.parametrize("A", generate_random_matrices())
def test_coo_construction(A, request):
    """Test the construction of a COO matrix."""
    A = A.tocoo()
    Ac = csparse.COOMatrix(A.data, A.row, A.col, shape=A.shape)
    assert_allclose(A.toarray(), Ac.toarray(), atol=1e-15)
    assert A.nnz == Ac.nnz
    assert max(1, Ac.nnz) == max(1, Ac.nzmax)


@pytest.mark.parametrize("A", generate_random_matrices())
def test_matrix_permutation(A):
    """Test matrix permutation."""
    M, N = A.shape
    rng = np.random.default_rng(56)
    p = rng.permutation(M)
    q = rng.permutation(N)
    C1 = A[p[:, np.newaxis], q]
    Ac = csparse.csc_from_scipy(A)
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


@pytest.mark.parametrize("A", generate_random_matrices())
def test_symmetric_matrix_permutation(A):
    """Test symmetric matrix permutation."""
    M, N = A.shape
    rng = np.random.default_rng(56)

    # Make a symmetric matrix
    N = min(M, N)
    B = A[:N, :N]
    p = rng.permutation(N)
    B = B + B.T

    C1 = sparse.triu(B[p[:, np.newaxis], p])
    C2 = csparse.csc_from_scipy(B).symperm(p)
    assert_allclose(C1.toarray(), C2.toarray(), atol=1e-15)


# -----------------------------------------------------------------------------
#         Test 4
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "A, B",
    list(generate_random_compatible_matrices(N_max=100, kind='multiply'))
)
def test_multiply(A, B):
    """Test the multiplication of a matrix with a vector."""
    C = A @ B
    D = csparse.csc_from_scipy(A) @ csparse.csc_from_scipy(B)
    assert_allclose(C.toarray(), D.toarray(), atol=1e-12)


# -----------------------------------------------------------------------------
#         Test 5
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "A, B",
    list(generate_random_compatible_matrices(N_max=100, kind='add'))
)
class TestAdd:
    """Test the addition of two matrices."""

    @pytest.fixture(autouse=True)
    def setup_method(self, A, B):
        """Set up the matrices for testing."""
        self.A = A
        self.B = B
        self.Ac = csparse.csc_from_scipy(A)
        self.Bc = csparse.csc_from_scipy(B)

    def test_add(self):
        """Test the addition of two matrices."""
        C = self.A + self.B
        D = self.Ac + self.Bc
        assert_allclose(C.toarray(), D.toarray(), atol=1e-12)

    def test_add_scaled(self):
        """Test the addition of a scaled matrix."""
        C = np.pi * self.A + self.B
        D = np.pi * self.Ac + self.Bc
        assert_allclose(C.toarray(), D.toarray(), atol=1e-12)

    def test_add_both_scaled(self):
        """Test the addition of a scaled matrix."""
        C = np.pi * self.A + 3 * self.B
        D = np.pi * self.Ac + 3 * self.Bc
        assert_allclose(C.toarray(), D.toarray(), atol=1e-12)


# -----------------------------------------------------------------------------
#         Test 14
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "A", list(generate_random_matrices(N_max=100, d_scale=0.1))
)
def test_droptol(A):
    """Test the drop tolerance operation on a random matrix."""
    # Shift the values to the range [-1, 1]
    A = sparse.csc_array((2 * A.data - 1, A.indices, A.indptr), shape=A.shape)

    drop_tol = 0.5  # drop entries with absolute value < 0.5

    B = csparse.csc_from_scipy(A).droptol(drop_tol)

    # About 4x as fast to filter A.data and eliminate zeros than to do the full
    # multiplication: A * (np.abs(A) > drop_tol).
    # The csparse method is a further 4x faster
    A.data[np.abs(A.data) <= drop_tol] = 0
    A.eliminate_zeros()

    assert all(np.abs(A.data) > drop_tol)
    assert all(np.abs(B.data) > drop_tol)

    assert_allclose(A.toarray(), B.toarray(), atol=1e-15)


# -----------------------------------------------------------------------------
#         Test 2D matrix input to gaxpy functions
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "gaxpy_func", [csparse.gaxpy_col, csparse.gaxpy_row, csparse.gaxpy_block]
)
def test_gaxpy_2D(gaxpy_func, subtests):
    """Test gaxpy with 2D matrix input."""
    rng = np.random.default_rng(56)

    for overunder in ['over', 'under']:
        with subtests.test(overunder=overunder):
            if overunder == 'over':
                M, N, K = 10, 7, 5
            else:  # under
                M, N, K = 5, 7, 10

            A = sparse.random_array(
                (M, N), density=0.8, format='csc', random_state=rng
            )
            X = rng.random((N, K))
            Y = rng.random((M, K))

            Z = A @ X + Y
            Q = gaxpy_func(A, X, Y)

            assert_allclose(Q, Z, atol=1e-15)


@pytest.mark.parametrize(
    "gatxpy_func", [csparse.gatxpy_col, csparse.gatxpy_row, csparse.gatxpy_block]
)
def test_gatxpy_2D(gatxpy_func, subtests):
    """Test gatxpy with 2D matrix input."""
    rng = np.random.default_rng(56)

    for overunder in ['over', 'under']:
        with subtests.test(overunder=overunder):
            if overunder == 'over':
                M, N, K = 10, 7, 5
            else:  # under
                M, N, K = 5, 7, 10

            A = sparse.random_array(
                (M, N), density=0.8, format='csc', random_state=rng
            )
            X = rng.random((M, K))
            Y = rng.random((N, K))

            Z = A.T @ X + Y
            Q = gatxpy_func(A, X, Y)

            assert_allclose(Q, Z, atol=1e-15)


# =============================================================================
# =============================================================================
