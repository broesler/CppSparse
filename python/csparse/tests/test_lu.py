#!/usr/bin/env python3
# =============================================================================
#     File: test_lu.py
#  Created: 2025-03-20 12:15
#   Author: Bernie Roesler
#
"""
Unit tests for the csparse.lu*() functions.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import os
import pytest

from numpy.testing import assert_allclose
from pathlib import Path
from scipy import linalg as la, sparse
from scipy.sparse import linalg as spla

from .helpers import generate_suitesparse_matrices

import csparse

ATOL = 1e-15  # testing tolerance

LU_NOPIVOT_FUNCS = [
    csparse.lu_rightr,
    csparse.lu_right,
]

LU_PIVOT_FUNCS = [
    csparse.lu_left,
    csparse.lu_rightpr,
    csparse.lu_rightprv,
    csparse.lu_rightp,
    csparse.lu_crout,
]

LU_FUNCS = LU_NOPIVOT_FUNCS + LU_PIVOT_FUNCS


@pytest.fixture
def A():
    """Define a dense, square matrix."""
    return csparse.davis_example_qr().toarray()


@pytest.mark.parametrize("order", ['Natural', 'ATANoDenseRows', 'ATA'])
def test_lu_interface(order):
    """Test the LU decomposition python interface."""
    Ac = csparse.davis_example_qr()
    A = Ac.toarray()

    # Test the LU decomposition with the default order
    L, U, p, q = csparse.lu(Ac, order=order)

    assert_allclose((L @ U).toarray(), A[p][:, q], atol=ATOL)


def lu_helper(A, lu_func):
    """Helper function to test the LU decomposition."""
    # Compare to scipy PLU = A
    P_, L_, U_ = la.lu(A)

    assert_allclose(P_ @ L_ @ U_, A, atol=ATOL)

    # Computes LU = PA
    P, L, U = lu_func(A)

    if lu_func is csparse.lu_rightprv:
        # P is a vector, so create the matrix
        P = np.eye(A.shape[0])[:, P]

    assert_allclose(P, P_.T, atol=ATOL)
    assert_allclose(L, L_, atol=ATOL)
    assert_allclose(U, U_, atol=ATOL)
    assert_allclose(L @ U, P @ A, atol=ATOL)


@pytest.mark.parametrize("lu_func", LU_FUNCS)
def test_nonpivoting_LU(A, lu_func):
    """Test the LU decomposition without pivoting."""
    lu_helper(A, lu_func)


# Only parametrize with functions that include pivoting
@pytest.mark.parametrize("lu_func", LU_PIVOT_FUNCS)
def test_pivoting_LU(A, lu_func):
    """Test the LU decomposition with pivoting."""
    # Permute the rows of A so pivoting is required
    seed = 56
    rng = np.random.default_rng(seed)
    A_in = A.copy()  # preserve the original matrix
    for i in range(A.shape[0]):
        print(f"Testing LU with permuted rows, {seed=}, {i=}")
        p = np.arange(A.shape[0])
        rng.shuffle(p)
        A = A_in[p]

        lu_helper(A, lu_func)


def test_1norm_estimate(A):
    """Test the 1-norm and condition number estimates."""
    # Notes are for csparse.davis_example_qr(10)
    As = sparse.csc_matrix(A)

    normd = la.norm(A, 1)
    norms = spla.norm(As, 1)
    norm_est = spla.onenormest(A)

    assert_allclose(normd, norms, atol=ATOL)
    assert_allclose(normd, norm_est, atol=ATOL)

    # Condition number == ||A||_1 * ||A^-1||_1
    condd = np.linalg.cond(A, 1)

    Ainv = la.inv(A)
    Asinv = spla.inv(As)

    normd_inv = la.norm(Ainv, 1)
    norms_inv = spla.norm(Asinv, 1)
    norm_est_inv = spla.onenormest(Ainv)

    # Test out condition number estimate
    # CSparse version:
    #
    # >> [L, U, P, Q] = lu(A);
    # >> norm1est(L, U, P, Q)  % CSparse 1-norm estimate
    # ans = 0.1153750055167834

    # >> cond1est(A)  % CSparse 1-norm condition number estimate
    # ans = 2.422875115852452

    # >> condest(A)  % built-in 1-norm condition number estimate
    # ans = 2.422875115852452

    # C++Sparse version:
    normc_inv = csparse.norm1est_inv(As)  # == 0.11537500551678347

    assert_allclose(normd_inv, norms_inv, atol=ATOL)
    assert_allclose(normd_inv, norm_est_inv, atol=ATOL)
    assert_allclose(normd_inv, normc_inv, atol=ATOL)

    κc = csparse.cond1est(As)          # == 2.422875115852453

    assert_allclose(condd, κc, atol=ATOL)

    print("---------- 1-norm estimate:")
    print("    normd:", normd)
    print("normd_inv:", normd_inv)
    print("    condd:", condd)


# -----------------------------------------------------------------------------
#         Test 7
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "problem",
    list(generate_suitesparse_matrices(square_only=True))
)
class TestLU:
    """Test class for the LU decomposition on SuiteSparse matrices."""
    # TODO abstract this fixture to a base class for all tests
    @pytest.fixture(scope='class', autouse=True)
    def setup_plot(self, request):
        """Set up the figure for plotting across tests."""
        cls = request.cls

        if not request.config.getoption('--make-figures'):
            cls.make_figures = False
            yield  # skip the setup if not making figures
            return

        cls.make_figures = True

        cls.fig, cls.axs = plt.subplots(num=1, nrows=3, ncols=4, clear=True)
        cls.fig.suptitle(f"LU Factors for {cls.problem.name}")

        # Run the tests
        yield

        # Teardown code (save the figure)
        cls.fig_dir = Path('test_figures/test_lu')
        os.makedirs(cls.fig_dir, exist_ok=True)

        cls.figure_path = (
            cls.fig_dir /
            f"{cls.problem.name.replace('/', '_')}.pdf"
        )
        print(f"Saving figure to {cls.figure_path}")
        cls.fig.savefig(cls.figure_path)

        plt.close(cls.fig)

    @pytest.mark.parametrize('kind', ['natural', 'colamd', 'amd'])
    def test_lu(self, problem, kind):
        """Test LU decomposition with natural ordering."""
        A = problem.A

        if kind == 'natural':
            permc_spec = 'NATURAL'
            order = 'Natural'
            tol = 1.0
            row = 0
        elif kind == 'colamd':
            permc_spec = 'COLAMD'
            order = 'ATANoDenseRows'
            tol = 1.0
            row = 1
        elif kind == 'amd':
            permc_spec = 'MMD_AT_PLUS_A'
            order = 'APlusAT'
            tol = 0.1
            row = 2

        try:
            lu = spla.splu(A, permc_spec=permc_spec, diag_pivot_thresh=tol)
        except RuntimeError as e:
            pytest.skip(f"scipy.sparse: {e}")

        L_, U_, p_, q_ = lu.L, lu.U, lu.perm_r, lu.perm_c

        # Get the minimum absolute value of the diagonal of U
        min_diag_U = np.min(np.abs(U_.diagonal()))
        print(f"{min_diag_U=:.4g}")

        if min_diag_U > 1e-14:
            L, U, p, q = csparse.lu(A, order=order, tol=tol)

            if self.make_figures:
                self.axs[row, 0].spy(A, markersize=1)
                self.axs[row, 1].spy(A[p], markersize=1)
                self.axs[row, 2].spy(L, markersize=1)
                self.axs[row, 3].spy(U, markersize=1)

            assert_allclose((L_ @ U_)[p_][:, q_].toarray(), A.toarray(), atol=1e-6)
            assert_allclose((L @ U).toarray(), A[p][:, q].toarray(), atol=1e-6)

# =============================================================================
# =============================================================================
