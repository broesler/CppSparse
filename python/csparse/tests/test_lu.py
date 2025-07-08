#!/usr/bin/env python3
# =============================================================================
#     File: test_lu.py
#  Created: 2025-03-20 12:15
#   Author: Bernie Roesler
#
"""Unit tests for the csparse.lu*() functions."""
# =============================================================================

import numpy as np
import pytest

from numpy.testing import assert_allclose
from pathlib import Path
from scipy import linalg as la, sparse
from scipy.sparse import linalg as spla

from .helpers import generate_suitesparse_matrices, BaseSuiteSparsePlot

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
    """Test the LU decomposition."""
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
    As = sparse.csc_array(A)

    normd = la.norm(A, 1)
    norms = spla.norm(As, 1)
    norms_est = spla.onenormest(A)

    assert_allclose(normd, norms, atol=ATOL)
    assert_allclose(normd, norms_est, atol=ATOL)

    # Condition number == ||A||_1 * ||A^-1||_1
    condd = np.linalg.cond(A, 1)

    Ainv = la.inv(A)
    Asinv = spla.inv(As)

    normd_inv = la.norm(Ainv, 1)
    norms_inv = spla.norm(Asinv, 1)
    norms_inv_est = spla.onenormest(Ainv)

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
    assert_allclose(normd_inv, norms_inv_est, atol=ATOL)
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
    list(generate_suitesparse_matrices(square_only=True)),
    indirect=True  # allow `problem` to be a fixture vs a parameter
)
class TestLU(BaseSuiteSparsePlot):
    """Test class for the LU decomposition on SuiteSparse matrices."""

    _nrows = 3
    _ncols = 4
    _fig_dir = Path('test_lu')
    _fig_title_prefix = 'LU Factors for '

    @pytest.mark.parametrize('kind', ['natural', 'colamd', 'amd'])
    def test_lu(self, kind, request):
        """Test LU decomposition with natural ordering."""
        A = self.problem.A

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
            request.cls.make_figures = False
            pytest.skip(f"scipy.sparse: {e}")  # catch singular matrix errors

        L_, U_, p_, q_ = lu.L, lu.U, lu.perm_r, lu.perm_c

        # Get the minimum absolute value of the diagonal of U
        min_diag_U = np.min(np.abs(U_.diagonal()))
        print(f"{min_diag_U=:.4g}")

        # if min_diag_U > 1e-14:
        L, U, p, q = csparse.lu(A, order=order, tol=tol)

        if self.make_figures:
            self.axs[row, 0].spy(A, markersize=1)
            self.axs[row, 1].spy(A[p], markersize=1)
            self.axs[row, 2].spy(L, markersize=1)
            self.axs[row, 3].spy(U, markersize=1)

        assert_allclose(
            (L_ @ U_)[p_][:, q_].toarray(),
            A.toarray(),
            atol=1e-6
        )

        assert_allclose(
            (L @ U).toarray(),
            A[p][:, q].toarray(),
            atol=1e-6
        )


# -----------------------------------------------------------------------------
#         Test 22
# -----------------------------------------------------------------------------
_N_trials = 200


@pytest.mark.parametrize(
    'problem',
    list(generate_suitesparse_matrices(N=_N_trials, square_only=True)),
    indirect=True
)
class TestCond1est(BaseSuiteSparsePlot):
    """Test the 1-norm condition number estimate."""

    _nrows = 1
    _ncols = 2
    _fig_dir = Path('test_cond1est')

    # Class variables to track plotting
    _i = 0
    _numpy_conds = np.zeros(_N_trials)
    _csparse_conds = np.zeros(_N_trials)
    _scipy_conds = np.zeros(_N_trials)

    def test_cond1est(self, request):
        """Test the condition number estimate and plot it."""
        A = self.problem.A

        κ_n = np.linalg.cond(A.toarray(), p=1)

        try:
            κ_c = csparse.cond1est(A)
            κ_s = csparse.scipy_cond1est(A)
        except RuntimeError as e:
            # splu may fail if the matrix is singular
            request.cls.make_figures = False
            pytest.skip(f"csparse: {e}")

        # Store the values in a string for logging
        cond_str = (
            f"  numpy:   {κ_n:.8e}\n"
            f"  scipy:   {κ_s:.8e}\n"
            f"  csparse: {κ_c:.8e}\n"
        )

        # Estimates are a lower bound on the exact condition number, but the
        # "<=" condition may not hold if they are close to the exact value.
        passed_any_check = False
        failure_reasons = []

        # All may be effectively infinite, but not necessarily "equal"
        try:
            huge_val = 1 / np.finfo(float).eps  # ~ 4.5e+15
            assert κ_n > huge_val
            assert κ_s > huge_val
            assert κ_c > huge_val
            passed_any_check = True
            print(f"all huge:\n{cond_str}")
        except AssertionError:
            failure_reasons.append("Not all estimates are huge.")
            print(f"NOT all huge:\n{cond_str}")

        # assert_allclose tests: |a - d| <= atol + rtol * |d|
        rtol = 1e-7
        atol = 0

        # Estimates are a lower bound on the exact condition number
        if not passed_any_check:
            try:
                # similar to assert_allclose, but with "less than or equal"
                assert κ_s - κ_n <= atol + rtol * abs(κ_n)
                assert κ_c - κ_n <= atol + rtol * abs(κ_n)
                passed_any_check = True
                print(f"less than or equal:\n{cond_str}")
            except AssertionError:
                failure_reasons.append("Not all estimates are less than exact.")
                print(f"NOT less than or equal:\n{cond_str}")

        if not passed_any_check:
            pytest.fail(
                f"Condition number estimates failed for {self.problem.name}\n"
                "\n".join(failure_reasons) + "\n" + cond_str
            )

        if self.make_figures:
            self.fig.suptitle('1-Norm Condition Number Estimate')

            cls = request.cls
            cls._numpy_conds[cls._i] = κ_n
            cls._scipy_conds[cls._i] = κ_s
            cls._csparse_conds[cls._i] = κ_c
            cls._i += 1

            self.axs[0].clear()
            self.axs[0].axline((0, 0), slope=1, color='k', linestyle='-.')
            self.axs[0].scatter(self._numpy_conds, self._csparse_conds,
                                marker='o', c='C3', alpha=0.5, zorder=3)
            self.axs[0].set(
                title='numpy vs csparse',
                xlabel='numpy cond',
                ylabel='csparse cond1est',
                xscale='log',
                yscale='log',
                aspect='equal'
            )

            self.axs[1].clear()
            self.axs[1].axline((0, 0), slope=1, color='k', linestyle='-.')
            self.axs[1].scatter(self._scipy_conds, self._csparse_conds,
                                marker='o', c='C3', alpha=0.5, zorder=3)
            self.axs[1].set(
                title='scipy vs csparse',
                xlabel='scipy cond1est',
                ylabel='csparse cond1est',
                xscale='log',
                yscale='log',
                aspect='equal'
            )

# =============================================================================
# =============================================================================
