#!/usr/bin/env python3
# =============================================================================
#     File: test_fillreducing.py
#  Created: 2025-06-12 19:41
#   Author: Bernie Roesler
#
"""
Test fill-reducing ordering algorithms.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import os
import pytest
import time

from pathlib import Path
from scipy import sparse
from scipy import linalg as la
from scipy.sparse import linalg as spla

from .helpers import (
    BaseSuiteSparsePlot,
    generate_suitesparse_matrices,
    generate_random_matrices,
    is_valid_permutation
)

import csparse


# -----------------------------------------------------------------------------
#         Test 15
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    'A',
    generate_random_matrices(N_max=200, d_scale=0.05, square_only=True)
)
def test_amd(A, request):
    """Test AMD fill-reducing ordering."""
    M, N = A.shape
    assert M == N, 'Matrix must be square for AMD ordering.'

    # Add a randomly-placed dense column
    k = np.random.randint(0, N)
    A = A.todok()
    A[:, k] = 1.0

    # TODO
    # * scipy AMD ordering? Use sparse.splu(A, permc_spec='MMD_ATA')?
    # * symbfact to compute lnz

    p = csparse.amd(A, order='APlusAT')

    assert is_valid_permutation(p)

    if request.config.getoption('--make-figures'):
        C = A + A.T + sparse.eye_array(N)

        fig, axs = plt.subplots(num=1, ncols=2, clear=True)

        axs[0].spy(C, markersize=1)
        axs[1].spy(C[p][:, p], markersize=1)
        # TODO scipy ordering?

        axs[0].set_title('C = A + A.T + I')
        axs[1].set_title('AMD Reordered C')

        fig_dir = Path('test_figures/test_amd_random')
        os.makedirs(fig_dir, exist_ok=True)

        test_id = request.node.name
        figure_path = fig_dir / f"{test_id}.pdf"
        print(f"Saving figure to {figure_path}")
        fig.savefig(figure_path)

        plt.close(fig)


@pytest.mark.parametrize(
    'problem',
    list(generate_suitesparse_matrices(N=200)),
    indirect=True
)
class TestAMD(BaseSuiteSparsePlot):
    """Test AMD fill-reducing ordering on SuiteSparse matrices."""
    _nrows = 2
    _ncols = 2
    _fig_dir = Path('test_amd_suitesparse')
    _fig_title_prefix = 'AMD for '

    @pytest.fixture(scope='class', autouse=True)
    def setup_problem(self, request, base_setup_problem):
        """Setup method to initialize the problem matrix."""
        cls = request.cls

        A = cls.problem.A
        cls.A_orig = A.copy()

        M, N = A.shape

        if M < N:
            A = A.T

        if M != N:
            A = A.T @ A

        cls.M, cls.N = A.shape

        cls.A = A
        print(f"A is shape {A.shape} with {A.nnz} nonzeros.")

    def test_symmetric_amd(self):
        """Test AMD fill-reducing ordering."""
        p = csparse.amd(self.A, order='APlusAT')

        assert is_valid_permutation(p)

        if self.make_figures:
            C = self.A + self.A.T + sparse.eye_array(self.N)
            self.axs[0, 0].spy(C, markersize=1)
            self.axs[0, 1].spy(C[p][:, p], markersize=1)

            self.axs[0, 0].set_title('C = A + A.T + I')
            self.axs[0, 1].set_title('AMD Reordered C')

    def test_colamd(self):
        """Test COLAMD fill-reducing ordering."""
        p = csparse.amd(self.A_orig, order='ATA')

        assert is_valid_permutation(p)

        if self.make_figures:
            self.axs[1, 0].spy(self.A_orig, markersize=1)
            self.axs[1, 1].spy(self.A_orig[:, p], markersize=1)

            self.axs[1, 0].set_title('Original A')
            self.axs[1, 1].set_title('csparse.amd(ATA)')


# -----------------------------------------------------------------------------
#         Test 24
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    'problem',
    list(generate_suitesparse_matrices(N=200, square_only=True)),
    indirect=True
)
class TestFiedler(BaseSuiteSparsePlot):
    """Test Fiedler's fill-reducing ordering on SuiteSparse matrices."""
    _nrows = 1
    _ncols = 3
    _fig_dir = Path('test_fielder')
    _fig_title_prefix = 'RCM vs. Fielder for '

    def test_fiedler(self):
        """Test Fiedler's fill-reducing ordering."""
        A = self.problem.A

        r_start = time.perf_counter()
        pr = sparse.csgraph.reverse_cuthill_mckee(A)
        r_end = time.perf_counter()

        f_start = time.perf_counter()

        try:
            pf, _, _ = csparse.fiedler(A)
        except ValueError as e:
            pytest.skip(f"csparse.fiedler: {e}")

        f_end = time.perf_counter()

        assert is_valid_permutation(pr)
        assert is_valid_permutation(pf)

        tr = r_end - r_start
        tf = f_end - f_start
        rel = tf / max(tr, 1e-6)
        print(f"time: RCM {tr:.2e}s   Fiedler {tf:.2e}s   ratio {rel:.2e}")

        # Evaluate the profile metric
        r_profile = csparse.profile(A[pr][:, pr])
        f_profile = csparse.profile(A[pf][:, pf])
        print(f"{A.shape}, "
              f"RCM profile: {r_profile}, "
              f"Fiedler profile: {f_profile}")

        if self.make_figures:
            Ab = A.astype(bool)
            Ab = Ab + Ab.T
            self.axs[0].spy(Ab, markersize=1)
            self.axs[1].spy(Ab[pr][:, pr], markersize=1)
            self.axs[2].spy(Ab[pf][:, pf], markersize=1)

            self.axs[0].set_title('Original A')
            self.axs[1].set_title('RCM')
            self.axs[2].set_title('Fiedler')


# -----------------------------------------------------------------------------
#         Test 25
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    'problem',
    list(generate_suitesparse_matrices(N=200, square_only=True)),
    indirect=True
)
class TestNestedDissection(BaseSuiteSparsePlot):
    """Test nested dissection ordering on SuiteSparse matrices."""
    _nrows = 1
    _ncols = 3
    _fig_dir = Path('test_nested_dissection')
    _fig_title_prefix = 'RCM vs. Nested Dissection for '

    def test_nested_dissection(self):
        """Test nested dissection fill-reducing ordering."""
        A = self.problem.A

        r_start = time.perf_counter()
        pr = sparse.csgraph.reverse_cuthill_mckee(A)
        r_end = time.perf_counter()

        f_start = time.perf_counter()

        try:
            pn = csparse.nested_dissection(A)
        except ValueError as e:
            pytest.skip(f"csparse.nested_dissection: {e}")

        f_end = time.perf_counter()

        assert is_valid_permutation(pr)
        assert is_valid_permutation(pn)

        tr = r_end - r_start
        tn = f_end - f_start
        rel = tn / max(tr, 1e-6)
        print(f"time: RCM {tr:.2e}s   ND {tn:.2e}s   ratio {rel:.2e}")

        # Evaluate the profile metric
        r_profile = csparse.profile(A[pr][:, pr])
        f_profile = csparse.profile(A[pn][:, pn])
        print(f"{A.shape}, "
              f"RCM profile: {r_profile}, "
              f"ND profile: {f_profile}")

        if self.make_figures:
            Ab = A.astype(bool)
            Ab = Ab + Ab.T
            self.axs[0].spy(Ab, markersize=1)
            self.axs[1].spy(Ab[pr][:, pr], markersize=1)
            self.axs[2].spy(Ab[pn][:, pn], markersize=1)

            self.axs[0].set_title('Original A')
            self.axs[1].set_title('RCM')
            self.axs[2].set_title('ND')


# -----------------------------------------------------------------------------
#         Test 26
# -----------------------------------------------------------------------------
_N_trials = 100


# NOTE pytest internals compare generated sparse matrix objects, so we get
# a SparseEfficiencyWarning. Ignore it for this test.
@pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")
@pytest.mark.parametrize(
    'problem',
    list(generate_random_matrices(N_trials=_N_trials, d_scale=0.1, N_max=100)),
    indirect=True
)
class TestDMSolve(BaseSuiteSparsePlot):
    """Test dm_solve function on random matrices."""
    _nrows = _ncols = 1
    _fig_dir = Path('test_dm_solve')

    _rng = np.random.default_rng(565656)

    # Class variables for plotting
    _i = 0
    _scipy_err = np.zeros(_N_trials)
    _dmsol_err = np.zeros(_N_trials)

    def test_dm_solve(self, request):
        """Test dm_solve on a random matrix."""
        cls = request.cls
        A = self.problem.A
        b = self._rng.random(A.shape[0])

        # Solve the system
        x_scipy = spla.lsqr(A, b)[0]
        x_dmsol = csparse.dm_solve(A, b)

        # Compute the error
        err_scipy = la.norm(A @ x_scipy - b)
        err_dmsol = la.norm(A @ x_dmsol - b)
        rel_err = np.exp(np.log(err_dmsol) - np.log(err_scipy))

        print(
            f"Trial {self._i}:\n"
            f"             scipy error: {err_scipy:.4e}\n"
            f"  csparse.dm_solve error: {err_dmsol:.4e}\n"
            f"                   ratio: {rel_err:.4e}"
        )

        eps = np.finfo(float).eps
        cls._scipy_err[cls._i] = max(err_scipy, eps)
        cls._dmsol_err[cls._i] = max(err_dmsol, eps)
        cls._i += 1

        if self.make_figures:
            self.axs.axline((0, 0), slope=1, color='k', linestyle='-.')
            self.axs.scatter(self._scipy_err, self._dmsol_err,
                             marker='o', c='C3', alpha=0.5, zorder=3)
            self.axs.set(
                title='scipy vs csparse',
                xlabel='scipy.sparse.linalg.lsqr error',
                ylabel='csparse.dm_solve error',
                xscale='log',
                yscale='log',
                aspect='equal'
            )


# =============================================================================
# =============================================================================
