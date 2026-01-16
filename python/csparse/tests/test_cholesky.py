#!/usr/bin/env python3
# =============================================================================
#     File: test_cholesky.py
#  Created: 2025-02-20 14:55
#   Author: Bernie Roesler
#
"""Unit tests for the python Cholesky algorithms."""
# =============================================================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import linalg as la
from scipy import sparse
from scipy.sparse import linalg as sla

try:
    from sksparse import cholmod
    HAS_CHOLMOD = True
except ImportError:
    HAS_CHOLMOD = False

import csparse

from .helpers import (
    BaseSuiteSparsePlot,
    generate_random_cholesky_matrices,
    generate_random_matrices,
    generate_suitesparse_matrices,
)

ATOL = 1e-15  # testing tolerance


@pytest.fixture
def A():
    """Define a dense, symmetric, positive definite matrix."""
    return csparse.davis_example_chol().toarray()


PYTHON_CHOL_FUNCS = [
    csparse.chol_up,
    csparse.chol_left,
    csparse.chol_left_amp,
    csparse.chol_right,
    csparse.chol_super
]


@pytest.mark.parametrize("use_postorder", [False, True])
@pytest.mark.parametrize("order", ['Natural', 'APlusAT'])
def test_cholesky_interface(order, use_postorder):
    """Test the Cholesky decomposition python interface."""
    A = csparse.davis_example_chol()

    L, p = csparse.chol(A, order, use_postorder)
    Ls = csparse.symbolic_cholesky(A, order, use_postorder).L
    Ll = csparse.leftchol(A, order, use_postorder).L
    Lr = csparse.rechol(A, order, use_postorder).L

    assert_allclose(L.indptr, Ls.indptr)
    assert_allclose(L.indices, Ls.indices)
    assert_allclose(L.toarray(), Ll.toarray())
    assert_allclose(Ll.toarray(), Lr.toarray())

    assert_allclose((L @ L.T).toarray(), A[p[:, np.newaxis], p].toarray(), atol=ATOL)


@pytest.mark.parametrize("chol_func", PYTHON_CHOL_FUNCS)
def test_python_cholesky(A, chol_func):
    """Test the Cholesky decomposition algorithms."""
    # Compute the Cholesky factor
    L_ = la.cholesky(A, lower=True)

    if chol_func == csparse.chol_super:
        # Define "supernodes" as ones to get the same result as left-looking
        s = np.ones(A.shape[0], dtype=int)
        L = csparse.chol_super(A, s, lower=True)
    else:
        L = chol_func(A, lower=True)

    # Check that algorithms work
    assert_allclose(L, L_, atol=ATOL)
    assert_allclose(L @ L.T, A, atol=ATOL)


# See also: test20.m
# NOTE pytest internals compare generated sparse matrix objects, so we get
# a SparseEfficiencyWarning. Ignore it for this test.
@pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")
@pytest.mark.parametrize(
    'problem',
    list(generate_random_matrices(
        N_trials=10,
        N_max=100,
        d_scale=0.1,
        square_only=True
    )),
    indirect=True
)
class TestCholeskyUpdate(BaseSuiteSparsePlot):
    """Test the Cholesky update and downdate algorithms.

    .. note:: These tests only cover the python implementations of the
    update/downdate functions, not the C++ cs::chol_update function.
    """

    _seed = 565656
    _rng = np.random.default_rng(_seed)

    _nrows = 1
    _ncols = 3
    _fig_dir = Path('test_chol_update')
    _fig_title_prefix = 'Cholesky Update for '

    @pytest.fixture(scope='class', autouse=True)
    def setup_problem(self, request, base_setup_problem):
        """Prepare the problem for testing."""
        cls = request.cls
        A = cls.problem.A
        N = A.shape[0]

        # Make sure the matrix is symmetric positive definite
        # (Scipy Cholesky factorization is only for dense matrices)
        cls.A = (A @ A.T + N * sparse.eye_array(N)).toarray()
        cls.N = N

        cls.parent = csparse.etree(sparse.csc_array(cls.A))

        cls.L = la.cholesky(cls.A, lower=True)
        assert_allclose(cls.L @ cls.L.T, cls.A, atol=ATOL)

    @pytest.fixture(scope='class', autouse=True)
    def make_plot(self, request, setup_problem, setup_plot):
        """Make a plot for the Cholesky update tests."""
        cls = request.cls
        if not cls.make_figures:
            return

        cls.axs[0].spy(cls.A, markersize=1)
        # cls.axs[1].treeplot(cls.parent)  # TODO
        cls.axs[2].spy(cls.L, markersize=1)

        cls.axs[0].set_title('Original Matrix A')
        # cls.axs[1].set_title('Tree plot of A')
        cls.axs[2].set_title('Cholesky Factor L')

    @pytest.fixture(scope='function')
    def setup_update(self, request, setup_problem):
        """Generate a new w and updated matrix for each test function call."""
        cls = request.cls

        # Generate random update with same pattern as a random column of L
        k = cls._rng.integers(0, cls.N)
        idx = np.nonzero(cls.L[:, k])[0]

        w = np.zeros((cls.N,))
        w[idx] = cls._rng.random(idx.size)
        wwT = np.outer(w, w)  # == w[:, np.newaxis] @ w[np.newaxis, :]

        A_up = cls.A + wwT

        return A_up, w

    @pytest.mark.parametrize('trial', range(10))
    def test_python_chol_update(self, trial, setup_update):
        """Test the python Cholesky update and downdate."""
        A_up, w = setup_update
        A, L = self.A, self.L

        L_up, w_up = csparse.chol_update(L, w)
        assert_allclose(L_up @ L_up.T, A_up, atol=ATOL)
        assert_allclose(la.solve(L, w), w_up, atol=ATOL)

        L_upd, w_upd = csparse.chol_updown(L, w, update=True)
        assert_allclose(L_up, L_upd, atol=ATOL)
        assert_allclose(L_upd @ L_upd.T, A_up, atol=ATOL)
        assert_allclose(la.solve(L, w), w_upd, atol=ATOL)

        # Just downdate back to the original matrix!
        # A_down = A.copy()  # A_down == A_up - wwT == A
        L_down, w_down = csparse.chol_downdate(L_up, w)
        assert_allclose(L_down @ L_down.T, A, atol=1e-14)
        assert_allclose(la.solve(L_up, w), w_down, atol=ATOL)

        L_downd, w_downd = csparse.chol_updown(L_up, w, update=False)
        assert_allclose(L_down, L_downd, atol=ATOL)
        assert_allclose(L_downd @ L_downd.T, A, atol=1e-14)
        assert_allclose(la.solve(L_up, w), w_downd, atol=ATOL)

    @pytest.mark.parametrize('trial', range(10))
    def test_cpp_chol_update(self, trial, setup_update):
        """Test the C++ Cholesky update and downdate."""
        A_up, w = setup_update
        L = sparse.csc_array(self.L)

        # Convert w to (N, 1) csc_array
        w = sparse.csc_array(w[:, np.newaxis])

        L_up = csparse.chol_update_(L, True, w, self.parent)
        assert_allclose((L_up @ L_up.T).toarray(), A_up, atol=ATOL)

        # Just downdate back to the original matrix!
        L_down = csparse.chol_update_(L_up, False, w, self.parent)
        assert_allclose((L_down @ L_down.T).toarray(), self.A, atol=1e-14)


# -----------------------------------------------------------------------------
#         Test 3
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "problem",
    list(generate_suitesparse_matrices(square_only=True)),
    indirect=True
)
class TestTrisolveCholesky(BaseSuiteSparsePlot):
    """Test triangular solvers with Cholesky factors."""

    _nrows = 2
    _ncols = 3
    _fig_dir = Path('test_trisolve_cholesky')
    _fig_title_prefix = 'Cholesky Factors for '

    @pytest.fixture(scope='class', autouse=True)
    def setup_problem(self, request, base_setup_problem):
        """Prepare the problem for testing."""
        cls = request.cls
        A = cls.problem.A
        cls.order = 'APlusAT'
        N = A.shape[1]

        # Make the matrix symmetric positive definite
        cls.A = A @ A.T + 2 * N * sparse.eye_array(N)

        # Get the Cholesky factorization using scipy (dense matrices only)
        try:
            cls.L0 = sparse.csc_array(la.cholesky(cls.A.toarray(), lower=True))
        except Exception:
            pytest.skip(f"Skipping {cls.problem.name}: Cholesky failure.")

        # RHS
        rng = np.random.default_rng(cls.problem.id)
        cls.b = rng.random(N)

        # Get the permutation from AMD
        p = csparse.amd(cls.A, order=cls.order)

        # Reorder the matrix
        cls.C = cls.A[p[:, np.newaxis], p]
        cls.κ = csparse.cond1est(cls.C)
        print(f"cond1est: {cls.κ:.4e} ({cls.problem.name})")

    def test_lsolve(self):
        """Test lsolve vs. scipy.linalg.spsolve."""
        x1 = sla.spsolve(self.L0, self.b)
        x2 = csparse.lsolve(self.L0, self.b)
        assert_allclose(x1, x2, atol=1e-12 * self.κ)

    def test_ltsolve(self):
        """Test ltsolve vs. scipy.linalg.spsolve."""
        x1 = sla.spsolve(self.L0.T, self.b)
        x2 = csparse.ltsolve(self.L0, self.b)
        assert_allclose(x1, x2, atol=1e-10 * self.κ)

    def test_usolve(self):
        """Test usolve vs. scipy.linalg.spsolve."""
        U = self.L0.T
        x1 = sla.spsolve(U, self.b)
        x2 = csparse.usolve(U, self.b)
        assert_allclose(x1, x2, atol=1e-10 * self.κ)

    def test_utsolve(self):  # (not in test3.m)
        """Test utsolve vs. scipy.linalg.spsolve."""
        U = self.L0.T
        x1 = sla.spsolve(U.T, self.b)
        x2 = csparse.utsolve(U, self.b)
        assert_allclose(x1, x2, atol=1e-10 * self.κ)

    def test_csparse_cholesky(self):
        """Test the C++Sparse Cholesky vs. scipy.linalg.cholesky."""
        L2 = csparse.chol(self.A).L
        if self.make_figures:
            self.axs[0, 0].spy(self.L0, markersize=1)
            self.axs[1, 0].spy(L2, markersize=1)
            self.axs[0, 0].set_title("chol(A)")
        assert_allclose(self.L0.toarray(), L2.toarray(), atol=1e-8 * self.κ)

    def test_cholesky_reordered(self):
        """Test the C++Sparse Cholesky with reordering."""
        L1 = sparse.csc_array(la.cholesky(self.C.toarray(), lower=True))
        L2 = csparse.chol(self.C).L
        if self.make_figures:
            self.axs[0, 1].spy(L1, markersize=1)
            self.axs[1, 1].spy(L2, markersize=1)
            self.axs[0, 1].set_title("chol(C)")
        assert_allclose(L1.toarray(), L2.toarray(), atol=1e-8 * self.κ)

    def test_cholesky_internal_reordering(self):
        """Test the C++Sparse Cholesky with internal reordering."""
        res = csparse.chol(self.A, order=self.order)
        L3 = res.L
        p3 = res.p
        C3 = self.A[p3[:, np.newaxis], p3]
        L4 = sparse.csc_array(la.cholesky(C3.toarray(), lower=True))
        if self.make_figures:
            self.axs[0, 2].spy(L4, markersize=1)
            self.axs[1, 2].spy(L3, markersize=1)
            self.axs[0, 2].set_title("chol(A[p[:, np.newaxis], p])")
        assert_allclose(L3.toarray(), L4.toarray(), atol=1e-8 * self.κ)


# -----------------------------------------------------------------------------
#         Test 6
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "L, b",
    list(generate_random_cholesky_matrices(N_trials=201, N_max=100))
)
@pytest.mark.parametrize("lower", [True, False])
def test_reachability(L, b, lower, request):
    """Test the reachability of the Cholesky factor."""
    test_id = request.node.callspec.id

    # Solve the system
    if not lower:
        L = L.T

    # returns "squeezed" array (N,) even if b is (N, 1)
    x = sla.spsolve(L, b)

    sr = csparse.reach_r(L, b)
    sz = csparse.reach_r(L, b)
    assert all(sr == sz)

    s2 = csparse.reach(L, b)
    assert all(s2 == sr)

    if lower:
        xs = csparse.lsolve(L, b)
        xd = csparse.lsolve(L, b.toarray())  # test with dense input
    else:
        xs = csparse.usolve(L, b)
        xd = csparse.usolve(L, b.toarray())  # test with dense input

    # Plot the results
    if request.config.getoption('--make-figures'):
        fig, ax = plt.subplots(num=1, clear=True)
        ax.spy(sparse.hstack([L, x[:, np.newaxis], xs, b]), markersize=1)
        ax.set_title(f"Reachability Test {test_id}")
        # Save the figure for reference
        fig_dir = Path('test_figures/test_reachability')
        fig_dir.mkdir(parents=True, exist_ok=True)
        # Take only the "random_dd" part of "random_dd::(M, N)::nnz"
        test_fileroot = test_id.split('::')[0] + ']'
        fig.savefig(fig_dir / f"reachability_{test_fileroot}.pdf")

    xi = np.nonzero(xs)[0]
    s = np.sort(sr)
    assert all(s == xi)

    # Check sparse and dense solutions
    # Ensure 1D array for comparison
    assert_allclose(xs.toarray(), xd, atol=1e-8)

    # Check vs. scipy
    assert_allclose(x, xd.squeeze(), atol=1e-8)  # works for upper


# -----------------------------------------------------------------------------
#         Test 11
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("order", ['Natural', 'APlusAT'])
@pytest.mark.parametrize(
    "problem",
    list(generate_suitesparse_matrices(N=200, square_only=True))
)
def test_rowcnt(problem, order):
    """Test Cholesky row counts."""
    A = problem.A
    N = A.shape[0]
    A = A.astype(bool).astype(float)  # "symbolic" all entries are ones

    # Permute the matrix using AMD (or Natural order)
    p = csparse.amd(A, order=order)
    A = A[p[:, np.newaxis], p]

    # Make sure A is positive definite
    A = A + A.T + N * sparse.eye_array(N)

    # Get the Cholesky factorization
    L = la.cholesky(A.toarray(), lower=True)
    rc_scipy = np.sum(L != 0, axis=1)

    parent = csparse.etree(A)
    post = csparse.post(parent)
    rc = csparse.rowcnt(sparse.csc_array(L), parent, post)

    assert all(rc == rc_scipy)


# -----------------------------------------------------------------------------
#         Test 13
# -----------------------------------------------------------------------------
def _cholmod_counts(A, ATA):
    kind = 'col' if ATA else None

    # Get the etree and postordering
    parent_, post_ = cholmod.etree(A, kind=kind, return_post=True)

    parent = csparse.etree(A, ATA=ATA)
    post = csparse.post(parent)

    assert_array_equal(parent, parent_)
    assert_array_equal(post, post_)

    # Get the column counts of L
    counts_ = cholmod.symbfact(A, kind=kind)[0]

    counts = csparse.chol_colcounts(A, ATA=ATA)
    assert_array_equal(counts, counts_)

    if not ATA:
        counts_triu = csparse.chol_colcounts(sparse.triu(A))
        assert_array_equal(counts_triu, counts_)


@pytest.mark.skipif(not HAS_CHOLMOD, reason="scikit-sparse not installed")
@pytest.mark.parametrize(
    "A",
    list(generate_random_matrices(
        N_trials=100,
        N_max=100,
        d_scale=0.1,
        square_only=True
    )),
)
def test_chol_counts(A, subtests):
    A = A.copy()  # don't modify the original matrix

    # Make sure A is symmetric positive definite
    A = (A + A.T) / 2
    A.setdiag(A.diagonal() + 1e-6)
    A = A.tocsc()

    for ATA in [False, True]:
        with subtests.test(ATA=ATA):
            _cholmod_counts(A, ATA)

    # One more test for non-square A
    M, N = A.shape
    rng = np.random.default_rng(565656)

    for overunder in ['M < N', 'M > N']:
        with subtests.test(overunder=overunder):
            if overunder == 'M < N' and M > 1:
                Mc = rng.integers(1, M)
                C = A[:Mc].copy()
            elif N > 1:
                Nc = rng.integers(1, N)
                C = A[:, :Nc].copy()
            else:
                C = A.copy()

            _cholmod_counts(C, ATA=True)

# =============================================================================
# =============================================================================
