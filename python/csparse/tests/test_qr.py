#!/usr/bin/env python3
# =============================================================================
#     File: test_qr.py
#  Created: 2025-02-20 11:49
#   Author: Bernie Roesler
#
"""
Unit tests for the csparse.qr() function.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import os
import pytest

from pathlib import Path
from numpy.testing import assert_allclose
from scipy import sparse
from scipy import linalg as la
from scipy.sparse import linalg as spla

from .helpers import generate_suitesparse_matrices

import csparse


ATOL = 1e-12  # random matrices need a bit larger tolerance than 1e-15

# Group the matrices for parameterization
N = 7  # arbitrary matrix size for testing
TEST_MATRICES = [
    ("Identity", sparse.eye_array(N).tocsc()),
    ("Diagonal", sparse.diags(np.arange(1, N)).tocsc()),
    ("Asymmetric Banded",
        sparse.diags([np.ones(N-1), np.arange(1, N+1)], [-1, 0]).tocsc()),
    ("Laplacian (Symmetric Banded)",
        sparse.diags([1, -2, 1], [-1, 0, 1]).tocsc()),
    ("Davis 8x8", csparse.davis_example_qr()),
    ("Davis 4x4", csparse.davis_example_small()),
    # See: Strang Linear Algebra p 203.
    ("Strang 3x3", sparse.csc_array(
        np.array([[1, 1, 2], [0, 0, 1], [1, 0, 0]], dtype=float)
    )),
    ("Davis 8x5 (M > N)", csparse.davis_example_qr()[:, :5]),
    ("Davis 5x8 (M < N)", csparse.davis_example_qr()[:5, :])
]


def categorize_shape(M, N):
    if M < N:
        return "under"
    elif M == N:
        return "square"
    else:
        return "over"


def generate_test_matrices():
    """Generate all matrices for testing."""
    # Fixed test matrices
    for case_name, A in TEST_MATRICES:
        M, N = A.shape
        shape_cat = categorize_shape(M, N)
        test_id = f"{shape_cat}::{case_name}"
        yield pytest.param(shape_cat, case_name, A, id=test_id)

    # Random test matrices
    seed = 565656
    rng = np.random.default_rng(seed)
    Ns = np.r_[2, 7, 10]
    Ms = (Ns * np.r_[0.6, 1, 1.4]).astype(int)
    N_runs = 10

    for N in Ns:
        for i in range(N_runs):
            for M in Ms:
                A = sparse.random(M, N,
                                  density=0.5, format='csc', random_state=rng)
                # ensure structural full rank
                A.setdiag(N * np.arange(1, min(M, N) + 1))

                shape_cat = categorize_shape(M, N)
                case_name = f"Random {M}x{N} ({seed=}, {i=})"
                test_id = f"{shape_cat}::{case_name}"
                yield pytest.param(shape_cat, case_name, A,
                                   id=test_id,
                                   marks=pytest.mark.random)


@pytest.mark.parametrize("order", ['Natural', 'ATA'])
@pytest.mark.parametrize("shape_cat, case_name, A", generate_test_matrices())
def test_csparse_qr(shape_cat, case_name, A, order):
    """Test QR decomposition with various matrices using parametrization."""
    A_dense = A.toarray()
    M, N = A.shape

    # ---------- Compute csparse QR
    V, beta, R, p, q = csparse.qr(A, order)

    if order == 'Natural':
        assert_allclose(q, np.arange(N))

    # Convert to numpy arrays
    V = V.toarray()
    R = R.toarray()

    p_inv = csparse.inv_permute(p)
    Q = csparse.apply_qright(V, beta, p)
    Ql = csparse.apply_qtleft(V, beta, p).T

    # Test the apply functions both get the same Q
    assert_allclose(Q, Ql, atol=ATOL)

    # ---------- scipy QR
    # Apply the row permutation to A_dense
    Apq = A_dense[p][:, q]
    (Qraw, tau), Rraw = la.qr(Apq, mode='raw')
    Q_, R_ = la.qr(Apq)
    # Handle case when M < N
    V_ = np.tril(Qraw, -1)[:, :M] + np.eye(M, min(M, N))
    Qr_ = csparse.apply_qright(V_, tau, p)

    # Now we get the same Householder vectors and weights
    assert_allclose(V, V_, atol=ATOL)
    assert_allclose(beta, tau, atol=ATOL)
    assert_allclose(R, R_, atol=ATOL)

    # Q is the same up to row permutation
    assert_allclose(Q, Qr_, atol=ATOL)
    assert_allclose(Q, Q_[p_inv], atol=ATOL)
    assert_allclose(Q[p], Q_, atol=ATOL)

    # Reproduce A = QR
    assert_allclose(Q_ @ R_, Apq, atol=ATOL)
    assert_allclose(Q_[p_inv] @ R_, A_dense[:, q], atol=ATOL)
    assert_allclose(Q @ R, A_dense[:, q], atol=ATOL)


@pytest.mark.parametrize("shape_cat, case_name, A", generate_test_matrices())
def test_apply_q(shape_cat, case_name, A):
    """Test application of the Householder reflectors."""
    A = A.toarray()  # only test with dense matrices
    M, N = A.shape

    (Qraw, tau), Rraw = la.qr(A, mode='raw')

    if M > N:
        # Extra rows of zeros are not included in the LAPACK output
        Rraw = np.vstack([Rraw, np.zeros((M - N, N))])

    # Check that the raw LAPACK output is as expected
    assert_allclose(np.triu(Qraw), Rraw, atol=ATOL)

    # Get the Householder reflectors from the raw LAPACK output
    V_ = np.tril(Qraw, -1)[:, :M] + np.eye(M, min(M, N))

    # Apply them to the identity to get back Q itself
    Q_r = csparse.apply_qright(V_, tau)
    Q_l = csparse.apply_qtleft(V_, tau).T
    assert_allclose(Q_r, Q_l, atol=ATOL)

    # Compare to the scipy output
    Q_, R_ = la.qr(A)
    assert_allclose(Q_r, Q_, atol=ATOL)

    # Ensure scipy is self-consistent
    assert_allclose(R_, Rraw, atol=ATOL)


@pytest.mark.parametrize("shape_cat, case_name, A", generate_test_matrices())
@pytest.mark.parametrize("qr_func", [csparse.qr_right, csparse.qr_left])
def test_qrightleft(shape_cat, case_name, A, qr_func):
    """Test the python QR decomposition algorithms."""
    A = A.toarray()  # only test with dense matrices
    M, N = A.shape

    if qr_func == csparse.qr_right and M < N:
        A = A.T  # transpose to make it M >= N

    M, N = A.shape

    # Test our own python QR decomposition
    V, beta, R = qr_func(A)
    Q = csparse.apply_qright(V, beta)

    # Compare to scipy
    (Qraw, tau), Rraw = la.qr(A, mode='raw')
    V = np.tril(Qraw, -1)[:, :M] + np.eye(M, min(M, N))

    assert_allclose(V, V, atol=ATOL)
    assert_allclose(beta, tau, atol=ATOL)

    # Compare to scipy's QR
    Q_, R_ = la.qr(A)
    assert_allclose(Q_, Q, atol=ATOL)
    assert_allclose(R_, R, atol=ATOL)

    # Reproduce A = QR
    assert_allclose(Q @ R, A, atol=ATOL)


# -----------------------------------------------------------------------------
#         Test 9
# -----------------------------------------------------------------------------
# TODO don't really need to exercise qr_(left|right) here. We could just remove
# this whole class and write it as a single test function for csparse.qr().
@pytest.mark.parametrize(
    "problem",
    list(generate_suitesparse_matrices()),
    indirect=True
)
class TestQR:
    """Test class for the QR decomposition on SuiteSparse matrices."""
    @pytest.fixture(scope='class')
    def problem(self, request):
        """Fixture to provide the problem instance."""
        return request.param

    @pytest.fixture(scope='class', autouse=True)
    def setup_problem(self, request, problem):
        """Set up the problem instance for the test class."""
        cls = request.cls
        problem = problem
        print(f"---------- {problem.name}")

        A = problem.A

        if A.shape[0] < A.shape[1]:
            A = A.T

        M, N = A.shape

        # rank = sparse.csgraph.structural_rank(A)

        A_orig = A.copy()

        q = csparse.amd(A, order='ATA')  # like colamd in MATLAB
        A = A[:, q]
        # Use the singular values to compare with R from each decomposition
        cls.sig = la.svdvals(A.toarray())

        # Compute scipy R factor
        _, R_ = la.qr(A.toarray())

        # TODO treeplot using csgraph?
        # [c, h, parent] = symbfact(A, 'col');
        # only uses `parent` in treeplot.

        # Compute csparse QR
        V, beta, R, p, q = csparse.qr(A)

        C = A.copy()
        M2 = V.shape[0]
        if M2 > M:
            # Add empty rows
            C = sparse.vstack([A, sparse.csc_array((M2 - M, N))])

        cls.C = C[p]
        cls.R = R

        # Make the plot
        if not request.config.getoption('--make-figures'):
            yield  # skip the setup if not making figures
            return

        fig, axs = plt.subplots(num=1, nrows=2, ncols=4, clear=True)
        fig.suptitle(f"QR Factors for {cls.problem.name}")

        axs[0, 0].spy(A, markersize=1)
        axs[0, 1].spy(cls.C, markersize=1)
        # treeplot(parent, ax=axs[0, 2])  # TODO
        axs[0, 3].spy(A_orig, markersize=1)
        axs[1, 0].spy(abs(R) > 0, markersize=1)
        axs[1, 1].spy(R_, markersize=1)
        axs[1, 2].spy(R, markersize=1)
        axs[1, 3].spy(V, markersize=1)

        axs[0, 0].set_title('A colamd')
        axs[0, 1].set_title('A permuted')
        axs[0, 2].set_title('A treeplot')
        axs[0, 3].set_title('A original')
        axs[1, 0].set_title('R abs > 0')
        axs[1, 1].set_title('R scipy')
        axs[1, 2].set_title('R csparse')
        axs[1, 3].set_title('V csparse')

        # Run the tests
        yield

        # Teardown code (save the figure)
        fig_dir = Path('test_figures/test_qr')
        os.makedirs(fig_dir, exist_ok=True)

        figure_path = fig_dir / f"{problem.name.replace('/', '_')}.pdf"
        print(f"Saving figure to {figure_path}")
        fig.savefig(figure_path)

        plt.close(fig)

    def test_qr(self):
        """Test the CSparse qr function vs scipy."""
        s = la.svdvals(self.R)
        err = la.norm(self.sig - s, ord=1) / self.sig[0]
        print(f"qr: {err:.2e}")
        assert err < 1e-12

    @pytest.mark.parametrize('qr_func', [csparse.qr_right, csparse.qr_left])
    def test_qr(self, qr_func):
        """Test left-looking QR decomposition."""
        V, beta, R = qr_func(self.C.toarray())
        s = la.svdvals(R)
        err = la.norm(self.sig - s, ord=1) / self.sig[0]
        print(f"qr_left: {err:.2e}")
        assert err < 1e-12

# =============================================================================
# =============================================================================
