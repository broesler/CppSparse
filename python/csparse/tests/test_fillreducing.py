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

from pathlib import Path
from scipy import sparse

from .helpers import (
    generate_suitesparse_matrices,
    generate_random_matrices
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

    # TODO def is_valid_permutation(p, N)
    assert np.array_equal(np.sort(p), np.arange(N)), \
        'Permutation is not a valid reordering.'

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
        assert False
        figure_path = fig_dir / f"{test_id}.pdf"
        print(f"Saving figure to {figure_path}")
        fig.savefig(figure_path)

        plt.close(fig)


@pytest.mark.parametrize(
    'problem',
    list(generate_suitesparse_matrices(N=200)),
    indirect=True
)
class TestAMD:
    """Test AMD fill-reducing ordering on SuiteSparse matrices."""
    @pytest.fixture(scope='class')
    def problem(self, request):
        """Fixture to provide the problem matrix."""
        return request.param

    @pytest.fixture(scope='class', autouse=True)
    def setup_class(self, request, problem):
        """Setup method to initialize the problem matrix."""
        cls = request.cls
        cls.problem = problem
        print(f"Testing matrix {problem.id} ({problem.name})")

        A = problem.A
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

        assert np.array_equal(np.sort(p), np.arange(self.N)), \
            'Permutation is not a valid reordering.'

    def test_colamd(self):
        """Test COLAMD fill-reducing ordering."""
        p = csparse.amd(self.A_orig, order='ATA')
        M, N = self.A_orig.shape

        assert np.array_equal(np.sort(p), np.arange(N)), \
            'Permutation is not a valid reordering.'

# =============================================================================
# =============================================================================
