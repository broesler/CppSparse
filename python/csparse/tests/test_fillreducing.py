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

from .helpers import generate_random_matrices

import csparse


# -----------------------------------------------------------------------------
#         Test 15
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    'A', generate_random_matrices(N_max=200, d_scale=0.05, square_only=True)
)
def test_amd(A, request):
    """Test AMD fill-reducing ordering."""
    M, N = A.shape
    assert M == N, 'Matrix must be square for AMD ordering.'

    # Add a randomly-placed dense column
    k = np.random.randint(0, N)
    A = A.todok()
    A[:, k] = 1.0

    # TODO scipy AMD ordering? Use sparse.splu(A, permc_spec='MMD_ATA')?

    p = csparse.amd(A, order='APlusAT')

    assert np.array_equal(np.sort(p), np.arange(N)), \
        'Permutation is not a valid reordering.'

    C = A + A.T + sparse.eye_array(N)

    if request.config.getoption('--make-figures'):
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

# =============================================================================
# =============================================================================
