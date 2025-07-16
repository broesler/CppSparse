#!/usr/bin/env python3
# =============================================================================
#     File: helpers.py
#  Created: 2025-05-28 15:44
#   Author: Bernie Roesler
#
"""Helper functions for the C++Sparse python tests."""
# =============================================================================

import pytest

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from scipy import sparse
from suitesparseget import suitesparseget as ssg


# -----------------------------------------------------------------------------
#         Matrix Generators
# -----------------------------------------------------------------------------
def generate_suitesparse_matrices(N=100, real_only=True, square_only=False):
    """Generate a list of SuiteSparse matrices."""
    df = ssg.get_index()

    # Get the list of the N smallest SuiteSparse matrices
    max_dim = df[['nrows', 'ncols']].max(axis=1)
    tf = df.loc[max_dim.sort_values().index]

    filters = (
        (tf['is_real'] if real_only else True) &
        (tf['nrows'] == tf['ncols'] if square_only else True)
    )

    tf = tf[filters]

    for idx, row in tf.head(N).iterrows():
        try:
            problem = ssg.get_problem(row=row, fmt='mat')
        except NotImplementedError as e:
            print(f"Skipping matrix {idx} due to: {e}")
            continue

        yield pytest.param(
            problem,
            id=f"{problem.id}::{problem.name}",
            marks=pytest.mark.suitesparse
        )


def generate_random_matrices(
    seed=565656,
    N_trials=100,
    N_max=10,
    square_only=True,
    d_scale=1
):
    """Generate a list of random sparse matrices of maximum size N x N."""
    rng = np.random.default_rng(seed)
    for trial in range(N_trials):
        # Generate a random sparse matrix
        if square_only:
            M = N = rng.integers(1, N_max, endpoint=True)
        else:
            M, N = rng.integers(1, N_max, size=2, endpoint=True)

        d = d_scale * rng.random()  # density

        A = sparse.random_array(
            (M, N),
            density=d,
            format='csc',
            random_state=rng
        )

        yield pytest.param(
            A,
            id=f"random_{trial:02d}::{A.shape}::{A.nnz}",
            marks=pytest.mark.random
        )


def generate_random_compatible_matrices(
    seed=565656, N_trials=100, N_max=10, kind='multiply'
):
    """Generate a list of random sparse matrices with compatible shapes."""
    rng = np.random.default_rng(seed)

    for trial in range(N_trials):
        # Generate a random sparse matrix
        M, N, K = rng.integers(1, N_max, size=3, endpoint=True)
        d = rng.random()  # density ∈ [0, 1]

        if kind == 'multiply':
            A_shape = (M, N)
            B_shape = (N, K)
        elif kind == 'add':
            A_shape = B_shape = (M, N)

        A = sparse.random_array(
            A_shape,
            density=d,
            format='csc',
            random_state=rng,
            data_sampler=rng.normal
        )

        B = sparse.random_array(
            B_shape,
            density=d,
            format='csc',
            random_state=rng,
            data_sampler=rng.normal
        )

        yield pytest.param(
            A, B,
            id=f"random_{trial:02d}::{A.shape}::{A.nnz}::{B.nnz}",
            marks=pytest.mark.random
        )


def generate_random_cholesky_matrices(seed=565656, N_trials=100, N_max=100):
    """Generate a list of random, square, lower-triangular matrices."""
    rng = np.random.default_rng(seed)

    for trial in range(N_trials):
        # Generate a random sparse matrix
        N = rng.integers(1, N_max, endpoint=True)
        d = 0.1 * rng.random()  # density ∈ [0, 0.1]

        A = sparse.random_array(
            (N, N),
            density=d,
            format='csc',
            random_state=rng,
            data_sampler=rng.normal
        )

        # Make it lower triangular
        D = sparse.diags(rng.random(N), 0, shape=(N, N))
        L = sparse.tril(A, -1) + D

        # RHS column vector
        b = sparse.random_array(
            (N, 1),
            density=d,
            format='csc',
            random_state=rng,
            data_sampler=rng.normal
        )

        yield pytest.param(
            L, b,
            id=f"random_{trial:02d}::{L.shape}::{L.nnz}::{b.nnz}",
            marks=pytest.mark.random
        )


def generate_pvec_params(seed=565656, N_trials=100, N_max=10):
    """Generate random permutation vectors and values."""
    rng = np.random.default_rng(seed)
    for i in range(N_trials):
        M = rng.integers(1, N_max, endpoint=True)
        p = rng.permutation(M)
        x = rng.random(M)
        yield pytest.param(
            p, x,
            id=f"trial_{i+1}_seed_{seed}",
            marks=pytest.mark.random
        )


# -----------------------------------------------------------------------------
#         Matrix Type Checking
# -----------------------------------------------------------------------------
def is_real(A):
    """Check if a sparse matrix is real-valued.

    Parameters
    ----------
    A : sparse.sparray
        The sparse matrix to check.

    Returns
    -------
    bool
        True if the matrix is real-valued, False otherwise.
    """
    return np.issubdtype(A.dtype, np.floating)


def is_complex(A):
    """Check if a sparse matrix is complex-valued.

    Parameters
    ----------
    A : sparse.sparray
        The sparse matrix to check.

    Returns
    -------
    bool
        True if the matrix is complex-valued, False otherwise.
    """
    return np.issubdtype(A.dtype, np.complexfloating)


def is_valid_permutation(p):
    """Check if a vector is a valid permutation."""
    return np.array_equal(np.sort(p), np.arange(len(p)))


# -----------------------------------------------------------------------------
#         Test Classes
# -----------------------------------------------------------------------------
class BaseSuiteSparseTest:
    """An abstract base class for tests."""

    @pytest.fixture(scope='class')
    def problem(self, request):
        """Fixture to provide the problem matrix."""
        return request.param

    @pytest.fixture(scope='class', autouse=True)
    def base_setup_problem(self, request, problem):
        """Initialize the problem."""
        cls = request.cls

        if isinstance(problem, ssg.MatrixProblem):
            cls.problem = problem
        elif isinstance(problem, sparse.sparray):
            A = problem
            cls.problem = ssg.MatrixProblem(
                id=f"random_{A.shape[0]}_{A.shape[1]}_{A.nnz}",
                name=f"Random {A.shape}, {A.nnz} nnz",
                A=A
            )
        else:
            raise TypeError(f"Expected MatrixProblem or sparse.sparray, "
                            f"got {type(problem)}")

        # print(f"Testing matrix {cls.problem.id} ({cls.problem.name})")
        # Subclasses should override this method to set up the problem


class BaseSuiteSparsePlot(BaseSuiteSparseTest):
    """An abstract base class for tests that require a plot."""

    # Default values for parameters
    _nrows = 1
    _ncols = 1
    _fig_dir = Path('test_suitesparse')
    _fig_title_prefix = ''

    @pytest.fixture(scope='class', autouse=True)
    def setup_plot(self, request, base_setup_problem):
        """Set up the problem and figure for plotting across tests."""
        cls = request.cls

        cls.make_figures = request.config.getoption('--make-figures')

        if not cls.make_figures:
            yield  # skip the setup if not making figures
            return

        cls.fig, cls.axs = plt.subplots(
            num=1,
            nrows=cls._nrows,
            ncols=cls._ncols,
            clear=True
        )
        cls.fig.suptitle(f"{cls._fig_title_prefix}{cls.problem.name}")
        cls.fig.set_size_inches((3 * cls._ncols, 4 * cls._nrows))

        def finalize_plot():
            """Finalize the plot after all tests."""
            if cls.make_figures:
                # Save the figure
                cls.fig_dir = Path('test_figures') / cls._fig_dir
                cls.fig_dir.mkdir(parents=True, exist_ok=True)

                cls.figure_path = (
                    cls.fig_dir /
                    f"{cls.problem.name.replace('/', '_')}.pdf"
                )
                print(f"Saving figure to {cls.figure_path}")
                cls.fig.savefig(cls.figure_path)

            # Clean up
            if hasattr(cls, 'fig') and cls.fig is not None:
                plt.close(cls.fig)
                del cls.fig
                del cls.axs

        # Make sure to finalize the plot after all tests
        request.addfinalizer(finalize_plot)

        # Run the tests
        yield

# =============================================================================
# =============================================================================
