#!/usr/bin/env python3
# =============================================================================
#     File: plot.py
#  Created: 2025-05-07 19:44
#   Author: Bernie Roesler
#
"""
Functions for plotting sparse matrices.
"""
# =============================================================================

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import MaxNLocator
from scipy.sparse import issparse

from csparse.csparse import dmperm
from csparse._fillreducing import scc_perm


def cspy(A, cmap='viridis_r', colorbar=True, ax=None, **kwargs):
    """Visualize a sparse or dense matrix with colored markers.

    This function is similar to `matplotlib.pyplot.spy`, but it colors the
    markers based on the value of the non-zero elements in the matrix.
    It can handle both dense NumPy arrays and SciPy sparse matrices.

    Parameters
    ----------
    A : array_like
        The 2D matrix to visualize. Can be a NumPy array, SciPy sparse
        matrix, or any object convertible to a 2D NumPy array.
    cmap : str or matplotlib.colors.Colormap, optional
        The colormap to use for coloring the markers, by default 'viridis_r'.
    colorbar : bool, optional
        Whether to display a colorbar, by default True.
    ax : matplotlib.axes.Axes, optional
        An existing Axes object to plot on. If None (default), the current axes
        are used.
    **kwargs?
        Additional keyword arguments passed directly to
        `matplotlib.pyplot.imshow`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes object used for plotting.
    cb : matplotlib.colorbar.Colorbar
        The colorbar object.

    See Also
    --------
    matplotlib.pyplot.spy : Plot the sparsity pattern of a 2D array.
    matplotlib.pyplot.imshow : Display data as an image, i.e., on a 2D regular
        raster.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> data = np.array([[1, 0, 2], [0, -3, 0], [4, 0, 0]])
    >>> ax = cspy(data, markersize=50)
    >>> plt.show()

    >>> from scipy.sparse import csr_array
    >>> sparse_data = csr_array(data)
    >>> ax = cspy(sparse_data, cmap='coolwarm')
    >>> plt.show()
    """
    if ax is None:
        ax = plt.gca()  # get current Axes if not provided

    fig = ax.figure

    if issparse(A):
        dense_matrix = A.toarray().astype(np.float64)
    else:
        try:
            dense_matrix = np.array(A, dtype=np.float64)
        except Exception as e:
            raise TypeError(
                "Input matrix must be a NumPy array, SciPy sparse matrix, "
                f"or convertible to a 2D NumPy array. Error: {e}"
            )

    if dense_matrix.ndim != 2:
        raise ValueError("Input matrix must be 2-dimensional.")

    M, N = dense_matrix.shape
    nnz = np.count_nonzero(dense_matrix)

    # Set zeros to NaN
    dense_matrix[dense_matrix == 0] = np.nan

    # Set plot limits and aspect ratio
    # Ensure limits are appropriate even for single row/column matrices
    ax.set_xlim(-0.75, N - 0.25 if N > 0 else 0.75)
    ax.set_ylim(M - 0.25 if M > 0 else 0.75, -0.75)  # inverted y-axis like spy

    ax.xaxis.tick_top()  # match spy's x-axis orientation
    # ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)

    # Use MaxNLocator to ensure integer ticks on both axes
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    if nnz == 0:
        ax.set_xlabel(f"{A.shape}, nnz = 0, density = 0")
        return ax

    ax.set_xlabel((f"{A.shape}, nnz = {nnz}, "
                   f"density = {nnz / (M * N):.2%}"))

    # Convert to a dense matrix and use imshow
    im = ax.imshow(dense_matrix, cmap=cmap, origin='upper', aspect='equal',
                   **kwargs)

    # Add a colorbar
    if colorbar:
        cb = fig.colorbar(im, ax=ax, shrink=0.8)
    else:
        cb = None

    return ax, cb


def dmspy(A, colored=True, seed=0, ax=None, **kwargs):
    """Plot the Dulmage-Mendelsohn (DM) ordering of a sparse matrix.

    Parameters
    ----------
    A : (M, N) array_like
        Matrix of M vectors in N dimensions.
    colored : bool, optional
        If True, color the points based on their values, by default True.
    seed : int, optional
        Random seed passed to `csparse.dmperm`, by default 0.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If `None`, the current axes are used.
    **kwargs
        Additional keyword arguments passed to `matplotlib.pyplot.spy`, or
        `csparse.cspy` (depending on `colored`).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes object used for plotting.
    cb : matplotlib.colorbar.Colorbar
        The colorbar object, if `colored` is True. Otherwise, None.
    """
    if ax is None:
        ax = plt.gca()

    # Compute the Dulmage-Mendelsohn (DM) ordering
    p, q, r, s, rr, cc, Nb = dmperm(A, seed=seed)

    # Plot the result
    S = A[p][:, q]

    if colored:
        ax, cb = cspy(S, ax=ax, **kwargs)
    else:
        ax.spy(S, **kwargs)
        cb = None

    # Set the title
    M, N = A.shape
    sprank = rr[3]
    m = np.nonzero(np.diff(rr))[0].size
    n = np.nonzero(np.diff(cc))[0].size

    ax.set_title(
        f"{M}-by-{N}, sprank: {sprank:d},\n"
        f"fine blocks: {Nb}, coarse blocks: {m, n}"
    )

    # Draw boxes around the blocks
    drawboxes(Nb, r, s, ax=ax)
    # Draw boxes around the singletons
    M, N = A.shape

    # Box around entire matrix
    # drawbox(0, M, 0, N, ec='C4', fc='none', lw=2, ax=ax)

    # TODO label the boxes
    drawbox(rr[0], rr[1], cc[0], cc[1], ec='C0', fc='none', lw=2, ax=ax)
    drawbox(rr[0], rr[1], cc[1], cc[2], ec='C1', fc='none', lw=2, ax=ax)
    drawbox(rr[1], rr[2], cc[2], cc[3],  ec='k', fc='none', lw=2, ax=ax)
    drawbox(rr[2], rr[3], cc[3], cc[4], ec='C2', fc='none', lw=2, ax=ax)
    drawbox(rr[3], rr[4], cc[3], cc[4], ec='C4', fc='none', lw=2, ax=ax)

    return ax, cb


def ccspy(A, colored=True, seed=0, ax=None, **kwargs):
    """Plot the connected components of a sparse matrix.

    Parameters
    ----------
    A : (M, N) array_like
        Matrix of M vectors in N dimensions.
    colored : bool, optional
        If True, color the points based on their values, by default True.
    seed : int, optional
        Random seed passed to `csparse.dmperm`, by default 0.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If `None`, the current axes are used.
    **kwargs
        Additional keyword arguments passed to `matplotlib.pyplot.spy`, or
        `csparse.cspy` (depending on `colored`).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes object used for plotting.
    cb : matplotlib.colorbar.Colorbar
        The colorbar object, if `colored` is True. Otherwise, None.
    """
    if ax is None:
        ax = plt.gca()

    # Find the strongly connected components
    p, q, r, s = scc_perm(A)
    S = A[p][:, q]

    Nb = r.size - 1

    if colored:
        ax, cb = cspy(S, ax=ax, **kwargs)
    else:
        ax.spy(S, **kwargs)
        cb = None

    M, N = A.shape
    ax.set_title(f"{M}-by-{N}, strongly connected components: {Nb:d}")

    drawboxes(Nb, r, s, ax=ax)

    return ax, cb


def drawboxes(Nb, r, s, ax=None, **kwargs):
    """Draw boxes around the blocks of a matrix.

    Parameters
    ----------
    Nb : int
        Number of blocks.
    r, s : (M,) array_like
        Row and column indices of the blocks, typically a result of the
        `dmperm` function.
    ax : matplotlib.axes.Axes, optional
    **kwargs : dict
        Additional keyword arguments passed to `matplotlib.patches.Rectangle`.

    Returns
    -------
    ax : (M, N) ndarray
        Matrix of M vectors in N dimensions
    """
    # Default styling
    opts = dict(ec='C3', fc='none', lw=2)
    opts.update(kwargs)

    if Nb > 1:
        r1 = r[:Nb]
        r2 = r[1:Nb+1]
        c1 = s[:Nb]
        c2 = s[1:Nb+1]

        kk = np.nonzero(
            (np.diff(c1) > 0) |
            (np.diff(c2) > 0) |
            (np.diff(r1) > 0) |
            (np.diff(r2) > 0)
        )[0]

        for k in kk:
            rect = patches.Rectangle(
                (c1[k] - 0.5, r1[k] - 0.5),  # shift center to corner of pixel
                width=c2[k] - c1[k],
                height=r2[k] - r1[k],
                **opts
            )
            ax.add_patch(rect)


def drawbox(r1, r2, c1, c2, ax=None, **kwargs):
    """Draw a box on the given axes."""
    if c2 < c1 or r2 < r1:
        return

    if ax is None:
        ax = plt.gca()

    opts = dict(edgecolor='k', facecolor='none', lw=2)
    opts.update(kwargs)

    # Draw a rectangle
    rect = patches.Rectangle(
        (c1 - 0.5, r1 - 0.5),  # shift center to corner of pixel
        width=c2 - c1,
        height=r2 - r1,
        **opts
    )
    ax.add_patch(rect)


if __name__ == '__main__':
    # --- Example Usage ---
    plt.close('all')

    # # 1. Dense NumPy array
    dense_matrix = np.array([
        [1.5, 0, 0, 2.1],
        [0, -3.3, 0, 0],
        [0, 0, 0, 0],
        [4.7, 0, 5.0, -0.5]
    ])
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    cspy(dense_matrix, cmap='coolwarm', ax=ax1)
    ax1.set_title("Dense Matrix")
    plt.show()

    # 2. SciPy sparse matrix (if SciPy is installed)
    from scipy.sparse import csr_array
    sparse_matrix_data = np.array([10, 20, -30, 40, 50, 60, -70, 80, 5, -15])
    row_ind = np.array([0, 0, 1, 2, 3, 3, 4, 4, 0, 1])
    col_ind = np.array([0, 2, 1, 3, 0, 3, 1, 2, 3, 3])
    # Ensure shape is large enough for all indices
    shape = (max(row_ind) + 1, max(col_ind) + 1)
    sparse_matrix = csr_array(
        (sparse_matrix_data, (row_ind, col_ind)),
        shape=shape
    )

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    cspy(sparse_matrix, cmap='plasma',
         vmin=-100, vmax=100, ax=ax2)
    ax2.set_title("Sparse Matrix")
    plt.show()

    # Example with a different sparse format (COO)
    from scipy.sparse import coo_array
    row = np.array([0, 3, 1, 0, 5, 5, 2])
    col = np.array([0, 3, 1, 2, 0, 2, 2])
    data = np.array([1, 2.5, 3.1, 4.9, -1.2, -5.5, 0.5])
    coo_m = coo_array((data, (row, col)), shape=(6, 4))
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    cspy(coo_m, cmap='viridis', ax=ax3)
    ax3.set_title("COO Sparse Matrix")
    plt.show()

    # 3. Matrix with all zeros
    zero_matrix = np.zeros((5, 5))
    fig4, ax4 = plt.subplots(figsize=(5, 4))
    cspy(zero_matrix, ax=ax4)
    ax4.set_title("Zero Matrix")
    plt.show()

    # 4. Empty matrix
    empty_matrix = np.array([[]])  # or np.empty((0,5)) or np.empty((5,0))
    fig5, ax5 = plt.subplots(figsize=(5, 4))
    cspy(empty_matrix, ax=ax5)
    ax5.set_title("Empty Matrix")
    plt.show()

    # 5. Larger random matrix (more sparse-like)
    np.random.seed(42)
    large_random_matrix = np.random.randn(25, 35)
    large_random_matrix[np.abs(large_random_matrix) < 0.8] = 0
    fig6, ax6 = plt.subplots(figsize=(8, 6))
    cspy(large_random_matrix, cmap='magma', ax=ax6)
    ax6.set_title("Larger Random Matrix")
    plt.show()

    # 6. Matrix with only positive values and custom normalization
    positive_matrix = np.abs(dense_matrix) + 1  # ensure all positive
    fig7, ax7 = plt.subplots(figsize=(6, 5))
    # Example of using vmin/vmax for color normalization
    cspy(positive_matrix, cmap='Reds',
         vmin=0, vmax=np.max(positive_matrix)+2, ax=ax7)
    ax7.set_title("Positive Values Matrix with vmin/vmax")
    plt.show()

    # 7. Using an existing Axes object
    fig8, (ax_spy, ax_cspy) = plt.subplots(1, 2, figsize=(12, 5))
    # Standard spy plot
    ax_spy.spy(dense_matrix)
    ax_spy.set_title("pyplot.spy")
    # Our cspy plot on the second axes
    cspy(dense_matrix, cmap='coolwarm', ax=ax_cspy)
    ax_cspy.set_title("cspy")
    plt.show()

# =============================================================================
# =============================================================================
