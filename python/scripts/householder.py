#!/usr/bin/env python3
# =============================================================================
#     File: householder.py
#  Created: 2025-02-19 10:20
#   Author: Bernie Roesler
#
"""
Demonstration of the different Householder vector computations.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la


def house(x, method='LAPACK'):
    r"""Compute the Householder reflection vector for a given vector x.

    The Householder reflection is defined as:

    .. math:: H = I - \beta v v^T

    where :math:`\beta = \frac{2}{v^T v}` and 
    :math:`v = \[1, v_2, \ldots, v_n\]^T`.

    The reflection is defined such that the product

    .. math:: Hx = \pm \|x\| e_1

    where :math:`e_1` is the first unit vector.
    
    The `method` argument determines which reflector is used. The LAPACK
    method defines:

    .. math:: v = \[ 1, \frac{x_2}{x_1 + \text{sign}(x_1) \|x\|} \]^T

    whereas the Davis method defines:

    .. math:: v = \[ x_1 - \|x\|, x_2 \]^T.

    The LAPACK method chooses the sign of the denominator to avoid
    cancellation, whereas the Davis method uses additional logic to compute the
    denominator in a numerically stable way in the case where :math:`x_1 < 0`.

    This function normalizes the reflector such that :math:`v_1 = 1` for both
    methods.

    Parameters
    ----------
    x : (N,) array_like
        The vector to be reflected.
    method : str in {'LAPACK', 'Davis'}, optional
        The method to use for computing the Householder vector.
        The 'LAPACK' method taken from LAPACK's DLARFG subroutine. It is the
        same algorithm described in Trefethen and Bau [Tref]_.
        The 'Davis' method is taken from Davis [Davis]_, which is 
        equivalent to Golub & Van Loan [GVL]_.

    Returns
    -------
    v : (N,) ndarray
        The Householder reflection vector.
    beta : float
        The scaling factor.
    s : float
        The L2-norm of the vector x.     References
    ----------
    .. [Tref] Trefethen, Lloyd and David Bau (1997).
        "Numerical Linear Algebra". Eq (10.5), and Algorithm 10.1.
    .. [Davis] Davis, Timothy A. (2006).
        "Direct Methods for Sparse Linear Systems", p 69 (`cs_house`).
    .. [GVL] Golub, Gene H. and Charles F. Van Loan (1996).
        "Matrix Computations". Algorithm 5.1.1.
    """
    v = np.copy(x)
    σ = np.sum(v[1:]**2)

    if σ == 0:
        s = np.abs(v[0])
        beta = 2 if v[0] <= 0 else 0  # make direction positive if x[0] < 0
        v[0] = 1                      # make the reflector a unit vector
    else:
        s = np.sqrt(v[0]**2 + σ)   # ||x||_2

        if method == 'LAPACK':
            a = v[0]
            b = -np.sign(a) * s
            β = (b - a) / b 
            # v = x + sign(x[0]) * ||x|| e_1
            # v /= v[0]
            v[0] = 1
            v[1:] /= (a - b)  # a - b == x[0] + sign(x[0]) * ||x||
        elif method == 'Davis':
            # These options compute equivalent values, but the v[0] > 0 case
            # is a more numerically stable option.
            v[0] = (v[0] - s) if v[0] <= 0 else (-σ / (v[0] + s))
            β = -1 / (s * v[0])
            # Normalize β and v s.t. v[0] = 1
            β *= v[0] * v[0]
            v /= v[0]
        else:
            raise ValueError(f"Unknown method '{method}'.")

    return v, β, s


def plot_vector(v, text=None, ax=None, **kwargs):
    """Plot a vector in 2D space."""
    if ax is None:
        ax = plt.gca()

    ax.quiver(0, 0, v[0], v[1],
              angles='xy', scale_units='xy', scale=1, 
              width=0.005,
              **kwargs)

    # Annotate vector
    if text is not None:
        kwargs.pop('label', None)  # not used in ax.text()
        ax.text(1.01*v[0], 1.01*v[1],
                f"{text}: ({v[0]:.2g}, {v[1]:.2g})",
                **kwargs)

    return ax


def annotate_vector(v, name='', ax=None, **kwargs):
    """Annotate a vector a name and its components."""
    if ax is None:
        ax = plt.gca()

    return ax


if __name__ == "__main__":
    # Show numerical *instability* of the naïve method
    ϵ = 1e-15  # a very small number

    x = np.r_[1 + ϵ, ϵ]  # a vector very close to e_1 == [1, 0]

    v = x - la.norm(x)  # cancellation occurs!
    with np.errstate(divide='ignore', invalid='ignore'):
        v /= v[0]
    
    print("v:")
    print("unstable:", v)  # [nan, -inf] divide by zero!

    # Show numerical *stability* of the Davis method
    v_D, _, _ = house(x, method='Davis')
    print("   Davis:", v_D)

    v_L, _, _ = house(x, method='LAPACK')
    print("  LAPACK:", v_L)

    # -------------------------------------------------------------------------
    #         Plot the vectors and reflectors
    # -------------------------------------------------------------------------
    # Create a vector with an easy norm
    x = np.r_[3., 4.]  # |x| == 5

    v_D, β_D, s_D = house(x, method='Davis')
    v_L, β_L, s_L = house(x, method='LAPACK')

    # Compute the Householder matrices
    I = np.eye(x.size)
    H_D = I - β_D * np.outer(v_D, v_D)
    H_L = I - β_L * np.outer(v_L, v_L)

    Hx_D = H_D @ x
    Hx_L = H_L @ x

    # Compare to hand calculations
    np.testing.assert_allclose(v_D, np.r_[1, -2])
    np.testing.assert_allclose(v_L, np.r_[1, 0.5])
    np.testing.assert_allclose(β_D, 0.4)
    np.testing.assert_allclose(β_L, 1.6)
    np.testing.assert_allclose(s_D, 5)
    np.testing.assert_allclose(s_L, 5)

    # Need atol when comparing to 0
    np.testing.assert_allclose(H_L, -H_D)
    np.testing.assert_allclose(Hx_D, np.r_[s_D, 0], atol=1e-15)
    np.testing.assert_allclose(Hx_L, np.r_[-s_L, 0], atol=1e-15)

    # ---------- Plot the results
    fig, ax = plt.subplots(num=1, clear=True)

    # Plot the axes
    ax.axhline(0, color='k', zorder=0)
    ax.axvline(0, color='k', zorder=0)

    # Plot the Householder planes (normal to the reflector)
    ax.axline((0, 0), (-v_D[1], v_D[0]), color='C2', linestyle='--', zorder=0)
    ax.axline((0, 0), (-v_L[1], v_L[0]), color='C0', linestyle='--', zorder=0)

    # Plot the vectors
    plot_vector(x, 'x', color='k')
    plot_vector(v_D, 'v', color='C2')
    plot_vector(v_L, 'v', color='C0')

    # Plot the reflected vectors
    plot_vector(Hx_D, 'Hx', color='C2')
    plot_vector(Hx_L, 'Hx', color='C0')

    # ax.legend()
    AX_LIM = 6
    ax.set(
        xlabel='$e_1$',
        ylabel='$e_2$',
        xlim=(-AX_LIM, AX_LIM),
        ylim=(-AX_LIM, AX_LIM),
        aspect='equal',
    )
    ax.grid(which='both')



# =============================================================================
# =============================================================================
