#!/usr/bin/env python3
# =============================================================================
#     File: householder.py
#  Created: 2025-02-19 10:20
#   Author: Bernie Roesler
#
r"""
Demonstration of the different Householder vector computations.

We show an example of an unstable Householder vector computation when the input
vector `x` is very close to the first unit vector.
We then compare the methods of Davis and LAPACK for computing the Householder
vector. For an input vector close to the first unit vector,

.. math:: x = [1 + \varepsilon, \varepsilon]^T

with :math:`0 < \varepsilon \ll 1`, the Davis method creates a large second
component of the reflector:

.. math::
    \begin{align}
        v &= [1, \frac{-2}{\varepsilon}]^T \\
        \beta &= \frac{\varepsilon^2}{2}
    \end{align}

and a correspondingly miniscule scaling factor. The LAPACK method contains
these values in a more numerically stable way:

.. math::
    \begin{align}
        v &= [1, \frac{\varepsilon}{2}]^T \\
        \beta &= 2,
    \end{align}

so that the scaling factor is essentially a constant, and the second component
of :math:`v` is proportional to :math:`\varepsilon`.
In fact, LAPACK contains :math:`1 \le \beta \le 2` for all inputs except
multiples of the unit vector, when :math:`\beta = 0`.

When the first component of `x` is negative, both methods compute the same
result.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

from pathlib import Path

SAVE_FIGS = False
verbose = False
fig_path = Path('../../plots/')


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

    In the case where :math:`x = \alpha e_1`, the LAPACK method sets
    :math:`\beta = 0` such that :math:`H = I`,
    whereas the Davis method sets :math:`\beta = 0`, if :math:`x_1 > 0`,
    and :math:`\beta = 2` if :math:`x_1 \le 0` to make the direction positive.

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
        The L2-norm of the vector x.

    References
    ----------
    .. [Tref] Trefethen, Lloyd and David Bau (1997).
        "Numerical Linear Algebra". Eq (10.5), and Algorithm 10.1.
    .. [Davis] Davis, Timothy A. (2006).
        "Direct Methods for Sparse Linear Systems", p 69 (`cs_house`).
    .. [GVL] Golub, Gene H. and Charles F. Van Loan (1996).
        "Matrix Computations". Algorithm 5.1.1.
    """
    if method == 'LAPACK':
        return _house_lapack(x)
    elif method == 'Davis':
        return _house_davis(x)
    else:
        raise ValueError(f"Unknown method '{method}'")


def _house_lapack(x):
    """Compute the Householder reflection vector using the LAPACK method."""
    v = np.copy(x)
    σ = np.sum(v[1:]**2)

    if σ == 0:
        s = v[0]  # if β = 0, H is the identity matrix, so Hx = x
        β = 0
        v[0] = 1  # make the reflector a unit vector
    else:
        norm_x = np.sqrt(v[0]**2 + σ)   # ||x||_2
        a = v[0]
        s = -np.sign(a) * norm_x
        β = (s - a) / s
        # v = x + sign(x[0]) * ||x|| e_1
        # v /= v[0]
        v[0] = 1
        v[1:] /= (a - s)  # a - b == x[0] + sign(x[0]) * ||x||

    return v, β, s


def _house_davis(x):
    """Compute the Householder reflection vector using the Davis method."""
    v = np.copy(x)
    σ = np.sum(v[1:]**2)

    if σ == 0:
        s = np.abs(v[0])           # ||x|| consistent with always-positive Hx
        β = 2 if v[0] <= 0 else 0  # make direction positive if x[0] < 0
        v[0] = 1                   # make the reflector a unit vector
    else:
        s = np.sqrt(v[0]**2 + σ)   # ||x||_2

        # These options compute equivalent values, but the v[0] > 0 case
        # is a more numerically stable option.
        v[0] = (v[0] - s) if v[0] <= 0 else (-σ / (v[0] + s))
        β = -1 / (s * v[0])

        # Normalize β and v s.t. v[0] = 1
        β *= v[0] * v[0]
        v /= v[0]

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
        ax.text(v[0] + 0.1, v[1] + 0.3,
                f"{text}: [{v[0]:.2g}, {v[1]:.2g}]",
                bbox={
                    'facecolor': 'white',
                    'alpha': 0.5,
                    'edgecolor': 'none',
                    # pad=5
                },
                **kwargs)

    return ax


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    #         Show numerical *instability* of the naïve method
    # -------------------------------------------------------------------------
    ϵ = 1e-15  # a very small number

    x = np.r_[1 + ϵ, ϵ]  # a vector very close to e_1 == [1, 0]

    v = x - la.norm(x)  # cancellation occurs!
    with np.errstate(divide='ignore', invalid='ignore'):
        v /= v[0]

    np.set_printoptions(suppress=False)
    print("x:", x)
    print("v:")
    print("unstable:", v)  # [nan, -inf] divide by zero!

    # Show numerical *stability* of the Davis method
    # NOTE that Davis method is "stable", but creates a *massive* second
    # component of the reflector [1, -2e+15]!
    v_D, β_D, _ = house(x, method='Davis')
    print("   Davis:", v_D)
    print(f"     β_D: {β_D:.2g}")

    # The LAPACK method is numerically stable and avoids cancellation, so it
    # computes a nicer reflector [1.0, 5e-16]
    v_L, β_L, _ = house(x, method='LAPACK')
    print("  LAPACK:", v_L)
    print(f"     β_L: {β_L:.2g}")

    np.set_printoptions(suppress=True)

    # -------------------------------------------------------------------------
    #         Numerical experiment of v_2 and β vs. ϵ
    # -------------------------------------------------------------------------
    N = 100
    epsilons = np.logspace(-15, 3, N)
    xs = np.c_[1 + epsilons, epsilons]  # (N, 2)

    # Vectorize the computation
    house_vecD = np.vectorize(lambda x: house(x, method='Davis'),
                              signature='(n)->(n),(),()')
    house_vecL = np.vectorize(lambda x: house(x, method='LAPACK'),
                              signature='(n)->(n),(),()')

    vDs_all, βDs, ss = house_vecD(xs)
    vLs_all, βLs,  _ = house_vecL(xs)
    vDs = vDs_all[:, 1]  # just get the 2nd component
    vLs = vLs_all[:, 1]

    # Plot the results
    CD = 'C2'  # color for Davis
    CL = 'C0'  # color for LAPACK

    fig, ax = plt.subplots(num=1, clear=True)
    fig.set_size_inches(5, 8, forward=True)

    ax.axhline(1, color='k', zorder=0)
    ax.plot(epsilons, epsilons, 'k-.', lw=1, label='$\\epsilon$')
    ax.plot(epsilons, ss, 'C3-.', lw=1, label='$||x||_2$')

    ax.plot(epsilons, np.abs(vDs), CD, label='$v_2$ (Davis)')
    ax.plot(epsilons, np.abs(vLs), CL, label='$v_2$ (LAPACK)')
    ax.plot(epsilons, np.abs(βDs), CD, ls='--', label='$\\beta$ (Davis)')
    ax.plot(epsilons, np.abs(βLs), CL, ls='--', label='$\\beta$ (LAPACK)')

    ax.legend(loc='lower right')
    ax.set(
        title=r'Householder reflector of $x = [1 + \epsilon, \epsilon]^T$',
        xscale='log',
        yscale='log',
        xlabel=r'$\epsilon$',
        ylabel=r'$|v_2|$, $|\beta|$',
        aspect='equal'
    )
    ax.grid(which='both')

    if SAVE_FIGS:
        fig.savefig(fig_path / 'householder_stability.pdf')

    # -------------------------------------------------------------------------
    #         Plot a nice example of the vectors and reflectors
    # -------------------------------------------------------------------------
    # Figure for vectors with 0 on y-axis
    fig2, axs2 = plt.subplots(num=2, ncols=2, clear=True)
    fig2.set_size_inches(8, 4, forward=True)

    # Figure for vectors with non-zeros
    fig3, axs3 = plt.subplots(num=3, nrows=2, ncols=2, clear=True)
    fig3.set_size_inches(9.4, 8, forward=True)

    xs = np.array(
        [[ 3.,  0.],
         [-3.,  0.],
         [-3.,  4.],  # both methods give the same result
         [ 3.,  4.],  # |x| == 5
         [-3., -4.],
         [ 3., -4.]]
    )

    for i, x in enumerate(xs):
        # Get the axes for the current plot
        if x[1] == 0:
            ax = axs2.flatten()[i]
        else:
            ax = axs3.flatten()[i-2]

        # Create functions so we can loop over the xs array?
        #   * function to create overall plot and labels given x and ax
        #   * function to plot v and Hx given x and the method name
        #   * testing function to compare the results of the two methods
        #
        # --> v and Hx used for plotting
        # --> β, s, H use for testing only
        v_D, β_D, s_D = house(x, method='Davis')
        v_L, β_L, s_L = house(x, method='LAPACK')

        # Compute the Householder matrices
        I = np.eye(x.size)
        H_D = I - β_D * np.outer(v_D, v_D)
        H_L = I - β_L * np.outer(v_L, v_L)

        Hx_D = H_D @ x
        Hx_L = H_L @ x

        if verbose:
            print("   x:", x)
            print("Hx_D:", Hx_D)
            print("Hx_L:", Hx_L)
            print("β_D:", β_D)
            print("β_L:", β_L)

        # Compare to hand calculations
        atol = 1e-15
        if np.allclose(x, np.r_[3, 4]):
            np.testing.assert_allclose(v_D, np.r_[1, -2])
            np.testing.assert_allclose(v_L, np.r_[1, 0.5])
            np.testing.assert_allclose(β_D, 0.4)
            np.testing.assert_allclose(β_L, 1.6)
            np.testing.assert_allclose(s_D, 5)
            np.testing.assert_allclose(s_L, -5)
            # Need atol when comparing to 0
            np.testing.assert_allclose(H_L, -H_D)
            np.testing.assert_allclose(Hx_D, np.r_[s_D, 0], atol=atol)
            np.testing.assert_allclose(Hx_L, np.r_[s_L, 0], atol=atol)
        elif np.allclose(x, np.r_[-3, 4]):
            np.testing.assert_allclose(v_D, np.r_[1, -0.5])
            np.testing.assert_allclose(v_L, np.r_[1, -0.5])
            np.testing.assert_allclose(β_D, 1.6)
            np.testing.assert_allclose(β_L, 1.6)
            np.testing.assert_allclose(s_D, 5)
            np.testing.assert_allclose(s_L, 5)
            # Need atol when comparing to 0
            np.testing.assert_allclose(H_L, H_D)
            np.testing.assert_allclose(Hx_D, np.r_[s_D, 0], atol=atol)
            np.testing.assert_allclose(Hx_L, np.r_[s_L, 0], atol=atol)

        # ---------- Get the raw LAPACK output for a test vector
        (Qraw, tau), _ = la.qr(np.c_[x], mode='raw')
        v = np.vstack([1.0, Qraw[1:]])
        H = np.eye(x.size) - tau * (v @ v.T)
        Hx = H @ x

        if verbose:
            print("LAPACK (via scipy.linalg.qr(mode='raw')):")
            print("x =", x)
            print("v =", v.flatten())
            print("beta =", tau[0])
            print("Hx =", Hx)

        np.testing.assert_allclose(
            Hx.flatten(),
            # np.r_[-np.sign(x[0])*la.norm(x), np.zeros(x.size - 1)],
            np.r_[Qraw[0, 0], np.zeros(x.size - 1)],
            atol=atol
        )

        # ---------- Plot the results
        # Plot the axes
        ax.axhline(0, color='k', zorder=0)
        ax.axvline(0, color='k', zorder=0)

        # Plot the Householder planes (normal to the reflector)
        ax.axline((0, 0), (-v_D[1], v_D[0]), color=CD, linestyle='--', zorder=0)
        ax.axline((0, 0), (-v_L[1], v_L[0]), color=CL, linestyle='--', zorder=0)

        # Plot the vectors
        plot_vector(x, 'x', ax=ax, color='k')
        plot_vector(v_D, 'v', ax=ax, color=CD, label='Davis')
        plot_vector(v_L, 'v', ax=ax, color=CL, label='LAPACK')

        # Plot the reflected vectors
        plot_vector(Hx_D, 'Hx', ax=ax, color=CD)
        plot_vector(Hx_L, 'Hx', ax=ax, color=CL)

        ss = ax.get_subplotspec()
        if ss.is_first_row() and ss.is_first_col():
            ax.legend(loc='lower left')

        AX_LIM = 6
        ax.set(
            xlabel='$e_1$',
            ylabel='$e_2$',
            xlim=(-AX_LIM, AX_LIM),
            ylim=(-AX_LIM, AX_LIM),
            aspect='equal',
        )
        ax.grid(which='both')

    if SAVE_FIGS:
        fig2.savefig(fig_path / 'householder_demo_y0.pdf')
        fig3.savefig(fig_path / 'householder_demo.pdf')


    # -------------------------------------------------------------------------
    #         Print the table of signs for the reflector
    # -------------------------------------------------------------------------
    # for x in xs:
    #     v_D, β_D, s_D = house(x, method='Davis')
    #     v_L, β_L, s_L = house(x, method='LAPACK')
    #     print(x, v_D, v_L, β_D, β_L, s_D, s_L)

    plt.show()


# =============================================================================
# =============================================================================
