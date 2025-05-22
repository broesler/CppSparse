#!/usr/bin/env python3
# =============================================================================
#     File: householder_experiment.py
#  Created: 2025-02-12 16:06
#   Author: Bernie Roesler

# =============================================================================

import matplotlib.pyplot as plt
import numpy as np

from scipy import linalg as la, sparse
from scipy.sparse import linalg as spla

from pathlib import Path

import csparse

SAVE_FIGS = False
data_path = Path('../../data/')
fig_path = Path('../../plots/')

# splu orders: 'NATURAL', 'MMD_ATA', 'MMD_AT_PLUS_A', 'COLAMD'
# reverse_cuthill_mckee: 'RCM'
# MATLAB orders: 'MATLAB_COLAMD', 'MATLAB_QRP'
permc_spec = 'MATLAB_COLAMD'

# -----------------------------------------------------------------------------
#         Load the Data
# -----------------------------------------------------------------------------
filename = Path('west0479')
full_path = data_path / filename

dtypes = np.dtype([
    ('rows', np.int32),
    ('cols', np.int32),
    ('vals', np.float64)
])

data = np.genfromtxt(full_path, delimiter=' ', dtype=dtypes)

A = sparse.coo_array((data['vals'], (data['rows'], data['cols']))).tocsc()

assert A.shape[0] == A.shape[1] == 479
assert A.nnz == 1888

# -----------------------------------------------------------------------------
#         Compute the QR decomposition of A with scipy
# -----------------------------------------------------------------------------
q = slice(None)

match permc_spec:
    case 'MMD_ATA' | 'MMD_AT_PLUS_A' | 'COLAMD':
        # Compute the permutation vector with SuperLU - doesn't work?
        lu = spla.splu(A, permc_spec=permc_spec)
        # p = lu.perm_r
        q = lu.perm_c
    case 'MATLAB_COLAMD':
        # Read the permutation vector from file
        q = np.genfromtxt(data_path / f"{filename}_amdq",
                          delimiter=',', dtype=int)
    case 'MATLAB_QRP':
        # Read the permutation vector from file
        q = np.genfromtxt(data_path / f"{filename}_qrq",
                          delimiter=',', dtype=int)
    case 'RCM':
        q = sparse.csgraph.reverse_cuthill_mckee(A)

# Permute the matrix
Aq = A[:, q].toarray()

# ---------- Get the scipy QR decomposition
Q_, R_ = la.qr(Aq)

np.testing.assert_allclose(Aq, Q_ @ R_, atol=1e-9)

# Get the Householder vectors
(Qraw, tau), _ = la.qr(Aq, mode='raw')
V_ = np.tril(Qraw, -1) + np.eye(Aq.shape[0])

Qr_ = csparse.apply_qright(V_, tau)

np.testing.assert_allclose(Q_, Qr_, atol=1e-9)

# ---------- Get the CSparse QR decomposition
V, beta, R, p, _ = csparse.qr(sparse.csc_array(Aq))

V = V.toarray()
R = R.toarray()

Q = csparse.apply_qright(V, beta, p)

np.testing.assert_allclose(Q @ R, Aq, atol=1e-9)

# ---------- Prep for plotting
# Filter small values
#   NOTE the difference is *not* a thresholding problem! 
#   These are the smallest values in the MATLAB/octave matrices with COLAMD
#   ordering and the proper number of non-zeros:
#
#   octave:5 >> min(abs(Q(Q > 0)))
#   ans = 9.3410e-30
#   octave:6 >> min(abs(V(V > 0)))
#   ans = Compressed Column Sparse (rows = 1, cols = 1, nnz = 1 [100%])
#
#     (1, 1) -> 3.3759e-17
#   octave:7 >> min(abs(R(R > 0)))
#   ans = Compressed Column Sparse (rows = 1, cols = 1, nnz = 1 [100%])
#
#     (1, 1) -> 5.7903e-24
#   octave:8 >> min(abs(Q2(Q2 > 0)))
#   ans = Compressed Column Sparse (rows = 1, cols = 1, nnz = 1 [100%])
#
#     (1, 1) -> 6.5061e-21
#   octave:9 >> min(abs(R2(R2 > 0)))
#   ans = Compressed Column Sparse (rows = 1, cols = 1, nnz = 1 [100%])
#
#     (1, 1) -> 9.8369e-20
#

# tol = np.finfo(float).eps  # ϵ = 2.220446049250313e-16
# Q_[np.abs(Q_) < tol] = 0
# V_[np.abs(V_) < tol] = 0
# R_[np.abs(R_) < tol] = 0

np.testing.assert_allclose(Q_ @ R_, Aq, atol=1e-9)

# Convert to sparse matrices
Aq = sparse.csc_array(Aq)

Q_ = sparse.csc_array(Q_)
V_ = sparse.csc_array(V_)
R_ = sparse.csc_array(R_)

Q = sparse.csc_array(Q)
V = sparse.csc_array(V)
R = sparse.csc_array(R)

# Check that these are correct? Book states "Q 38,070 vs V 3,906"
# It seems like COLAMD is not working correctly. The householder_explicit.m
# file computes nnz values that are similar to the book with q = colamd(A).
print(f"{permc_spec=}")
# print(f"density of A: {A.nnz / (A.shape[0] * A.shape[1]):.4g}")
# print("A.nnz:", A.nnz)

print("Q_.nnz:", Q_.nnz)
print("V_.nnz:", V_.nnz)
print("R_.nnz:", R_.nnz)

print("Q.nnz:", Q.nnz)
print("V.nnz:", V.nnz)
print("R.nnz:", R.nnz)

# With COLAMD ordering from file:
# scipy.linalg.qr numbers are nowhere near MATLAB
# We need a tol = XXX to get the same nnz values
# Q_.nnz: 114811
# V_.nnz: 21791
# R_.nnz: 33403
#
# csparse numbers almost exactly match MATLAB
# Q.nnz: 38074
# V.nnz: 3909
# R.nnz: 7311

# -----------------------------------------------------------------------------
#         Plots
# -----------------------------------------------------------------------------
fig, axs = plt.subplots(num=1, nrows=1, ncols=2, clear=True)
fig.set_size_inches(6.4, 3.8, forward=True)
fig.suptitle(f"Natural vs. {permc_spec} Ordering")

for ax, M, title in zip(axs.flat, [A, Aq], ['A', 'A[:, q]']):
    csparse.cspy(M, norm='log', ax=ax)
    ax.set_title(title)
    ax.set_xlabel(f"nnz = {M.nnz}")

if SAVE_FIGS:
    fig.savefig(fig_path / f"{filename}_{permc_spec}_A.pdf")


fig, axs = plt.subplots(num=2, nrows=1, ncols=2, clear=True)
fig.set_size_inches(6.4, 3.8, forward=True)
fig.suptitle(f"SciPy with {permc_spec} Ordering")

for ax, M, title in zip(axs.flat, [Q_, V_ + R_], ['Q', 'V + R']):
    csparse.cspy(M, norm='log', ax=ax)
    ax.set_title(title)
    ax.set_xlabel(f"nnz = {M.nnz}")


if SAVE_FIGS:
    fig.savefig(fig_path / f"{filename}_{permc_spec}_QR_scipy.pdf")


fig, axs = plt.subplots(num=3, nrows=1, ncols=2, clear=True)
fig.set_size_inches(6.4, 3.8, forward=True)
fig.suptitle(f"csparse with {permc_spec} Ordering")

for ax, M, title in zip(axs.flat, [Q, V + R], ['Q', 'V + R']):
    csparse.cspy(M, norm='log', ax=ax)
    ax.set_title(title)
    ax.set_xlabel(f"nnz = {M.nnz}")


if SAVE_FIGS:
    fig.savefig(fig_path / f"{filename}_{permc_spec}_QR_csparse.pdf")

plt.show()

# =============================================================================
# =============================================================================
