#!/usr/bin/env python3
# =============================================================================
#     File: fillreducing_example.py
#  Created: 2025-04-23 09:16
#   Author: Bernie Roesler
#
"""Test fill-reducing python functions."""
# =============================================================================

import numpy as np

import csparse

A = csparse.davis_example_amd()

print("A:")
print(A.toarray())

# Compute the Fiedler vector of the graph
p, v, d = csparse.fiedler(A)

print("--- Fiedler vector:")
print(f"   permutation: {p}")
print(f"Fiedler vector: {v}")
print(f"    eigenvalue: {d}")


# Compute an edge separator of A
a, b = csparse.edge_separator(A)

print("--- Edge separator:")
print(f"a: {a}")
print(f"b: {b}")


# Get a node separator from the edge separator
s, a_s, b_s = csparse.node_from_edge_sep(A, a, b)

print("--- Node separator:")
print(f"s: {s}")
print(f"a_s: {a_s}")
print(f"b_s: {b_s}")


# Directly get a node separator
s_, a_s_, b_s_ = csparse.node_separator(A)

print("--- Node separator (direct):")
print(f"s_: {s_}")
print(f"a_s_: {a_s_}")
print(f"b_s_: {b_s_}")

np.testing.assert_allclose(s, s_)
np.testing.assert_allclose(a_s, a_s_)
np.testing.assert_allclose(b_s, b_s_)


# Compute the nested dissection ordering of A
p = csparse.nested_dissection(A)

print("--- Nested dissection ordering:")
print(f"p: {p}")



# =============================================================================
# =============================================================================
