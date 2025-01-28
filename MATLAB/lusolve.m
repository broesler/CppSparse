%!/usr/bin/env python3
% =============================================================================
%     File: lusolve.m
%  Created: 2025-01-21 10:01
%   Author: Bernie Roesler
%
% =============================================================================

% Create the matrices
A = repmat(1:6, 6, 1);
L = tril(A)
U = triu(A)

% Create the permutation vectors/matrices
p = 1 + [5 3 0 1 4 2];
q = 1 + [1 4 0 2 5 3];

P = eye(size(p)(2))(p, :);
Q = eye(size(q)(2))(:, q);

PLQ = P * L * Q
PUQ = P * U * Q

x = [1:6]';

% Create the RHS
bL = L * x
bU = U * x

PbU = PUQ * x
PbL = PLQ * x

assert (L \ bL == x)
assert (U \ bU == x)
assert (PLQ \ PbL == x)
assert (PUQ \ PbU == x)

% =============================================================================
% =============================================================================
% vim: ft=matlab
