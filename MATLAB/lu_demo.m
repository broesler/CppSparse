%===============================================================================
%     File: lu_demo.m
%  Created: 2025-04-24 20:41
%   Author: Bernie Roesler
%
%  Description: Test LU decomposition and solve Ax = b to compare MATLAB
%    behavior with python and cs_lu.
%
%===============================================================================

clear; close all;

A = davis_example_qr();
N = size(A, 1);

for i = 1:N
    A(i, i) += 10;
end

expect = [1:N]';

p = 1 + [5, 1, 7, 0, 2, 6, 4, 3]';  % arbitrary permutation
Ap = A(p, :);

% Permuted Solve
b = A * expect;
bp = b(p);

assert(norm(bp - Ap * expect) < 1e-14, 'Permutation failed');

[L, U, P] = lu(full(Ap));
Q = eye(N);

assert(norm(L * U - P * Ap) < 1e-14, 'LU decomposition failed');

% Solve Ax = b
Pb = P * bp;
y = L \ Pb;
QTx = U \ y;
x = Q * QTx;

assert(norm(x - expect) < 1e-14, 'LU solve failed');

% Repeat with vector form of permutation
[L, U, pv] = lu(full(Ap), 'vector');

assert(norm(L * U - Ap(pv, :)) < 1e-14, 'LU decomposition failed');

[pv_, ~] = find(P');

assert(norm(pv - pv_) < 1e-14, 'LU permutation failed');


% Test solution with column permutation
[L, U, P, Q] = lu(A);

assert(norm(L * U - P * A * Q) < 1e-14, 'LU decomposition failed');

% Solve Ax = b
Pb = P * b;
y = L \ Pb;
QTx = U \ y;
x = Q * QTx;

assert(norm(x - expect) < 1e-14, 'LU solve failed');

% Repeat with vector form of permutation
[L, U, pv, qv] = lu(A, 'vector');

assert(norm(L * U - A(pv, qv)) < 1e-14, 'LU decomposition failed');

% Solve Ax = b
Pb = b(pv);
y = L \ Pb;
QTx = U \ y;
x = QTx(inv_permute(qv));  % == Q * QTx requires inverse of qv

assert(norm(x - expect) < 1e-14, 'LU solve failed');

%===============================================================================
%===============================================================================
